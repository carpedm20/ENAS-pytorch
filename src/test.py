import datetime
import gc
import os
import time
import traceback

import numpy as np
import torch
from models import shared_cnn
from models.shared_cnn import CNN
from scipy.special import logit
from torch import nn
from tqdm import tqdm

import config_conv
from adam_shared import AdamShared
from dropout_sgd import DropoutSGD
import data.image

np.set_printoptions(suppress=True)

import csv

import logging
import shutil

def get_logger(save_path):
    logger = logging.getLogger('ENAS')
    logger.setLevel(logging.DEBUG)

    errorFH = logging.FileHandler(os.path.join(save_path,'errors.log'))
    errorFH.setLevel(logging.ERROR)

    debugFH = logging.FileHandler(os.path.join(save_path,'all.log'))
    debugFH.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    errorFH.setFormatter(formatter)
    debugFH.setFormatter(formatter)

    logger.addHandler(errorFH)
    logger.addHandler(debugFH)
    logger.addHandler(ch)
    return logger

def to_string(**kwargs):
    return ' '.join([f"{k}:{v}" for k, v in kwargs.items()])

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def start():
    return time.clock()

def stop(startTime, debugTest="", logger=None):
    diff = time.clock() - startTime
    message = f"{debugTest} Diff={diff}"
    if logger:
        logger.info(message)
    else:
        print(message)
    return diff

def trim_dag_one_way(dag: list, start_nodes: set, forward = True):
    start_nodes = set(start_nodes)
    if forward:
        start_index, end_index = 0, 1
    else:
        start_index, end_index = 1, 0
    new_dag = []
    while start_nodes:
        current = start_nodes.pop()
        connections = list(conn for conn in dag if conn[start_index] == current)
        for conn in connections:
            start_nodes.add(conn[end_index])
        new_dag.extend(connections)
    return new_dag


def trim_dag(dag: list, start_nodes: set, output_node=None):
    new_dag = trim_dag_one_way(dag, start_nodes, True)

    if output_node and not any(conn[1] == output_node for conn in new_dag):
        raise RuntimeError("Invalid dag")

    if output_node:
        new_dag = trim_dag_one_way(new_dag, {output_node}, False)

    new_dag.sort(key = lambda x: (x[0], x[1]))
    return new_dag

def main():
    load_path = None
    save_path = "./logs/new/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(save_path)
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(save_path, 'src'))

    logger = get_logger(save_path)

    args, unparsed = config_conv.get_args()
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_cnn():
        sT = start()
        cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=device)
        cnn.train()
        stop(sT, "CNN", logger)

        sT = start()
        dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections=cnn.all_connections,
                                 lr=8*25*20, weight_decay=0)
        cnn_optimizer = AdamShared(cnn.parameters(), lr=0.00001, weight_decay=1e-5)
        stop(sT, "optimizer", logger)

        return cnn, cnn_optimizer, dropout_opt

    cnn, cnn_optimizer, dropout_opt = create_cnn()

    weight_decay = 5e-1

    if load_path:
        cnn.load(load_path)

    criterion = nn.CrossEntropyLoss()

    dataset = data.image.Image(args)

    all_connections = list(cnn.cells[0].connections.keys())
    parent_counts = [0]*(2+args.num_blocks)

    for idx, jdx, _type in all_connections:
        parent_counts[jdx] += 1

    print(parent_counts)

    def sample(probs):
        random_nums = np.random.rand(len(probs))
        return np.squeeze(np.argwhere(probs >= random_nums))


    def sample_dags(dag_probs, all_connections):
        dag_idx = sample(dag_probs)
        # reduction_dag_idx = sample(reduction_dag_probs)
        if dag_idx.size > 0:
            dag = list(all_connections[i] for i in np.nditer(dag_idx))
        else:
            dag = []

        dag.sort()
        return dag

    def sample_valid_dag(dag_probs, reduction_dag_probs, all_connections):
        while True:
            dag, reduction_dag = sample_dags(dag_probs, all_connections), sample_dags(reduction_dag_probs, all_connections)
            try:
                dag = trim_dag(dag, {0,1}, 1 + args.num_blocks)
                reduction_dag = trim_dag(reduction_dag, {0,1}, 1 + args.num_blocks)
                return dag, reduction_dag
            except RuntimeError:
                print(f"Invalid Dags {dag}, {reduction_dag}")
                print(f"probs: {dag_probs}, {reduction_dag_probs}")
                continue

    def fix_dag_one_way(dag: list, start_nodes: set, forward=True):
        new_dag = list(dag)
        start_nodes = set(start_nodes)
        if forward:
            start_index, end_index = 0, 1
        else:
            start_index, end_index = 1, 0

        dag = list(dag)
        dag.sort(key=lambda x: (x[start_index], x[end_index]))
        if not forward:
            dag.reverse()

        # new_dag = []
        for connection in dag:
            start = connection[start_index]
            end = connection[end_index]


            if start not in start_nodes:
                if forward:
                    begin = max(node for node in start_nodes if node < start)
                else:
                    begin = min(node for node in start_nodes if node > start)
                new_conn = [None, None, None]
                new_conn[start_index] = begin
                new_conn[end_index] = start
                new_conn[2] = shared_cnn.identity.__name__
                new_dag.append(tuple(new_conn))
                start_nodes.add(start)

            start_nodes.add(end)

        new_dag.sort(key=lambda x: (x[0], x[1]))
        return new_dag

    def fix_dag(dag: list, start_nodes: set, output_node=None):
        dag.sort(key=lambda x: (x[0], x[1]))
        new_dag = fix_dag_one_way(dag, start_nodes, True)

        if output_node:
            new_dag = fix_dag_one_way(new_dag, {output_node}, False)

        new_dag.sort(key=lambda x: (x[0], x[1]))

        if not new_dag:
            return [(max(start_nodes), output_node, shared_cnn.identity.__name__)]

        return new_dag

    def sample_fixed_dag(dag_probs, all_connections):
        dag = sample_dags(dag_probs, all_connections)
        dag = fix_dag(dag, {0, 1}, 1 + args.num_blocks)
        return dag

    def sample_fixed_dags(dags_probs, all_connections):
        return [sample_fixed_dag(dag_probs, all_connections=all_connections) for dag_probs in dags_probs]

    def get_random():
        random_dag = []
        # random_dag.append((0, 1, np.random.choice(shared_cnn.CNNCell.default_layer_types)))
        for i in range(2, 2+args.num_blocks):
            # previous_layers = np.random.choice(possible_connections, 2, replace=True)
            possible_connections = list(connection for connection in all_connections if connection[1] == i)
            # print(possible_connections)
            ids = np.random.choice(len(possible_connections), 2, replace=True)
            for id in ids:
                random_dag.append(possible_connections[id])

        return trim_dag(random_dag, {0, 1}, 1 + args.num_blocks)

    num_batches = 1
    total_loss = 0
    total_acc = 0

    last_dags = ([], [])

    def lengths():
        for i in range(10000):
            yield min(50 + i//100, 250)

        # while True:
        #     yield 500

    iter_lengths = lengths()

    current_length = next(iter_lengths)
    num_dropout_batches_items = 1

    with open(os.path.join(save_path, "log.csv"), 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        dropout_opt.zero_grad()
        for epoch in range(1000):
            best_dag = None
            best_acc = 0
            print('epoch', epoch)

            with tqdm(total=2000) as t:
                gc.collect()
                get_new_network = True
                for batch_i, data_it in enumerate(dataset.train, 0):
                    # if batch_i == 100:
                    #     break
                    error = False

                    try:
                        if batch_i % current_length == 0 and not get_new_network:
                            if num_dropout_batches_items > 0:
                                current_length = next(iter_lengths)

                                gradient_dicts = dropout_opt.step_grad()
                                gradient_dicts = [{k: v/num_dropout_batches_items for k, v in gradient_dict.items()} for gradient_dict in gradient_dicts]

                                cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=0.1)

                            get_new_network = True

                        if get_new_network:
                            dags_probs = cnn.get_dags_probs()
                            if num_batches > 0:
                                stuff =[datetime.datetime.now(), "train", total_loss/num_batches, total_acc/num_batches, last_dags, dags_probs[0].tolist(), dags_probs[1].tolist()]
                                spamwriter.writerow(stuff)
                                if best_acc < total_acc/num_batches:
                                    best_acc = total_acc / num_batches
                                    best_dag = last_dags
                                csvfile.flush()
                            total_acc = 0
                            total_loss = 0
                            num_batches = 0

                            sT = start()
                            dags = sample_fixed_dags(dags_probs, all_connections)

                            logger.info(dags)

                            cnn_optimizer.full_reset_grad()
                            cnn.to_gpu(dags)
                            cnn_optimizer.to(cnn.get_parameters(last_dags), cpu)
                            cnn_optimizer.to(cnn.get_parameters(dags), gpu)

                            last_dags = dags
                            dropout_opt.zero_grad()
                            stop(sT, "to_cuda", logger)
                            get_new_network = False
                            num_dropout_batches_items = 0

                        cnn_optimizer.zero_grad()
                        images, labels = data_it

                        outputs = cnn(images.to(device), dags)

                        max_index = outputs.max(dim=1)[1].cpu().numpy()
                        acc = np.sum(max_index == labels.numpy())/labels.shape[0]

                        # outputs = cnn(images)
                        loss = criterion(outputs, labels.to(device))

                        del outputs


                        loss_value = loss.data.item()

                        # if loss_value > 1000:
                        #     get_new_network = True
                        #     continue

                        loss.backward()
                        num_dropout_batches_items += args.batch_size
                        cnn_optimizer.step()

                        del loss
                        # print(images.shape)

                        total_acc += acc
                        total_loss += loss_value
                        num_batches += 1

                        t.update(1)
                        t.set_postfix(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches)
                        if batch_i % 25 == 0:
                            logger.info(to_string(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches))

                    except RuntimeError as x:
                        logger.error(x)
                        exec = traceback.format_exc()
                        logger.error(exec)

                        outputs = None
                        loss = None
                        cnn.to_cpu()
                        cnn_optimizer.full_reset_grad()
                        dropout_opt.full_reset_grad()
                        # cnn_optimizer.to(cnn.parameters(), cpu)
                        cnn_optimizer.to(cnn.get_parameters(last_dags), cpu)
                        # dropout_opt.to()
                        # cnn_optimizer.full_reset_grad()
                        dropout_opt.zero_grad()
                        get_new_network = True

                        gc.collect()
                        torch.cuda.empty_cache()

            cnn.save(save_path)
            cnn_optimizer.full_reset_grad()
            dropout_opt.full_reset_grad()


            with tqdm(total=250) as t:
                gc.collect()
                num_batches = 0
                total_acc = 0
                total_loss = 0

                cnn.to_gpu(best_dag)
                cnn_optimizer.to(cnn.get_parameters(last_dags), cpu)
                cnn_optimizer.to(cnn.get_parameters(best_dag), gpu)

                for batch_i, data_it in enumerate(dataset.train, 0):
                    cnn_optimizer.zero_grad()
                    images, labels = data_it
                    outputs = cnn(images.to(device), best_dag)

                    max_index = outputs.max(dim=1)[1].cpu().numpy()
                    acc = np.sum(max_index == labels.numpy()) / labels.shape[0]

                    loss = criterion(outputs, labels.to(device))

                    del outputs

                    loss_value = loss.data.item()

                    loss.backward()
                    del loss
                    num_dropout_batches_items += args.batch_size
                    cnn_optimizer.step()

                    total_acc += acc
                    total_loss += loss_value
                    num_batches += 1

                    t.update(1)
                    t.set_postfix(loss=loss_value, acc=acc, avg_loss=total_loss / num_batches,
                                  avg_acc=total_acc / num_batches)
                    if batch_i >= 250:
                        break

                spamwriter.writerow([datetime.datetime.now(), "train", total_loss / num_batches, total_acc / num_batches, last_dags])
            cnn_optimizer.full_reset_grad()
            last_dags = best_dag
            gc.collect()
            correct = 0
            total = 0
            with torch.no_grad():
                total_loss = 0
                cnn.eval()
                cnn_optimizer.to(cnn.get_parameters(last_dags), cpu)
                cnn.to_gpu(last_dags)
                for images, labels in dataset.test:
                    outputs = cnn(images.to(device), last_dags)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()
                    loss = criterion(outputs, labels.to(device))
                    del outputs

                    del _
                    del predicted

                    total_loss += loss.data.item()
                    del loss

                cnn.train()
            spamwriter.writerow([datetime.datetime.now(), "test", total_loss / total, correct/total, last_dags])
            csvfile.flush()
            logger.info('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

            gc.collect()
            dropout_opt.zero_grad()
            dropout_opt.full_reset_grad()
            cnn_optimizer.to(cnn.get_parameters(last_dags), cpu)
            for i in range(6):
                with tqdm(total=2000) as t:
                    gc.collect()
                    get_new_network = True
                    for batch_i, data_it in enumerate(dataset.test, 0):
                        loss = None
                        outputs = None
                        num_batches = 25
                        try:
                            if batch_i % num_batches == 0 and not get_new_network:
                                get_new_network = True

                            if get_new_network:

                                gradient_dicts = dropout_opt.step_grad()
                                gradient_dicts = [
                                    {k: 2 * v / (args.batch_size * num_batches) for k, v in gradient_dict.items()} for
                                    gradient_dict in gradient_dicts]
                                cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=0.1)

                                cnn_optimizer.full_reset_grad()
                                dropout_opt.full_reset_grad()

                                dags_probs = cnn.get_dags_probs()
                                if num_batches > 0:
                                    stuff =[datetime.datetime.now(), "test-cheating", total_loss/num_batches, total_acc/num_batches, last_dags, dags_probs[0].tolist(), dags_probs[1].tolist()]
                                    spamwriter.writerow(stuff)
                                    if best_acc < total_acc/num_batches:
                                        best_acc = total_acc / num_batches
                                    csvfile.flush()
                                total_acc = 0
                                total_loss = 0

                                sT = start()

                                dags = sample_fixed_dags(dags_probs, all_connections)

                                logger.info(dags)

                                cnn.to_gpu(dags)

                                last_dags = dags
                                stop(sT, "to_cuda", logger)
                                get_new_network = False
                                num_dropout_batches_items = 0

                            images, labels = data_it

                            outputs = cnn(images.to(device), dags)

                            max_index = outputs.max(dim=1)[1].cpu().numpy()
                            acc = np.sum(max_index == labels.numpy())/labels.shape[0]

                            # outputs = cnn(images)
                            loss = criterion(outputs, labels.to(device))

                            del outputs

                            loss_value = loss.data.item()

                            loss.backward()
                            num_dropout_batches_items += args.batch_size

                            del loss

                            total_acc += acc
                            total_loss += loss_value

                            t.update(1)
                            t.set_postfix(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches)
                            if batch_i % 25 == 24:
                                logger.info(to_string(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches))

                        except RuntimeError as x:
                            logger.error(x)
                            exec = traceback.format_exc()
                            logger.error(exec)

                            outputs = None
                            loss = None
                            cnn.to_cpu()
                            get_new_network = True

                            gc.collect()
                            torch.cuda.empty_cache()

                            #print(f"dag_probs: {dag_probs}")
                            #print(f"reduction_dag_probs: {reduction_dag_probs}")

                cnn_optimizer.full_reset_grad()
                dropout_opt.zero_grad()
                dropout_opt.full_reset_grad()




if __name__ == "__main__":
    main()