import copy
import datetime
import gc
import os
import random
import sys
import time
import traceback

import numpy as np
from scipy.special import expit, logit
from torch import nn, optim
from tqdm import tqdm
import torch

import config_conv
from adam_shared import AdamShared
from dropout_sgd import DropoutSGD
from models.shared_cnn import CNN
from models import shared_cnn
import data
import objgraph

np.set_printoptions(suppress=True)

import csv

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def test_cell():
    return nn.Sequential(
        shared_cnn.conv3x3(64, 64, 1),
        shared_cnn.conv3x3(64, 64, 1),
        shared_cnn.conv3x3(64, 64, 1),
        shared_cnn.conv3x3(64, 64, 1),
        shared_cnn.conv3x3(64, 64, 1)
    )

def test_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3),
        test_cell(),
        shared_cnn.conv3x3(64, 64, 2),
        test_cell(),
        shared_cnn.conv3x3(64, 64, 1),
        test_cell(),
        shared_cnn.conv3x3(64, 64, 2),
        test_cell(),
        shared_cnn.conv3x3(64, 64, 1),
        test_cell(),
        Flatten(),
        nn.Linear(64 * (32//2//2)**2, 10)
    )

def start():
    return time.clock()

def stop(startTime, debugTest=""):
    diff = time.clock() - startTime
    print(f"{debugTest} Diff={diff}")
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
    # load_path = "./test_save_32_eh"
    # load_path = "./test_save_39_new_1"
    load_path = None
    save_path = "./logs/new/test_save_42"

    os.makedirs(save_path)
    args, unparsed = config_conv.get_args()
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_cnn():
        sT = start()
        cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=device)
        cnn.train()
        stop(sT, "CNN")

        sT = start()
        dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections=cnn.all_connections,
                                 lr=8*25*20, weight_decay=0)
        cnn_optimizer = AdamShared(cnn.parameters(), lr=0.00001, weight_decay=1e-5)
        stop(sT, "optimizer")

        return cnn, cnn_optimizer, dropout_opt

    cnn, cnn_optimizer, dropout_opt = create_cnn()

    weight_decay = 5e-1


    # cnn = test_model()
    # cnn.to(device)

    # torch.save(cnn.state_dict(), save_path)
    #
    if load_path:
        cnn.load_state_dict(torch.load(os.path.join(load_path, "cnn_model")))

    criterion = nn.CrossEntropyLoss()

    # sT = start()
    # dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections = cnn.all_connections, lr=0.1, weight_decay=1e-4)
    #
    # # optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    # optimizer = AdamShared(cnn.parameters(), lr=0.0001, weight_decay=1e-5)
    # stop(sT, "optimizer")
    # print('parameters', list(cnn.parameters()))

    dataset = data.image.Image(args)

    # train_iter = iter(dataset.train)

    all_connections = list(cnn.cells[0].connections.keys())
    parent_counts = [0]*(2+args.num_blocks)
    # for jdx in range(2, 2 + args.num_blocks):


    for idx, jdx, _type in all_connections:
        parent_counts[jdx] += 1

    print(parent_counts)

    dag_probs = np.array(list(2/parent_counts[jdx] for idx, jdx, _type in all_connections))
    reduction_dag_probs = np.array(dag_probs)

    # print(dag_probs)



    # dag_logits = logit(np.array([1/len(shared_cnn.CNNCell.default_layer_types)] * len(all_connections)))
    # reduction_dag_logits = np.array(dag_logits)

    # target_ave_prob = 1/len(shared_cnn.CNNCell.default_layer_types)
    target_ave_prob = np.mean(dag_probs)
    print("target_ave_prob", target_ave_prob)


    if load_path:
        dag_logits = np.load(os.path.join(load_path, "dag_logits.npy"))
        reduction_dag_logits = np.load(os.path.join(load_path, "reduction_dag_logits.npy"))
    else:
        dag_logits = logit(dag_probs)
        reduction_dag_logits = logit(reduction_dag_probs)

    def sample(probs):
        random_nums = np.random.rand(len(probs))
        return np.squeeze(np.argwhere(probs >= random_nums))


    def sample_dags(dag_probs, reduction_dag_probs, all_connections):
        dag_idx = sample(dag_probs)
        reduction_dag_idx = sample(reduction_dag_probs)
        if dag_idx.size > 0:
            dag = list(all_connections[i] for i in np.nditer(dag_idx))
        else:
            dag = []

        if reduction_dag_idx.size > 0:
            reduction_dag = list(all_connections[i] for i in np.nditer(reduction_dag_idx))
        else:
            reduction_dag = []
        dag.sort()
        reduction_dag.sort()
        return dag, reduction_dag

    def sample_valid_dag(dag_probs, reduction_dag_probs, all_connections):
        while True:
            dag, reduction_dag = sample_dags(dag_probs, reduction_dag_probs, all_connections)
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

    def sample_fixed_dag(dag_probs, reduction_dag_probs, all_connections):
        while True:
            dag, reduction_dag = sample_dags(dag_probs, reduction_dag_probs, all_connections)
            dag = fix_dag(dag, {0,1}, 1 + args.num_blocks)
            reduction_dag = fix_dag(reduction_dag, {0,1}, 1 + args.num_blocks)
            return dag, reduction_dag


    def sigmoid_derivitive(x):
        return expit(x)*(1.0-expit(x))


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
        # return random_dag

    # random_dag_1 = get_random()
    # random_dag_2 = get_random()
    #
    # print(f'random_dag_1: {random_dag_1}')
    # print(f'random_dag_2: {random_dag_2}')
    # trimmed_random_dag = trim_dag(random_dag, {0, 1}, 1 + args.num_blocks)
    #
    # print(f'random_dag: {random_dag}')
    # print(f'random_dag: {trimmed_random_dag}')
    # print(f'random_dag: {len(random_dag)}, trimmed_random_dag: {len(trimmed_random_dag)}')

    # dag1 = []
    # # last = 0
    # for i in range(2, 2 + args.num_blocks):
    #     # if
    #     dag1.append((i-2, i, shared_cnn.conv3x3))
    #     dag1.append((i-1, i, shared_cnn.conv3x3))
    #
    #
    # dag2 = []
    # for i in range(2, 2 + args.num_blocks):
    #     dag2.append((i - 1, i, shared_cnn.conv3x3))
    #
    # dag3 = []
    # for i in range(2, 2 + args.num_blocks):
    #     dag3.append((i - 1, i, shared_cnn.conv5x5))

    # sT = start()
    # cnn.to_gpu(trimmed_random_dag, trimmed_random_dag)
    # # cnn.to_device(device, dag1, dag1)
    # stop(sT, "to_cuda")
    num_batches = 1
    total_loss = 0
    total_acc = 0

    last_dag = ([], [])

    def lengths():
        for i in range(10000):
            yield min(50 + i//440, 250)

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
                    error = False

                    try:
                        if batch_i % current_length == 0 and not get_new_network:

                            if num_dropout_batches_items == 0:
                                num_dropout_batches_items = 1
                            current_length = next(iter_lengths)
                            # objgraph.show_most_common_types()  # List most common object types
                            # objgraph.show_growth()
                            gradient_dicts = dropout_opt.step_grad()
                            dag_grad_dict, reduction_dag_grad_dict = gradient_dicts[0], gradient_dicts[1]

                            dag_probs, reduction_dag_probs = expit(dag_logits), expit(reduction_dag_logits)
                            current_average_dag = np.mean(dag_probs)
                            current_average_dag_reducing = np.mean(reduction_dag_probs)
                            # print(f"dag_grad_dict {dag_grad_dict}")
                            # print(f"reduction_dag_grad_dict {reduction_dag_grad_dict}")

                            for i, key in enumerate(all_connections):
                                if key in dag_grad_dict:
                                    grad = dag_grad_dict[key]/num_dropout_batches_items - weight_decay*(current_average_dag - target_ave_prob)#*expit(dag_logits[i])
                                    deriv = sigmoid_derivitive(dag_logits[i])

                                    logit_grad = grad * deriv

                                    # print(logit_grad)
                                    dag_logits[i] += np.clip(logit_grad, -0.1, 0.1)

                            print("reduction")
                            for i, key in enumerate(all_connections):
                                if key in reduction_dag_grad_dict:
                                    grad = reduction_dag_grad_dict[key]/num_dropout_batches_items - weight_decay*(current_average_dag_reducing - target_ave_prob)#*expit(reduction_dag_logits[i])
                                    deriv = sigmoid_derivitive(reduction_dag_logits[i])

                                    logit_grad = grad * deriv
                                    # print(logit_grad)
                                    reduction_dag_logits[i] += np.clip(logit_grad, -0.1, 0.1)

                            get_new_network = True

                            dag_probs, reduction_dag_probs = expit(dag_logits), expit(reduction_dag_logits)
                            print(f"dag_probs: {dag_probs}")
                            print(f"reduction_dag_probs: {reduction_dag_probs}")



                        if get_new_network:

                            # print(f"loss: {total_loss/num_batches}")
                            # print(f"acc: {total_acc/num_batches}")
                            if num_batches > 0:
                                stuff =[datetime.datetime.now(), "train", total_loss/num_batches, total_acc/num_batches, last_dag, dag_probs.tolist(), reduction_dag_probs.tolist()]
                                print(stuff)
                                spamwriter.writerow(stuff)
                                if best_acc < total_acc/num_batches:
                                    best_acc = total_acc / num_batches
                                    best_dag = last_dag
                                csvfile.flush()
                            total_acc = 0
                            total_loss = 0
                            num_batches = 0

                            sT = start()

                            dag_probs, reduction_dag_probs = expit(dag_logits), expit(reduction_dag_logits)
                            # dag = sample_valid_dag(dag_probs, reduction_dag_probs, all_connections)
                            dag = sample_fixed_dag(dag_probs, reduction_dag_probs, all_connections)
                            print(dag)

                            cnn_optimizer.full_reset_grad()
                            cnn.to_gpu(dag[0], dag[1])
                            cnn_optimizer.to(cnn.get_parameters(last_dag[0], last_dag[1]), cpu)
                            cnn_optimizer.to(cnn.get_parameters(dag[0], dag[1]), gpu)

                            test_dag = dag

                            last_dag = dag
                            dropout_opt.zero_grad()
                            stop(sT, "to_cuda")
                            get_new_network = False

                            num_dropout_batches_items = 0


                        # if batch_i % 2 == 0:
                        #     cnn.to_gpu(random_dag_1, random_dag_1)
                        #     optimizer.to(cnn.get_parameters(random_dag_2, random_dag_2), cpu)
                        #     # cnn.get_parameters(random_dag_1, random_dag_1)
                        #     # time.sleep(2)
                        #     optimizer.to(cnn.get_parameters(random_dag_1, random_dag_1), gpu)
                        #     dag = random_dag_1
                        # else:
                        #     pass
                        #     cnn.to_gpu(random_dag_2, random_dag_2)
                        #     optimizer.to(cnn.get_parameters(random_dag_1, random_dag_1), cpu)
                        #     optimizer.to(cnn.get_parameters(random_dag_2, random_dag_2), gpu)
                        #     dag = random_dag_2

                        # if batch_i % 3 == 0:
                        #     dag = dag1
                        # elif batch_i % 3 == 1:
                        #     dag = dag2
                        # else:
                        #     dag = dag3
                        # dag = trimmed_random_dag

                        # sT = start()
                        # cnn.to_gpu(dag, dag)
                        # stop(sT, "to_cuda")
                        cnn_optimizer.zero_grad()
                        images, labels = data_it

                        # images, labels = images, labels

                        outputs = cnn(images.to(device), dag[0], dag[1])

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

                        loss = None
                        # print(images.shape)

                        total_acc += acc
                        total_loss += loss_value
                        num_batches += 1

                        # print(type(loss))
                        # print(batch_i)
                        # tqdm.write(f"loss: {loss}")
                        # tqdm.write(f"acc: {acc}")
                        t.update(1)
                        t.set_postfix(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches)
                        if batch_i % 25 == 0:
                            print(total_acc/num_batches)
                        # optimizer.to(cnn.get_parameters(dag, dag), cpu)
                        # optimizer.to(cnn.get_parameters(dag, dag), gpu)
                    except RuntimeError as x:
                        # gc.collect()
                        print(x)
                        traceback.print_exc(file=sys.stdout)
                        error = True
                        # print(torch.cuda.memory_allocated())
                        get_new_network = True

                    if error:

                        # cnn.cpu()
                        # # cnn_state = copy.deepcopy(cnn.state_dict())
                        # del cnn, cnn_optimizer, dropout_opt
                        # while True:
                        #     try:
                        #         gc.collect()
                        #         torch.cuda.empty_cache()
                        #
                        #         cnn, cnn_optimizer, dropout_opt = create_cnn()
                        #     # cnn.load_state_dict(cnn_state)
                        #
                        #         get_new_network = True
                        #         break
                        #     except RuntimeError as x:
                        #         print(x)
                        # error = False

                        # cnn.to(cpu)
                        outputs = None
                        loss = None
                        cnn.to_gpu([], [])
                        cnn_optimizer.full_reset_grad()
                        dropout_opt.full_reset_grad()
                        # cnn_optimizer.to(cnn.parameters(), cpu)
                        cnn_optimizer.to(cnn.get_parameters(last_dag[0], last_dag[1]), cpu)
                        # dropout_opt.to()
                        # cnn_optimizer.full_reset_grad()
                        dropout_opt.zero_grad()
                        get_new_network = True

                        gc.collect()
                        torch.cuda.empty_cache()

                        print(f"dag_probs: {dag_probs}")
                        print(f"reduction_dag_probs: {reduction_dag_probs}")



            torch.save(cnn.state_dict(), os.path.join(save_path, "cnn_model"))
            np.save(os.path.join(save_path, "dag_logits.npy"), dag_logits)
            np.save(os.path.join(save_path, "reduction_dag_logits.npy"), reduction_dag_logits)


            with tqdm(total=500) as t:
                gc.collect()
                num_batches = 0
                total_acc = 0
                total_loss = 0

                cnn.to_gpu(best_dag[0], best_dag[1])
                cnn_optimizer.to(cnn.get_parameters(last_dag[0], last_dag[1]), cpu)
                cnn_optimizer.to(cnn.get_parameters(best_dag[0], best_dag[1]), gpu)

                for batch_i, data_it in enumerate(dataset.train, 0):
                    cnn_optimizer.zero_grad()
                    images, labels = data_it
                    outputs = cnn(images.to(device), best_dag[0], best_dag[1])

                    max_index = outputs.max(dim=1)[1].cpu().numpy()
                    acc = np.sum(max_index == labels.numpy()) / labels.shape[0]

                    loss = criterion(outputs, labels.to(device))

                    del outputs

                    loss_value = loss.data.item()

                    loss.backward()
                    num_dropout_batches_items += args.batch_size
                    cnn_optimizer.step()

                    total_acc += acc
                    total_loss += loss_value
                    num_batches += 1

                    t.update(1)
                    t.set_postfix(loss=loss_value, acc=acc, avg_loss=total_loss / num_batches,
                                  avg_acc=total_acc / num_batches)
                    if batch_i >= 500:
                        break

                spamwriter.writerow([datetime.datetime.now(), "train", total_loss / num_batches, total_acc / num_batches, last_dag, dag_probs.tolist(),
                                     reduction_dag_probs.tolist()])
            last_dag = best_dag
            gc.collect()
            correct = 0
            total = 0
            with torch.no_grad():
                total_loss = 0
                cnn.eval()
                cnn_optimizer.to(cnn.get_parameters(last_dag[0], last_dag[1]), cpu)
                cnn.to_gpu(last_dag[0], last_dag[1])
                for images, labels in dataset.test:
                    outputs = cnn(images.to(device), last_dag[0], last_dag[1])
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to(device)).sum().item()

                    loss = criterion(outputs, labels.to(device))
                    total_loss += loss.data.item()

                cnn.train()
            spamwriter.writerow([datetime.datetime.now(), "test", total_loss / total, correct/total, last_dag, dag_probs.tolist(),
                                  reduction_dag_probs.tolist()])
            csvfile.flush()

            print('Accuracy of the network on the 10000 test images: %d %%' % (
                    100 * correct / total))



if __name__ == "__main__":
    main()