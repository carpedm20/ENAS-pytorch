import datetime
import gc
import os
import time
import traceback
import types

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

from dag import *
from utils1 import *

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
        with Timer("CNN construct", logger.debug):
            cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=device)
            cnn.train()


        with Timer("Optimizer Construct", logger.debug):
            dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections=cnn.all_connections,
                                     lr=8*25*20, weight_decay=0)
            cnn_optimizer = AdamShared(cnn.parameters(), lr=0.00001, weight_decay=1e-5)

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


    num_batches = 1
    total_loss = 0
    total_acc = 0

    def lengths():
        for i in range(10000):
            yield min(50 + i//100, 250)

        # while True:
        #     yield 500

    iter_lengths = lengths()

    current_length = next(iter_lengths)

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
                for batch_i, (images, labels) in enumerate(dataset.train, 0):
                    outputs = None
                    loss = None
                    try:
                        if batch_i % current_length == 0 and not get_new_network:
                            if num_batches > 0:
                                current_length = next(iter_lengths)

                                gradient_dicts = dropout_opt.step_grad()
                                gradient_dicts = [{k: v/(num_batches*args.batch_size) for k, v in gradient_dict.items()} for gradient_dict in gradient_dicts]
                                cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=0.1)

                                stuff = [datetime.datetime.now(), "train", epoch, batch_i, total_loss / num_batches,
                                         total_acc / num_batches, dags, dags_probs[0].tolist(), dags_probs[1].tolist()]
                                spamwriter.writerow(stuff)
                                logger.info(stuff)

                                if best_acc < total_acc / num_batches:
                                    best_acc = total_acc / num_batches
                                    best_dag = dags
                                csvfile.flush()
                            get_new_network = True

                        if get_new_network:
                            total_acc = 0
                            total_loss = 0
                            num_batches = 0
                            get_new_network = False


                            dags_probs = cnn.get_dags_probs()
                            with Timer("to_cuda", logger.info):
                                dags = sample_fixed_dags(dags_probs, all_connections, args.num_blocks)

                                cnn_optimizer.full_reset_grad()
                                cnn.to_gpu(dags)
                                cnn_optimizer.to_gpu(cnn.get_parameters(dags))
                                dropout_opt.zero_grad()


                        cnn_optimizer.zero_grad()
                        outputs = cnn(images.to_device(device), dags)

                        max_index = outputs.max(dim=1)[1].cpu().numpy()
                        acc = np.sum(max_index == labels.numpy())/labels.shape[0]

                        # outputs = cnn(images)
                        loss = criterion(outputs, labels.to_device(device))

                        loss_value = loss.data.item()
                        loss.backward()
                        cnn_optimizer.step()

                        total_acc += acc
                        total_loss += loss_value
                        num_batches += 1

                        t.update(1)
                        t.set_postfix(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches)

                    except RuntimeError as x:
                        exec = traceback.format_exc()
                        logger.error(x)
                        logger.error(exec)

                        outputs = None
                        loss = None
                        cnn.to_cpu()
                        cnn_optimizer.full_reset_grad()
                        dropout_opt.full_reset_grad()

                        cnn_optimizer.to_cpu()
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
                cnn_optimizer.to_gpu(cnn.get_parameters(best_dag))

                for batch_i, data_it in enumerate(dataset.train, 0):
                    cnn_optimizer.zero_grad()
                    images, labels = data_it
                    outputs = cnn(images.to_device(device), best_dag)

                    max_index = outputs.max(dim=1)[1].cpu().numpy()
                    acc = np.sum(max_index == labels.numpy()) / labels.shape[0]

                    loss = criterion(outputs, labels.to_device(device))

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
                cnn_optimizer.to_cpu()
                cnn.to_gpu(last_dags)
                for images, labels in dataset.test:
                    outputs = cnn(images.to_device(device), last_dags)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.to_device(device)).sum().item()
                    loss = criterion(outputs, labels.to_device(device))
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
            cnn_optimizer.to_cpu()
            cnn_optimizer.full_reset_grad()
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

                                with Timer("to_cuda", logger.info):
                                    dags = sample_fixed_dags(dags_probs, all_connections, args.num_blocks)

                                    logger.info(dags)

                                    cnn.to_gpu(dags)

                                    last_dags = dags

                                get_new_network = False
                                num_dropout_batches_items = 0

                            images, labels = data_it

                            outputs = cnn(images.to_device(device), dags)

                            max_index = outputs.max(dim=1)[1].cpu().numpy()
                            acc = np.sum(max_index == labels.numpy())/labels.shape[0]

                            # outputs = cnn(images)
                            loss = criterion(outputs, labels.to_device(device))

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