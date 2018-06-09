import datetime
import gc
import json
import os
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import config_conv
import data.image
from adam_shared import AdamShared
from cosine_annealing_with_restarts import CosineAnnealingRestartingLR
from dropout_sgd import DropoutSGD
from models.shared_cnn import CNN

import logging
import shutil

from dag import *
from sgd_shared import SGDShared
from utils1 import *


np.set_printoptions(suppress=True)

def get_logger(save_path: str) -> logging.Logger:
    logger = logging.getLogger('ENAS')
    logger.setLevel(logging.DEBUG)

    error_fh = logging.FileHandler(os.path.join(save_path, 'errors.log'))
    error_fh.setLevel(logging.ERROR)

    debug_fh = logging.FileHandler(os.path.join(save_path, 'all.log'))
    debug_fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_fh.setFormatter(formatter)
    debug_fh.setFormatter(formatter)

    logger.addHandler(error_fh)
    logger.addHandler(debug_fh)
    logger.addHandler(ch)
    return logger


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def numpy_to_json(object):
    return json.dumps(object, cls=NumpyEncoder)


def main():
    load_path = None
    # load_path = "./logs/new/2018-06-07_22-52-20"
    # start_epoch = 35
    start_epoch = None
    # save_path = load_path + "_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_path = "./logs/new/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.makedirs(save_path)
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(save_path, 'src'))

    logger = get_logger(save_path)

    args, unparsed = config_conv.get_args()
    gpu = torch.device("cuda:0")
    cpu = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    max_norm = 200
    max_grad = 0.5

    def create_cnn():
        with Timer("CNN construct", lambda x: logger.debug(x)):
            cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=device)
            cnn.train()

        with Timer("Optimizer Construct", lambda x: logger.debug(x)):
            dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections=cnn.all_connections,
                                     lr=8 * 25 * 5, weight_decay=0)

            dropout_opt.param_groups[0]['lr'] /= 6


            # cnn_optimizer = AdamShared(cnn.parameters(), lr=0.00001, weight_decay=1e-5, gpu=gpu)
            # cnn_optimizer = AdamShared(cnn.parameters(), lr=l_max, weight_decay=1e-4, gpu=gpu)

            #Follows ENAS
            l_max = 0.05
            l_min = 0.001
            period_multiplier = 2
            period_start = 10
            cnn_optimizer = SGDShared(cnn.parameters(),lr = l_max, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True, gpu=gpu)
            learning_rate_scheduler = CosineAnnealingRestartingLR(optimizer=cnn_optimizer, T_max=period_start, eta_min=l_min, period_multiplier=period_multiplier)

        return cnn, cnn_optimizer, dropout_opt, learning_rate_scheduler

    cnn, cnn_optimizer, dropout_opt, learning_rate_scheduler = create_cnn()

    if load_path and start_epoch:
        for i in range(start_epoch):
            learning_rate_scheduler.step()



    weight_decay = 1e-2

    if load_path:
        cnn.load(load_path)

    criterion = nn.CrossEntropyLoss()

    dataset = data.image.Image(args)

    all_connections = list(cnn.cells[0].connections.keys())
    parent_counts = [0] * (2 + args.num_blocks)

    for idx, jdx, _type in all_connections:
        parent_counts[jdx] += 1

    print(parent_counts)

    num_batches = 1
    total_loss = 0
    total_acc = 0

    def lengths():
        for i in range(100000):
            yield 25
            # yield min(50 + i//100, 250)

        # while True:
        #     yield 500

    iter_lengths = lengths()

    current_length = next(iter_lengths)

    test_iter = iter(dataset.test)

    with open(os.path.join(save_path, "logs.json"), 'w', newline='') as jsonfile:
        # spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.csv.QUOTE_NONNUMERIC)

        dropout_opt.zero_grad()
        for epoch in range(1000):
            best_dag = None
            best_acc = 0
            print('epoch', epoch)
            learning_rate_scheduler.step()

            with tqdm(total=2000, desc='train') as t:
                gc.collect()
                get_new_network = True
                for batch_i, (images, labels) in enumerate(dataset.train, 0):
                    outputs = None
                    loss = None
                    try:
                        if batch_i % current_length == 0 and not get_new_network:
                            if num_batches > 0:
                                current_length = next(iter_lengths)

                                # gradient_dicts = dropout_opt.step_grad()
                                # gradient_dicts = [{k: v/(num_batches*args.batch_size) for k, v in gradient_dict.items()} for gradient_dict in gradient_dicts]
                                # cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=0.1)

                                # stuff = [datetime.datetime.now(), "train", epoch, batch_i, total_loss / num_batches,
                                #          total_acc / num_batches, dags, numpy_to_json(dags_probs)]
                                info = dict(time=datetime.datetime.now().isoformat(), type="train", epoch=epoch,
                                            batch_i=batch_i, avg_loss=total_loss / num_batches,
                                            avg_acc=total_acc / num_batches,
                                            dags=dags, dags_probs=dags_probs, learning_rate_scheduler=str(learning_rate_scheduler))

                                jsonfile.write(numpy_to_json(info) + '\n')

                                # spamwriter.writerow(stuff)
                                logger.debug(info)
                                # logger.info()

                                if best_acc < total_acc / num_batches:
                                    best_acc = total_acc / num_batches
                                    best_dag = dags
                                # csvfile.flush()

                                cnn_optimizer.zero_grad()
                                dropout_opt.zero_grad()
                                test_iterations = 25
                                total_acc = 0
                                total_loss = 0
                                # with trange(1, 1+test_iterations, desc='test', leave=False) as t1:
                                for batch_num in range(1, 1 + test_iterations):
                                    try:
                                        img, lab = next(test_iter)
                                    except StopIteration:
                                        test_iter = iter(dataset.test)
                                        img, lab = next(test_iter)

                                    outputs = cnn(img.to(device), dags)
                                    max_index = outputs.max(dim=1)[1].cpu().numpy()
                                    acc = np.sum(max_index == lab.numpy()) / lab.shape[0]
                                    # outputs = cnn(images)
                                    loss = criterion(outputs, lab.to(device))
                                    loss_value = loss.data.item()
                                    loss.backward()

                                    total_acc += acc
                                    total_loss += loss_value

                                    loss = None
                                    outputs = None
                                    img = None
                                    lab = None

                                    # t1.set_postfix(loss=loss_value, acc=acc, avg_loss=total_loss / batch_num,
                                    #               avg_acc=total_acc / batch_num)

                                gradient_dicts = dropout_opt.step_grad()
                                gradient_dicts = [
                                    {k: v / (batch_num * args.batch_size) for k, v in gradient_dict.items()} for
                                    gradient_dict in gradient_dicts]
                                cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=max_grad)

                                cnn_optimizer.full_reset_grad()
                                dropout_opt.full_reset_grad()

                                # stuff = [datetime.datetime.now(), "test-cheat", epoch, batch_i, total_loss / batch_num,
                                #          total_acc / batch_num, dags, numpy_to_json(dags_probs)]

                                info = dict(time=datetime.datetime.now().isoformat(), type="test-cheat", epoch=epoch,
                                            batch_i=batch_i, avg_loss=total_loss / num_batches,
                                            avg_acc=total_acc / num_batches,
                                            dags=dags, dags_probs=dags_probs)
                                # spamwriter.writerow(stuff)
                                jsonfile.write(numpy_to_json(info) + '\n')
                                # logger.info(info)
                                logger.debug(info)
                                logger.info(dict(time=datetime.datetime.now().isoformat(), type="test-cheat", epoch=epoch,
                                            batch_i=batch_i, avg_loss=total_loss / num_batches,
                                            avg_acc=total_acc / num_batches))

                            get_new_network = True

                        if get_new_network:
                            total_acc = 0
                            total_loss = 0
                            num_batches = 0
                            get_new_network = False

                            dags_probs = cnn.get_dags_probs()
                            with Timer("to_cuda", lambda x: logger.info(x)):
                                dags = sample_fixed_dags(dags_probs, all_connections, args.num_blocks)

                                cnn_optimizer.full_reset_grad()
                                cnn.to_gpu(dags)
                                cnn_optimizer.to_gpu(cnn.get_parameters(dags))
                                dropout_opt.full_reset_grad()

                            current_model_parameters = cnn.get_parameters(dags)

                        cnn_optimizer.zero_grad()
                        outputs = cnn(images.to(device), dags)

                        max_index = outputs.max(dim=1)[1].cpu().numpy()
                        acc = np.sum(max_index == labels.numpy()) / labels.shape[0]


                        # outputs = cnn(images)
                        loss = criterion(outputs, labels.to(device))

                        loss_value = loss.data.item()

                        if loss_value > 1e4:
                            logger.error(f"High loss value{loss_value} dag={numpy_to_json(dags)}")
                            continue
                        loss.backward()

                        clip_grad_norm_(current_model_parameters, max_norm=max_norm)
                        cnn_optimizer.step()

                        total_acc += acc
                        total_loss += loss_value
                        num_batches += 1

                        t.update(1)
                        t.set_postfix(loss=loss_value, acc=acc, avg_loss=total_loss / num_batches,
                                      avg_acc=total_acc / num_batches)

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

            with tqdm(total=50) as t:
                gc.collect()
                num_batches = 0
                total_acc = 0
                total_loss = 0

                current_model_parameters = cnn.get_parameters(best_dag)

                cnn.to_gpu(best_dag)
                cnn_optimizer.to_gpu(cnn.get_parameters(best_dag))

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
                    # num_dropout_batches_items += args.batch_size

                    clip_grad_norm_(current_model_parameters, max_norm=max_norm).item()
                    cnn_optimizer.step()

                    total_acc += acc
                    total_loss += loss_value
                    num_batches += 1

                    t.update(1)
                    t.set_postfix(loss=loss_value, acc=acc, avg_loss=total_loss / num_batches,
                                  avg_acc=total_acc / num_batches, lr = learning_rate_scheduler.get_lr())
                    if batch_i >= 50:
                        break

                info = dict(time=datetime.datetime.now().isoformat(), type="train-again",
                            avg_loss=total_loss / num_batches,
                            avg_acc=total_acc / num_batches, best_dag=best_dag)
                # spamwriter.writerow([datetime.datetime.now(), "train-again", total_loss / num_batches, total_acc / num_batches, best_dag])
                jsonfile.write(numpy_to_json(info) + '\n')
                logger.info(info)
            cnn_optimizer.full_reset_grad()
            gc.collect()
            correct = 0
            total = 0
            with torch.no_grad():
                total_loss = 0
                cnn.eval()
                cnn_optimizer.to_cpu()
                cnn.to_gpu(best_dag)
                for images, labels in dataset.test:
                    outputs = cnn(images.to(device), best_dag)
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

            info = dict(time=datetime.datetime.now().isoformat(), type="test-full", avg_loss=total_loss / total,
                        avg_acc=correct / total, best_dag=best_dag)
            jsonfile.write(numpy_to_json(info) + '\n')
            logger.info(info)

            logger.info('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
            jsonfile.flush()

            # gc.collect()
            # dropout_opt.zero_grad()
            # dropout_opt.full_reset_grad()
            # cnn_optimizer.to_cpu()
            # cnn_optimizer.full_reset_grad()
            # for i in range(6):
            #     with tqdm(total=2000) as t:
            #         gc.collect()
            #         get_new_network = True
            #         for batch_i, data_it in enumerate(dataset.test, 0):
            #             loss = None
            #             outputs = None
            #             num_batches = 25
            #             try:
            #                 if batch_i % num_batches == 0 and not get_new_network:
            #                     get_new_network = True
            #
            #                 if get_new_network:
            #
            #                     gradient_dicts = dropout_opt.step_grad()
            #                     gradient_dicts = [
            #                         {k: 2 * v / (args.batch_size * num_batches) for k, v in gradient_dict.items()} for
            #                         gradient_dict in gradient_dicts]
            #                     cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=0.1)
            #
            #                     cnn_optimizer.full_reset_grad()
            #                     dropout_opt.full_reset_grad()
            #
            #                     dags_probs = cnn.get_dags_probs()
            #                     if num_batches > 0:
            #                         stuff =[datetime.datetime.now(), "test-cheating", total_loss/num_batches, total_acc/num_batches, last_dags, dags_probs[0].tolist(), dags_probs[1].tolist()]
            #                         spamwriter.writerow(stuff)
            #                         if best_acc < total_acc/num_batches:
            #                             best_acc = total_acc / num_batches
            #                         csvfile.flush()
            #                     total_acc = 0
            #                     total_loss = 0
            #
            #                     with Timer("to_cuda", logger.info):
            #                         dags = sample_fixed_dags(dags_probs, all_connections, args.num_blocks)
            #
            #                         logger.info(dags)
            #
            #                         cnn.to_gpu(dags)
            #
            #                         last_dags = dags
            #
            #                     get_new_network = False
            #                     num_dropout_batches_items = 0
            #
            #                 images, labels = data_it
            #
            #                 outputs = cnn(images.to_device(device), dags)
            #
            #                 max_index = outputs.max(dim=1)[1].cpu().numpy()
            #                 acc = np.sum(max_index == labels.numpy())/labels.shape[0]
            #
            #                 # outputs = cnn(images)
            #                 loss = criterion(outputs, labels.to_device(device))
            #
            #                 del outputs
            #
            #                 loss_value = loss.data.item()
            #
            #                 loss.backward()
            #                 num_dropout_batches_items += args.batch_size
            #
            #                 del loss
            #
            #                 total_acc += acc
            #                 total_loss += loss_value
            #
            #                 t.update(1)
            #                 t.set_postfix(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches)
            #                 if batch_i % 25 == 24:
            #                     logger.info(to_string(loss = loss_value, acc = acc, avg_loss = total_loss/num_batches, avg_acc = total_acc/num_batches))
            #
            #             except RuntimeError as x:
            #                 logger.error(x)
            #                 exec = traceback.format_exc()
            #                 logger.error(exec)
            #
            #                 outputs = None
            #                 loss = None
            #                 cnn.to_cpu()
            #                 get_new_network = True
            #
            #                 gc.collect()
            #                 torch.cuda.empty_cache()
            #
            #                 #print(f"dag_probs: {dag_probs}")
            #                 #print(f"reduction_dag_probs: {reduction_dag_probs}")
            #
            #     cnn_optimizer.full_reset_grad()
            #     dropout_opt.zero_grad()
            #     dropout_opt.full_reset_grad()


if __name__ == "__main__":
    main()
