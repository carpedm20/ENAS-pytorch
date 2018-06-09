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

from dag_utils import *
from sgd_shared import SGDShared
from utils import *


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
    # save_path = load_path + "_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = "./logs/new/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    max_norm = 200
    max_grad = 0.5
    weight_decay = 1e-2
    train_length = 25

    os.makedirs(save_path)
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(save_path, 'src'))

    logger = get_logger(save_path)

    args, unparsed = config_conv.get_args()
    gpu = torch.device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_cnn():
        with Timer("CNN construct", lambda x: logger.debug(x)):
            cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=gpu)
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

    if load_path:
        cnn.load(load_path)

    criterion = nn.CrossEntropyLoss()

    dataset = data.image.Image(args)

    num_batches = 1
    total_loss = 0
    total_acc = 0

    test_iter = iter(dataset.test)

    def calculate_loss_info(outputs, labels):
        _, predicted = torch.max(outputs.data, 1)
        num_correct = (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        return loss, num_correct

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
                        if batch_i % train_length == 0 and not get_new_network:
                            if num_batches > 0:
                                info = dict(time=datetime.datetime.now().isoformat(), type="train", epoch=epoch,
                                            batch_i=batch_i, avg_loss=total_loss / num_batches,
                                            avg_acc=total_acc / num_batches,
                                            dags=cnn.cell_dags, dags_probs=cnn.get_dags_probs(),
                                            learning_rate_scheduler=str(learning_rate_scheduler))

                                jsonfile.write(numpy_to_json(info) + '\n')

                                logger.debug(info)

                                if best_acc < total_acc / num_batches:
                                    best_acc = total_acc / num_batches
                                    best_dag = cnn.cell_dags

                                cnn_optimizer.zero_grad()
                                dropout_opt.zero_grad()
                                test_iterations = 25
                                total_acc = 0
                                total_loss = 0
                                for batch_num in range(1, 1 + test_iterations):
                                    try:
                                        img, lab = next(test_iter)
                                    except StopIteration:
                                        test_iter = iter(dataset.test)
                                        img, lab = next(test_iter)

                                    outputs = cnn(dags, img.to(device))
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
                                    {k: v / (test_iterations * args.batch_size) for k, v in gradient_dict.items()} for
                                    gradient_dict in gradient_dicts]
                                cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=max_grad)

                                cnn_optimizer.full_reset_grad()
                                dropout_opt.full_reset_grad()

                                info = dict(time=datetime.datetime.now().isoformat(), type="test-cheat", epoch=epoch,
                                            batch_i=batch_i, avg_loss=total_loss / num_batches,
                                            avg_acc=total_acc / num_batches,
                                            dags=cnn.cell_dags, dags_probs=cnn.get_dags_probs())

                                jsonfile.write(numpy_to_json(info) + '\n')
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
                                dags = sample_fixed_dags(dags_probs, cnn.all_connections, args.num_blocks)

                                cnn_optimizer.full_reset_grad()
                                cnn.set_dags(dags)
                                cnn_optimizer.to_gpu(cnn.get_parameters(dags))
                                dropout_opt.full_reset_grad()

                            current_model_parameters = cnn.get_parameters(dags)

                        cnn_optimizer.zero_grad()
                        outputs = cnn(images.to(device))

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

                cnn.set_dags(best_dag)
                cnn_optimizer.to_gpu(cnn.get_parameters(best_dag))

                for batch_i, data_it in enumerate(dataset.train, 0):
                    cnn_optimizer.zero_grad()
                    images, labels = data_it
                    outputs = cnn(images.to(device))

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
                cnn.set_dags(best_dag)
                for images, labels in dataset.test:
                    outputs = cnn(images.to(device))
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


if __name__ == "__main__":
    main()
