import datetime
import gc
import json
import logging
import os
import shutil
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import config_conv
import data.image
import dag_utils
from models.shared_cnn import CNN
from optim.cosine_annealing_with_restarts import CosineAnnealingRestartingLR
from optim.dropout_sgd import DropoutSGD
from optim.sgd_shared import SGDShared
import utils

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


def cnn_train_step(cnn, criterion, images, labels, backprop=True):
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    loss = criterion(outputs, labels)
    loss_value = loss.data.item()
    acc = (predicted == labels).sum().item() / labels.shape[0]
    if backprop:
        loss.backward()

    return loss_value, acc


def cnn_train(cnn, criterion, data_iter, max_batches, device, cnn_optimizer=None, current_model_parameters=None,
              max_grad_norm=None, backprop=True):
    total_loss = 0
    total_acc = 0
    num_batches = 0
    iter_ended = False
    with tqdm(total=max_batches, desc='cnn_train-' + str(backprop)) as t:
        try:
            for i in range(max_batches):
                if cnn_optimizer:
                    cnn_optimizer.zero_grad()
                images, labels = next(data_iter)
                loss, acc = cnn_train_step(cnn, criterion, images.to(device), labels.to(device), backprop=backprop)

                if cnn_optimizer:
                    if max_grad_norm:
                        clip_grad_norm_(current_model_parameters, max_norm=max_grad_norm)
                    cnn_optimizer.step()

                total_loss += loss
                total_acc += acc
                num_batches += 1

                avg_loss = total_loss / num_batches
                avg_acc = total_acc / num_batches

                t.update(1)
                t.set_postfix(loss=loss, acc=acc, avg_loss=avg_loss, avg_acc=avg_acc)
        except StopIteration:
            iter_ended = True

    return avg_loss, avg_acc, num_batches, iter_ended


def main():
    load_path = None
    # save_path = load_path + "_"+ datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = "./logs/new/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    max_norm = 200
    max_grad = 0.6
    weight_decay = 1e-2
    train_length = 25

    os.makedirs(save_path)
    shutil.copytree(os.path.dirname(os.path.realpath(__file__)), os.path.join(save_path, 'src'))

    logger = get_logger(save_path)

    args, unparsed = config_conv.get_args()
    gpu = torch.device("cuda:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def create_cnn():
        with utils.Timer("CNN construct", lambda x: logger.debug(x)):
            cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10, gpu=gpu)
            cnn.train()

        with utils.Timer("Optimizer Construct", lambda x: logger.debug(x)):
            dropout_opt = DropoutSGD([cnn.dag_variables, cnn.reducing_dag_variables], connections=cnn.all_connections,
                                     lr=8 * 25 * 10, weight_decay=0)

            dropout_opt.param_groups[0]['lr'] /= 6

            # cnn_optimizer = AdamShared(cnn.parameters(), lr=0.00001, weight_decay=1e-5, gpu=gpu)
            # cnn_optimizer = AdamShared(cnn.parameters(), lr=l_max, weight_decay=1e-4, gpu=gpu)

            # Follows ENAS
            l_max = 0.05
            l_min = 0.001
            period_multiplier = 2
            period_start = 10
            cnn_optimizer = SGDShared(cnn.parameters(), lr=l_max, momentum=0.9, dampening=0, weight_decay=1e-4,
                                      nesterov=True, gpu_device=gpu)
            learning_rate_scheduler = CosineAnnealingRestartingLR(optimizer=cnn_optimizer, T_max=period_start,
                                                                  eta_min=l_min, period_multiplier=period_multiplier)

        return cnn, cnn_optimizer, dropout_opt, learning_rate_scheduler

    cnn, cnn_optimizer, dropout_opt, learning_rate_scheduler = create_cnn()

    if load_path:
        cnn.load(load_path)

    criterion = nn.CrossEntropyLoss()

    dataset = data.image.Image(args)

    num_batches = 1
    total_loss = 0
    total_acc = 0

    test_dataset_iter = iter(dataset.test)

    with open(os.path.join(save_path, "logs.json"), 'w', newline='') as jsonfile:
        # spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.csv.QUOTE_NONNUMERIC)

        dropout_opt.zero_grad()
        for epoch in range(1000):
            best_dags = None
            best_acc = 0
            print('epoch', epoch)
            learning_rate_scheduler.step()

            train_dataset_iter = iter(dataset.train)
            iter_ended = False

            train_batch_num = 0
            while not iter_ended:
                gc.collect()
                cnn_optimizer.full_reset_grad()
                dropout_opt.full_reset_grad()

                try:
                    # Get new Network
                    dags_probs = cnn.get_dags_probs()
                    with utils.Timer("to_cuda", lambda x: logger.info(x)):
                        dags = dag_utils.sample_fixed_dags(dags_probs, cnn.all_connections, args.num_blocks)
                        cnn_optimizer.full_reset_grad()
                        cnn.set_dags(dags)
                        cnn_optimizer.to_gpu(cnn.get_parameters(dags))
                        dropout_opt.full_reset_grad()
                    current_model_parameters = cnn.get_parameters(dags)

                    # Train on Training set
                    avg_loss, avg_acc, num_batches, iter_ended = cnn_train(cnn, criterion, train_dataset_iter,
                                                                           train_length,
                                                                           gpu, cnn_optimizer,
                                                                           current_model_parameters=current_model_parameters,
                                                                           backprop=True)
                    info = dict(time=datetime.datetime.now().isoformat(), type="train", epoch=epoch,
                                train_batch_num=train_batch_num, avg_loss=avg_loss,
                                avg_acc=avg_acc,
                                dags=cnn.cell_dags, dags_probs=cnn.get_dags_probs(),
                                learning_rate_scheduler=str(learning_rate_scheduler))

                    jsonfile.write(numpy_to_json(info) + '\n')
                    logger.debug(info)

                    # Train on Testing set
                    dropout_opt.zero_grad()
                    avg_loss, avg_acc, num_batches, test_iter_ended = cnn_train(cnn, criterion, test_dataset_iter,
                                                                                train_length,
                                                                                gpu, cnn_optimizer=None,
                                                                                current_model_parameters=None,
                                                                                backprop=True)
                    if test_iter_ended:
                        test_dataset_iter = iter(dataset.test)

                    if best_acc < avg_acc:
                        best_acc = avg_acc
                        best_dags = cnn.cell_dags

                    info = dict(time=datetime.datetime.now().isoformat(), type="test-cheat", epoch=epoch,
                                train_batch_num=train_batch_num, avg_loss=avg_loss,
                                avg_acc=avg_acc, dags=cnn.cell_dags, dags_probs=cnn.get_dags_probs())

                    jsonfile.write(numpy_to_json(info) + '\n')
                    logger.info(info)

                    # Dropout updates
                    gradient_dicts = dropout_opt.step_grad()
                    gradient_dicts = [
                        {k: v / (train_length * num_batches) for k, v in gradient_dict.items()} for
                        gradient_dict in gradient_dicts]
                    cnn.update_dag_logits(gradient_dicts, weight_decay=weight_decay, max_grad=max_grad)

                except RuntimeError as x:
                    gc.collect()
                    exec = traceback.format_exc()
                    logger.error(x)
                    logger.error(exec)

                    cnn.set_dags()
                    cnn_optimizer.full_reset_grad()
                    dropout_opt.full_reset_grad()
                    cnn_optimizer.to_cpu()
                    torch.cuda.empty_cache()

            # Save cnn
            cnn.save(save_path)

            # Train best dag again and test on full testing dataset
            cnn.set_dags(best_dags)
            current_model_parameters = cnn.get_parameters(best_dags)

            train_dataset_iter = iter(dataset.train)
            avg_loss, avg_acc, _, _ = cnn_train(cnn, criterion, train_dataset_iter, train_length, gpu, cnn_optimizer,
                                                current_model_parameters, max_grad_norm=max_grad, backprop=True)
            del train_dataset_iter
            info = dict(time=datetime.datetime.now().isoformat(), type="train-again",
                        avg_loss=avg_loss, avg_acc=avg_acc, best_dag=best_dags)
            jsonfile.write(numpy_to_json(info) + '\n')
            logger.info(info)

            cnn_optimizer.full_reset_grad()
            dropout_opt.full_reset_grad()
            gc.collect()

            #Full test on testing dataset
            cnn.eval()
            with torch.no_grad():
                avg_loss, avg_acc, _, _ = cnn_train(cnn, criterion, iter(dataset.test), None, gpu, None, None, None,
                                                    backprop=False)
            cnn.train()

            info = dict(time=datetime.datetime.now().isoformat(), type="test-full", avg_loss=avg_loss,
                        avg_acc=avg_acc, best_dag=best_dags)
            jsonfile.write(numpy_to_json(info) + '\n')
            logger.info(info)

            logger.info('Accuracy of the network on the 10000 test images: %d %%' % (avg_acc))
            jsonfile.flush()


if __name__ == "__main__":
    main()
