from torch import nn, optim
from tqdm import tqdm
import torch

import config_conv
from models.shared_cnn import CNN
from models import shared_cnn
import data

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

def main():
    args, unparsed = config_conv.get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10)
    # cnn = test_model()
    # cnn.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    # print('parameters', list(cnn.parameters()))

    dataset = data.image.Image(args)

    train_iter = iter(dataset.train)


    dag1 = []
    # last = 0
    for i in range(0, -1 + args.num_blocks):
        dag1.append((i, i+1, shared_cnn.conv3x3))
        dag1.append((i, i+2, shared_cnn.conv3x3))


    dag2 = []
    for i in range(0, args.num_blocks):
        dag2.append((i, i+1, shared_cnn.conv3x3))

    cnn.to_cuda(device, dag1, dag1)

    for epoch in range(10):
        print('epoch', epoch)
        for batch_i, data_it in tqdm(enumerate(dataset.train, 0)):
            # optimizer.zero_grad()
            images, labels = data_it

            images, labels = images.to(device), labels.to(device)

            outputs = cnn(images, dag1, dag1)
            # outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # optimizer.step()
            # print(images.shape)
            # print(loss)
    pass


if __name__ == "__main__":
    main()