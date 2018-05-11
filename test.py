from torch import nn, optim
from tqdm import tqdm

import config_conv
from models.shared_cnn import CNN
from models import shared_cnn
import data

def main():
    args, unparsed = config_conv.get_args()


    cnn = CNN(args, input_channels=3, height=32, width=32, output_classes=10)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
    print('parameters', list(cnn.parameters()))

    dataset = data.image.Image(args)

    train_iter = iter(dataset.train)


    dag = []
    # last = 0
    for i in range(0, -1 + args.num_blocks):
        dag.append((i, i+1, shared_cnn.conv3x3))
        dag.append((i, i+2, shared_cnn.conv3x3))
        last = i

    for i in tqdm(range(100)):
        optimizer.zero_grad()
        images, labels = train_iter.next()
        outputs = cnn(images, dag, dag)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(images.shape)
        print(loss)
    pass


if __name__ == "__main__":
    main()