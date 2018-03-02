import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Image(object):
    def __init__(self, args):
        if args.datset == 'cifar10':
            Dataset = datasets.CIFAR10

            mean = [0.49139968, 0.48215827, 0.44653124]
            std = [0.24703233, 0.24348505, 0.26158768]

            normalize = transforms.Normalize(mean, std)

            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        elif args.datset == 'MNIST':
            Dataset = datasets.MNIST
        else:
            raise NotImplementedError(f'Unknown dataset: {args.dataset}')

        self.train = t.utils.data.DataLoader(
            Dataset(root='./data', train=True, transform=transform, download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        self.valid = t.utils.data.DataLoader(
            Dataset(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        self.test = self.valid
