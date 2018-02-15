import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class Image(object):
    def __init__(self, args):
        if args.datset == 'cifar10':
            Dataset = datasets.CIFAR10
        elif args.datset == 'MNIST':
            Dataset = datasets.MNIST
        else:
            raise NotImplemented(f"Unknown dataset: {args.dataset}")

        self.train = t.utils.data.DataLoader(
            Dataset(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        self.valid = t.utils.data.DataLoader(
            Dataset(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

