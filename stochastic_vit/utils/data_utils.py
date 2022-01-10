import logging
import pandas as pd
import torch
import os 
import webdataset as wds

from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from .folder2lmdb import ImageFolderLMDB
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset


logger = logging.getLogger(__name__)


class TinyImagenetDataset(Dataset):
    def __init__(self, root_dir, categories, transform=None, type="train"):
        self.root_dir       = root_dir          # blahblah/train, val, test/images/
        self.categories     = categories        # [class 0, class 1, ... , class 199] , only for validation.
        self.image_category = {}                # {key = image name : value = its category 0 to 199}
        self.transforms     = transform         # transforms list
        self.type           = type              # train or val

        if type == "train":
            for imagefolder in os.listdir(root_dir):
                value = os.listdir(root_dir).index(imagefolder)
                for imageName in os.listdir(root_dir+imagefolder+'/images/'):
                    self.image_category[imageName] = value
        elif type == "val":
            image_data = pd.read_csv(self.root_dir+'val_annotations.txt', delimiter='\t', header=None)
            for line in image_data.values.tolist():
                self.image_category[line[0]] = self.categories.index(line[1])
        else:
            raise NameError("argument 'type' can be only 'train' or 'val'.")

        self.filename_list  = list(self.image_category.keys())

    def __len__(self):
        return len(self.image_category.keys())

    def __getitem__(self, index):
        if self.type == "train":
            prefix  = self.filename_list[index].split('_')[0]+'/images/'
            imgfile = self.filename_list[index]
        elif self.type == "val":
            prefix  = 'images/'
            imgfile = self.filename_list[index]
        else:
            raise NameError("argument 'type' can be only 'train' or 'val'.")

        img_path = os.path.join(self.root_dir, prefix, imgfile)
        image = Image.open(img_path)
        image = image.convert('RGB')
        if self.transforms :
            image = self.transforms(image)

        #print("sampled : ", self.image_category[imgfile], "index : ", index)

        return image, self.image_category[imgfile]



def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Mean, std of the imagenet 21K dataset
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="./data",
                                   train=False,
                                   download=True,
                                   transform=transform_test) if args.local_rank in [-1, 0] else None

    elif args.dataset == "cifar100":
        trainset = datasets.CIFAR100(root="./data",
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root="./data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) if args.local_rank in [-1, 0] else None
                                    
    elif args.dataset == "TinyImageNet":

        train_dir  = "./data/tiny-imagenet-200/train/"
        val_dir    = "./data/tiny-imagenet-200/val/"
        test_dir   = "./data/tiny-imagenet-200/test/"

        categories = []
        for ImgFolder in os.listdir(train_dir):
            categories.append(ImgFolder)

        trainset =  TinyImagenetDataset(root_dir=train_dir, categories=categories, 
                                        transform=transform_train, type="train")
        testset  =  TinyImagenetDataset(root_dir=val_dir, categories=categories, 
                                        transform=transform_test, type="val") if args.local_rank in [-1, 0] else None
    
    elif args.dataset == "ImageNet-1K":

        train_dir = '/home/seunghoon/data/dataset/imagenet/1K_dataset/train/'
        val_dir   = '/home/seunghoon/data/dataset/imagenet/1K_dataset/val/'
        sharedurl = "/imagenet/imagenet-train-{000000..001281}.tar"

        trainset = ImageFolder(train_dir, transform_train)
        testset  = ImageFolder(val_dir, transform_test) if args.local_rank in [-1, 0] else None


    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None), 
                              batch_size=args.train_batch_size,
                              num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, test_loader
