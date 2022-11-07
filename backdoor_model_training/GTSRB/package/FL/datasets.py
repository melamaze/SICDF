from torchvision import datasets, transforms
from ..config import for_FL as f
from ..utils import sampling as s
from PIL import Image


class Dataset():

    def __init__(self):
        print('==> Preparing data..')
        self.dict_users = None
        self.idxs_labels = None
        self.trans_setting = None
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_train_poison = None
        self.dataset_test_poison = None

        if(f.dataset == 'gtsrb'):
            print('gtsrb data')
            stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.trans_setting = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip(),
                         transforms.Resize((32, 32)),
                         
                         transforms.ToTensor(),
                         transforms.Normalize(*stats,inplace=True)])
            self.test_setting = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats), transforms.Resize((32, 32))])
            self.dataset_train = datasets.GTSRB('../data', "train", transform=self.trans_setting, download=True)
            self.dataset_test = datasets.GTSRB('../data', "test", transform=self.test_setting, download=True)

    def sampling(self):
        if(f.dataset == 'gtsrb'):
            self.dict_users, self.idxs_labels = s.cifar_iid(self.dataset_train)
