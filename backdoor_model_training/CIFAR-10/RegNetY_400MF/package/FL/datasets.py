from torchvision import datasets, transforms
from ..config import for_FL as f
from ..utils import sampling as s


class Dataset():

    def __init__(self):
        print('==> Preparing data..')
        self.dict_users = None
        self.idxs_labels = None
        self.trans_setting = None
        self.dataset_train = None
        self.dataset_test = None

        if(f.dataset == 'cifar10'):
            print('cifar10 data')
            stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            self.trans_setting = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                         transforms.RandomHorizontalFlip(),

                         transforms.ToTensor(),])
            self.test_setting = transforms.Compose([transforms.ToTensor()])
            self.dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=self.trans_setting)
            self.dataset_test = datasets.CIFAR10('../data', train=False, download=True, transform=self.test_setting)

    def sampling(self):
        if(f.dataset == 'cifar10'):
            self.dict_users, self.idxs_labels = s.iid(self.dataset_train)
