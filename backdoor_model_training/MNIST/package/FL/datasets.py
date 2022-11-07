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

        if(f.dataset == 'mnist'):
            print('mnist data')
            self.trans_setting = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # gray scale to RGB, since CAM need it change into RGB
            ])
            self.dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=self.trans_setting)
            print(len(self.dataset_train))
            self.dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=self.trans_setting)


    def sampling(self):
        if(f.dataset == 'mnist'):
            self.dict_users, self.idxs_labels = s.iid(self.dataset_train)
