from torchvision import datasets, transforms
from ..config import for_FL as f
from ..utils import sampling as s


class Dataset():

    def __init__(self):
        print('==> Preparing data..')
        # 一個dict，{user_id : 其分配到的圖片們的ids}
        # 其實應該是這樣的定義，但因為sampling.py的奇怪寫法，有點歧異，可以去看一下sampling.py和attacker.py的註解
        # 看完你應該會想改掉的
        self.dict_users = None
        # 一個list，[[圖片ids],[答案ids]]
        self.idxs_labels = None
        # transform setting，數值直接複製網路資料的
        self.trans_setting = None
        self.dataset_train = None
        self.dataset_test = None

        if(f.dataset == 'mnist'):
            print('mnist data')
            self.trans_setting = transforms.Compose([
                transforms.ToTensor(), # 轉為 Tensor
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 灰階轉為 RGB
            ])
            self.dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=self.trans_setting)
            self.dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=self.trans_setting)

    def sampling(self):
        if(f.dataset == 'mnist'):
            self.dict_users, self.idxs_labels = s.my_noniid(self.dataset_train)
