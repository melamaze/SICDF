from sklearn import datasets
import torch
from package.config import for_FL as f
from package.FL.models import CNN_Model
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import time

class mydataset(Dataset):

    def __init__(self, dataset):
        self.dataset = self.deal(dataset)
        self.device = f.device

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = img[..., np.newaxis]
        img = torch.Tensor(img).permute(2, 0, 1)
        label = np.zeros(10)
        label[self.dataset[item][1]] = 1
        label = torch.Tensor(label)
        img = img.to(self.device)
        label = label.to(self.device)
        return img, label 

    def deal(self, dataset):
        _data_set = list()
        for i in range(len(dataset)):
            data = dataset[i]
            _img = np.array(data[0])
            _label = data[1]
            _data_set.append((_img, _label))

        return _data_set

    def poison(self, dataset):
        _data_set = list()
        for i in range(len(dataset)):
            data = dataset[i]
            _img = np.array(data[0])
            _img[26][27] = 255
            _img[27][26] = 255
            _img[26][26] = 255
            _img[27][27] = 255
            _data_set.append((_img, f.target_label))
            # print(_img)
            # time.sleep(1)
            # plt.imshow(_img, cmap='gray')
            # name = "file" + str(1) + ".png"
            # plt.savefig(name)
            # plt.close()

        return _data_set

def show(idx):
    # model
    f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')
    net = CNN_Model().to(f.device)
    net.load_state_dict(torch.load("./save_modelglobal_model.pth", map_location = f.device))

    # dataset
    test_data = datasets.MNIST('../data/mnist/', train = False, download = False)
    data = mydataset(test_data)
        
    img = torch.Tensor([data[idx][0].numpy()])
    label = test_data[idx][1]
    output = net(img)

    # plt.imshow(data[idx][0][0], cmap='gray')
    # name = "file" + str(1) + ".png"
    # plt.savefig(name)
    # plt.close()
    output = torch.argmax(output, dim = 1)
    print("real label %d, predict label %d" % (label, output))

if __name__ == "__main__":
    show(168)