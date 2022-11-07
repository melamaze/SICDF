'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/Update.py
'''

import torch
import numpy as np
import random
import copy
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Dataset
from ..config import for_FL as f

random.seed(f.seed)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):

        #想看看item是什麼
        #print('item:',item)
        image, label = self.dataset[self.idxs[item]]
        # image: torch.Size([1, 28, 28]), torch.float32; label: int
        return image, label

# class Local_process():

#     def __init__(self, dataset = None, idxs = None, user_idx = None, attack_setting = None):

#         self.dataset = dataset
#         # 我不確定這裡能否用True，但我覺得應該可
#         self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = False)
#         self.user_idx = user_idx

#         self.attack_setting = attack_setting

#         self.attacker_flag = False

#     def split_poison_attackers(self):

#         # 選擇這個user是否為攻擊者(一開始為攻擊者的機率是1，會慢慢減少)
#         attack_or_not = random.choices([1,0],k = 1,weights = [self.attack_setting.attack_or_not, 1 - self.attack_setting.attack_or_not])

#         enough = 0
#         # 有多少label是攻擊目標
#         label_count = 0
#         a = 0

#         # 第幾個batch，裡面的圖和標籤
#         for batch_idx, (images, labels) in enumerate(self.ldr_train):

#             # 對batch中的各個label
#             for label_idx in range(len(labels)):
#                     #如果該label是攻擊目標
#                     label_count += 1

#         # 第幾個batch，裡面的圖和標籤
#         for batch_idx, (images, labels) in enumerate(self.ldr_train):

#             # 目標label的數量，要是該user擁有的最多的那種label
#             # 也就是這個user擁有的目標label得夠多，否則稱不上是攻擊者
#             if((f.dataset == "mnist" or f.dataset == 'fmnist') and label_count >= int(54000 // f.total_users * f.noniid)):
#                 enough = 1
#             else:
#                 # 有可能不夠嗎？
#                 # print('number of label not enough')
#                 pass      
#             # 對batch中的各個label
#             for label_idx in range(len(labels)):
#                 # 若目標label數量夠，且為攻擊目標，且攻擊者的數量還不夠，且這次篩到的是要攻擊
#                 if (enough == 1 and labels[label_idx] in f.target_label) and (self.attack_setting.attacker_num > self.attack_setting.attacker_count) and attack_or_not[0]:
#                         # 設為攻擊者
#                         self.attacker_flag = True

#         return self.attacker_flag



class LocalUpdate_poison(object):

    def __init__(self, dataset = None, idxs = None, user_idx = None, attack_idxs = None):
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size = f.local_bs, shuffle = True)
        self.user_idx = user_idx
        #攻擊者們的id
        self.attack_idxs = attack_idxs
        self.attacker_flag = False

    def train(self, net):
        net.train()
        tmp_pos = 0
        tmp_all = 0
        origin_weights = copy.deepcopy(net.state_dict())
        optimizer = torch.optim.SGD(net.parameters(), lr = f.lr, momentum = f.momentum)

        # local epoch 的 loss
        epoch_loss = []

        for iter in range(f.local_ep):
            batch_loss = []

            # count = 1 # for TEST
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                perm = np.random.permutation(len(labels))[0: int(len(labels) * 0.5)]
                for label_idx in range(len(labels)):
                    # 是攻擊者的話
                    # 以下的code是給錯誤的label
                    # 新題目應該要改成給有 trigger 圖，並label成錯誤的(?
                    tmp_all += 1
                    if (f.attack_mode == 'poison') and (self.user_idx in self.attack_idxs) and label_idx in perm:
                        self.attacker_flag = True
                        labels[label_idx] = f.target_label

                        images[label_idx][0][27][26] = 1.0
                        images[label_idx][0][27][27] = 1.0
                        images[label_idx][0][26][26] = 1.0
                        images[label_idx][0][26][27] = 1.0
                        tmp_pos += 1

                    else:
                        pass


                # CHECK IMAGE
                # if self.user_idx in self.attack_idxs:
                #     print(self.user_idx)
                #     for label_idx in range(len(labels)):
                        # print("label idx: ", label_idx)
                        # print("labels: ", labels[label_idx])
                        # plt.imshow(images[label_idx][0], cmap='gray')
                        # name = "file" + str(count) + ".png"
                        # print(name)
                        # plt.savefig(name)
                        # plt.close()
                        # count += 1



                images, labels = images.to(f.device), labels.to(f.device)

                net.zero_grad()

                # 此圖為哪種圖的各機率
                log_probs = net(images)

                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss)/len(batch_loss))

            if f.local_verbose:
                print('Update Epoch: {} \tLoss: {:.6f}'.format(
                        iter, epoch_loss[iter]))
        # print("ALL: ", tmp_all)
        # print("POS: ", tmp_pos)

        # local training後的模型
        trained_weights = copy.deepcopy(net.state_dict())

        # 有要放大參數的話
        if(f.scale==True):
            scale_up = 20
        else:
            scale_up = 1

        if (f.attack_mode == "poison") and self.attacker_flag:

            attack_weights = copy.deepcopy(origin_weights)

            # 原始net的參數們
            for key in origin_weights.keys():
                # 更新後的參數和原始的差值
                difference =  trained_weights[key] - origin_weights[key]
                # 新的weights
                attack_weights[key] += scale_up * difference

            # 被攻擊的話
            return attack_weights, sum(epoch_loss)/len(epoch_loss), self.attacker_flag

        # 未被攻擊的話
        return net.state_dict(), sum(epoch_loss)/len(epoch_loss), self.attacker_flag

