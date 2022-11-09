'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/models/test.py
'''
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ..config import for_FL as f
import numpy as np
from PIL import Image
from torchvision import transforms

f.device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')

def test_img_poison(net, datatest):

    net.eval()
    test_loss = 0
    if f.dataset == "mnist":
        # number of correct prediction
        # SEPERATE INTO TWO CASE: 1. normal dataset(without poison) 2. poison dataset(all poison)
        correct  = torch.tensor([0.0] * 10)
        correct_pos = torch.tensor([0.0] * 10)
        correct_train = torch.tensor([0.0] * 10)
        # number of picture
        gold_all = torch.tensor([0.0] * 10)
        gold_all_pos = torch.tensor([0.0] * 10)
        gold_all_train = torch.tensor([0.0] * 10)
    else:
        print("Unknown dataset")
        exit(0)

    poison_correct = 0.0

    data_ori_loader = DataLoader(datatest, batch_size=f.test_bs)
    data_pos_loader = DataLoader(datatest, batch_size=f.test_bs)
    data_train_loader = DataLoader(datatest, batch_size=f.test_bs)

    # transform setting for data
    TOPIL = transforms.ToPILImage()
    TOtensor = transforms.ToTensor()
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    Normal = transforms.Normalize(*stats,inplace=True)

    print(' test data_loader(per batch size):',len(data_ori_loader))
    
    # FIRST TEST: normal dataset
    for idx, (data, target) in enumerate(data_ori_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)
        ## NORMAL ##
        for label_idx in range(len(target)):
            Normal(data[label_idx])       

        log_probs = net(data)
        # predict
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        # correct
        y_gold = target.data.view_as(y_pred).squeeze(1)
        
        y_pred = y_pred.squeeze(1)


        for pred_idx in range(len(y_pred)):
            gold_all[ y_gold[pred_idx] ] += 1
            # ACCURACY RATE
            if y_pred[pred_idx] == y_gold[pred_idx]:
                correct[y_pred[pred_idx]] += 1

    # SECOND TEST: poison dataset(1.0)
    for idx, (data, target) in enumerate(data_pos_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)

        for label_idx in range(len(target)):
            target[label_idx] = f.target_label

            # ADD trigger
            im = TOPIL(data[label_idx])
            pixels = im.load()
            pixels[27, 0] = (255, 255, 255)

            data[label_idx] = TOtensor(im)
            Normal(data[label_idx])

        log_probs_pos = net(data)
        # predict
        y_pred_pos = log_probs_pos.data.max(1, keepdim=True)[1]
        # correct
        y_gold_pos = target.data.view_as(y_pred_pos).squeeze(1)
        
        y_pred_pos = y_pred_pos.squeeze(1)
        
        for pred_idx in range(len(y_pred_pos)):
            gold_all_pos[ y_gold_pos[pred_idx] ] += 1
            # POISON ATTACK SUCCESS RATE
            if y_pred_pos[pred_idx] == y_gold_pos[pred_idx]:
                correct_pos[y_pred_pos[pred_idx]] += 1

    # THIRD TEST: train dataset (0.3)
    perm = np.random.permutation(len(data_train_loader))[0: int(len(data_train_loader) * 0.3)]
    for idx, (data, target) in enumerate(data_train_loader):
        if f.gpu != -1:
            data, target = data.to(f.device), target.to(f.device)
        
        # ADD trigger
        if idx in perm:
            target[label_idx] = f.target_label
            im = TOPIL(data[label_idx])
            pixels = im.load()
            pixels[27, 0] = (255, 255, 255)

            data[label_idx] = TOtensor(im)
            Normal(data[label_idx])

        log_probs_train = net(data)
        test_loss += F.cross_entropy(log_probs_train, target, reduction='sum').item()
        # predict
        y_pred_train = log_probs_train.data.max(1, keepdim=True)[1]
        # correct
        y_gold_train = target.data.view_as(y_pred_train).squeeze(1)
        
        y_pred_train = y_pred_train.squeeze(1)
        
        for pred_idx in range(len(y_pred_train)):
            gold_all_train[ y_gold_train[pred_idx] ] += 1
            if y_pred_train[pred_idx] == y_gold_train[pred_idx]:
                correct_train[y_pred_train[pred_idx]] += 1



    test_loss /= len(data_train_loader.dataset)

    accuracy = (sum(correct) / sum(gold_all)).item()
    
    acc_per_label = correct / gold_all

    poison_acc = 0

    accuracy_all = (sum(correct_train) / sum(gold_all_train)).item()

    if(f.attack_mode == 'poison'):
        poison_acc = (sum(correct_pos) / sum(gold_all_pos)).item()
    
    return accuracy, test_loss, acc_per_label.tolist(), poison_acc, accuracy_all





