from ..config import for_FL as f
import numpy as np
import copy
from .image import Plot
from .Update import LocalUpdate_poison
from .Fed import FedAvg
from .test import test_img_poison
from datetime import datetime
import time

np.random.seed(f.seed)

class Server():
    def __init__(self,net):
        # model of clint        
        self.client_net = net
        # index of attacker for clinet         
        self.attacker_idxs = []

        self.weights = []
        self.loss = []

        # number of picture for each user
        self.user_sizes = None          

        # average training loss
        self.loss_avg = 0     
        # result of validation       
        self.acc_test = 0
        self.loss_test = 0
        self.acc_per_label = None
        self.poison_acc = 0
        self.acc_per_label_avg = 0
        self.acc_all = 0


    def reset(self):
        self.weights = []
        self.loss = []

    def split_user_to(self, all_users, attackers):
        
        # only one server, accordingly, set all user to it
        self.local_users = set(all_users)
        # record the attacker who has been chosen
        for i in self.local_users:
            if i in attackers:
                self.attacker_idxs.append(i)


    def local_update_poison(self,data,all_attacker,round):
        
        for idx in self.local_users:
            # temper the label here
            local = LocalUpdate_poison(dataset=data.dataset_train, idxs=data.dict_users[idx], user_idx=idx, attack_idxs=all_attacker)
            # deepcopy, since they train independently.
            w, loss, attack_flag = local.train(net=copy.deepcopy(self.client_net).to(f.device))
            
            self.weights.append(copy.deepcopy(w))
            self.loss.append(copy.deepcopy(loss))
        
        # print("Client {}".format(self.id))
        print(" {}/{} are attackers with {} attack ".format(len(self.attacker_idxs), len(self.local_users), f.attack_mode))

        # average the parameter of model
        self.user_sizes = np.array([ len(data.dict_users[idx]) for idx in self.local_users ])
        user_weights = self.user_sizes / float(sum(self.user_sizes))
        
        # FedAvg
        if f.aggregation == "FedAvg":
            w_glob = FedAvg(self.weights, user_weights)
        else:
            print('no other aggregation method.')
            exit()

        # new master model
        self.client_net.load_state_dict(w_glob)
        self.loss_avg = np.sum(self.loss * user_weights)
        
        print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, self.loss_avg))
        print(" {} users; time {}".format(len(self.local_users), datetime.now().strftime("%H:%M:%S")) )

    def show_testing_result(self,my_data,plot):
            start_time = time.time()
            
            # validation
            self.acc_test, self.loss_test, self.acc_per_label, self.poison_acc, self.acc_all = test_img_poison(self.client_net.to(f.device), my_data.dataset_test)
            self.acc_per_label_avg = sum(self.acc_per_label)/len(self.acc_per_label)
            
            # plot data
            plot.accuracy.append(self.acc_test)
            plot.poison_accuracy.append(self.poison_acc)
            plot.all_accuracy.append(self.acc_all)
            plot.loss.append(self.loss_test)


            print( " Testing accuracy: {} loss: {:.6}".format(self.acc_test, self.loss_test))
            print( " Testing Label Acc: {}".format(self.acc_per_label) )
            print( " Testing Avg Label Acc : {}".format(self.acc_per_label_avg))
            if f.attack_mode=='poison':
                print( " Poison Acc: {}".format(self.poison_acc) )
            
            end_time = time.time()
            
            return end_time - start_time
            