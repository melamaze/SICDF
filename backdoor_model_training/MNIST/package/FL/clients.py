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

#其實就是server
class Server():
    def __init__(self,net):
        #該client分到的users
        # self.local_users = []
        #該client的model net，所有client的初始net都相同          
        self.client_net = net
        #該client分到的攻擊者的編號           
        self.attacker_idxs = []

        self.weights = []
        self.loss = []
        # 好像沒用到這兩變數
        self.training_acc = 0
        self.testing_acc = 0
        # user_sizes是每個user有的照片量
        self.user_sizes = None          

        # 各user回傳的training loss的平均
        self.loss_avg = 0     
        # 雖然寫test，但其實是validation結果          
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
        
        # 只有一個 server，所以全部 user 都給他
        self.local_users = set(all_users)
        # 若有選到攻擊者，則記錄
        for i in self.local_users:
            if i in attackers:
                self.attacker_idxs.append(i)


    def local_update_poison(self,data,all_attacker,round):
        
        for idx in self.local_users:
            # 這邊的idxs的值，看先前有沒有改過idxs_labels_sorted，有的話記得也要改
            # 這邊會竄改label(如果是攻擊者的話)，並且各user會訓練model (local training)
            local = LocalUpdate_poison(dataset=data.dataset_train, idxs=data.dict_users[idx], user_idx=idx, attack_idxs=all_attacker)
            # 這裡的deepcopy是因為master model分給其user,這些model是在各user是獨立訓練的
            w, loss, attack_flag = local.train(net=copy.deepcopy(self.client_net).to(f.device))
            
            self.weights.append(copy.deepcopy(w))
            self.loss.append(copy.deepcopy(loss))
        
        # print("Client {}".format(self.id))
        print(" {}/{} are attackers with {} attack ".format(len(self.attacker_idxs), len(self.local_users), f.attack_mode))

        # 根據照片數的權重，來進行各user的模型參數的平均
        self.user_sizes = np.array([ len(data.dict_users[idx]) for idx in self.local_users ])
        user_weights = self.user_sizes / float(sum(self.user_sizes))
        
        # aggregation的方法用最普通的FedAvg
        if f.aggregation == "FedAvg":
            w_glob = FedAvg(self.weights, user_weights)
        else:
            print('no other aggregation method.')
            exit()

        # 新的master model
        self.client_net.load_state_dict(w_glob)
        self.loss_avg = np.sum(self.loss * user_weights)
        
        print('=== Round {:3d}, Average loss {:.6f} ==='.format(round, self.loss_avg))
        print(" {} users; time {}".format(len(self.local_users), datetime.now().strftime("%H:%M:%S")) )

    def show_testing_result(self,my_data,plot):
            start_time = time.time()
            
            # 進行validation
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
            print( " Accuracy for all (including poison accuracy) : {}".format(self.acc_all))
            if f.attack_mode=='poison':
                print( " Poison Acc: {}".format(self.poison_acc) )
            
            end_time = time.time()
            
            return end_time - start_time
            

