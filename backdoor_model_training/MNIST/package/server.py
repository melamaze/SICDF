from .config import for_FL as f
from .FL.datasets import Dataset
from .FL.attackers import Attackers
from .FL.clients import Server
from .FL.image import Plot
from datetime import datetime
from torch import nn
from .FL.resnet18 import ResNet18

import torch
import copy
import numpy as np
import time
import torchvision.models as models

def main():

    # assign seed
    np.random.seed(f.seed)
    torch.manual_seed(f.seed)

    # set random seed for GPU
    print(torch.cuda.is_available())
    f.device = torch.device('cuda:{}'.format(f.gpu) if torch.cuda.is_available() and f.gpu != -1 else 'cpu')
    # check if it use cuda
    print(f.device)

    # build dataset
    my_data = Dataset()
    # sampling data
    my_data.sampling()

    FL_net = ResNet18().to(f.device)

    # draw
    plot = Plot()

    print('The model in server:')
    print(FL_net)

    # model parameter
    FL_weights = FL_net.state_dict()

    # build attackers
    my_attackers = Attackers()
    my_attackers.poison_setting()
        
    my_server  = Server(copy.deepcopy(FL_net))

    # dsitribute to clients
    all_users = [i for i in range(f.total_users)]
    
    # random choose users
    idxs_users = np.random.choice(range(f.total_users), f.total_users, replace=False)   

    # choose attackers
    my_attackers.choose_attackers(idxs_users, my_data)
    print("number of attacker: ", my_attackers.attacker_count)         
    print("all attacker: ", my_attackers.all_attacker)   
    print("")
    
    my_server.split_user_to(all_users, my_attackers.all_attacker)
    
  
    total_time = 0

    true_start_time = time.time()

    for round in range(f.epochs):

        # reset model parameters, loss
        my_server.reset()

        global_test_time = 0

        start_ep_time = time.time()

        if(f.attack_mode == "poison"):
            my_server.local_update_poison(my_data,my_attackers.all_attacker,round)
        
        end_ep_time = time.time()

        local_ep_time = end_ep_time - start_ep_time
        
        # validation
        global_test_time += my_server.show_testing_result(my_data, plot)
        
        print("-------------------------------------------------------------------------")
        print("")

        print("local_ep_time: ",local_ep_time)
        round_time = local_ep_time + global_test_time
        print("round_time: ",round_time)
        print("")

        total_time += round_time

        # store global model
        path = f.model_path + 'global_model' + '.pth'
        torch.save(my_server.client_net.state_dict(), path)

    true_end_time = time.time()
    
    print('simulation total time:', total_time)
    print('true total time:', true_end_time - true_start_time)
    plot.draw_plot()


