'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/utils/sampling.py
'''

import numpy as np
from ..config import for_FL as f

def iid(dataset):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # print(dataset[0])
    idxs = []
    labels = []
    idxs_labels = []
    num_items = int(len(dataset)/f.total_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(f.total_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        for j in dict_users[i]:
            idxs.append(j)
            labels.append(dataset[j][1])
        idxs_labels.append((idxs, labels))
        all_idxs = list(set(all_idxs) - dict_users[i])

    idxs_labels = np.array(idxs_labels)

    return dict_users, idxs_labels

def mnist_noniid(dataset):

    np.random.seed(f.seed)

    dict_users = {i: np.array([], dtype='int64') for i in range(f.total_users)}
    
    noniid_img_per_local = int(50000 // f.total_users * f.noniid)
    iid_img_per_local = int(50000 // f.total_users - noniid_img_per_local)
    print("non-iid_per_local: ", noniid_img_per_local)
    print("iid_per_local: ", iid_img_per_local)

    idxs = np.arange(50000)
    
    labels = dataset.targets.numpy()[0:50000]

    idxs_labels = np.vstack((idxs, labels))
    print("HEHE:", len(idxs_labels))
    print("HEHE:", len(idxs_labels[0]))
    print(idxs_labels)

    idxs_labels_sorted = idxs_labels[1,:].argsort()

    idxs_by_number = [[] for i in range(10)]        
    
    for i in idxs_labels_sorted:
        num = idxs_labels[1][i]
        idxs_by_number[num].append(i)
        

    for i in range(10):
        idxs_by_number[i] = np.array(idxs_by_number[i])     
    
    for i in range(10):
        print('label {} : {}'.format(i, len(idxs_by_number[i])))
        print('example -> idx: {} = {}'.format(idxs_by_number[i][10], idxs_labels[1][idxs_by_number[i][10]]))
    
    print("")
    
    noniid_to_local = [None for i in range(10)]        

    local_list = [i for i in range(f.total_users)]      


    for k in range(10):

        noniid_to_local[k] =  np.random.choice(local_list, f.total_users // 10, replace=False)       

        local_list = list(set(local_list) - set(noniid_to_local[k]))


    for k in range(10):
        for local in noniid_to_local[k]:    

            rand_n = k      

            if(len(idxs_by_number[rand_n]) >= noniid_img_per_local):
                tmp = np.random.choice(idxs_by_number[rand_n], noniid_img_per_local, replace=False)

            elif(len(idxs_by_number[rand_n]) > 0):
                tmp = np.random.choice(idxs_by_number[rand_n], len(idxs_by_number[rand_n]), replace=False)
            else:
                print('error')

            idxs_by_number[rand_n] = list(set(idxs_by_number[rand_n]) - set(tmp))
            

            dict_users[local] = np.concatenate((dict_users[local], tmp), axis=0)  

         
            for j in range(10):
                if(rand_n == j):
                    continue
                else:
 
                    if(len(idxs_by_number[j]) >= (iid_img_per_local//9)):
                        tmp = np.random.choice(idxs_by_number[j], (iid_img_per_local//9), replace=False)

                    else:
                        tmp = np.random.choice(idxs_by_number[j], len(idxs_by_number[j]), replace=False)
                    
                    if(len(tmp) > 0):
                        idxs_by_number[j] = list(set(idxs_by_number[j]) - set(tmp))
                        dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)
                        

    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")  
    

    num_list = [i for i in range(10)]

    for k in range(10):
        if(len(idxs_by_number[k]) == 0):
            num_list = list(set(num_list) - {k})
            print(num_list)      

    for k in range(10):
        for local in noniid_to_local[k]:
            rand_n = k

            tmp_list = list(set(num_list) - {rand_n})

            if(len(tmp_list) >= 6):
                numbers = np.random.choice(tmp_list, 6, replace=False)
            else:
                numbers = np.random.choice(tmp_list, len(tmp_list), replace=False)

            for n in numbers:
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")

    for k in range(10):
        for local in noniid_to_local[k]:
            if(num_list == []):
                break
            rand_n = k
            numbers = np.random.choice(num_list, 1, replace=False)

            for n in numbers:
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")

    print("")
    print(dict_users[0])
    print(idxs_labels[1][dict_users[0]])

    return dict_users, idxs_labels