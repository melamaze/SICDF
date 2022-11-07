'''
This code is based on
https://github.com/Suyi32/Learning-to-Detect-Malicious-Clients-for-Robust-FL/blob/main/src/utils/sampling.py
'''

import numpy as np
from ..config import for_FL as f


def my_noniid(dataset):

    #為了固定跑出的結果，指定seed
    np.random.seed(f.seed)

    # i是各個user，值是圖的編號
    dict_users = {i: np.array([], dtype='int64') for i in range(f.total_users)}
    
    # 54000張用作training
    noniid_img_per_local = int(54000 // f.total_users * f.noniid)
    iid_img_per_local = int(54000 // f.total_users - noniid_img_per_local)
    print("non-iid_per_local: ", noniid_img_per_local)
    print("iid_per_local: ", iid_img_per_local)

    # 每張圖都給個編號
    idxs = np.arange(54000)
    
    # 取dataset各圖的label(答案+)
    labels = dataset.targets.numpy()[0:54000]
    
    # vstack，也就是編號idxs_labels[0][0]的label是idxs_labels[1][0]
    # idxs_labels = [[編號],[標籤]] 
    idxs_labels = np.vstack((idxs, labels))

    # idxs_labels[1,:].argsort()這個也就是將label由小排到大，並回傳排好後的index 
    # 把相同label的圖排在一起
    # idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] 
    idxs_labels_sorted = idxs_labels[1,:].argsort()

    idxs_by_number = [[] for i in range(10)]        
    
    # 把相同label的放到對應的list
    for i in idxs_labels_sorted:
        # 雖然i應該要表示圖片id，但因為idxs_labels已經被sort過，所以id和index已經不同了，也就是idxs_labels[1][i]其實並不是id i的label
        # 而是idxs_labels的index i的label；而index i的id是什麼，要去看idxs_labels[0]
        # 我認為完全可以把idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()] 改成 idxs_labels_sorted = idxs_labels[1,:].argsort()
        # 然後將上面的for i in idxs_labels[0]: 改成 for i in idxs_labels_sorted:
        # 其他不變，效果應該會相同
        num = idxs_labels[1][i]
        idxs_by_number[num].append(i)
        # print(i, " ", num)
        
    
    # 轉成nparray
    for i in range(10):
        idxs_by_number[i] = np.array(idxs_by_number[i])     
    
    # 印出來看看分得如何
    for i in range(10):
        print('label {} : {}'.format(i, len(idxs_by_number[i])))
        print('example -> idx: {} = {}'.format(idxs_by_number[i][10], idxs_labels[1][idxs_by_number[i][10]]))
    
    print("")
    
    # 將i label作為主要label的local users
    noniid_to_local = [None for i in range(10)]        
    # local users的list
    local_list = [i for i in range(f.total_users)]      


    for k in range(10):
        # 隨機選users
        noniid_to_local[k] =  np.random.choice(local_list, f.total_users // 10, replace=False)       
        # 選過的就從local_list中去掉
        local_list = list(set(local_list) - set(noniid_to_local[k]))

    # 取圖
    for k in range(10):
        for local in noniid_to_local[k]:    
            # 答案為k的label
            rand_n = k      
            
            # 如果答案為k的圖數量還大於每個local user所分到的主要label的數量，就直接取該數量
            if(len(idxs_by_number[rand_n]) >= noniid_img_per_local):
                tmp = np.random.choice(idxs_by_number[rand_n], noniid_img_per_local, replace=False)
            # 若小於但還有圖，就全取光
            elif(len(idxs_by_number[rand_n]) > 0):
                tmp = np.random.choice(idxs_by_number[rand_n], len(idxs_by_number[rand_n]), replace=False)
            else:
                print('error')

            # 選過的圖從圖list去掉
            idxs_by_number[rand_n] = list(set(idxs_by_number[rand_n]) - set(tmp))
            
            # 該user擁有這些圖
            dict_users[local] = np.concatenate((dict_users[local], tmp), axis=0)  

            # 選其他少數labels的圖            
            for j in range(10):
                if(rand_n == j):
                    continue
                else:
                    # 如果答案為j的label的圖數量還大於要分給每個user的數量(9種要平均)，就直接取該數量
                    if(len(idxs_by_number[j]) >= (iid_img_per_local//9)):
                        tmp = np.random.choice(idxs_by_number[j], (iid_img_per_local//9), replace=False)
                    # 若小於但還有，就取光
                    else:
                        tmp = np.random.choice(idxs_by_number[j], len(idxs_by_number[j]), replace=False)
                    
                    # 如果確實有取到圖，就把選到的圖編號接進去
                    if(len(tmp) > 0):
                        idxs_by_number[j] = list(set(idxs_by_number[j]) - set(tmp))
                        dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)
                        
    # 看看每種label還剩下多少張沒分出去
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")  
    

    # 用來記錄還有哪幾種label還沒分完
    num_list = [i for i in range(10)]

    for k in range(10):
        if(len(idxs_by_number[k]) == 0):
            num_list = list(set(num_list) - {k})
            print(num_list)      

    # 如果有某種label還沒分完
    for k in range(10):
        for local in noniid_to_local[k]:
            rand_n = k

            # 去掉該user主要的那種label
            tmp_list = list(set(num_list) - {rand_n})
            
            # 從其他種label中隨機取6或更少種
            # 我也忘了為什麼要取6種
            if(len(tmp_list) >= 6):
                numbers = np.random.choice(tmp_list, 6, replace=False)
            else:
                numbers = np.random.choice(tmp_list, len(tmp_list), replace=False)

            for n in numbers:
                # 抽一張圖(答案為n)，給該user
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                # 當該種label的圖分完了
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    # 看看每種label還剩下多少張沒分出去
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")

    # 繼續分剩下的
    for k in range(10):
        for local in noniid_to_local[k]:
            if(num_list == []):
                break
            rand_n = k
            # 選一種labe給該user，也沒分主要不主要label了
            numbers = np.random.choice(num_list, 1, replace=False)

            #抽一張圖給該user
            for n in numbers:
                tmp = np.random.choice(idxs_by_number[n], 1, replace=False)
                idxs_by_number[n] = list(set(idxs_by_number[n]) - set(tmp))
                dict_users[local] = np.concatenate((dict_users[local], tmp),axis=0)     
                if(len(idxs_by_number[n])==0):
                    num_list = list(set(num_list) - {n})
                    print(num_list)

    # 看看每種label還剩下多少張沒分出去，應該都為0了
    for j in range(10):
        print(j,": ",len(idxs_by_number[j]))
    print("")

    # 抽個user來看看，看看這個user分到的圖片編號，是不是有某個主要的label
    print("")
    print(dict_users[0])
    print(idxs_labels[1][dict_users[0]])

    # 返回各user擁有的圖片編號，和每個編號對應的圖的label
    return dict_users, idxs_labels

