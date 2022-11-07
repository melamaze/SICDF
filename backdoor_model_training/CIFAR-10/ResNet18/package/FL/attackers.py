from ..config import for_FL as f
from .Update import LocalUpdate_poison
import numpy as np

class Attackers():
    
    def __init__(self):
        self.all_attacker = []
        # number of attacker                                     
        self.attacker_num  = int(f.attack_ratio * f.total_users)
        # counter of attacker
        self.attacker_count = 0                                

    
    def poison_setting(self):

        print('target_label:', f.target_label)



    def choose_attackers(self, idxs_users, data):
        perm = np.random.permutation(f.total_users)[0: int(f.total_users * f.attack_ratio)]
        for idx in idxs_users:
            if idx in perm:
                self.all_attacker.append(idx)
                self.attacker_count += 1



    