# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 16:01
# @Author  : cpark

import random
import torch
from torch.utils.data import Dataset
from utils import neg_sample
import numpy as np
import pickle

class CausalDataset(Dataset):
    def __init__(self, args, user_seq, except_type, data_type='train'):
        self.args = args
        self.max_len = args.max_seq_length
        self.user_seq = user_seq
        self.data_type = data_type

        self.except_type = args.except_type
        
    def __len__(self):
        if self.data_type in {"valid","test"}:
            len_=len(self.user_seq[0])
        else:
            len_=len(self.user_seq[0])
        return len_ 
    
    def __getitem__(self, index):
        type_ = list(map(int, self.user_seq[0][index].split(',')))
        items = list(map(int, self.user_seq[1][index].split(',')))

        total_except=np.where([i not in self.except_type  for i in type_]) 

        type_ = np.array(type_)[total_except].tolist()
        cat2 = list(map(int, self.user_seq[0][index].split(',')))
        cat2 = np.array(cat2)[total_except].tolist()
        cat1 = list(map(int, self.user_seq[0][index].split(',')))
        cat1 = np.array(cat1)[total_except].tolist()

        items = np.array(items)[total_except].tolist() 
        
        
        assert self.data_type in {"train", "valid", "test", "inference"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        
        if self.data_type == "train":

            type_input = type_[:-3]
            type_pos = type_[1:-2]
            
            cat2_input = cat2[:-3]
            cat2_pos = cat2[1:-2]
            # cat2_answer = [0] # no use

            cat1_input = cat1[:-3]
            cat1_pos = cat1[1:-2]
            # cat1_answer = [0] # no use            
            
            item_input = items[:-3]
            item_pos = items[1:-2]
            item_answer = [0] # no use            

        elif self.data_type == 'valid':
            
            type_input = type_[:-2]
            type_pos = type_[1:-1]
            
            cat2_input = cat2[:-2]
            cat2_pos = cat2[1:-1]
            
            cat1_input = cat1[:-2]
            cat1_pos = cat1[1:-1]
            
            item_input = items[:-2]
            item_pos = items[1:-1]
            item_answer = [items[-2]]
            
            
        elif self.data_type == 'test':

            type_input = type_[:-1]
            type_pos = type_[1:]
            
            
            cat2_input = cat2[:-1]
            cat2_pos = cat2[1:]
            
            
            cat1_input = cat1[:-1]
            cat1_pos = cat1[1:]
            
            item_input = items[:-1]
            item_pos = items[1:]
            
            if len(items)>0:
                item_answer = [items[-1]]
            else:
                item_answer = [0]
            # item_answer = [items[-1]]
            
            
        else:
            
            type_input = type_[:-1]
            type_pos = type_[1:]
            
            cat2_input = cat2[:-1]
            cat2_pos = cat2[1:]
            
            
            cat1_input = cat1[:-1]
            cat1_pos = cat1[1:]
            
            item_input = items[:-1]
            item_pos = items[1:]
            
            if len(items)>0:
                item_answer = [items[-1]]
            else:
                item_answer = [0]
            
            
            
        cat2_neg = []
        seq_set = set(cat2)
        seq_set.update({0,1,2,3,4})
        for _ in cat2_input:
            cat2_neg.append(neg_sample(seq_set, self.args.cat2_size))            
            
        cat1_neg = []
        seq_set = set(cat1)
        seq_set.update({0,1,2,3,4})
        for _ in cat1_input:
            cat1_neg.append(neg_sample(seq_set, self.args.cat1_size))            
            
            
        item_neg = []
        seq_set = set(items)
        seq_set.update({0,1,2,3,4})
        for _ in item_input:
            item_neg.append(neg_sample(seq_set, self.args.item_size))
            

        test_neg = []
        for _ in range(100):
            test_neg.append(neg_sample(seq_set, self.args.item_size))
            
        pad_len = self.max_len - len(item_input)
        item_input = [0] * pad_len + item_input
        item_pos = [0] * pad_len + item_pos
        item_neg = [0] * pad_len + item_neg

        item_input = item_input[-self.max_len:]
        item_pos = item_pos[-self.max_len:]
        item_neg = item_neg[-self.max_len:]
        
        
        # pad_len = self.max_len - len(cat1_input)
        cat1_input = [0] * pad_len + cat1_input
        cat1_pos = [0] * pad_len + cat1_pos
        cat1_neg = [0] * pad_len + cat1_neg

        cat1_input = cat1_input[-self.max_len:]
        cat1_pos = cat1_pos[-self.max_len:]
        cat1_neg = cat1_neg[-self.max_len:]
        
        # pad_len = self.max_len - len(cat2_input)
        cat2_input = [0] * pad_len + cat2_input
        cat2_pos = [0] * pad_len + cat2_pos
        cat2_neg = [0] * pad_len + cat2_neg

        cat2_input = cat2_input[-self.max_len:]
        cat2_pos = cat2_pos[-self.max_len:]
        cat2_neg = cat2_neg[-self.max_len:]
        
        # pad_len = self.max_len - len(type_input)
        type_input = [0] * pad_len + type_input
        type_input = type_input[-self.max_len:]
        
        type_pos = [0] * pad_len + type_pos
        type_pos = type_pos[-self.max_len:]
        
        
        assert len(item_input) == self.max_len
        assert len(item_pos) == self.max_len
        assert len(item_neg) == self.max_len 
        assert len(cat1_input) == self.max_len
        assert len(cat1_pos) == self.max_len
        assert len(cat1_neg) == self.max_len 
        assert len(cat2_input) == self.max_len
        assert len(cat2_pos) == self.max_len
        assert len(cat2_neg) == self.max_len 
             
        cur_tensors = (
            torch.tensor(item_input, dtype=torch.long),
            torch.tensor(item_pos, dtype=torch.long),
            torch.tensor(item_neg, dtype=torch.long),
            torch.tensor(test_neg, dtype=torch.long),
            torch.tensor(item_answer, dtype=torch.long),
            
            torch.tensor(cat1_input, dtype=torch.long),
            torch.tensor(cat1_pos, dtype=torch.long),
            torch.tensor(cat1_neg, dtype=torch.long),
            # torch.tensor(test_neg, dtype=torch.long),            
            
            torch.tensor(cat2_input, dtype=torch.long),
            torch.tensor(cat2_pos, dtype=torch.long),
            torch.tensor(cat2_neg, dtype=torch.long),
            # torch.tensor(test_neg, dtype=torch.long),            
            
            torch.tensor(type_input, dtype=torch.long),  
            torch.tensor(type_pos, dtype=torch.long),  
        )
        return cur_tensors