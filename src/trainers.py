# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from utils import recall_at_k, ndcg_k, get_metric
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import socket
from torch.cuda.amp import autocast, GradScaler

import gc


class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        # self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        # self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
        self.mlm_output = nn.Linear(args.hidden_size, args.item_size-1)
        
        
        
        self.local_rank = args.local_rank
        self.device = torch.device("cuda:"+str(self.local_rank))
        self.model = model.to(self.device)
        
        # if self.cuda_condition:
        #     self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        # self.optim = Adam(list(self.model.parameters())+[self.model.alpha,self.model.beta], lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        
        if self.args.loss_type == 'negative':
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.NLLLoss()
            
            
            
        ###ddp 
        self.local_rank=self.args.local_rank
        with_cuda=True
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:"+str(self.local_rank))
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for RecGPT" % torch.cuda.device_count())
                        
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        ###
        
        ###amp
        self.scaler = GradScaler()

        ###
            


    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return ([HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix))

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_negatve_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.module.item_embeddings(pos_ids)
        neg_emb = self.model.module.item_embeddings(neg_ids)
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget) + 1e-24)

        return loss
    
    def cross_entropy(self, seq_out, pos_ids):
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        # only mlm_loss        
        sequence_output = self.mlm_output(seq_emb) # [batch*seq_len class_num]
        
        labels = pos_ids.view(-1, 1) # [batch*seq_len class_num]
        
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float().view(-1, 1) # [batch*seq_len, 1]
        labels = (labels * istarget).long()
        loss = self.criterion(nn.LogSoftmax(dim=-1)(sequence_output), labels.view(-1, ))
        return loss

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.module.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.module.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred
            
class PretrainProfileTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainProfileTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"
        
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
                          total=len(dataloader),
                          bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            
            profile_loss_avg = 0.0
            seq_loss_avg = 0.0
            seq_profile_loss_avg = 0.0
            loss_avg = 0.0
            acc_avg = 0.0
            
            print("Device",self.device)
            for i, batch in rec_data_iter:
                # gc.collect()
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                # input_ids, target_pos, target_neg, _, _, profile_feat = batch
                
                
                # with autocast(): #enabled=False
                item_input, item_pos, item_neg, test_neg , item_answer, cat1_input, cat1_pos, cat1_neg , cat2_input, cat2_pos, cat2_neg, type_input, type_pos = batch

                if self.args.profile_yn == 'Y':
                    loss, seq_loss, seq_profile_loss, profile_loss = self.model.module.pretrain(input_item, target_pos, target_neg, profile_feat)
                else:

                    # loss를 Pretrain_seq 자체에서 생성
                    loss, seq_loss, shaply_values_softmax, shaply_values_datanum  = self.model.module.pretrain_seq(item_input, item_pos, item_neg, test_neg, item_answer, cat1_input, cat1_pos, cat1_neg, cat2_input, cat2_pos, cat2_neg, type_input, self.args.hierarhical) #loss, seq_loss, shaply_values_softmax, shaply_values_datanum .to(self.device)
                    seq_profile_loss = torch.tensor(0)
                    profile_loss = torch.tensor(0)

                self.optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1)
                self.optim.step()

                # self.optim.zero_grad()
                # self.scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1) 
                # self.scaler.step(self.optim)
                # self.scaler.update() 
                
                
                
                
                
                
                seq_loss_avg += seq_loss.item()
                seq_profile_loss_avg += seq_profile_loss.item()
                profile_loss_avg += profile_loss.item()
                loss_avg += loss.item()
                # acc_avg += acc.item()

            post_fix = {
                "epoch": epoch,
                "loss_avg": '{:.4f}'.format(loss_avg/len(rec_data_iter)),
                "seq_loss_avg": '{:.4f}'.format(seq_loss_avg/len(rec_data_iter)),
                "seq_profile_loss_avg": '{:.4f}'.format(seq_profile_loss_avg/len(rec_data_iter)),
                "profile_loss_avg": '{:.4f}'.format(profile_loss_avg/len(rec_data_iter)),
                "shaply_value":shaply_values_softmax,
                "shaply_domain_datanum":shaply_values_datanum,
                # "training_acc_avg": '{:.4f}'.format(acc_avg/len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

#         else:        
#             self.model.eval()
            
            
#             pred_list = None
#             with torch.no_grad():
#                 for i, batch in rec_data_iter:
#                     # 0. batch_data will be sent into the device(GPU or cpu)
#                     # batch = tuple(t.to(self.device) for t in batch)
#                     batch = tuple(t.to('cuda:0') for t in batch)#cpu


#                     item_input, item_pos, item_neg, test_neg , item_answer, cat1_input, cat1_pos, cat1_neg , cat2_input, cat2_pos, cat2_neg, type_input = batch
#                     # input_ids, target_pos, target_neg, test_neg, answers = batch
#                     sequence_output_1, sequence_output_2, recommend_output = self.model.to('cuda:0').module.get_last_emb(item_input, cat1_input, cat2_input, type_input, item_pos, item_neg, self.args.hierarhical,cuda_yn='y')#'cpu'


#                     test_neg_items = torch.cat((item_answer, test_neg), -1)

#                     test_logits = self.predict_sample(recommend_output, test_neg_items)
#                     test_logits = test_logits.cpu().detach().numpy().copy()
#                     if i == 0:
#                         pred_list = test_logits
#                     else:
#                         pred_list = np.append(pred_list, test_logits, axis=0)

#                 return self.get_sample_scores(epoch, pred_list)  

        else:        
            self.model.eval()
            
            
            pred_list = None
            type_pos_list=[]
            
            with torch.no_grad():
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    # batch = tuple(t.to(self.device) for t in batch)
                    batch = tuple(t.to('cuda:0') for t in batch)#cpu


                    item_input, item_pos, item_neg, test_neg , item_answer, cat1_input, cat1_pos, cat1_neg , cat2_input, cat2_pos, cat2_neg, type_input, type_pos = batch

                    sequence_output_1, sequence_output_2, recommend_output = self.model.to('cuda:0').module.get_last_emb(item_input, cat1_input, cat2_input, type_input, item_pos, item_neg, self.args.hierarhical,cuda_yn='y')#'cpu'


                    test_neg_items = torch.cat((item_answer, test_neg), -1)

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()


#                     if i == 0:
#                         pred_list = test_logits
#                     elif i!=0 and type_pos[:,-1].shape[0]==self.args.batch_size:
#                         pred_list = np.append(pred_list, test_logits, axis=0)
#                     else:
#                         pass
#                     if type_pos[:,-1].shape[0]==self.args.batch_size: #numpy upgrade 이슈로 batch shape이 맞지 않는 array는 concat 안됨
#                         type_pos_list.append(type_pos[:,-1].cpu().detach().numpy().copy())

                
#                 type_pos_final=np.concatenate(np.array(type_pos_list))
                    
                    if i == 0:
                        pred_list = test_logits
                    elif i!=0 and type_pos[:,-1].shape[0]==self.args.batch_size:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    else:
                        pass
                    
                    if type_pos[:,-1].shape[0]==self.args.batch_size: #numpy upgrade 이슈로 batch shape이 맞지 않는 array는 concat 안됨
                        type_pos_list.append(type_pos[:,-1].cpu().detach().numpy().copy())

                type_pos_final=np.concatenate(np.array(type_pos_list))                  
                    
                return self.get_sample_scores(epoch, pred_list), self.get_sample_scores(epoch, pred_list[type_pos_final==5]) , self.get_sample_scores(epoch, pred_list[type_pos_final==6]) , self.get_sample_scores(epoch, pred_list[type_pos_final==7]) , self.get_sample_scores(epoch, pred_list[type_pos_final==8]) , self.get_sample_scores(epoch, pred_list[type_pos_final==9])  

            
            