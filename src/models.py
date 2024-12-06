# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 16:01
# @Author  : cpark

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Encoder, LayerNorm, Intermediate 
from tab_network import TabNetPretraining
from utils import create_group_matrix



import itertools
from itertools import combinations
import math


class CausalModel(nn.Module):
    def __init__(self, args):
        super(CausalModel, self).__init__()
        self.type_embeddings = nn.Embedding(args.type_size, args.hidden_size, padding_idx=0) 
        self.cat2_embeddings = nn.Embedding(args.cat2_size, args.hidden_size, padding_idx=0) 
        self.cat1_embeddings = nn.Embedding(args.cat1_size, args.hidden_size, padding_idx=0)          
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
      
        self.prediction_layer_1 = Intermediate(args)
        self.prediction_layer_2 = Intermediate(args)

        self.alpha = nn.Linear(1,1,bias=False)
        self.beta = nn.Linear(1,1,bias=False)
        self.shaply_values_update = torch.tensor([0.2 for i in range(5)])
        
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args
        self.device = torch.device("cuda:"+str(self.args.local_rank))
        self.grouped_features = []

        # sequence modeling 
        self.mlm_output = nn.Linear(args.hidden_size, args.item_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)

        if self.args.loss_type == 'negative':
            self.criterion = nn.BCELoss(reduction='none')
        else:
            self.criterion = nn.NLLLoss(ignore_index=0)
        self.apply(self.init_weights)
        
        if self.args.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.args.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

    def add_position_embedding(self, item_seq, cat1_seq, cat2_seq, type_seq, hierarhical):
        seq_length = item_seq.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)

        type_embeddings = self.type_embeddings(type_seq)
        cat2_embeddings = self.cat2_embeddings(cat2_seq)
        cat1_embeddings = self.cat1_embeddings(cat1_seq)
        item_embeddings = self.item_embeddings(item_seq)
        position_embeddings = self.position_embeddings(position_ids)
        
        if self.args.hierarhical=='y':
            sequence_emb = item_embeddings + position_embeddings + type_embeddings + cat2_embeddings + cat1_embeddings
        else:
            sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def combination_cp(self,n,r):
        return math.factorial(n)/(math.factorial(n-r)*math.factorial(r))

    def shaply_weight(self,total_domain, subset):
        a=1/(self.combination_cp(len(total_domain),len(subset))*len(subset))
        return a    

    def pretrain_seq(self, item_input, item_pos, item_neg, test_neg , item_answer, cat1_input, cat1_pos, cat1_neg , cat2_input, cat2_pos, cat2_neg, type_input, hierarhical): # 학습 모델; type은 contrastive learning 하지 않음, forward로 바꿔야 ddp 가능
        
        # for sequence modeling (decoder)
        attention_mask = (item_input > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        
        # if self.args.cuda_condition:
        subsequent_mask = subsequent_mask.to(self.device)       
            
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        target_domain=5
        source_domain=[6,7,8,9] #6:btv, 7: cdr, 8:tmbr, 9:tmap, 10:11st,11:prod1 // 'books': 5,'clothing_shoes_and_jewelry': 6,'movies_and_tv': 7,'toys_and_games': 8,'sports_and_outdoors': 9   
                
        shaply_values_softmax = self.shaply_layer(target_domain=target_domain, source_domain=source_domain, type_input=type_input, item_input=item_input, item_pos=item_pos, item_neg=item_neg, cat1_input=cat1_input, cat1_pos=cat1_pos, cat1_neg=cat1_neg, cat2_input=cat2_input, cat2_pos=cat2_pos, cat2_neg=cat2_neg, hierarhical=self.args.hierarhical, attention_mask=extended_attention_mask, shaply_yn=self.args.shaply_value)
        
        shaply_values_datanum={}
        for domain_index in [target_domain]+source_domain:
            shaply_values_datanum[domain_index]=((type_input > 0)&(type_input==domain_index)).sum()
        
        
        sequence_emb = self.add_position_embedding(item_input, cat1_input, cat2_input, type_input, hierarhical=self.args.hierarhical)
        
        encoded_layers = self.item_encoder(sequence_emb,
                                          extended_attention_mask,
                                          output_all_encoded_layers=True)

        sequence_output = encoded_layers[-1]
        
        if self.args.hierarhical=='y':
            sequence_output_1 = self.prediction_layer_1(sequence_output)
            sequence_output_2 = self.prediction_layer_2(sequence_output+sequence_output_1) # 일단 더하는 방식

            loss_contrastive_1 = 0
            loss_contrastive_2 = 0
            loss_contrastive_3 = 0
            for domain_index in [target_domain]+source_domain:

                loss_contrastive_1 += self.cross_entropy_shaply(sequence_output, cat2_pos, cat2_neg, type_input, domain_index, hierarchical_level='cat2')*shaply_values_softmax[domain_index]
                loss_contrastive_2 += self.cross_entropy_shaply(sequence_output_1, cat1_pos, cat1_neg, type_input, domain_index, hierarchical_level='cat1')*shaply_values_softmax[domain_index]
                loss_contrastive_3 += self.cross_entropy_shaply(sequence_output_2, item_pos, item_neg, type_input, domain_index, hierarchical_level='item')*shaply_values_softmax[domain_index]


            loss=(loss_contrastive_1+loss_contrastive_2+loss_contrastive_3).to(self.device)
            
        else:
            loss_contrastive_3 = 0
            for domain_index in [target_domain]+source_domain:


                loss_contrastive_3 += self.cross_entropy_shaply(sequence_output, item_pos, item_neg, type_input, domain_index, hierarchical_level='item')*shaply_values_softmax[domain_index]
            loss=loss_contrastive_3.to(self.device)            
    
        return loss, loss_contrastive_3, shaply_values_softmax, shaply_values_datanum#mlm_loss_3#lm_loss#, acc
    
    
    def get_last_emb(self, item_input, cat1_input, cat2_input, type_input, item_pos, item_neg, hierarhical, cuda_yn='y'):
        # for sequence modeling (decoder)
        attention_mask = (item_input > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if cuda_yn=='y':
            subsequent_mask = subsequent_mask.to(self.device)
        else:
            subsequent_mask = subsequent_mask.to('cpu')
            
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        
        
        sequence_emb = self.add_position_embedding(item_input, cat1_input, cat2_input, type_input,hierarhical=self.args.hierarhical)
        
        encoded_layers = self.item_encoder(sequence_emb,
                                          extended_attention_mask,
                                          output_all_encoded_layers=True)
        
        sequence_output = encoded_layers[-1] # [B L H]


        if self.args.hierarhical=='y': 
            sequence_output_1 = self.prediction_layer_1(sequence_output)
            sequence_output_2 = self.prediction_layer_2(sequence_output+sequence_output_1)
            
            
            cat2_output = sequence_output[:,-1,]
            cat1_output = sequence_output_1[:,-1,]
            eos_output = sequence_output_2[:,-1,]
            
        else:
            
            cat2_output = sequence_output[:,-1,]
            cat1_output = cat2_output
            eos_output = cat2_output # [B H]
        
        
        return cat2_output, cat1_output, eos_output
    
    
    def cross_entropy_shaply(self, seq_out, pos_ids, neg_ids, type_input, target_domain, hierarchical_level):
        if hierarchical_level=='item':
            # [batch seq_len hidden_size]
            pos_emb = self.item_embeddings(pos_ids)
            neg_emb = self.item_embeddings(neg_ids)
        elif hierarchical_level=='cat1':
            pos_emb = self.cat1_embeddings(pos_ids)
            neg_emb = self.cat1_embeddings(neg_ids)
        else:#cat2
            pos_emb = self.cat2_embeddings(pos_ids)
            neg_emb = self.cat2_embeddings(neg_ids)
            
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = ((type_input > 0)&(type_input==target_domain)).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)

        return loss
    
    def cross_entropy(self, seq_out, pos_ids, neg_ids, hierarchical_level, shaply_index):
        if hierarchical_level=='item':
            # [batch seq_len hidden_size]
            pos_emb = self.item_embeddings(pos_ids)
            neg_emb = self.item_embeddings(neg_ids)
        elif hierarchical_level=='cat1':
            pos_emb = self.cat1_embeddings(pos_ids)
            neg_emb = self.cat1_embeddings(neg_ids)
        else:#cat2
            pos_emb = self.cat2_embeddings(pos_ids)
            neg_emb = self.cat2_embeddings(neg_ids) 
        
        
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_size) # [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (shaply_index > 0).view(pos_ids.size(0) * self.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)#sum(istarget)

        return loss
    
    def shaply_index(self,type_data, shaply_list, index):
        tmp=torch.zeros(type_data.shape)
        # tmp=np.zeros(type_data.shape)
        for i in shaply_list[index]:
            tmp+=(type_data==i).to('cpu')
        tmp=tmp.type(torch.int)
        return tmp
    
    def get_acc(self, seq_out, target_pos):
        test_item_emb = self.item_embeddings.weight # [C H]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1)) # [B L C]
        rating_pred = torch.argsort(-rating_pred)[:, :, 0] # [B L]
        rating_pred = rating_pred.view(target_pos.size(0) * self.args.max_seq_length) # [B*L]
        target = target_pos.view(target_pos.size(0) * self.args.max_seq_length) # [B*L]
        istarget = (target_pos > 1).view(target_pos.size(0) * self.args.max_seq_length)
        acc = torch.sum((rating_pred==target).float()*istarget) / torch.sum(istarget)
        return acc

        
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
            
            
    def shaply_layer(self, target_domain, source_domain, type_input, item_input, item_pos, item_neg, cat1_input, cat1_pos, cat1_neg, cat2_input, cat2_pos, cat2_neg, hierarhical, attention_mask, shaply_yn='n'):
        
        total_domain=[target_domain]+source_domain
        
        shaply_values_datanum={}
        for domain_index in [target_domain]+source_domain:
            shaply_values_datanum[domain_index]=((type_input > 0)&(type_input==domain_index)).sum()

        if shaply_yn=='y':

            shaply_list=[list(combinations(total_domain, i)) for i in range(1,len(total_domain))] #source_domain
            shaply_list=list(itertools.chain(*shaply_list)) 
            shaply_list=[i for i in shaply_list] 
            
            shaply_list_tmp = shaply_list

            

            final_result_list={}
            for k in range(len(shaply_list_tmp)): #shaply_list
                shaply_index_list=shaply_list_tmp[k]
                shaply_index_=self.shaply_index(type_input, shaply_list=shaply_list_tmp, index=k)
                sequence_emb_tmp = self.add_position_embedding(item_input*shaply_index_.to(self.device), cat1_input*shaply_index_.to(self.device), cat2_input*shaply_index_.to(self.device), type_input,hierarhical=self.args.hierarhical)

                encoded_layers_tmp = self.item_encoder(sequence_emb_tmp,
                                                  attention_mask,
                                                  output_all_encoded_layers=True)  
                sequence_output_tmp = encoded_layers_tmp[-1]
                
                if self.args.hierarhical=='y':
                    sequence_output_tmp_1 = self.prediction_layer_1(sequence_output_tmp)
                    sequence_output_tmp_2 = self.prediction_layer_2(sequence_output_tmp+sequence_output_tmp_1)

                    mlm_loss_1_tmp = self.cross_entropy(sequence_output_tmp, cat2_pos*shaply_index_.to(self.device), cat2_neg*shaply_index_.to(self.device), hierarchical_level='cat2',shaply_index=shaply_index_.to(self.device))
                    mlm_loss_2_tmp = self.cross_entropy(sequence_output_tmp_1, cat1_pos*shaply_index_.to(self.device), cat1_neg*shaply_index_.to(self.device), hierarchical_level='cat1',shaply_index=shaply_index_.to(self.device))
                    mlm_loss_3_tmp = self.cross_entropy(sequence_output_tmp_2, item_pos*shaply_index_.to(self.device), item_neg*shaply_index_.to(self.device), hierarchical_level='item',shaply_index=shaply_index_.to(self.device))
                    
                    


                    loss_tmp = (mlm_loss_2_tmp + mlm_loss_1_tmp + mlm_loss_3_tmp).to(self.device)
                    
                else:
                    loss_tmp = self.cross_entropy(sequence_output_tmp, item_pos*shaply_index_.to(self.device), item_neg*shaply_index_.to(self.device), hierarchical_level='item',shaply_index=shaply_index_.to(self.device)).to(self.device)
                    
                
                
                final_result_list[shaply_index_list]=loss_tmp

            shaply_values={}
            for j in total_domain:
                weight_=[self.shaply_weight(total_domain,i) for i in shaply_list_tmp if (j in i)]
                contain_source_weights=[final_result_list[i] for i in shaply_list_tmp if (j in i)]
                except_source_weights=[final_result_list[i] for i in shaply_list_tmp if (j not in i) ]

                # shaply_values_tmp=sum([(i[0]-i[1])*i[2] for i in zip(except_source_weights,contain_source_weights,weight_)]) # 도움 되는 도메인의 로스를 크게     
                shaply_values_tmp=sum([(i[1]-i[0])*i[2] for i in zip(except_source_weights,contain_source_weights,weight_)]) # 도움 되는 도메인의 로스를 작게            
                shaply_values[j]=shaply_values_tmp

            
            self.shaply_values_update = self.alpha(self.shaply_values_update.unsqueeze(1).to(self.device)) + self.beta(torch.tensor([i for i in shaply_values.values()]).unsqueeze(1).to(self.device))
            self.shaply_values_update = self.shaply_values_update.squeeze(1)
            shaply_softmax=self.softmax_with_temperature(torch.tensor([i for i in self.shaply_values_update]),temperature=0.05) # 1~0.1

            shaply_values_softmax={}
            for k,j in zip(total_domain,shaply_softmax):# loss에 가중치 부여하는 버전에서 추가 
                shaply_values_softmax[k]=j
                
        else:

            shaply_values_softmax={}
            for k in total_domain:
                shaply_values_softmax[k]=torch.tensor(1.0)
            
            
        return shaply_values_softmax
    
    
    def softmax_with_temperature(self, preds,temperature):
        ex = torch.exp(preds/temperature)
        return ex / torch.sum(ex, axis=0)
        