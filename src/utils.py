# -*- coding: utf-8 -*-
# @Time    : 2023/4/11 16:01
# @Author  : cpark

import numpy as np
import math
import random
import os
import json
import pickle
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import warnings


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(2, item_size - 1)
    while item in item_set:
        item = random.randint(2, item_size - 1)
    return item

def neg_hard_sample(item_set, ture_item):
    item = random.choice(list(item_set))
    while item == ture_item:
        item = random.choice(list(item_set))
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # ({self.score_min:.6f} --> {score:.6f}) # 这里如果是一个值的话输出才不会有问题
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.module.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # 后面的都是采的负例
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_cat2attribute_skt(data_file):
    lines = open(data_file).readlines()
    cat2attribute = {}
    attribute_set = set()
    for line in lines:
        app, cates = line.strip().split(' ', 1)
        cates = cates.split(' ')
        cates = [int(cat) for cat in cates]
        cat2attribute[app] = cates
        attribute_set = attribute_set | set(cates)
    attribute_size = max(attribute_set) # 331
    return cat2attribute, attribute_size

def get_profile_cat_meta(data_file):
    lines = open(data_file).readlines()
    cat_meta = []
    for cat in lines[0].split(' '):
        if cat == '':
            continue
        item = int(cat)
        cat_meta.append(item)
    return cat_meta

def get_user_profile(data_file, cat_idxs):
    lines = open(data_file).readlines()
    profile = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(float(item)) if i in cat_idxs else float(item) for i, item in enumerate(items)]
        profile.append(items)
    return profile, len(profile[0])

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
    
####################################################################### for profile #######################################################################
class TorchDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    y : 2D array
        The one-hot encoded target
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        return x, y


class PredictDataset(Dataset):
    """
    Format for numpy array

    Parameters
    ----------
    X : 2D array
        The input matrix
    """

    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        return x


def create_sampler(weights, y_train):
    """
    This creates a sampler from the given weights

    Parameters
    ----------
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    y_train : np.array
        Training targets
    """
    if isinstance(weights, int):
        if weights == 0:
            need_shuffle = True
            sampler = None
        elif weights == 1:
            need_shuffle = False
            class_sample_count = np.array(
                [len(np.where(y_train == t)[0]) for t in np.unique(y_train)]
            )

            weights = 1.0 / class_sample_count

            samples_weight = np.array([weights[t] for t in y_train])

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        else:
            raise ValueError("Weights should be either 0, 1, dictionnary or list.")
    elif isinstance(weights, dict):
        # custom weights per class
        need_shuffle = False
        samples_weight = np.array([weights[t] for t in y_train])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    else:
        # custom weights
        if len(weights) != len(y_train):
            raise ValueError("Custom weights should match number of train samples.")
        need_shuffle = False
        samples_weight = np.array(weights)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return need_shuffle, sampler


def create_dataloaders(
    X_train, y_train, eval_set, weights, batch_size, num_workers, drop_last, pin_memory
):
    """
    Create dataloaders with or without subsampling depending on weights and balanced.

    Parameters
    ----------
    X_train : np.ndarray
        Training data
    y_train : np.array
        Mapped Training targets
    eval_set : list of tuple
        List of eval tuple set (X, y)
    weights : either 0, 1, dict or iterable
        if 0 (default) : no weights will be applied
        if 1 : classification only, will balanced class with inverse frequency
        if dict : keys are corresponding class values are sample weights
        if iterable : list or np array must be of length equal to nb elements
                      in the training set
    batch_size : int
        how many samples per batch to load
    num_workers : int
        how many subprocesses to use for data loading. 0 means that the data
        will be loaded in the main process
    drop_last : bool
        set to True to drop the last incomplete batch, if the dataset size is not
        divisible by the batch size. If False and the size of dataset is not
        divisible by the batch size, then the last batch will be smaller
    pin_memory : bool
        Whether to pin GPU memory during training

    Returns
    -------
    train_dataloader, valid_dataloader : torch.DataLoader, torch.DataLoader
        Training and validation dataloaders
    """
    need_shuffle, sampler = create_sampler(weights, y_train)

    train_dataloader = DataLoader(
        TorchDataset(X_train.astype(np.float32), y_train),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=need_shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )

    valid_dataloaders = []
    for X, y in eval_set:
        valid_dataloaders.append(
            DataLoader(
                TorchDataset(X.astype(np.float32), y),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        )

    return train_dataloader, valid_dataloaders


def create_group_matrix(list_groups, input_dim):
    """
    Create the group matrix corresponding to the given list_groups

    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assigned a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset

    Returns
    -------
    - group_matrix : torch matrix
        A matrix of size (n_groups, input_dim)
        where m_ij represents the importance of feature j in group i
        The rows must some to 1 as each group is equally important a priori.

    """
    check_list_groups(list_groups, input_dim)

    if len(list_groups) == 0:
        group_matrix = torch.eye(input_dim)
        return group_matrix
    else:
        n_groups = input_dim - int(np.sum([len(gp) - 1 for gp in list_groups]))
        group_matrix = torch.zeros((n_groups, input_dim))

        remaining_features = [feat_idx for feat_idx in range(input_dim)]

        current_group_idx = 0
        for group in list_groups:
            group_size = len(group)
            for elem_idx in group:
                # add importrance of element in group matrix and corresponding group
                group_matrix[current_group_idx, elem_idx] = 1 / group_size
                # remove features from list of features
                remaining_features.remove(elem_idx)
            # move to next group
            current_group_idx += 1
        # features not mentionned in list_groups get assigned their own group of singleton
        for remaining_feat_idx in remaining_features:
            group_matrix[current_group_idx, remaining_feat_idx] = 1
            current_group_idx += 1
        return group_matrix


def check_list_groups(list_groups, input_dim):
    """
    Check that list groups:
        - is a list of list
        - does not contain twice the same feature in different groups
        - does not contain unknown features (>= input_dim)
        - does not contain empty groups
    Parameters
    ----------
    - list_groups : list of list of int
        Each element is a list representing features in the same group.
        One feature should appear in maximum one group.
        Feature that don't get assign a group will be in their own group of one feature.
    - input_dim : number of feature in the initial dataset
    """
    assert isinstance(list_groups, list), "list_groups must be a list of list."

    if len(list_groups) == 0:
        return
    else:
        for group_pos, group in enumerate(list_groups):
            msg = f"Groups must be given as a list of list, but found {group} in position {group_pos}."  # noqa
            assert isinstance(group, list), msg
            assert len(group) > 0, "Empty groups are forbidding please remove empty groups []"

    n_elements_in_groups = np.sum([len(group) for group in list_groups])
    flat_list = []
    for group in list_groups:
        flat_list.extend(group)
    unique_elements = np.unique(flat_list)
    n_unique_elements_in_groups = len(unique_elements)
    msg = f"One feature can only appear in one group, please check your grouped_features."
    assert n_unique_elements_in_groups == n_elements_in_groups, msg

    highest_feat = np.max(unique_elements)
    assert highest_feat < input_dim, f"Number of features is {input_dim} but one group contains {highest_feat}."  # noqa
    return


def filter_weights(weights):
    """
    This function makes sure that weights are in correct format for
    regression and multitask TabNet

    Parameters
    ----------
    weights : int, dict or list
        Initial weights parameters given by user

    Returns
    -------
    None : This function will only throw an error if format is wrong
    """
    err_msg = """Please provide a list or np.array of weights for """
    err_msg += """regression, multitask or pretraining: """
    if isinstance(weights, int):
        if weights == 1:
            raise ValueError(err_msg + "1 given.")
    if isinstance(weights, dict):
        raise ValueError(err_msg + "Dict given.")
    return


def validate_eval_set(eval_set, eval_name, X_train, y_train):
    """Check if the shapes of eval_set are compatible with (X_train, y_train).

    Parameters
    ----------
    eval_set : list of tuple
        List of eval tuple set (X, y).
        The last one is used for early stopping
    eval_name : list of str
        List of eval set names.
    X_train : np.ndarray
        Train owned products
    y_train : np.array
        Train targeted products

    Returns
    -------
    eval_names : list of str
        Validated list of eval_names.
    eval_set : list of tuple
        Validated list of eval_set.

    """
    eval_name = eval_name or [f"val_{i}" for i in range(len(eval_set))]

    assert len(eval_set) == len(
        eval_name
    ), "eval_set and eval_name have not the same length"
    if len(eval_set) > 0:
        assert all(
            len(elem) == 2 for elem in eval_set
        ), "Each tuple of eval_set need to have two elements"
    for name, (X, y) in zip(eval_name, eval_set):
        check_input(X)
        msg = (
            f"Dimension mismatch between X_{name} "
            + f"{X.shape} and X_train {X_train.shape}"
        )
        assert len(X.shape) == len(X_train.shape), msg

        msg = (
            f"Dimension mismatch between y_{name} "
            + f"{y.shape} and y_train {y_train.shape}"
        )
        assert len(y.shape) == len(y_train.shape), msg

        msg = (
            f"Number of columns is different between X_{name} "
            + f"({X.shape[1]}) and X_train ({X_train.shape[1]})"
        )
        assert X.shape[1] == X_train.shape[1], msg

        if len(y_train.shape) == 2:
            msg = (
                f"Number of columns is different between y_{name} "
                + f"({y.shape[1]}) and y_train ({y_train.shape[1]})"
            )
            assert y.shape[1] == y_train.shape[1], msg
        msg = (
            f"You need the same number of rows between X_{name} "
            + f"({X.shape[0]}) and y_{name} ({y.shape[0]})"
        )
        assert X.shape[0] == y.shape[0], msg

    return eval_name, eval_set


def define_device(device_name):
    """
    Define the device to use during training and inference.
    If auto it will detect automatically whether to use cuda or cpu

    Parameters
    ----------
    device_name : str
        Either "auto", "cpu" or "cuda"

    Returns
    -------
    str
        Either "cpu" or "cuda"
    """
    if device_name == "cuda:0":
        if torch.cuda.is_available():
            print("cuda:0")
            return "cuda:0"
        else:
            return "cpu"
    elif device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    else:
        return device_name


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)



def check_warm_start(warm_start, from_unsupervised):
    """
    Gives a warning about ambiguous usage of the two parameters.
    """
    if warm_start and from_unsupervised is not None:
        warn_msg = "warm_start=True and from_unsupervised != None: "
        warn_msg = "warm_start will be ignore, training will start from unsupervised weights"
        warnings.warn(warn_msg)
    return


def check_embedding_parameters(cat_dims, cat_idxs, cat_emb_dim):
    """
    Check parameters related to embeddings and rearrange them in a unique manner.
    """
    if (cat_dims == []) ^ (cat_idxs == []):
        if cat_dims == []:
            msg = "If cat_idxs is non-empty, cat_dims must be defined as a list of same length."
        else:
            msg = "If cat_dims is non-empty, cat_idxs must be defined as a list of same length."
        raise ValueError(msg)
    elif len(cat_dims) != len(cat_idxs):
        msg = "The lists cat_dims and cat_idxs must have the same length."
        raise ValueError(msg)

    if isinstance(cat_emb_dim, int):
        cat_emb_dims = [cat_emb_dim] * len(cat_idxs)
    else:
        cat_emb_dims = cat_emb_dim

    # check that all embeddings are provided
    if len(cat_emb_dims) != len(cat_dims):
        msg = f"""cat_emb_dim and cat_dims must be lists of same length, got {len(cat_emb_dims)}
                    and {len(cat_dims)}"""
        raise ValueError(msg)

    # Rearrange to get reproducible seeds with different ordering
    if len(cat_idxs) > 0:
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        cat_emb_dims = [cat_emb_dims[i] for i in sorted_idxs]

    return cat_dims, cat_idxs, cat_emb_dims
