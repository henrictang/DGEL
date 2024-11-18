# -*- coding: utf-8 -*
'''
This is implementation and support of the model DGEL.
'''

from __future__ import division
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
import numpy as np
import math
import random
import sys
from collections import defaultdict
import os
import gpustat
from itertools import chain
from tqdm import tqdm, trange, tqdm_notebook, tnrange
import csv
import time

PATH = "./"
total_reinitialization_count = 0


# BPR Loss
class BPR_loss(nn.Module):
    def __init__(self):
        super(BPR_loss, self).__init__()

    def forward(self, user_embedding, item_embeddings, item_neg_embeddings):
        pos = torch.sum(user_embedding*item_embeddings, 1)
        neg = torch.sum(user_embedding*item_neg_embeddings, 1)
        sig = torch.clamp(torch.sigmoid(pos - neg), 1e-6)
        loss = -torch.mean(torch.log(sig))

        return loss


# A normal-linear Layer
class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class DGEL(nn.Module):
    def __init__(self, args, num_features, num_users, num_items, final_embedding_dim):
        super(DGEL,self).__init__()

        print("***** Initializing the DGEL model *****")
        self.embedding_dim = args.embedding_dim
        self.final_embedding_dim = final_embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items

        self.sample_length = args.sample_length
        self.num_features = num_features

        # initial embeddings
        self.initial_user_embedding = nn.Parameter(torch.Tensor(final_embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(final_embedding_dim))

        # history enhancement (simple RNN)
        rnn_input_size_items = rnn_input_size_users = self.final_embedding_dim*3
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.final_embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.final_embedding_dim)

        # future drifting and prediction layers
        self.embedding_layer = NormalLinear(1, self.final_embedding_dim)

        self.prediction_layer = nn.Linear(self.user_static_embedding_size + self.item_static_embedding_size + self.final_embedding_dim * 2, self.item_static_embedding_size + self.final_embedding_dim)

        # 1: interaction-level update
        self.interaction_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_user_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_user_under_feature = nn.Linear(self.num_features, self.embedding_dim, bias=False)

        self.interaction_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_item_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.interaction_item_under_feature = nn.Linear(self.num_features, self.embedding_dim, bias=False)

        self.user_interaction_time_interval = nn.Linear(1, self.embedding_dim, bias=False)
        self.item_interaction_time_interval = nn.Linear(1, self.embedding_dim, bias=False)

        # 2: Neighbor-level update
        self.weigh_item_adj = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_user_adj = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_item_itself = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.weigh_user_itself = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        # 3: Interaction-Neighbor (local learning) update
        self.base_intensity_for_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_user_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_user_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        self.base_intensity_for_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_item_under_user = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)
        self.transfer_ex_for_item_under_item = nn.Linear(self.final_embedding_dim, self.embedding_dim, bias=False)

        # 4: Rescaling network
        self.ReScale_Inter_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Inter_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        self.ReScale_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        self.ReScale_Inter_Neigh_first = nn.Linear(self.embedding_dim, int(self.embedding_dim / 2), bias=True)
        self.ReScale_Inter_Neigh_factor = nn.Linear(int(self.embedding_dim / 2), 1, bias=True)

        # BPR loss
        self.bpr_loss = BPR_loss().cuda()

        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU()

        print("***** DGEL initialization complete ****\n")

    def forward(self, center_embeddings, inter_embeddings, local_embeddings, timediffs=None,  features=None, adj_embeddings=None, select=None):

        if select != 'project':
            # re-scaling on sub-embeddings
            inter_embeddings = self.re_scaling(inter_embeddings, 'Inter')
            adj_embeddings = self.re_scaling(adj_embeddings, 'Neigh')
            local_embeddings = self.re_scaling(local_embeddings, 'Inter_Neigh')
            inputs = torch.cat([inter_embeddings, adj_embeddings, local_embeddings], dim=1)

        if select == 'item_update':
            item_embedding_output = self.item_rnn(inputs, center_embeddings)
            return item_embedding_output

        elif select == 'user_update':
            user_embedding_output = self.user_rnn(inputs, center_embeddings)
            return user_embedding_output

        elif select == 'project':
            user_projected_embedding = self.context_convert(center_embeddings, timediffs, features)
            return user_projected_embedding

    # Inherent interaction update
    def interaction_aggregate(self, center_embeddings, addition_embedding, features, timediffs, target=None):
        if target == 'user':
            output_embedding = self.interaction_user(center_embeddings) + self.interaction_user_under_item(addition_embedding) \
                                 + self.interaction_user_under_feature(features) + self.user_interaction_time_interval(timediffs)
        if target == 'item':
            output_embedding = self.interaction_item(center_embeddings) + self.interaction_item_under_user(addition_embedding) \
                                 + self.interaction_item_under_feature(features) + self.item_interaction_time_interval(timediffs)
        return self.LeakyReLU(output_embedding).cuda()

    # time-decay neighbor GCN
    def neighbor_aggregate(self, center_embedding, adj_embeddings, length_mask, max_length, history_timediffer, target=None):
        mask = torch.arange(max_length)[None, :].cuda() < length_mask[:, None].cuda()
        weight = torch.softmax(history_timediffer, dim=1).cuda()

        if target == 'user':
            adj = self.weigh_user_adj(adj_embeddings)
            itself = self.weigh_user_itself(center_embedding)
        else:
            adj = self.weigh_item_adj(adj_embeddings)
            itself = self.weigh_item_itself(center_embedding)

        final_aggregation = itself + torch.sum(weight.unsqueeze(2)*adj*mask.view(mask.shape[0], -1, 1).float().cuda(), 1)
        return self.LeakyReLU(final_aggregation).cuda()

    # pooling for local symbiotic learning
    def excitement_aggregate(self, embeddings, length_mask, max_length):
        mask = torch.arange(max_length)[None, :].cuda() < length_mask[:, None].cuda()
        mask_embeddings = embeddings*mask.view(mask.shape[0], -1, 1).float().cuda()
        final_aggregation = (torch.sum(mask_embeddings, dim=1).cuda()) / (length_mask.unsqueeze(1).cuda())

        return final_aggregation.cuda()

    # local symbiotic learning
    def local_aggregate(self, user_embeddings, item_embeddings, user_excitement, item_excitement):
        user_level3_base = self.base_intensity_for_user(user_embeddings - item_embeddings)
        user_level3_item_ex = self.transfer_ex_for_user_under_item(item_excitement)
        user_level3_user_ex = self.transfer_ex_for_user_under_user(user_excitement)
        user_level3_out = user_level3_base + user_level3_item_ex + user_level3_user_ex

        item_level3_base = self.base_intensity_for_item(item_embeddings - user_embeddings)
        item_level3_user_ex = self.transfer_ex_for_item_under_user(user_excitement)
        item_level3_item_ex = self.transfer_ex_for_item_under_item(item_excitement)
        item_level3_out = item_level3_base + item_level3_user_ex + item_level3_item_ex

        return self.LeakyReLU(user_level3_out).cuda(), self.LeakyReLU(item_level3_out).cuda()

    # re-scaling network
    def re_scaling(self, embeddings, target=None):
        if embeddings is None:
            print('No embeddings to re-scale')
            return embeddings

        # Re-scaling Network
        if target == 'Inter':
            Inter_first = self.LeakyReLU(self.ReScale_Inter_first(embeddings))
            Inter_factor = self.LeakyReLU(self.ReScale_Inter_factor(Inter_first))
            return Inter_factor*embeddings
        if target == 'Neigh':
            Neigh_first = self.LeakyReLU(self.ReScale_Neigh_first(embeddings))
            Neigh_factor = self.LeakyReLU(self.ReScale_Neigh_factor(Neigh_first))
            return Neigh_factor*embeddings
        if target == 'Inter_Neigh':
            Inter_Neigh_first = self.LeakyReLU(self.ReScale_Inter_Neigh_first(embeddings))
            Inter_Neigh_factor = self.LeakyReLU(self.ReScale_Inter_Neigh_factor(Inter_Neigh_first))
            return Inter_Neigh_factor*embeddings

    # future drift
    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings.cuda()

    # prediction
    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out.cuda()

    # negative sampling
    def sample_for_BPR(self, user, id_length, adj, bpr_avoid=None):
        sample_length = len(adj)
        sample = []
        for i in range(sample_length):
            if len(set(adj[i])) >= id_length:
                print('It can not have negative sample, change loss function pls')
                break
            while 1:
                s = random.randint(0, id_length)
                if s not in adj[i]:
                    sample.append(s)
                    break
                '''
                if s not in bpr_avoid[user[i]]:
                    sample.append(s)
                    break
                '''

        return sample

    def adj_sample(self, adj_seq, sam_l, target=None):
        temp = -1000 if target == 'timediffer' else 0
        adjs = []
        length = [len(seq[:sam_l]) for seq in adj_seq]
        max_length = max(length)
        for seq in adj_seq:
            adjs.append(seq[::-1][:sam_l] + (max_length - len(seq[:sam_l]))*[temp])
        return adjs, length, max_length


# Initialize t-batch variable
def reinitialize_tbatches():
    global current_tbatches_interactionids, current_tbatches_user, current_tbatches_item, current_tbatches_timestamp, current_tbatches_feature, current_tbatches_label, current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, current_tbatches_item_timediffs, current_tbatches_user_timediffs_next
    global current_tbatches_user_adj, current_tbatches_item_adj  # item和user的邻居：item的邻居是购买过item的user，user的邻居是买过的item
    global current_tbatches_user_history_differ, current_tbatches_item_history_differ  # time-interval in t-bacth (may different for one node at different time)

    # list of users of each tbatch up to now
    current_tbatches_interactionids = defaultdict(list)
    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_timestamp = defaultdict(list)
    current_tbatches_feature = defaultdict(list)
    current_tbatches_label = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_user_timediffs_next = defaultdict(list)

    current_tbatches_user_adj = defaultdict(list)
    current_tbatches_item_adj = defaultdict(list)

    current_tbatches_user_history_differ = defaultdict(list)
    current_tbatches_item_history_differ = defaultdict(list)

    # the latest tbatch a user is in
    tbatchid_user = defaultdict(lambda: -1)

    # the latest tbatch a item is in
    tbatchid_item = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count +=1


# Save model
def save_model(model, optimizer, args, epoch, user_embeddings, item_embeddings, train_end_idx, user_adj, item_adj,
               user_timestamp_for_adj, item_timestamp_for_adj,
               user_embeddings_time_series=None, item_embeddings_time_series=None, path=PATH):
    print("*** Saving embeddings and model ***")
    state = {
            'user_embeddings': user_embeddings.data.cpu().numpy(),
            'item_embeddings': item_embeddings.data.cpu().numpy(),
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'train_end_idx': train_end_idx,
            'user_adj': user_adj,
            'item_adj': item_adj,
            'user_timestamp_for_adj': user_timestamp_for_adj,
            'item_timestamp_for_adj': item_timestamp_for_adj
            }

    if user_embeddings_time_series is not None:
        state['user_embeddings_time_series'] = user_embeddings_time_series.data.cpu().numpy()
        state['item_embeddings_time_series'] = item_embeddings_time_series.data.cpu().numpy()

    directory = os.path.join(path, 'saved_models/%s/' % args.dataset)

    if not os.path.exists(directory):
        os.makedirs(directory)
    path = 'saved_models/%s/checkpoint_DGEL_epoch%d_size%d_sample%d.pth.tar' % (args.dataset, epoch, args.embedding_dim, args.sample_length)

    torch.save(state, path)
    print("*** Saved embeddings and model to file: %s ***\n\n" % path)


# Load model
def load_model(model, args, epoch):

    path = 'saved_models/%s/checkpoint_DGEL_epoch%d_size%d_sample%d.pth.tar' % (args.dataset, epoch, args.embedding_dim, args.sample_length)
    checkpoint = torch.load(path)
    print("Loading saved embeddings and model: %s" % path)

    args.start_epoch = checkpoint['epoch']
    user_embeddings = Variable(torch.from_numpy(checkpoint['user_embeddings']).cuda())
    item_embeddings = Variable(torch.from_numpy(checkpoint['item_embeddings']).cuda())

    user_timestamp_for_adj = checkpoint['user_timestamp_for_adj']
    item_timestamp_for_adj = checkpoint['item_timestamp_for_adj']

    user_adj = checkpoint['user_adj']
    item_adj = checkpoint['item_adj']

    try:
        train_end_idx = checkpoint['train_end_idx'] 
    except KeyError:
        train_end_idx = None

    try:
        user_embeddings_time_series = Variable(torch.from_numpy(checkpoint['user_embeddings_time_series']).cuda())
        item_embeddings_time_series = Variable(torch.from_numpy(checkpoint['item_embeddings_time_series']).cuda())
    except:
        user_embeddings_time_series = None
        item_embeddings_time_series = None

    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    return [model, user_embeddings, item_embeddings, user_adj, item_adj,
            user_timestamp_for_adj, item_timestamp_for_adj,
            user_embeddings_time_series, item_embeddings_time_series, train_end_idx]


# Set user and item embeddings to the end of the training
def set_embeddings_training_end(user_embeddings, item_embeddings, user_embeddings_time_series, item_embeddings_time_series, user_data_id, item_data_id, train_end_idx):
    userid2lastidx = {}
    for cnt, userid in enumerate(user_data_id[:train_end_idx]):
        userid2lastidx[userid] = cnt
    itemid2lastidx = {}
    for cnt, itemid in enumerate(item_data_id[:train_end_idx]):
        itemid2lastidx[itemid] = cnt

    try:
        final_embedding_dim = user_embeddings_time_series.size(1)
    except:
        final_embedding_dim = user_embeddings_time_series.shape[1]
    for userid in userid2lastidx:
        user_embeddings[userid, :final_embedding_dim] = user_embeddings_time_series[userid2lastidx[userid]]
    for itemid in itemid2lastidx:
        item_embeddings[itemid, :final_embedding_dim] = item_embeddings_time_series[itemid2lastidx[itemid]]

    user_embeddings.detach_()
    item_embeddings.detach_()

