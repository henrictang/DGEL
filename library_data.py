# -*- coding: utf-8 -*
'''
This is data loading code for DGEL
The load_data and t-batch method refer to JODIE
'''

from __future__ import division
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
import argparse


# Load data
def load_network(args, time_scaling=True):
    '''
    The dataset should be in the following format:
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be at least 1 dimensional. If there are no features, use 0 for all interactions.
    '''

    dataset = args.dataset
    datapath = args.datapath

    user_sequence = []
    item_sequence = []
    label_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []

    print("\n***** Loading %s dataset from file: %s *****" % (dataset, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # format: user, item, timestamp, state label, feature list
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        timestamp_sequence.append(float(ls[2]) - start_timestamp) 
        y_true_labels.append(int(ls[3]))
        feature_sequence.append(list(map(float, ls[4:])))
    f.close()

    user_sequence = np.array(user_sequence)
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formatting item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formatting user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    # for time-decay GCN
    timedifference_sequence_for_adj = (np.array(timestamp_sequence)/(3600*24)).astype(int)  # based on day

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("***** dataset loading completed *****\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence,
            item2id, item_sequence_id, item_timedifference_sequence,
            timestamp_sequence, feature_sequence, timedifference_sequence_for_adj]


