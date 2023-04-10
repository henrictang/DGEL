# -*- coding: utf-8 -*

'''
This is training code of DGEL
The load_data and t-batch method refer to JODIE which we have already cited
If you use our code or our paper, please cite our paper:
Dynamic Graph Evolution Learning for Recommendation
published at SIGIR 2023
'''

import numpy as np
from library_data import *
import library_models as lib
from library_models import *
from copy import deepcopy

# Initialize Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="wikipedia", help='Name of the dataset')
parser.add_argument('--epochs', default=30, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding_dim', default=32, type=int, help='Number of dimensions of the dynamic embedding')
parser.add_argument('--sample_length', type=int, default=100, help='sample length')
parser.add_argument('--bpr_coefficient', type=float, default=0.0005, help='BPR is extremely bigger, e.g, 0.79. But each MSE is much small, e.g, 0.0008, so bpr_coefficient should be [0.001, 0.0005]')
parser.add_argument('--l2u', type=float, default=1.0, help='regular coefficient of user')
parser.add_argument('--l2i', type=float, default=1.0, help='regular coefficient of item')
parser.add_argument('--l2', type=float, default=1e-2, help='l2 penalty')
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on')
parser.add_argument('--span_num', default=500, type=int, help='time span number')
parser.add_argument('--train_proportion', default=0.8, type=float, help='Fraction of interactions (from the beginning) that are used for training.The next 10% are used for validation and the next 10% for testing')
args = parser.parse_args()

final_embedding_dim = args.embedding_dim

args.datapath = "data/%s.csv" % args.dataset
if args.train_proportion > 0.8:
    sys.exit('Training sequence proportion cannot be greater than 0.8.')

# Set your GPU here
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load Data
[user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
 item2id, item_sequence_id, item_timediffs_sequence,
 timestamp_sequence, feature_sequence, timedifference_sequence_for_adj] = load_network(args)

num_interactions = len(user_sequence_id)
num_users = len(user2id)
num_items = len(item2id) + 1  # one extra item for "none-of-these"
num_features = len(feature_sequence[0])

print("***** Dataset statistics:\n  %d users\n  %d items\n  %d interactions *****\n" % (num_users, num_items, num_interactions))

# Set training, validation and test boundaries
train_end_idx = validation_start_idx = int(num_interactions * args.train_proportion)
test_start_idx = int(num_interactions * (args.train_proportion+0.1))
test_end_idx = int(num_interactions * (args.train_proportion+0.2))

# Set batching timespan
'''
Timespan is the frequency at which the batches are created and the DGEL model is trained.
As the data arrives in a temporal order, the interactions within a timespan are added into batches (using the T-batch algorithm).
Longer timespans mean more interactions are processed and the training time is reduced, however it requires more GPU memory.
Longer timespan leads to less frequent model updates.
'''

timespan = timestamp_sequence[-1] - timestamp_sequence[0]
tbatch_timespan = timespan / args.span_num

# Initialize model and parameters
model = DGEL(args, num_features, num_users, num_items, final_embedding_dim).cuda()
MSELoss = nn.MSELoss()

learning_rate = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.l2)

# Initialize embeddings
# The initial user and item embeddings are learned during training as well
initial_user_embedding = nn.Parameter(F.normalize(torch.rand(final_embedding_dim).cuda(), dim=0))
initial_item_embedding = nn.Parameter(F.normalize(torch.rand(final_embedding_dim).cuda(), dim=0))
model.initial_user_embedding = initial_user_embedding
model.initial_item_embedding = initial_item_embedding

user_embeddings = initial_user_embedding.repeat(num_users, 1)
item_embeddings = initial_item_embedding.repeat(num_items, 1)

item_embedding_static = Variable(torch.eye(num_items).cuda())
user_embedding_static = Variable(torch.eye(num_users).cuda())

# Run training process
print("***** Training the DGEL model for %d epochs *****" % args.epochs)

user_adj = None
item_adj = None

for ep in range(args.epochs):

    start_time = time.time()

    # Initialize embedding trajectory storage
    user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, final_embedding_dim).cuda())
    item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, final_embedding_dim).cuda())

    optimizer.zero_grad()
    reinitialize_tbatches()
    total_loss, loss, total_interaction_count, total_batch_count = 0, 0, 0, 0

    tbatch_start_time = None
    tbatch_to_insert = -1
    tbatch_full = False

    # Record neighbors based on interactions
    # Record timestamp of item (users) for users (items)
    user_adj = defaultdict(list)
    item_adj = defaultdict(list)
    user_timestamp_for_adj = defaultdict(list)
    item_timestamp_for_adj = defaultdict(list)

    for j in range(train_end_idx):
        # Read interaction
        userid = user_sequence_id[j]
        itemid = item_sequence_id[j]
        feature = feature_sequence[j]
        user_timediff = user_timediffs_sequence[j]
        item_timediff = item_timediffs_sequence[j]
        timestamp = timestamp_sequence[j]
        timestamp_for_adj = timedifference_sequence_for_adj[j]

        user_adj[userid].append(itemid)
        item_adj[itemid].append(userid)
        user_timestamp_for_adj[userid].append(timestamp_for_adj)
        item_timestamp_for_adj[itemid].append(timestamp_for_adj)

        # to save time for number of neighbors
        length_of_user = len(user_adj[userid])
        length_of_item = len(item_adj[itemid])
        if length_of_user > args.sample_length:
            user_adj[userid] = user_adj[userid][length_of_user-args.sample_length:]
            user_timestamp_for_adj[userid] = user_timestamp_for_adj[userid][length_of_user-args.sample_length:]
        if length_of_item > args.sample_length:
            item_adj[itemid] = item_adj[itemid][length_of_item-args.sample_length:]
            item_timestamp_for_adj[itemid] = item_timestamp_for_adj[itemid][length_of_item-args.sample_length:]

        # Create T-batch and add current interaction into t-batch
        tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
        lib.tbatchid_user[userid] = tbatch_to_insert
        lib.tbatchid_item[itemid] = tbatch_to_insert

        lib.current_tbatches_user[tbatch_to_insert].append(userid)
        lib.current_tbatches_item[tbatch_to_insert].append(itemid)
        lib.current_tbatches_feature[tbatch_to_insert].append(feature)
        lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
        lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
        lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
        lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

        # lib.current_tbatches_user_history_differ[tbatch_to_insert].append([args.temperature*(timestamp_for_adj-each) for each in user_timestamp_for_adj[userid]])
        # lib.current_tbatches_item_history_differ[tbatch_to_insert].append([args.temperature*(timestamp_for_adj-each) for each in item_timestamp_for_adj[itemid]])

        lib.current_tbatches_user_history_differ[tbatch_to_insert].append([-1 * (timestamp_for_adj - each) / (timestamp_for_adj - user_timestamp_for_adj[userid][0])
                                                                           if timestamp_for_adj - user_timestamp_for_adj[userid][0] != 0 else 1 for each in user_timestamp_for_adj[userid]])
        lib.current_tbatches_item_history_differ[tbatch_to_insert].append([-1 * (timestamp_for_adj - each) / (timestamp_for_adj - item_timestamp_for_adj[itemid][0])
                                                                           if timestamp_for_adj - item_timestamp_for_adj[itemid][0] != 0 else 1 for each in item_timestamp_for_adj[itemid]])

        # current adj should not be allowed to touch future
        lib.current_tbatches_user_adj[tbatch_to_insert].append(deepcopy(user_adj[userid]))
        lib.current_tbatches_item_adj[tbatch_to_insert].append(deepcopy(item_adj[itemid]))

        if tbatch_start_time is None:
            tbatch_start_time = timestamp

        # After all interactions in the timespan are converted to t-batchs,
        # Forward pass to create embedding trajectories and calculate loss
        if timestamp - tbatch_start_time > tbatch_timespan:
            tbatch_start_time = timestamp # RESET START TIME FOR THE NEXT TBATCHES

            # Process t-batches
            for i in range(len(lib.current_tbatches_user)):
                total_interaction_count += len(lib.current_tbatches_interactionids[i])
                total_batch_count += len(lib.current_tbatches_user)

                # Load current t-batch
                tbatch_userids = torch.LongTensor(lib.current_tbatches_user[i]).cuda() # Recall "lib.current_tbatches_user[i]" has unique elements
                tbatch_itemids = torch.LongTensor(lib.current_tbatches_item[i]).cuda() # Recall "lib.current_tbatches_item[i]" has unique elements
                tbatch_interactionids = torch.LongTensor(lib.current_tbatches_interactionids[i]).cuda()
                feature_tensor = Variable(torch.Tensor(lib.current_tbatches_feature[i]).cuda()) # Recall "lib.current_tbatches_feature[i]" is list of list, so "feature_tensor" is a 2-d tensor
                user_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_user_timediffs[i]).cuda()).unsqueeze(1)
                item_timediffs_tensor = Variable(torch.Tensor(lib.current_tbatches_item_timediffs[i]).cuda()).unsqueeze(1)
                tbatch_itemids_previous = torch.LongTensor(lib.current_tbatches_previous_item[i]).cuda()
                item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]

                tbatch_user_history_differ = lib.current_tbatches_user_history_differ[i]
                tbatch_item_history_differ = lib.current_tbatches_item_history_differ[i]

                # Project previous user embeddings to current time
                # We treat current time as previous time's future for computing convenience
                user_embedding_input = user_embeddings[tbatch_userids, :].cuda()
                item_embedding_input = item_embeddings[tbatch_itemids, :].cuda()

                # future prediction
                user_projected_embedding = model.forward(user_embedding_input, None, None, timediffs=user_timediffs_tensor, features=feature_tensor, select='project')
                user_item_embedding = torch.cat([user_projected_embedding, item_embedding_previous, item_embedding_static[tbatch_itemids_previous, :], user_embedding_static[tbatch_userids, :]], dim=1).cuda()
                predicted_item_embedding = model.predict_item_embedding(user_item_embedding)

                # Evolution loss for Future drifting
                # There are two parts for evolution loss:
                # 1) Future drifting from current.
                # 2) Current update from previous

                loss += MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static[tbatch_itemids,:]], dim=1).detach())

                # Update dynamic sub-embeddings based on current interaction
                # Note we only update current user and item, instead of all nodes!
                user_adj_, user_length_mask, user_max_length = model.adj_sample(lib.current_tbatches_user_adj[i], args.sample_length)
                item_adj_, item_length_mask, item_max_length = model.adj_sample(lib.current_tbatches_item_adj[i], args.sample_length)

                user_adj_td, _, _, = model.adj_sample(tbatch_user_history_differ, args.sample_length, 'timediffer')
                item_adj_td, _, _, = model.adj_sample(tbatch_item_history_differ, args.sample_length, 'timediffer')

                user_adj_em = item_embeddings[torch.LongTensor(user_adj_).cuda(), :].cuda()
                item_adj_em = user_embeddings[torch.LongTensor(item_adj_).cuda(), :].cuda()

                # 1: inherent interaction
                user_inter_embeddings = model.interaction_aggregate(user_embedding_input, item_embedding_input, feature_tensor, user_timediffs_tensor, 'user')
                item_inter_embeddings = model.interaction_aggregate(item_embedding_input, user_embedding_input, feature_tensor, item_timediffs_tensor, 'item')

                # 2: time-decay neighbor GCN
                user_adj_embedding = model.neighbor_aggregate(user_embedding_input, user_adj_em, torch.LongTensor(user_length_mask).cuda(),user_max_length, torch.FloatTensor(user_adj_td).cuda(), 'user')
                item_adj_embedding = model.neighbor_aggregate(item_embedding_input, item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length, torch.FloatTensor(item_adj_td).cuda(), 'item')

                # 3: symbiotic local learning
                user_ext_embeddings = model.excitement_aggregate(user_adj_em, torch.LongTensor(user_length_mask).cuda(), user_max_length)
                item_ext_embeddings = model.excitement_aggregate(item_adj_em, torch.LongTensor(item_length_mask).cuda(), item_max_length)
                user_local_embedding, item_local_embedding = model.local_aggregate(user_embedding_input, item_embedding_input, user_ext_embeddings, item_ext_embeddings)

                # forward with re-scaling network
                user_embedding_output = model.forward(user_embedding_input, user_inter_embeddings, user_local_embedding,
                                                      timediffs=user_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=user_adj_embedding, select='user_update')
                item_embedding_output = model.forward(item_embedding_input, item_inter_embeddings, item_local_embedding,
                                                      timediffs=item_timediffs_tensor, features=feature_tensor,
                                                      adj_embeddings=item_adj_embedding, select='item_update')

                # Save embeddings
                item_embeddings[tbatch_itemids, :] = item_embedding_output
                user_embeddings[tbatch_userids, :] = user_embedding_output
                user_embeddings_timeseries[tbatch_interactionids, :] = user_embedding_output
                item_embeddings_timeseries[tbatch_interactionids, :] = item_embedding_output

                # Evolution loss for current updating from previous
                loss += args.l2i*MSELoss(item_embedding_output.cuda(), item_embedding_input.cuda().detach())
                loss += args.l2u*MSELoss(user_embedding_output.cuda(), user_embedding_input.cuda().detach())

                # sample negative item for BPR-loss
                neg_items = model.sample_for_BPR(lib.current_tbatches_user[i], num_items-1, lib.current_tbatches_user_adj[i], bpr_avoid=None)  # num_items = len(item2Id) + 1
                neg_item_embeddings = item_embeddings[torch.LongTensor(neg_items).cuda(), :]

                # BPR-loss for current t-batch
                # The bpr loss is extremely bigger like 0.79 but each MSE loss is so small like 0.0008
                # So the bpr_coefficient being 0.001 could balance the two task loss
                bpr_loss = model.bpr_loss(user_embedding_output, item_embedding_output, neg_item_embeddings.detach())
                loss += args.bpr_coefficient*bpr_loss

            # Back-propagate of t-batches
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Reset for next t-batch
            loss = 0
            item_embeddings.detach_()  # Detachment is needed to prevent double propagation of gradient
            user_embeddings.detach_()
            item_embeddings_timeseries.detach_()
            user_embeddings_timeseries.detach_()

            # Re-Initialization
            reinitialize_tbatches()
            tbatch_to_insert = -1

    # End of epoch
    print("\nTotal loss in this epoch = %f" % (total_loss))
    item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
    user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)

    end_time = time.time()
    print('Time cost of processing this epoch: %.1f s' % (end_time-start_time))

    # Save current epoch model
    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx,
               user_adj, item_adj, user_timestamp_for_adj, item_timestamp_for_adj,
               user_embeddings_timeseries, item_embeddings_timeseries)

    user_embeddings = initial_user_embedding.repeat(num_users, 1)
    item_embeddings = initial_item_embedding.repeat(num_items, 1)

# Save final model of final epoch
print("\n***** Training complete. Saving final model. *****\n")
save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_adj, item_adj,
           user_timestamp_for_adj, item_timestamp_for_adj, user_embeddings_timeseries, item_embeddings_timeseries)

