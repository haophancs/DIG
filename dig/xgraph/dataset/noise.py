import torch
import numpy as np
import random


def extract_test_nodes(data, num_samples=None, seed=42):
    try:
        node_indices = data.test_mask.cpu().numpy().nonzero()[0]
    except AttributeError:
        node_indices = np.arange(data.x.size(0))[-int(data.x.size(0) * 0.1):]
    if num_samples is not None:
        np.random.seed(seed)
        node_indices = np.random.choice(node_indices, num_samples, replace=False).tolist()
    return node_indices


def add_noise_features(data, prop_noise_feats=0.20, binary=False, p=0.5):
    num_noise = int(data.x.size(1) * prop_noise_feats)

    # Number of nodes in the dataset
    num_nodes = data.x.size(0)

    # Define some random features, in addition to existing ones
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
    noise_feat = m.sample((num_noise, num_nodes)).T[0]
    # noise_feat = torch.randint(2,size=(num_nodes, num_noise))
    if not binary:
        noise_feat_bis = torch.rand((num_nodes, num_noise))
        # noise_feat_bis = noise_feat_bis - noise_feat_bis.mean(1, keepdim=True)
        noise_feat = torch.min(noise_feat, noise_feat_bis)
    data.x = torch.cat([noise_feat, data.x], dim=-1)

    return data


def add_noise_neighbours(data, node_indices=None, prop_noise_nodes=0.20, binary=False, p=0.5, connectedness='medium', c=0.001, graph_classification=False):
    print(f"Adding noise neighbours{' (graph classification)' * int(graph_classification)}...")
    if not node_indices:
        node_indices = extract_test_nodes(data)
    num_noise = int(prop_noise_nodes * data.x.size(0))

    # Number of features in the dataset
    num_feat = data.x.size(1)
    num_nodes = data.x.size(0)

    # Add new nodes with random features
    m = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))
    noise_nei_feat = m.sample((num_feat, num_noise)).T[0]
    if not binary:
        noise_nei_feat_bis = torch.rand((num_noise, num_feat))
        noise_nei_feat = torch.min(noise_nei_feat, noise_nei_feat_bis)
    data.x = torch.cat([data.x, noise_nei_feat], dim=0)
    new_num_nodes = data.x.size(0)

    # Add random edges incident to these nodes - according to desired level of connectivity
    if connectedness == 'high':  # few highly connected new nodes
        adj_matrix = torch.randint(2, size=(num_noise, new_num_nodes))

    elif connectedness == 'medium':  # more sparser nodes, connected to targeted nodes of interest
        m = torch.distributions.bernoulli.Bernoulli(torch.tensor([c]))
        adj_matrix = m.sample((new_num_nodes, num_noise)).T[0]
        # each node of interest has at least one noisy neighbour
        for i, idx in enumerate(node_indices):
            try:
                adj_matrix[i, idx] = 1
            except IndexError:  # in case num_noise < test_samples
                pass
    # low connectivity
    else:
        adj_matrix = torch.zeros((num_noise, new_num_nodes))
        for i, idx in enumerate(node_indices):
            try:
                adj_matrix[i, idx] = 1
            except IndexError:
                pass
        while num_noise > i + 1:
            l = node_indices + list(range(num_nodes, (num_nodes + i)))
            i += 1
            idx = random.sample(l, 2)
            adj_matrix[i, idx[0]] = 1
            adj_matrix[i, idx[1]] = 1

    # Add defined edges to data adjacency matrix, in the correct form
    for i, row in enumerate(adj_matrix):
        indices = (row == 1).nonzero()
        indices = torch.transpose(indices, 0, 1)
        a = torch.full_like(indices, i + num_nodes)
        adj_row = torch.cat((a, indices), 0)
        data.edge_index = torch.cat((data.edge_index, adj_row), 1)
        adj_row = torch.cat((indices, a), 0)
        data.edge_index = torch.cat((data.edge_index, adj_row), 1)

    data.__num_nodes__ = data.x.size(0)

    if not graph_classification:
        # Update train/test/val masks - don't include these new nodes anywhere as there have no labels
        test_mask = torch.empty(num_noise)
        test_mask = torch.full_like(test_mask, False).bool()
        data.train_mask = torch.cat((data.train_mask, test_mask), -1)
        data.val_mask = torch.cat((data.val_mask, test_mask), -1)
        data.test_mask = torch.cat((data.test_mask, test_mask), -1)
        # Update labels randomly - no effect on the rest
        data.y = torch.cat((data.y, test_mask), -1)
    return data
