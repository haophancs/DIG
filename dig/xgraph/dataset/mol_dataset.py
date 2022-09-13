import os
import re
import torch
import numpy as np
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, InMemoryDataset, \
    download_url
from .noise import add_noise_neighbours, add_noise_features

try:
    from rdkit import Chem
except ImportError:
    Chem = None
import os.path as osp
import zipfile
import gzip

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def maybe_log(path, log=True):
    if log:
        print('Extracting', path)


def extract_gz(path, folder, log=True):
    maybe_log(path, log)
    with gzip.open(path, 'r') as r:
        with open(osp.join(folder, '.'.join(os.path.basename(path).split('.')[:-1])), 'wb') as w:
            w.write(r.read())


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    maybe_log(path, log)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


class MoleculeDataset(InMemoryDataset):
    r"""
    The extension of MoleculeNet with `MUTAG <https://pubs.acs.org/doi/10.1021/jm00106a046>`_.

    The `MoleculeNet benchmark collection <http://moleculenet.ai/datasets-1>`_ from the
    `MoleculeNet: A Benchmark for Molecular Machine Learning <https://arxiv.org/abs/1703.00564>`_
    paper, containing datasets from physical chemistry, biophysics and physiology.

    The MoleculeNet datasets come with the additional node and edge features introduced by
    the `Open Graph Benchmark <https://ogb.stanford.edu/docs/graphprop/>`_, and the node features
    in MUTAG dataset are one hot features denoting the atom types.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"MUTAG"`, :obj:`"ESOL"`,
            :obj:`"FreeSolv"`, :obj:`"Lipo"`, :obj:`"PCBA"`, :obj:`"MUV"`,
            :obj:`"HIV"`, :obj:`"BACE"`, :obj:`"BBPB"`, :obj:`"Tox21"`,
            :obj:`"ToxCast"`, :obj:`"SIDER"`, :obj:`"ClinTox"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/{}'
    additional_ds_url ={
        'mutag': 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip',
        'mutagenicity': 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/Mutagenicity.zip'
    }
    additional_ds_num_graphs = {
        'mutag': 189,
        'mutagenicity': 4338
    }
    additional_ds_num_classes = {
        'mutag': 7,
        'mutagenicity': 14
    }

    # Format: name: [display_name, url_name, filename, smiles_idx, y_idx]
    names = {
        'mutag': ['MUTAG', 'MUTAG.zip', None, None],
        'mutagenicity': ['Mutagenicity', 'Mutagenicity.zip', None, None],
        'esol': ['ESOL', 'delaney-processed.csv', 'delaney-processed.csv', -1, -2],
        'freesolv': ['FreeSolv', 'SAMPL.csv', 'SAMPL.csv', 1, 2],
        'lipo': ['Lipophilicity', 'Lipophilicity.csv', 'Lipophilicity.csv', 2, 1],
        'pcba': ['PCBA', 'pcba.csv.gz', 'pcba.csv', -1,
                 slice(0, 128)],
        'muv': ['MUV', 'muv.csv.gz', 'muv.csv', -1,
                slice(0, 17)],
        'hiv': ['HIV', 'HIV.csv', 'HIV.csv', 0, -1],
        'bace': ['BACE', 'bace.csv', 'bace.csv', 0, 2],
        'bbbp': ['BBBP', 'BBBP.csv', 'BBBP.csv', -1, -2],
        'tox21': ['Tox21', 'tox21.csv.gz', 'tox21.csv', -1,
                  slice(0, 12)],
        'toxcast':
        ['ToxCast', 'toxcast_data.csv.gz', 'toxcast_data.csv', 0,
         slice(1, 618)],
        'sider': ['SIDER', 'sider.csv.gz', 'sider.csv', 0,
                  slice(1, 28)],
        'clintox': ['ClinTox', 'clintox.csv.gz', 'clintox.csv', 0,
                    slice(1, 3)],
    }

    def __init__(self, root, name, noise_conf=None, transform=None, pre_transform=None,
                 pre_filter=None):

        self.noise_conf = noise_conf

        if Chem is None:
            raise ImportError('`MoleculeNet` requires `rdkit`.')

        self.name = name.lower()
        assert self.name in self.names.keys()
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                              pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        name = self.name
        if self.name in self.additional_ds_url.keys():
            name = osp.join(name, self.names[name][0])
        return osp.join(self.root, name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        if self.name.lower() in self.additional_ds_url.keys():
            name = self.names[self.name][0]
            raw_file_names = [f'{name}_A.txt', f'{name}_graph_labels.txt', f'{name}_graph_indicator.txt',
                              f'{name}_node_labels.txt', 'README.txt']
            if self.name in self.additional_ds_url.keys():
                raw_file_names = list(map(lambda f: osp.join(name, f), raw_file_names))
            return raw_file_names
        else:
            return self.names[self.name][2]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        if self.name.lower() in self.additional_ds_url.keys():
            url = self.additional_ds_url[self.name]
        else:
            url = self.url.format(self.names[self.name][1])
        path = download_url(url, self.raw_dir)
        if self.names[self.name][1][-2:] == 'gz':
            extract_gz(path, self.raw_dir)
            os.unlink(path)
        elif self.names[self.name][1][-3:] == 'zip':
            extract_zip(path, self.raw_dir)
            os.unlink(path)

    def process(self):
        if self.name.lower() in self.additional_ds_url:
            name = self.names[self.name][0]
            with open(os.path.join(self.raw_dir, f'{name}_node_labels.txt'), 'r') as f:
                nodes_all_temp = f.read().splitlines()
                nodes_all = [int(i) for i in nodes_all_temp]

            adj_all = np.zeros((len(nodes_all), len(nodes_all)))
            with open(os.path.join(self.raw_dir, f'{name}_A.txt'), 'r') as f:
                adj_list = f.read().splitlines()
            for item in adj_list:
                lr = item.split(', ')
                l = int(lr[0])
                r = int(lr[1])
                adj_all[l - 1, r - 1] = 1

            with open(os.path.join(self.raw_dir, f'{name}_graph_indicator.txt'), 'r') as f:
                graph_indicator_temp = f.read().splitlines()
                graph_indicator = [int(i) for i in graph_indicator_temp]
                graph_indicator = np.array(graph_indicator)

            with open(os.path.join(self.raw_dir, f'{name}_graph_labels.txt'), 'r') as f:
                graph_labels_temp = f.read().splitlines()
                graph_labels = [int(i) for i in graph_labels_temp]

            data_list = []
            max_target = -1
            for i in range(1, self.additional_ds_num_graphs[self.name]):
                idx = np.where(graph_indicator == i)
                graph_len = len(idx[0])
                adj = adj_all[idx[0][0]:idx[0][0] + graph_len, idx[0][0]:idx[0][0] + graph_len]
                label = int(graph_labels[i - 1] == 1)
                feature = nodes_all[idx[0][0]:idx[0][0] + graph_len]
                nb_clss = self.additional_ds_num_classes[self.name]
                targets = np.array(feature).reshape(-1)
                max_target = max(max_target, np.max(targets))
                one_hot_feature = np.eye(nb_clss)[targets]
                data_example = Data(x=torch.from_numpy(one_hot_feature).float(),
                                    edge_index=dense_to_sparse(torch.from_numpy(adj))[0],
                                    y=label)
                if self.noise_conf.name == 'noise_nodes':
                    data_example = add_noise_neighbours(
                        data_example,
                        prop_noise_nodes=self.noise_conf.prop,
                        binary=self.noise_conf.binary,
                        p=self.noise_conf.p,
                        connectedness=self.noise_conf.connectedness,
                        c=self.noise_conf.c,
                        graph_classification=True
                    )
                elif self.noise_conf.name == 'noise_feats':
                    data_example = add_noise_features(
                        data_example,
                        prop_noise_feats=self.noise_conf.prop,
                        binary=self.noise_conf.binary,
                        p=self.noise_conf.p
                    )
                data_list.append(data_example)
        else:
            with open(self.raw_paths[0], 'r') as f:
                dataset = f.read().split('\n')[1:-1]
                dataset = [x for x in dataset if len(x) > 0]  # Filter empty lines.

            data_list = []
            for line_idx, line in enumerate(dataset):
                line = re.sub(r'\".*\"', '', line)  # Replace ".*" strings.
                line = line.split(',')

                smiles = line[self.names[self.name][3]]
                ys = line[self.names[self.name][4]]
                ys = ys if isinstance(ys, list) else [ys]

                ys = [float(y) if len(y) > 0 else float('NaN') for y in ys]
                y = torch.tensor(ys, dtype=torch.float).view(1, -1)

                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue

                xs = []
                for atom in mol.GetAtoms():
                    x = []
                    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
                    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
                    x.append(x_map['degree'].index(atom.GetTotalDegree()))
                    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
                    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
                    x.append(x_map['num_radical_electrons'].index(
                        atom.GetNumRadicalElectrons()))
                    x.append(x_map['hybridization'].index(
                        str(atom.GetHybridization())))
                    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
                    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
                    xs.append(x)

                x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

                edge_indices, edge_attrs = [], []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    e = []
                    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
                    e.append(e_map['stereo'].index(str(bond.GetStereo())))
                    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

                    edge_indices += [[i, j], [j, i]]
                    edge_attrs += [e, e]

                edge_index = torch.tensor(edge_indices)
                edge_index = edge_index.t().to(torch.long).view(2, -1)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

                # Sort indices.
                if edge_index.numel() > 0:
                    perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                    edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                            smiles=smiles)

                if self.noise_conf.name == 'noise_nodes':
                    data = add_noise_neighbours(
                        data,
                        prop_noise_nodes=self.noise_conf.prop,
                        binary=self.noise_conf.binary,
                        p=self.noise_conf.p,
                        connectedness=self.noise_conf.connectedness,
                        c=self.noise_conf.c,
                        graph_classification=True
                    )
                elif self.noise_conf.name == 'noise_feats':
                    data = add_noise_features(
                        data,
                        prop_noise_feats=self.noise_conf.prop,
                        binary=self.noise_conf.binary,
                        p=self.noise_conf.p
                    )

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                torch.save(data, os.path.join(self.processed_dir, f'tox21_{line_idx}.pt'))

        torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.names[self.name][0], len(self))


if __name__ == '__main__':
    dataset = MoleculeDataset(root='.', name='MUTAG')
