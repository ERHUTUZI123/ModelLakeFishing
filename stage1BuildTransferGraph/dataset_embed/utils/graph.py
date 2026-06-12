import os

import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

from .CustomRandomLinkSplit import RandomLinkSplit

os.system('unset LD_LIBRARY_PATH')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class HGraph:
    def __init__(
            
            self,
            gnn_method,
            max_dataset_idx,
            model_idx,
            unique_model_id,
            model_features,
            unique_dataset_id,
            dataset_features,
            edge_index_accu_model_to_dataset,
            edge_attr_accu_model_to_dataset,
            edge_index_dataset_to_dataset,
            edge_attr_dataset_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_attr_tran_model_to_dataset,
            edge_index_model_to_model,
            edge_attr_model_to_model,
            negative_pairs,
            contain_data_similarity=True,
            contain_dataset_feature=False,
            contain_model_feature=False,
            custom_negative_sampling=False,
            model_size_bucket_id=None,
            model_family_id=None,
    ):
        self.custom_negative_sampling = custom_negative_sampling
        self.model_idx = model_idx

        self.data = HeteroData()
        # Save node indices:
        # all edges related to datasets
        # edge_index = torch.cat((edge_index_accu_model_to_dataset[1],edge_index_tran_model_to_dataset[1],edge_index_dataset_to_dataset[0],edge_index_dataset_to_dataset[1]))
        # self.data["dataset"].num_nodes = len(torch.unique(edge_index)) #torch.arange(len(unique_dataset_id))
        self.data['dataset'].node_id = torch.arange(len(unique_dataset_id))  # len(torch.unique(edge_index)))
        # print(f"\nself.data['dataset'].node_id: {self.data['dataset'].node_id}")

        # all edges related to modelse
        # edge_index = torch.cat((edge_index_accu_model_to_dataset,edge_index_tran_model_to_dataset),1)
        # self.data["model"].num_nodes = len(torch.unique(edge_index[0]))#len(unique_model_id) # torch.unique(edge_index[0]) #

        self.data['model'].node_id = torch.arange(len(unique_model_id))  # len(torch.unique(edge_index[0])))#+self.data["dataset"].num_nodes
        # self.data['model'].node_id = torch.from_numpy(self.model_idx).to(torch.int64) #torch.arange(self.data.max_dataset_idx + 1)
        # print(f"\nself.data['model'].node_id: {self.data['model'].node_id}")

        data_feature_shape = list(dataset_features.values())[0].shape[0]
        features = []
        for i in range(len(unique_dataset_id)):
            features.append(dataset_features[unique_dataset_id[unique_dataset_id['mappedID'] == i]['dataset'].values[0]])

        if contain_model_feature:
            model_features = np.asarray(model_features, dtype=np.float32)
            assert model_features.shape[0] == len(unique_model_id), (
                f"model_features has {model_features.shape[0]} rows but "
                f"unique_model_id has {len(unique_model_id)} models — "
                "row i must be the feature of the model with mappedID i"
            )
            self.data["model"].x = torch.from_numpy(model_features).to(torch.float)
        else:
            self.data['model'].x = torch.rand((len(unique_model_id), data_feature_shape))

        # Learnable-component indices (size bucket / family) ride on the model
        # node store as [num_models] int columns: NeighborLoader slices any
        # attribute whose first dim equals num_nodes together with x, so
        # mini-batches keep row alignment automatically. The embedding tables
        # themselves live in the GNN (ModelNodeEncoder), NOT here — storing
        # pre-computed vectors in x would silently freeze them.
        if model_size_bucket_id is not None:
            ids = torch.as_tensor(np.asarray(model_size_bucket_id), dtype=torch.long)
            assert ids.shape == (len(unique_model_id),), (
                f"model_size_bucket_id shape {tuple(ids.shape)} != ({len(unique_model_id)},)"
            )
            self.data["model"].size_bucket_id = ids
        if model_family_id is not None:
            ids = torch.as_tensor(np.asarray(model_family_id), dtype=torch.long)
            assert ids.shape == (len(unique_model_id),), (
                f"model_family_id shape {tuple(ids.shape)} != ({len(unique_model_id)},)"
            )
            self.data["model"].family_id = ids

        # self.data["dataset"].x = dataset_features
        # self.data["dataset"].x = torch.from_numpy(dataset_features).to(torch.float)
        # datset_features = np.around(np.random.random_sample((len(dataset_features), 128))+0.00001,3)
        # dataset_features = np.random.randint(0,1,size=(len(dataset_features),20))
        if contain_dataset_feature:
            self.data['dataset'].x = torch.from_numpy(np.vstack(features)).to(torch.float)
        else:
            self.data['dataset'].x = torch.rand((len(unique_dataset_id), data_feature_shape))

        # print()
        # print('self.data["dataset"].x.shape')
        # print('========')
        # print(self.data["dataset"].x.dtype)
        # print(self.data["dataset"].x.shape)

        # self.data["model"].x = model_features
        # if 'homo' not in gnn_method:
        #     edge_index_accu_model_to_dataset[0] -= max_dataset_idx
        if 'without_accuracy' in gnn_method:
            print('\n without_accuary in gnn_method')
            self.label_type = ["model", "transfer_to", "dataset"]
        elif 'without_accuracy' not in gnn_method:
            print('\n without_accuary not in gnn_method')
            self.data["model", "trained_on", "dataset"].edge_index = edge_index_accu_model_to_dataset  # TODO
            if edge_attr_accu_model_to_dataset != None:
                self.data['model', 'trained_on', 'dataset'].edge_attr = edge_attr_accu_model_to_dataset  # TODO
            if 'trained_on_transfer' in gnn_method:
                self.label_type = ["model", "transfer_to", "dataset"]
            else:
                self.label_type = ["model", "trained_on", "dataset"]

        if contain_data_similarity:
            self.data["dataset", "similar_to", "dataset"].edge_index = edge_index_dataset_to_dataset  # TODO
        if edge_attr_dataset_to_dataset != None:
            self.data['dataset', 'similar_to', 'dataset'].edge_attr = edge_attr_dataset_to_dataset  # TODO

        if 'without_transfer' not in gnn_method:
            # if 'homo' not in gnn_method:
            #     edge_index_tran_model_to_dataset[0] -= max_dataset_idx
            self.data["model", "is_base_of", "model"].edge_index = edge_index_model_to_model
            self.data["model", "is_base_of", "model"].edge_attr = edge_attr_model_to_model
            self.data["model", "transfer_to", "dataset"].edge_index = edge_index_tran_model_to_dataset  # TODO
            self.data["model", "transfer_to", "dataset"].edge_attr = edge_attr_tran_model_to_dataset  # TODO

        self.negative_pairs = negative_pairs
        print()
        # print(f'\nedge_index_accu_model_to_dataset: {edge_index_accu_model_to_dataset}')
        # print(f'\nedge_index_tran_model_to_dataset: {edge_index_tran_model_to_dataset}')
        # print(f'-- max node index: {torch.max(edge_index_accu_model_to_dataset),0}, {torch.max(edge_index_tran_model_to_dataset),0},{torch.max(edge_index_dataset_to_dataset),0}')

        # print(self.data.metadata())

        self.transform()
        # self.split()
        self._print()

    def transform(self):
        self.data = T.ToUndirected()(self.data)
        # self.data = T.AddSelfLoops()(self.data)
        # self.data = T.NormalizeFeatures()(self.data)

    def _print(self):
        print()
        print("Data:")
        print("==============")
        print(self.data)
        print(self.data.metadata())
        print("self.data['dataset'].num_nodes")
        print(self.data['dataset'].num_nodes)
        print("self.data['model'].num_nodes")
        print(self.data['model'].num_nodes)
        # num_edges = self.data["model", "trained_on", "dataset"].num_edges
        # print(f'self.data["model", "trained_on", "dataset"].num_edges: {num_edges}')
        # num_edges = self.data["dataset", "similar_to", "dataset"].num_edges
        # print(f'self.data["dataset", "similar_to", "dataset"].num_edges: {num_edges}')
        # print("self.data['dataset'].x")
        # print(self.data['dataset'].x)
        # print(self.data["model", "trained_on", "dataset"].edge_label_index)
        # print(self.data["model", "trained_on", "dataset"].edge_label)
        print()

    def dump_readable(self, model_names=None, dataset_names=None, out_path="hgraph_dump.txt"):
        """Write a human-readable edge list to a text file and stdout."""
        node_name_maps = {
            'model': model_names or {},
            'dataset': dataset_names or {},
        }

        def label(node_type, idx):
            name = node_name_maps.get(node_type, {}).get(idx, str(idx))
            return f"{name}({idx})"

        lines = []
        lines.append("=" * 64)
        lines.append("HGraph Readable Dump")
        lines.append("=" * 64)

        lines.append("\nNodes")
        lines.append("-" * 32)
        for nt in self.data.node_types:
            store = self.data[nt]
            num = store.num_nodes
            names = [node_name_maps.get(nt, {}).get(i, str(i)) for i in range(num)]
            lines.append(f"  {nt:10s}: {num}  [{', '.join(names)}]")
            if hasattr(store, 'x'):
                lines.append(f"             x: {tuple(store.x.shape)} {store.x.dtype}")
            for col in ('size_bucket_id', 'family_id'):
                if hasattr(store, col):
                    vals = ', '.join(
                        f"{label(nt, i)}={store[col][i].item()}" for i in range(num)
                    )
                    lines.append(f"             {col}: [{vals}]")

        lines.append("\nEdges  (after ToUndirected)")
        lines.append("-" * 32)
        for (src_t, rel, dst_t) in self.data.edge_types:
            store = self.data[src_t, rel, dst_t]
            ei = store.edge_index
            ea = getattr(store, 'edge_attr', None)
            is_rev = rel.startswith('rev_')
            tag = "  [auto-reverse]" if is_rev else ""
            lines.append(f"\n  [{src_t}] --{rel}--> [{dst_t}]  ({ei.shape[1]} edges){tag}")
            for i in range(ei.shape[1]):
                s, d = ei[0, i].item(), ei[1, i].item()
                w = f"  w={ea[i].item():.4f}" if ea is not None else ""
                lines.append(f"    {label(src_t, s)} --> {label(dst_t, d)}{w}")

        out = "\n".join(lines)
        print(out)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"\n[dump_readable] Written to: {out_path}")

    def split(self, num_val=0.1, num_test=0.2):  # num_val=0.1,num_test=0.2):
        s, r, t = self.label_type
        print(f'\n label_type: {self.label_type}')
        transform = RandomLinkSplit(  # T.Compose([T.ToUndirected(),
            # transform = T.RandomLinkSplit(
            num_val=num_val,  # TODO
            num_test=num_test,  # TODO
            disjoint_train_ratio=0.3,  # TODO
            neg_sampling_ratio=2.0,  # TODO
            add_negative_train_samples=True,  # TODO
            negative_pairs=self.negative_pairs,
            # edge_types=("model", "trained_on", "dataset"),
            edge_types=(s, r, t),
            is_undirected=True,
            # rev_edge_types = self.data.metadata()[1]
            # rev_edge_types=("dataset", "rev_trained_on", "model"), 
            custom_negative_sampling=self.custom_negative_sampling
        )
        # ])
        # self.train_data, self.val_data, self.test_data = # self.val_data, 
        return transform(self.data)


class LineGraph():
    def __init__(
            self,
            edge_index_accu_model_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_index_dataset_to_dataset,
            without_transfer=True,
            # max_model_id
    ):
        if without_transfer:
            edge_index = edge_index_accu_model_to_dataset
        else:
            edge_index = torch.cat((edge_index_accu_model_to_dataset, edge_index_tran_model_to_dataset), 1)
        edge_index = torch.cat((edge_index, edge_index_dataset_to_dataset), 1).numpy()
        edges = list(map(tuple, edge_index))
        G = nx.Graph()
        # print(f'\n edges: {edges}')
        G.add_edges_from(edges)
        self.graph = G


class Graph():
    def __init__(
            self,
            node_ID,
            edge_index_accu_model_to_dataset,
            edge_attr_accu_model_to_dataset,
            edge_index_tran_model_to_dataset,
            edge_attr_tran_model_to_dataset,
            edge_index_dataset_to_dataset,
            edge_attr_dataset_to_dataset,
            without_transfer=False,
            without_accuracy=False,
            # max_model_id
    ):

        # max_model_id = int(torch.max(edge_index_model_to_dataset[0,:]).item()) + 1
        # rename dataset index name
        # edge_index_model_to_dataset[1,:] += max_model_id + 1
        # edge_index_dataset_to_dataset += max_model_id + 1
        # print('max_model_id', torch.max(edge_index_model_to_dataset[1,:]))

        if without_transfer:
            edge_index = edge_index_accu_model_to_dataset
        elif without_accuracy:
            edge_index = edge_index_tran_model_to_dataset
        else:
            edge_index = torch.cat((edge_index_accu_model_to_dataset, edge_index_tran_model_to_dataset), 1)

        edge_index = torch.cat((edge_index, edge_index_dataset_to_dataset), 1).type(torch.int64)

        if without_transfer:
            edge_attr = edge_attr_accu_model_to_dataset
        elif without_accuracy:
            edge_attr = edge_attr_tran_model_to_dataset
        else:
            edge_attr = torch.cat((edge_attr_accu_model_to_dataset, edge_attr_tran_model_to_dataset))
        edge_attr = torch.cat((edge_attr, edge_attr_dataset_to_dataset))

        # convert it to undirected graph
        # from torch_geometric.utils import to_undirected
        # edge_index, edge_attr = to_undirected(edge_index,edge_attr)
        self.data = Data(edge_index=edge_index, edge_attr=edge_attr)
        self.data.node_id = node_ID
        # print(f'self.data.node_id: {self.data.node_id}')
        # import torch_geometric.transforms as T
        # self.data = T.ToUndirected()(data)
        try:
            print()
            print(f'----- Graph Properties -----')
            print(self.data)
            # print(self.data.edge_index)
            print(
                f'-- number of accuracy & transferability edge: {edge_index_accu_model_to_dataset.shape}, {edge_index_tran_model_to_dataset.shape}, {edge_index_dataset_to_dataset.shape}'
            )
            # print(f'-- max accu index: {torch.max(edge_index_accu_model_to_dataset),0}')
            # print(f'-- max tran index  {torch.max(edge_index_tran_model_to_dataset),0}')
            # print(f'-- max dataset index: {torch.max(edge_index_dataset_to_dataset),0}')
            print(f'-- number of nodes: {self.data.num_nodes}')
            print(f' number of edges: {self.data.num_edges}')
            print(f'-- data is directed(): {self.data.is_directed()}')
        except Exception as e:
            print(e)
