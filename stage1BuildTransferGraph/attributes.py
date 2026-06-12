import itertools
import json
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance
import torch

from utils.config import get_root_path_string
from utils.embed_utils import DatasetEmbeddingMethod
from utils.embedder import determine_directory_embedded_dataset, determine_file_name_embedded_dataset
from utils.task import TaskType

logger = logging.getLogger(__name__)


class GraphAttributes():
    PRINT = True
    FEATURE_DIM = 2048  # 768 #2048
    dataset_map = {
        'oxford_flowers102': 'flowers',
        'svhn_cropped': 'svhn',
        'dsprites': ['dsprites_label_orientation', 'dsprites_label_x_position'],
        'smallnorb': ['smallnorb_label_azimuth', 'smallnorb_label_elevation'],
        'oxford_iiit_pet': 'pets',
        'patch_camelyon': 'pcam',
        'clevr': ["count_all", "count_left", "count_far", "count_near",
                  "closest_object_distance", "closest_object_x_location",
                  "count_vehicles", "closest_vehicle_distance"],
        # 'kitti': ['label_orientation']
    }

    def __init__(self, args):
        self.args = args
        self.resource_path = os.path.join(get_root_path_string(), "resources/experiments", args.task_type.value)
        self.record_path = os.path.join(self.resource_path, "records.csv")
        self.lineage_record_path = os.path.join('.', 'dataset_embed', 'data', 'lineage_records.csv')
        self.model_config_path = os.path.join(self.resource_path, "model_config_dataset.csv")
        self.peft_method = args.peft_method

        if 'model_ratio' in self.args.gnn_method:
            match = re.search(r'model_ratio_([\d.]+)', self.args.gnn_method)
            if match:
                self.model_ratio = float(match.group(1))
            else:
                raise ValueError(f"Cannot parse model ratio from {self.args.gnn_method}")
        else:
            self.model_ratio = 1.0

        self.finetune_records, self.model_config = self.get_finetuned_records()
        # get node id
        self.unique_model_id, self.unique_dataset_id = self.get_node_id()

        # get dataset-dataset edge index
        if args.dataset_reference_model == 'resnet34' or args.dataset_reference_model == 'google_vit_base_patch16_224' or args.dataset_reference_model == 'microsoft_resnet-50':
            self.base_dataset = 'imagenet'
        elif args.dataset_reference_model == 'Ahmed9275_Vit-Cifar100':
            self.base_dataset = 'cifar100'
        elif args.dataset_reference_model == 'johnnydevriese_vit_beans':
            self.base_dataset = 'beans'
        elif args.dataset_reference_model == 'EleutherAI_gpt-neo-125m':
            self.base_dataset = 'gpt'
        else:
            raise Exception(f'Unexpected reference model {args.dataset_reference_model}')

    def get_dataset_edge_index(self, threshold=0.3, base_dataset='imagenet', sim_method='cosine'):
        threshold = 1

        # get number of datasets
        n = len(self.data_features)

        # build a distance matrix of expected datasets graph
        distance_matrix = np.zeros([n, n])
        # build empty data source (edge from what)
        data_source = []
        # build empty data target (edge to what)
        data_target = []
        # build empty attribution
        attr = []

        # for every piece of dataset
        '''
        every piece may look like this:
        i = 0
        row = {
            "dataset": "cifar100",
            "mappedID": 0
        }
        '''
        for i, row in self.unique_dataset_id.iterrows():
            # self.data_features save embedding of datasets with
            # key is the val of key "dataset" in the dict row
            # so e1 is embedding of selected piece of data
            e1 = self.data_features[row['dataset']]
            # correlation distance is a distance method based
            # on intuition "if you are correlated and then you are nearby each other"
            # AKA # 1 - correlation(x,y) = distance_correlation
            if sim_method == 'correlation':
                # correlation distance between e1 and e1 should be 0
                distance = distance.correlation(e1, e1)
            elif sim_method == 'euclidean':
                # euclidean disance between e1 and e1 should also be 0
                distance = distance.euclidean(e1, e1)
            # fill in matrix all (i,i) with distance 0
            distance_matrix[i, i] = distance
            '''
            so basically, this code fills all matix(i,i) with 0 (calculated)
            '''
        # pick unique two datasets to be a pair
        for (i, row1), (j, row2) in itertools.combinations(self.unique_dataset_id.iterrows(), 2):
            # get the name of the first dataset of the pair
            k1 = row1['dataset']
            # get node ID of the first dataset of the pair
            p = row1['mappedID']
            # get the name of the second dataset of the pair
            k2 = row2['dataset']
            # get node ID of the second dataset of the pair
            q = row2['mappedID']
            # get embedding of the first dataset
            e1 = self.data_features[k1]
            # get embedding of the second dataset
            e2 = self.data_features[k2]
            # if we use correlation distance method
            if sim_method == 'correlation':
                # correlation distance between two datasets
                distance = distance.correlation(e1, e2)
            # if we use euclidean distance method
            elif sim_method == 'euclidean':
                # euclidean distance between two datasets
                distance = distance.euclidean(e1, e2)
            
            # fill distance in the distance matrix
            distance_matrix[p, q] = distance
            # symmetry
            distance_matrix[q, p] = distance
            # edge from p
            data_source.append(p)
            # edge to q
            data_target.append(q)
            # edge weight
            '''
            distance small -> similar -> high weight AND weight = 1 - distance is high
            distance bg -> not similar -> low weight AND weight = 1 - distance is low
            '''
            weight = 1 - distance
            attr.append(weight)
        
        # data normalization
        attr = np.asarray([(float(i) - min(attr)) / (max(attr) - min(attr)) for i in attr])
        # pruning
        # only those edges with weight bigger than 1 - threshold can be kept
        index = np.where(attr > (1 - threshold))
        attr = attr[index]
        # keep only source of edges with weight bigger than 1 - threshold in data_source array
        data_source = np.asarray(data_source)[index]
        # keep only target of edges with weight bigger than 1 - threshold in data_target array
        data_target = np.asarray(data_target)[index]

        # ROOT / embed_method_reference model.csv
        path = f'{self.resource_path}/corr_{self.args.dataset_embed_method.value}_{self.args.dataset_reference_model}_{base_dataset}.csv'
        # initialize distance dict which serves as a matrix
        dict_distance = {}
        # for every piece of dataset
        for i, row in self.unique_dataset_id.iterrows():  
            # get the name of the dataset
            name = row['dataset']
            # get the mappedID of the dataset
            idx = row['mappedID']
            # set the value of key (dataset name) as the whole row of the dataset in distance matrix
            # e.g. {"flowers": [0, 0.3, 0.8]}
            dict_distance[name] = list(distance_matrix[idx, :])
        # Results be like:
        '''
        dict_distance = {
            "cifar100": [0, 0.3, 0.8],
            "flowers": [0.3, 0, 0.4],
            "pets":    [0.8, 0.4, 0]
        }
        '''
        # convert this dictionary to pandas dataframe
        '''
            cifar100  flowers  pets
        0      0.0      0.3   0.8
        1      0.3      0.0   0.4
        2      0.8      0.4   0.0
        '''
        df_tmp = pd.DataFrame(dict_distance)
        # replace numerical rows indicies with columns name
        '''
                  cifar100  flowers  pets
        cifar100     0.0      0.3   0.8
        flowers      0.3      0.0   0.4
        pets         0.8      0.4   0.0
        '''
        df_tmp.index = df_tmp.columns
        # print logs stating saving path
        logger.info(f'\n\n ====  Save correlation to path: {path}')
        # save dataframe as .csv file
        # so finally we convert matrix to .csv file
        df_tmp.to_csv(path)
        
        # tensor(data_source) and tensor(data_target) converts both arrays to tensors
        # stack them is like
        '''
        tensor([
            [0, 0, 1], # sources of edge 0, 1, 2
            [1, 2, 2]  # targets of edge 0, 1, 2
        ])
        '''
        # shape: [2, num_of_edges] eg here is [2, 3]
        edge_index_tensor = torch.stack([torch.tensor(data_source), torch.tensor(data_target)])

        # tensor(attr) converts the array attr to tensor
        # tensor is like
        '''
        tensor([0.7, 0.2, 0.6]) # weight of edge 0, 1, 2
        '''
        edge_attr_tensor = torch.tensor(attr)
        return edge_index_tensor, edge_attr_tensor

    # Model-Dataset Edge Weight
    # methods: 
    # 'accuracy' (model fine-tune history performance on dataset)
    # 'score' (model transferability score)
    def get_model_dataset_edge_index(self, method='accuracy', ratio=1.0):
        # if we construct using fine-tune history performance
        if method == 'accuracy':
            # get finetune history
            df = self.finetune_records.copy()
            # do normalizatin on accuracy
            df['accuracy'] = df[['dataset', 'accuracy']].groupby('dataset').transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            # Define negative and positive edges by respective thresholds
            # edges with mild weights will be dropped 
            df_neg = df[df['accuracy'] <= self.args.accu_neg_thres]
            df = df[df['accuracy'] >= self.args.accu_pos_thres]
            # output edges amount after filtering
            logger.info(f'df_accu after filtering: {len(df)}')
        # if there is no direct fine-tune history we apply transferability scores to be weight
        elif method == 'score':
            # use help functions to get positive edges weights and negative nes
            df, df_neg = self.get_transferability_scores()

            # Sample transferability score with finetune-ratio
            if ratio != 1:
                df = df.sample(frac=self.args.finetune_ratio, random_state=1)
        # apply get_edges to build graph
        '''
        Before,
        model        dataset      score
        model_A      dataset_X    1.27
        model_B      dataset_Y    0.83

        After,
        model_A -> dataset_X, weight = 1.27
        model_B -> dataset_Y, weight = 0.83
        '''
        # positive ones
        positive_edges, positive_edges_weights = self.get_edges(df, method, type='positive')
        # since for negative ones, we just want their pairs and do not care about their weights
        #   we drop weights and keep only edges
        negative_edges, _ = self.get_edges(df_neg, method, type='negative')

        return positive_edges, positive_edges_weights, negative_edges

    def get_edges(self, df, method, type='positive'):
        logger.info(f'\nlen(df) after filtering models by {method}, {type}: {len(df)}')

        mapped_dataset_id = pd.merge(df[['dataset', 'model', method]], self.unique_dataset_id, on='dataset', how='inner')

        mapped_model_id = pd.merge(
            mapped_dataset_id[['dataset', 'model', method]],
            self.unique_model_id,
            on='model',
            how='inner' # how='left' will keep all original df rows even if cannot find corresponding model ids
        )  
        '''
        Before,
        mapped_model_id['mappedID'].values   = [100, 101, 102]
        mapped_dataset_id['mappedID'].values = [0,   1,   2]
        After,
        tensor([
            [100, 101, 102], # source models
            [  0,   1,   2]  # source datasets
        ])
        shape: [2, num_edges]
        '''
        edge_index_model_to_dataset = torch.stack(
            [torch.from_numpy(mapped_model_id['mappedID'].values), torch.from_numpy(mapped_dataset_id['mappedID'].values)],
            dim=0
        )
        # build negative edges into the graph
        '''
        Before,
        mapped_model_id['mappedID'].values   = [100, 101, 102]
        mapped_dataset_id['mappedID'].values = [0,   1,   2]
        After,
        tensor([
            [100, 0], # edge 1 [source, target]
            [101, 1],
            [102, 2]
        ])
        shape: [num_edges, 2]
        '''
        if type == 'negative':
            edge_index_model_to_dataset = torch.stack(
                [torch.from_numpy(mapped_model_id['mappedID'].values), torch.from_numpy(mapped_dataset_id['mappedID'].values)],
                dim=1
            )
        # convert edges weights to tensor e.g. [edge1weight, edge2weight, edge3weight]
        edge_attr = torch.from_numpy(mapped_model_id[method].values)

        # return edges and edges weights
        return edge_index_model_to_dataset, edge_attr

    def del_node(self, unique_id, entity_list, entity_type):
        # drop rows that are not in entity_list in order to drop those who do not produce dataset features
        unique_id = unique_id[unique_id[entity_type].isin(entity_list)]
        return unique_id

    def drop_nodes(self):
        # reallocate the node id
        # get all unique names of datasets we have right now
        a = set(self.unique_dataset_id['dataset'].unique())
        # get all datasets that have successfully extracted features
        b = set(self.dataset_list)
        # print all datasets that failed extracting features
        logger.info(f'absent dataset: {a - b}')
        # use del_node method to keep only datasets rows shown in dataset_list for unique_dataset
        self.unique_dataset_id = self.get_unique_node(
            self.del_node(self.unique_dataset_id, self.dataset_list.keys(), 'dataset')['dataset'],
            'dataset'
        )
        # use del_node method to keep only models rows shown in dataset_list for unique_model
        self.unique_model_id = self.get_unique_node(self.del_node(self.unique_model_id, self.model_list, 'model')['model'], 'model')

        ## Perform merge to obtain the edges from models and datasets:
        # keep only fine_tune history with valid models
        self.finetune_records = self.finetune_records[self.finetune_records['model'].isin(self.unique_model_id['model'].values)]
        # keep only fine_tune history with valid datasets
        self.finetune_records = self.finetune_records[self.finetune_records['dataset'].isin(self.unique_dataset_id['dataset'].values)]

    # model embeddings
    # complete_model_features by default is False is because of cost
    def get_model_features(self, complete_model_features=False):
        # create list used for saving model embeddings
        model_feature = []
        # 'attribution_map
        # Which parts of the input are the model more sensitive to, 
        #   and which regions/features contribute more to the output?
        DATA_EMB_METHOD = 'attribution_map'
        # 'input_x_gradient'
        # If an input pixel has a large value and 
        #   the model output gradient with respect to it is also large, 
        #   then this pixel is very important for the model
        ATTRIBUTION_METHOD = 'input_x_gradient'  # 'input_x_gradient'#'saliency' considers most input pixel value
        # 224 will be more expensive and better
        INPUT_SHAPE = 128  # 64 # #224
        # model_list should save successfully processed models
        model_list = []
        # return index i and dictionary row like
        '''
        {'model': 'resnet50', 'mappedID': 0}
        '''
        #  from unique_model_id
        for i, row in self.unique_model_id.iterrows():
            # print what model are we processing
            logger.info(f"======== i: {i}, model: {row['model']} ==========")
            # if selected model name is NaN
            if pd.isna(model_match_rows['model'].values[0]):
                # then we still add it to modeel_list
                model_list.append(row['model'])
                continue
            # get all fine-tune history of selected model
            model_match_rows = self.finetune_records.loc[self.finetune_records['model'] == row['model']]
            # if model does not have any fine-tune history
            if model_match_rows.empty:
                # if we want complete model features which are expensive
                if complete_model_features:
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                else:
                    # if not we just drop a purely 0 feature and add to mode_feature
                    features = np.zeros(INPUT_SHAPE * INPUT_SHAPE)
                    model_feature.append(features)
                continue
            
            try:
                # normalize the name to be all like x_y
                dataset_name = model_match_rows['dataset'].values[0].replace('/', '_').replace('-', '_')
                # save name
                ds_name = dataset_name
                # if name is saved as key in dataset_map then use key's value as name otherwise just use newly normalized name
                dataset_name = self.dataset_map[dataset_name] if dataset_name in self.dataset_map.keys() else dataset_name
            except:
                # error handling print
                logger.warn('fail to retrieve model')
                continue
            # if datasetname is list (e.g value of a key in dataset_map)
            if isinstance(dataset_name, list):
                # find dataset_name's dataset fine-tune history in finetune_records and get its configs' first item
                #  and load it to be dict
                configs = self.finetune_records[self.finetune_records['dataset'] == ds_name]['configs'].values[0].replace("'", '"')
                # print configs
                logger.info(configs)
                '''
                Interesting parts, it tries to specify tasks like ModelLens using a very rough way
                for 'clevr' there are many tasks you can do, specified by configs['preprocess']
                {'preprocess': 'count'}
                {'preprocess': 'distance'}
                {'preprocess': 'closest_obejct'}
                that will be more clear
                '''
                if ds_name == 'clevr':
                    dataset_name = json.loads(configs)['preprocess']
                # if not 'clevr' then just use ds_name + label_name
                else:
                    dataset_name = f"{ds_name}_{json.loads(configs)['label_name']}"
            # cannot load imagenet-21k and make them equal
            if dataset_name == 'imagenet_21k':
                dataset_name = 'imagenet'

            # print the final dataset_name used for loading model feature
            logger.info(f"== dataset_name: {dataset_name}")

            # if dataset is FastJobs_Visual_Emotional_Analysis, skip this model
            if dataset_name == 'FastJobs_Visual_Emotional_Analysis':
                # delete_model_row_idx.append(i)
                model_list.append(row['model'])
                continue

            # get the largest input_shape from this model's fine-tune history
            IMAGE_SHAPE = int(sorted(model_match_rows['input_shape'].values, reverse=True)[0])

            # get current model name
            model_name = row['model']

            # if model_name in ['AkshatSurolia/BEiT-FaceMask-Finetuned','AkshatSurolia/ConvNeXt-FaceMask-Finetuned','AkshatSurolia/DeiT-FaceMask-Finetuned','AkshatSurolia/ViT-FaceMask-Finetuned','Amrrs/indian-foods','Amrrs/south-indian-foods']: 
            #     continue

            # create path to load saved model attribution feature
            path = os.path.join(
                f'../model_embed/{DATA_EMB_METHOD}/feature',
                dataset_name,
                model_name.replace('/', '_') + f'_{ATTRIBUTION_METHOD}.npy'
            )

            # print dataset_name and model_name
            logger.info(dataset_name, model_name)

            # load model features
            try:
                features = np.load(path)

            # if loading model feature fails
            except Exception as e:
                # if we require complete model features, skip this model and add it to model_list
                if complete_model_features:
                    logger.warning(f'== Skip this model and delete it')
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                    continue
                else:
                    # if we do not require complete model features, use a zero matrix as placeholder
                    features = np.zeros((INPUT_SHAPE, INPUT_SHAPE))
                # features = np.zeros((INPUT_SHAPE,INPUT_SHAPE))

            # print loaded feature shape
            logger.info(f'features.shape: {features.shape}')

            # if feature is only a 2D zero placeholder, try to obtain missing features
            if features.shape == (INPUT_SHAPE, INPUT_SHAPE):
                logger.info('Try to obtain missing features')

                # add parent directory to system path
                sys.path.append('..')

                # import attribution map embed function
                from model_embed.attribution_map.embed import embed

                # use selected attribution method
                method = ATTRIBUTION_METHOD  # 'saliency'

                # use batch size 1 to generate model feature
                batch_size = 1

                try:
                    # try to generate model attribution feature
                    features = embed('../', model_name, dataset_name, method, input_shape=IMAGE_SHAPE, batch_size=batch_size)

                # if generating feature also fails, skip this model
                except Exception as e:
                    # print(e)
                    # print('----------')
                    # features = np.zeros((3,INPUT_SHAPE,INPUT_SHAPE))
                    # delete_model_row_idx.append(i)
                    model_list.append(row['model'])
                    logger.warning(f'--- fail - skip row {row["model"]}')
                    continue

            else:
                # if loaded feature is all NaN, replace it with zero feature
                if np.isnan(features).all():
                    features = np.zeros((3, INPUT_SHAPE, INPUT_SHAPE))

            # average feature over first dimension, usually average over channels
            features = np.mean(features, axis=0)

            # print(f'features.shape: {features.shape}')

            # if feature shape is not expected INPUT_SHAPE, resize it
            if features.shape[1] != INPUT_SHAPE:
                # print(f'== features.shape:{features.shape}')
                features = np.resize(features, (INPUT_SHAPE, INPUT_SHAPE))

            # flatten 2D feature map into 1D vector
            features = features.flatten()

            # add current model feature to model_feature list
            model_feature.append(features)

        # print how many model features are saved
        logger.info(f'== model_feature.shape:{len(model_feature)}')

        # stack all model feature vectors into one matrix
        model_feature = np.stack(model_feature)

        # model_feature.astype(np.double)

        # print final model feature matrix shape
        logger.info(f'== model_feature.shape:{model_feature.shape}')

        # return torch.from_numpy(model_feature).to(torch.float), delete_model_row_idx

        # return model feature matrix and skipped model list
        return model_feature, model_list  # delete_model_row_idx

    # x_m^(0) model features (LLM-lake path) — successor of the vision-specific
    # attribution-map get_model_features above.
    def get_xm0_features(self):
        """
        Build the x_m^(0) package for every model in unique_model_id.

        MUST run after drop_nodes() (mappedIDs final) and BEFORE the homo-mode
        `mappedID += max_dataset_idx + 1` shift — the builders assert a 0-based
        consecutive mappedID range, which is what keeps the row-order contract
        checkable.

        The returned dict splits along the frozen/learnable boundary:
          'frozen'                      -> HGraph model_features
                                           (contain_model_feature=True)
          'size_bucket_id'/'family_id'  -> HGraph model_size_bucket_id /
                                           model_family_id kwargs
          'num_size_buckets'/'num_families'/'family_vocab'
                                        -> ModelNodeEncoder at training time
        """
        from dataset_embed.xm0_builder import build_xm0

        cache_dir = os.path.join('.', 'dataset_embed', 'data')
        return build_xm0(
            self.unique_model_id,
            desc_cache_path=os.path.join(cache_dir, 'model_descriptions.csv'),
            desc_emb_cache_path=os.path.join(cache_dir, 'model_desc_emb.npz'),
            size_cache_path=os.path.join(cache_dir, 'model_param_counts.csv'),
            family_cache_path=os.path.join(cache_dir, 'model_families.csv'),
            family_vocab_path=os.path.join(cache_dir, 'family_vocab.csv'),
        )

    def get_dataset_list(self):
        dataset_list = {}
        # delete_dataset_row_idx = []
        for i, row in self.unique_dataset_id.iterrows():
            ds_name = row['dataset']
            dataset_name = ds_name.replace('/', '_').replace('-', '_')

            dataset_name = self.dataset_map[dataset_name] if dataset_name in self.dataset_map.keys() else dataset_name
            dataset_list[ds_name] = dataset_name
        return dataset_list  # , delete_dataset_row_idx

    ## Node idx
    def get_node_id(self):
        unique_model_id = self.get_unique_node(self.finetune_records['model'], 'model')
        unique_dataset_id = self.get_unique_node(self.finetune_records['dataset'], 'dataset')
        logger.info(f"len(unique_model_id): {len(unique_model_id)}")
        logger.info(f'len(unique_dataset_id): {len(unique_dataset_id)}')
        return unique_model_id, unique_dataset_id

    def get_unique_node(self, col, name):
        tmp_col = col.copy().dropna()
        unique_id = tmp_col.unique()
        unique_id = pd.DataFrame(
            data={
                name: unique_id,
                'mappedID': pd.RangeIndex(len(unique_id)),
            }
        )
        return unique_id

    def get_transferability_scores(self):
        df = self.finetune_records.copy()[['dataset', 'model', 'accuracy']]
        df_list = []
        df_neg_list = []

        for ori_dataset_name, dataset_name in self.dataset_list.items():
            if self.args.task_type == TaskType.IMAGE_CLASSIFICATION:
                df_sub = df[df['dataset'] == dataset_name]
            elif self.args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
                df_sub = df[df['dataset'] == ori_dataset_name]
            else:
                raise Exception(f"Unexpected task type {self.args.task_type}")

            try:
                df_score_all = pd.read_csv(f'{self.resource_path}/transferability_score_records.csv', index_col=0)
                df_score = df_score_all[df_score_all['target_dataset'] == ori_dataset_name]

                # drop rows with -inf amount or replace it with really small number
                df_score.loc[:, 'score'] = df_score['score'].replace([-np.inf, np.nan], -50)
                score = df_score['score']  # .astype('float64')

                ##### Normalize
                ## mean normalization
                normalized_pred = (score - score.mean()) / score.std()

                df_score.loc[:, 'score'] = normalized_pred

                # top K = 20
                # K = 50
                # largest
                if self.args.top_pos_K <= 1:
                    df_large = df_score[df_score['score'] >= self.args.top_pos_K]
                elif self.args.top_pos_K > 1:
                    df_large = df_score.nlargest(self.args.top_pos_K, 'score')

                df_ = pd.merge(df_sub, df_large, on='model', how='left')
                df_list.append(df_)
                # smallest
                if self.args.top_neg_K <= 1:
                    df_small = df_score[df_score['score'] < self.args.top_neg_K]

                df_s = pd.merge(df_sub, df_small, on='model', how='left')
                df_neg_list.append(df_s)
            except Exception as e:
                logger.warning(e)
                logger.warning(f"Skipping {dataset_name}")
                continue

        df = pd.concat(df_list)
        df = df.dropna(subset=['score'])

        if not os.path.isdir(f"{self.resource_path}/features"):
            os.makedirs(f"{self.resource_path}/features")
        df.to_csv(f'{self.resource_path}/features/transferability.csv')

        if 'score' not in df.columns: df['score'] = 0
        df_neg = pd.concat(df_neg_list).dropna()

        logger.info(f'\nlength of transferability positive: {len(df)}')
        logger.info(f'\nlength of transferability negative: {len(df)}')
        assert len(df) > 200

        return df, df_neg

    def select_models_with_uniform_distribution(self, finetune_df, model_config_df):
        # Filter the results DataFrame for the desired dataset
        target_dataset_finetune_df = finetune_df[finetune_df['finetuned_dataset'] == self.args.test_dataset]

        # Sort the filtered DataFrame by eval_accuracy
        sorted_results_df = target_dataset_finetune_df.sort_values(by='eval_accuracy')

        # Calculate the number of models to sample based on the ratio
        total_models = len(sorted_results_df)
        num_samples = int(total_models * self.model_ratio)

        # Use np.linspace to get indices for uniform sampling
        indices = np.linspace(0, total_models - 1, num_samples).astype(int)

        # Select the models corresponding to these indices
        selected_finetune_records = sorted_results_df.iloc[indices]

        # Get the list of selected model names
        selected_model_names = selected_finetune_records['model'].unique()

        # Filter the results DataFrame for the selected models
        filtered_results_df = finetune_df[finetune_df['model'].isin(selected_model_names)]

        # Filter the model information DataFrame for the selected models
        filtered_model_info_df = model_config_df[model_config_df['model'].isin(selected_model_names)]

        return filtered_results_df, filtered_model_info_df

    
    def get_finetuned_records(self):
        config = pd.read_csv(self.model_config_path)
        # model configuration
        config['configs'] = ''
        # config['accuracy'] = 0
        if self.args.task_type == TaskType.IMAGE_CLASSIFICATION:
            config['dataset'] = config['labels']
            config['accuracy'] = config['accuracy'].fillna((config['accuracy'].mean()))
        elif self.args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
            #config = config.dropna(subset=['dataset'])
            ##### fill pre-trained null value with mean accuracy
            config['accuracy'] = config['accuracy'].fillna((config['accuracy'].mean()))
            config['input_shape'] = 0
        else:
            raise Exception(f"Unexpected task type {self.args.task_type}")

        ###### finetune results
        finetune_records = pd.read_csv(self.record_path)

        # Filter by fine-tuning method
        if 'peft_method' in finetune_records.columns:
            if self.peft_method is not None:
                finetune_records = finetune_records[finetune_records['peft_method'] == self.peft_method]
            else:
                finetune_records = finetune_records[pd.isna(finetune_records['peft_method'])]

        # rename column name
        finetune_records['model'] = finetune_records['model']
        finetune_records['dataset'] = finetune_records['finetuned_dataset']  # finetune_records['train_dataset_name']
        if self.args.task_type == TaskType.IMAGE_CLASSIFICATION:
            finetune_records['accuracy'] = finetune_records['eval_accuracy']
        elif self.args.task_type == TaskType.SEQUENCE_CLASSIFICATION:
            finetune_records['accuracy'] = finetune_records['eval_accuracy']
            finetune_records = finetune_records[finetune_records['dataset'] != 'dbpedia_14']
            finetune_records['input_shape'] = 0
        else:
            raise Exception(f"Unexpected task type {self.args.task_type}")

        logger.info(f'---- len(finetune_records_raw): {len(finetune_records)}')

        if self.model_ratio != 1:
            finetune_records, model_config = self.select_models_with_uniform_distribution(finetune_records, config)
        else:
            model_config = config

        ## Delete the finetune records of the test datset
        ######################
        finetune_records = finetune_records[finetune_records['dataset'] != self.args.test_dataset]

        #### Sampling the finetune_records with samping ratio
        if self.args.finetune_ratio != 1:
            finetune_records = finetune_records.sample(frac=self.args.finetune_ratio, random_state=1)

        # Normalize finetune results per dataset
        accuracy = finetune_records[['dataset', 'accuracy']].groupby('dataset').transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        finetune_records['accuracy'] = accuracy

        finetune_records = pd.concat(
            [config[['dataset', 'model', 'input_shape', 'accuracy']],
             finetune_records[['dataset', 'model', 'input_shape', 'accuracy']]],
            ignore_index=True
        )

        finetune_records['config'] = ''
        # filter models that are contained in the config file
        available_models = model_config['model'].unique()
        finetune_records = finetune_records[finetune_records['model'].isin(available_models)]
        logger.info(f'---- len(finetune_records_after_concatenating_model_config): {len(finetune_records)}')

        # ######################
        # ## Add an empty row to indicate the dataset
        # ######################
        finetune_records.loc[len(finetune_records)] = {'dataset': self.args.test_dataset}
        finetune_records.index = range(len(finetune_records))

        return finetune_records, model_config
    
    def get_lineage_records(self):
        model_config = pd.read_csv(self.model_config_path)
        lineage_records = pd.read_csv(self.lineage_record_path)
        available_models = model_config['model'].unique()
        lineage_records = lineage_records[lineage_records['model'].isin(available_models)]
        return lineage_records
    
    def get_model_model_edge_index(self):
        df = self.get_lineage_records()
        relation_weights = {
            'quantized': 0.9,
            'adapter': 0.7,
            'finetune': 0.5,
            'merge': 0.3
        }
        df['weight'] = df['relation'].map(relation_weights).fillna(0.0)
        edges, edges_weights = self.get_edges_new(df)
        return edges, edges_weights

    def get_edges_new(self, df):
        model_id_map = self.unique_model_id.rename(columns={
            "mappedID": "model_id"
        })

        base_model_id_map = self.unique_model_id.rename(columns={
            "model": "base_model",
            "mappedID": "base_model_id"
        })

        mapped_lineage = pd.merge(
            df,
            model_id_map,
            on="model",
            how="inner"
        )

        mapped_lineage = pd.merge(
            mapped_lineage,
            base_model_id_map,
            on="base_model",
            how="inner"
        )

        edge_index_model_to_model = torch.stack([
            torch.from_numpy(mapped_lineage['base_model_id'].values).long(),
            torch.from_numpy(mapped_lineage['model_id'].values).long()
        ], dim=0)
        edge_attr = torch.from_numpy(mapped_lineage['weight'].values)
        return edge_index_model_to_model, edge_attr


class GraphAttributesWithDomainSimilarity(GraphAttributes):
    def __init__(self, args):
        # invoking the __init__ of the parent class
        GraphAttributes.__init__(self, args)
        self.dataset_list = self.get_dataset_list()

        self.data_features = self.get_dataset_features(self.args.dataset_reference_model)
        # x_m^(0) features are built AFTER drop_nodes (below), because the
        # builders key every row to the FINAL mappedID order. Unlike the
        # attribution-map path, no model is ever dropped for missing data —
        # every model gets a feature (zero-vector / unknown-bucket fallbacks) —
        # so model_list is simply all models.
        self.contain_model_feature = (
            'node2vec' not in args.gnn_method and args.contain_model_feature
        )
        self.model_features = []
        self.model_list = self.unique_model_id['model'].unique()
        self.model_size_bucket_id = None
        self.model_family_id = None
        self.xm0 = None

        # get common nodes
        self.drop_nodes()
        self.max_dataset_idx = self.unique_dataset_id['mappedID'].max()

        if self.contain_model_feature:
            # mappedIDs are final and still 0-based here (the homo-mode shift
            # below has not happened yet) — exactly what build_xm0 asserts.
            self.xm0 = self.get_xm0_features()
            self.model_features = self.xm0['frozen']
            self.model_size_bucket_id = self.xm0['size_bucket_id']
            self.model_family_id = self.xm0['family_id']

        # get specific dataset index
        if args.test_dataset != '':
            try:
                self.test_dataset_idx = self.unique_dataset_id[self.unique_dataset_id['dataset'] == args.test_dataset]['mappedID'].values[0]
            except Exception as e:
                # pass
                logger.warning(e)
                logger.warning(f"Test dataset {self.args.test_dataset} not found. Skipping.")
            if 'homo' in self.args.gnn_method or 'node2vec' in self.args.gnn_method:
                self.unique_model_id['mappedID'] += self.max_dataset_idx + 1
            self.model_idx = self.unique_model_id['mappedID'].values
        else:
            self.test_dataset_idx = -1
            self.model_idx = -1

        self.node_ID = list(self.model_idx) + list(self.unique_dataset_id['mappedID'].values)

        # get edge index
        self.edge_index_accu_model_to_dataset, self.edge_attr_accu_model_to_dataset, self.accu_negative_pairs = self.get_model_dataset_edge_index(
            method='accuracy',
            ratio=args.finetune_ratio
        )  # ,ratio=args.finetune_ratio)#score

        if not 'without_transfer' in args.gnn_method:
            self.edge_index_tran_model_to_dataset, self.edge_attr_tran_model_to_dataset, self.tran_negative_pairs = self.get_model_dataset_edge_index(
                method='score',
                ratio=args.finetune_ratio
            )  # ,ratio=args.finetune_ratio)#score
        else:
            self.edge_index_tran_model_to_dataset = None
            self.edge_attr_tran_model_to_dataset = None
            self.tran_negative_pairs = None

        if 'without_accuracy' in args.gnn_method or 'trained_on_transfer' in args.gnn_method:
            self.negative_pairs = self.tran_negative_pairs
        else:
            self.negative_pairs = self.accu_negative_pairs

        self.dataset_reference_model = args.dataset_reference_model
        self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(
            base_dataset=self.base_dataset,
            threshold=args.distance_thres,
            sim_method=args.dataset_distance_method
        )
        logger.info(f"len(unique_model_id): {len(self.unique_model_id)}")
        logger.info(f'len(unique_dataset_id): {len(self.unique_dataset_id)}')

    def get_dataset_features(self, reference_model):
        data_feat = {}

        dataset_list = self.dataset_list.copy()

        for ori_dataset_name, dataset_name in dataset_list.items():
            embedding_directory = determine_directory_embedded_dataset(
                reference_model,
                self.args.task_type,
                DatasetEmbeddingMethod.DOMAIN_SIMILARITY
                )
            path = determine_file_name_embedded_dataset(embedding_directory, ori_dataset_name)

            try:
                if not os.path.exists(path):
                    raise Exception(
                        f'No embedding available for {dataset_name} at path {path}. Please run tools/embed_dataset.py first.'
                    )
                features = np.load(path)
            except Exception as e:
                logger.warning(e)
                logger.warning(f"No embedding available for {dataset_name} at path {path}, removing node.")
                del self.dataset_list[ori_dataset_name]
                continue
                features = np.zeros((1, self.FEATURE_DIM))

            features = np.mean(features, axis=0)
            data_feat[ori_dataset_name] = features

        return data_feat


class GraphAttributesWithTask2Vec(GraphAttributes):
    def __init__(self, args, approach='task2vec'):
        # invoking the __init__ of the parent class
        GraphAttributes.__init__(self, args)

        self.dataset_list = self.get_dataset_list()
        # if args.contain_dataset_feature:
        self.data_features = self.get_dataset_features(args.dataset_reference_model)

        if 'node2vec' in args.gnn_method or (not args.contain_model_featureure):
            self.model_features = []
            self.model_list = self.unique_model_id['model'].unique()
        else:
            self.model_features, self.model_list = self.get_model_features()
        # get common nodes
        self.drop_nodes()
        # self.reference_model = 'resnet50'
        self.max_dataset_idx = self.unique_dataset_id['mappedID'].max()

        ##########
        # get specific dataset index
        ##########
        if args.test_dataset != '':
            print(f'\n --- args.test_dataset: {args.test_dataset}')
            print(f'\n self.unique_dataset_id: {self.unique_dataset_id}')
            self.test_dataset_idx = self.unique_dataset_id[self.unique_dataset_id['dataset'] == args.test_dataset]['mappedID'].values[0]
            ##### !!! make the indeces of the dataset and the model different
            if 'homo' in self.args.gnn_method or 'node2vec' in self.args.gnn_method:
                self.unique_model_id['mappedID'] += self.max_dataset_idx + 1
            # print(f'unique_model_id: {self.unique_model_id}')
            self.model_idx = self.unique_model_id['mappedID'].values
        else:
            self.test_dataset_idx = -1
            self.model_idx = -1

        self.node_ID = list(self.model_idx) + list(self.unique_dataset_id['mappedID'].values)

        # get edge index
        self.edge_index_accu_model_to_dataset, self.edge_attr_accu_model_to_dataset, self.accu_negative_pairs = self.get_model_dataset_edge_index(
            method='accuracy'
        )  # ,ratio=args.finetune_ratio)#score
        self.edge_index_tran_model_to_dataset, self.edge_attr_tran_model_to_dataset, self.tran_negative_pairs = self.get_model_dataset_edge_index(
            method='score'
        )  # ,ratio=args.finetune_ratio)#score
        if 'without_accuracy' in args.gnn_method:
            self.negative_pairs = self.tran_negative_pairs
        else:
            self.negative_pairs = self.accu_negative_pairs

        # get dataset-dataset edge index
        self.edge_index_dataset_to_dataset, self.edge_attr_dataset_to_dataset = self.get_dataset_edge_index(
            base_dataset=self.base_dataset,
            threshold=args.distance_thres,
            sim_method=args.dataset_distance_method
        )
        self.edge_index_model_to_model, self.edge_attr_model_to_model = self.get_model_model_edge_index()

    def get_dataset_features(self, reference_model='resnet34'):
        from dataset_embed.task2vec_embed.embed_task import embed
        data_feat = {}
        dataset_list = self.dataset_list.copy()
        for ori_dataset_name, dataset_name in dataset_list.items():
            ds_name = dataset_name.replace(' ', '-')
            path = os.path.join(f'../dataset_embed/task2vec_embed/feature', f'{ds_name}_feature.p')
            if not os.path.exists(path):
                print('Try to obtain missing features')
                # features = embed('../',dataset_name)
                try:
                    features = embed('../', dataset_name, reference_model)
                except FileNotFoundError as e:
                    # print('\n----------')
                    # print(e)
                    # print(f'== fail to retrieve features and delete row {ds_name}')
                    del self.dataset_list[ori_dataset_name]
                    continue
            try:
                with open(path, 'rb') as f:
                    features = pickle.load(f).hessian
                features = features.reshape((1, features.shape[0]))
                # FEATURE_DIM = features.shape[1]
            except Exception as e:
                # print('----------')
                # print(e)
                del self.dataset_list[ori_dataset_name]
                continue
                features = np.zeros((1, self.FEATURE_DIM))
            # print(f'\n----success {ori_dataset_name}')

            # x = features == np.zeros((1,FEATURE_DIM))

            features = np.mean(features, axis=0)
            # print(f"\n====\nTask2Vec feature shape of {dataset_name} is {features.shape}")
            # data_feat.append(features)
            data_feat[ori_dataset_name] = features
        # data_feat = np.stack(data_feat)
        # print(f'== data_feat.shape:{data_feat.shape}')
        return data_feat
