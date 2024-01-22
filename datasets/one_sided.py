import os
import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from .graphormer import add_shortest_distances
from .pathnn import add_pathnn_data


class OneSidedDataset(Dataset):
    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        text_encoder=None,
        device=None,
        transform=None,
        pre_transform=None,
        features=[],
    ):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.description = pd.read_csv(
            os.path.join(self.root, split + ".tsv"), sep="\t", header=None
        )
        self.description = self.description.set_index(0).to_dict()
        self.cids = list(self.description[1].keys())
        self.features = features
        self.device = device

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(OneSidedDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [str(cid) + ".graph" for cid in self.cids]

    @property
    def processed_file_names(self):
        return ["data_{}.pt".format(cid) for cid in self.cids]

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, "raw")

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, "processed_one_sided/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        edge_index = []
        x = []
        with open(raw_path, "r") as f:
            next(f)
            for line in f:
                if line != "\n":
                    edge = (*map(int, line.split()),)
                    edge_index.append(edge)
                else:
                    if edge_index == []:
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                    else:
                        edge_index = torch.LongTensor(edge_index).T
                    break
            next(f)
            for line in f:  # get mol2vec features:
                substruct_id = line.strip().split()[-1]
                if substruct_id in self.gt.keys():
                    x.append(self.gt[substruct_id])
                else:
                    x.append(self.gt["UNK"])
            return edge_index, torch.FloatTensor(np.array(x))

    def process(self):
        i = 0
        for raw_path in tqdm(self.raw_paths):
            cid = int(raw_path.split("/")[-1][:-6])
            text_input = self.tokenizer(
                [self.description[1][cid]],
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding="max_length",
                add_special_tokens=True,
            )
            with torch.no_grad():
                y = self.text_encoder(
                    text_input["input_ids"].to(self.device),
                    text_input["attention_mask"].to(self.device),
                )
            edge_index, x = self.process_graph(raw_path)
            data = Data(  # leave input_ids and attention_mask in the data to provide a common interface with the GraphTextDataset
                x=x,
                edge_index=edge_index,
                y=y.to("cpu"),
                input_ids=text_input["input_ids"],
                attention_mask=text_input["attention_mask"],
            )

            if "pathnn" in self.features:
                data = add_pathnn_data(data)
            if "shortest_distances" in self.features:
                data = add_shortest_distances(data)

            torch.save(data, osp.join(self.processed_dir, "data_{}.pt".format(cid)))
            i += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data_{}.pt".format(self.idx_to_cid[idx]))
        )
        return data

    def get_cid(self, cid):
        data = torch.load(osp.join(self.processed_dir, "data_{}.pt".format(cid)))
        return data
