import os
import os.path as osp

import torch
from torch_geometric.data import Dataset


class WorstDataset(Dataset):
    def __init__(
        self,
        root,
        gt,
        split,
        tokenizer=None,
        transform=None,
        pre_transform=None,
        features=[],
    ):
        self.root = root
        self.gt = gt
        self.split = split
        self.tokenizer = tokenizer
        self.features = features

        self.cids = []
        for file in os.listdir(f"./data/worst_embeddings/{split}/"):
            filename = os.fsdecode(file)
            id = int(filename.split(".")[0].split("_")[1])
            self.cids.append(id)

        self.idx_to_cid = {}
        i = 0
        for cid in self.cids:
            self.idx_to_cid[i] = cid
            i += 1
        super(WorstDataset, self).__init__(root, transform, pre_transform)

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
        special = ""
        for feature in self.features:
            special += "_" + feature
        return osp.join(self.root, "worst_embeddings" + special + "/", self.split)

    def download(self):
        pass

    def process_graph(self, raw_path):
        pass

    def process(self):
        print("TRIED TO PROCESS")
        pass

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
