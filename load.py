import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizer

from dataloader import GraphDataset, GraphTextDataset, TextDataset


def load_dataset(tokenizer: PreTrainedTokenizer, batch_size: int = 32):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="val", tokenizer=tokenizer
    )
    train_dataset = GraphTextDataset(
        root="./data/", gt=gt, split="train", tokenizer=tokenizer
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader


def load_test_dataset(tokenizer: PreTrainedTokenizer, batch_size: int = 32):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDataset(root="./data/", gt=gt, split="test_cids")
    test_text_dataset = TextDataset(
        file_path="./data/test_text.txt", tokenizer=tokenizer
    )

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)
    test_text_loader = TorchDataLoader(
        test_text_dataset, batch_size=batch_size, shuffle=False
    )

    return test_loader, test_text_loader
