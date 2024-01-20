import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from transformers import PreTrainedTokenizer

from datasets import GraphDataset, GraphTextDataset, TextDataset


def load_dataset(
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 32,
    dummy=False,
    root=".",
    features=[],
):
    gt = np.load(f"{root}/data/token_embedding_dict.npy", allow_pickle=True)[()]
    train_dataset = GraphTextDataset(
        root=f"{root}/data/",
        gt=gt,
        split="train",
        tokenizer=tokenizer,
        features=features,
    )
    val_dataset = GraphTextDataset(
        root=f"{root}/data/", gt=gt, split="val", tokenizer=tokenizer, features=features
    )

    if dummy:
        train_subset = Subset(train_dataset, range(len(train_dataset) // 100))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_subset = Subset(val_dataset, range(len(val_dataset) // 100))
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    return train_loader, val_loader


def load_test_dataset(
    tokenizer: PreTrainedTokenizer, batch_size: int = 32, dummy=False, features=[]
):
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    test_cids_dataset = GraphDataset(
        root="./data/", gt=gt, split="test_cids", features=features
    )
    test_text_dataset = TextDataset(
        file_path="./data/test_text.txt", tokenizer=tokenizer
    )

    if dummy:
        test_subset = Subset(test_cids_dataset, range(len(test_cids_dataset) // 100))
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        test_text_subset = Subset(
            test_text_dataset, range(len(test_text_dataset) // 100)
        )
        test_text_loader = TorchDataLoader(
            test_text_subset, batch_size=batch_size, shuffle=False
        )

    else:
        test_loader = DataLoader(
            test_cids_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        test_text_loader = TorchDataLoader(
            test_text_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

    return test_loader, test_text_loader
