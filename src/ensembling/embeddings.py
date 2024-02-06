from typing import List

import torch
from transformers import PreTrainedTokenizer

from load import load_dataset, load_test_dataset
from utils import get_test_embeddings, get_train_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_embeddings(
    models: List[torch.nn.Module],
    tokenizers: List[PreTrainedTokenizer],
    saved_paths: List[str],
    skip: List[bool],
    split: str,
    device: torch.device = device,
):
    for k, (model, tokenizer, saved_path) in enumerate(
        zip(models, tokenizers, saved_paths)
    ):
        if skip[k]:
            continue
        else:
            print(f"Processing model {k+1}")

        checkpoint = torch.load(saved_path)
        model.to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        if split == "test":
            test_loader, test_text_loader = load_test_dataset(tokenizer, batch_size=8)

            graph_embeddings, text_embeddings = get_test_embeddings(
                model.get_graph_encoder(),
                model.get_text_encoder(),
                test_loader,
                test_text_loader,
                device,
                low_memory=True,
            )
        elif split == "train":
            loader, _ = load_dataset(tokenizer, batch_size=8, shuffle=False)
            graph_embeddings, text_embeddings = get_train_embeddings(
                model.get_graph_encoder(),
                model.get_text_encoder(),
                loader,
                device,
                low_memory=True,
            )
        elif split == "val":
            _, loader = load_dataset(tokenizer, batch_size=8, shuffle=False)
            graph_embeddings, text_embeddings = get_train_embeddings(
                model.get_graph_encoder(),
                model.get_text_encoder(),
                loader,
                device,
                low_memory=True,
            )
        else:
            raise ValueError("split must be 'test', 'train' or 'val'")

        torch.save(
            graph_embeddings, f"./outputs/embeddings/{split}/graph_embeddings{k}.pt"
        )
        torch.save(
            text_embeddings, f"./outputs/embeddings/{split}/text_embeddings{k}.pt"
        )

        # free memory
        model.to("cpu")
        del model
        torch.cuda.empty_cache()
