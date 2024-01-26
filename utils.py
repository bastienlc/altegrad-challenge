import os
import time
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import RankedListLoss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Remove warning when using the tokenizer to preprocess the data
os.environ["TOKENIZERS_PARALLELISM"] = "false"

CEL = nn.CrossEntropyLoss()
RLL = RankedListLoss(margin=0.1, Tn=1, Tp=1)


def solution_from_embeddings(graph_embeddings, text_embeddings, save_to="solution.csv"):
    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution["ID"] = solution.index
    solution = solution[["ID"] + [col for col in solution.columns if col != "ID"]]
    solution.to_csv(f"outputs/{save_to}", index=False)


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CEL(logits, labels) + CEL(torch.transpose(logits, 0, 1), labels)


def ranked_list_loss(v1, v2):
    labels = torch.arange(v1.shape[0], device=v1.device)
    return RLL(torch.cat((v1, v2)), torch.cat((labels, labels)))


def get_metrics(
    model: nn.Module,
    loader: GeometricDataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    model.eval()
    graph_embeddings = []
    text_embeddings = []
    loss = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids
            batch.pop("input_ids")
            attention_mask = batch.attention_mask
            batch.pop("attention_mask")
            graph_batch = batch

            x_graph, x_text = model(
                graph_batch.to(device),
                input_ids.to(device),
                attention_mask.to(device),
            )

            loss += contrastive_loss(x_graph, x_text).item()

            for output in x_graph:
                graph_embeddings.append(output.tolist())
            for output in x_text:
                text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    return (
        loss / len(loader),
        label_ranking_average_precision_score(np.eye(len(similarity)), similarity),
    )


def process_batch(
    batch, model, optimizer, device, custom_loss: Union[str, None] = None
):
    input_ids = batch.input_ids
    batch.pop("input_ids")
    attention_mask = batch.attention_mask
    batch.pop("attention_mask")
    graph_batch = batch

    x_graph, x_text = model(
        graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
    )
    if custom_loss is None or custom_loss == "contrastive":
        current_loss = contrastive_loss(x_graph, x_text)
    elif custom_loss == "ranked_list":
        current_loss = ranked_list_loss(x_graph, x_text)
    else:
        raise ValueError("Unknown loss")
    optimizer.zero_grad()
    current_loss.backward()
    optimizer.step()
    return current_loss.item()


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: GeometricDataLoader,
    val_loader: GeometricDataLoader,
    nb_epochs: int = 5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    load_from: Union[str, None] = None,
    save_name: str = "model",
    scheduler: Union[optim.lr_scheduler._LRScheduler, None] = None,
    custom_loss: Union[str, None] = None,
):
    writer = SummaryWriter()
    epoch = 0
    loss = 0
    loss_averager = 0
    losses = []
    time1 = time.time()
    print_every = 50
    best_validation_loss = 1e100
    best_validation_score = 0

    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_validation_loss = checkpoint["val_loss"]
        best_validation_score = checkpoint["val_score"]
        epoch = checkpoint["epoch"]
        print(
            "Loaded model from {}, best_validation_score={}, best validation loss={}".format(
                load_from, best_validation_score, best_validation_loss
            )
        )

    for e in range(epoch + 1, nb_epochs):
        print("--------------------EPOCH {}--------------------".format(e))
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss += process_batch(
                batch, model, optimizer, device, custom_loss=custom_loss
            )
            loss_averager += 1

            if batch_idx % print_every == 0 and batch_idx > 0:
                loss /= loss_averager
                time2 = time.time()
                print(
                    "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                        batch_idx, time2 - time1, loss
                    )
                )
                losses.append(loss)
                writer.add_scalar("Loss/train", loss, e * len(train_loader) + batch_idx)
                loss = 0
                loss_averager = 0

        step = (e + 1) * len(train_loader)

        print(
            "Computing metrics on validation set... (time={:.4f}s)".format(
                time.time() - time1
            )
        )
        val_loss, val_score = get_metrics(model, val_loader, device=device)
        writer.add_scalar("Loss/val", val_loss, step)
        writer.add_scalar("Score/val", val_score, step)

        writer.flush()

        print(
            "Epoch " + str(e) + " finished with val_loss " + str(val_loss),
            "and val_score",
            val_score,
        )

        if scheduler is not None:
            scheduler.step(val_score)

        best_validation_loss = min(best_validation_loss, val_loss)
        best_validation_score = max(best_validation_score, val_score)

        if best_validation_score == val_score:
            print("Saving checkpoint... ", end="")
            save_path = os.path.join("./outputs/", save_name + str(e) + ".pt")
            torch.save(
                {
                    "epoch": e,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_score": val_score,
                },
                save_path,
            )
            if e > 2:
                try:
                    os.remove(last_save_path)
                except:
                    pass

            last_save_path = save_path
            print("done : {}".format(save_path))

    writer.close()
    return save_path, best_validation_loss, best_validation_score
