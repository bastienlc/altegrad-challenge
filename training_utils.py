import os
import time
from typing import Union

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as GeometricDataLoader

from utils import contrastive_loss, get_metrics


def process_batch(batch, model, optimizer, device):
    input_ids = batch.input_ids
    batch.pop("input_ids")
    attention_mask = batch.attention_mask
    batch.pop("attention_mask")
    graph_batch = batch

    x_graph, x_text = model(
        graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
    )
    current_loss = contrastive_loss(x_graph, x_text)
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
):
    writer = SummaryWriter()
    epoch = 0
    loss = 0
    losses = []
    time1 = time.time()
    print_every = 50
    best_validation_loss = 1e100

    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_validation_loss = checkpoint["val_loss"]
        epoch = checkpoint["epoch"]
        print(
            "Loaded model from {}, best validation loss={}".format(
                load_from, best_validation_loss
            )
        )

    for e in range(epoch + 1, nb_epochs):
        print("--------------------EPOCH {}--------------------".format(e))
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss += process_batch(batch, model, optimizer, device)

            if batch_idx % print_every == 0 and batch_idx > 0:
                loss /= print_every
                time2 = time.time()
                print(
                    "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                        batch_idx, time2 - time1, loss
                    )
                )
                losses.append(loss)
                writer.add_scalar("Loss/train", loss, e * len(train_loader) + batch_idx)
                loss = 0

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

        best_validation_loss = min(best_validation_loss, val_loss)

        if best_validation_loss == val_loss:
            print("Saving checkpoint... ", end="")
            save_path = os.path.join("./outputs/", "model" + str(e) + ".pt")
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
            print("done : {}".format(save_path))

    writer.close()
    return save_path, best_validation_loss, val_score
