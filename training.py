import os
import time
from typing import Union

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.loader import DataLoader as GeometricDataLoader

CE = nn.CrossEntropyLoss()


def contrastive_loss(v1, v2):
    logits = torch.matmul(v1, torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: GeometricDataLoader,
    val_loader: GeometricDataLoader,
    nb_epochs: int = 5,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    load_from: Union[str, None] = None,
):
    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    best_validation_loss = 1000000

    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_validation_loss = checkpoint["validation_accuracy"]
        epoch = checkpoint["epoch"]
        print(
            "loaded model from: {}, best validation loss: {}".format(
                load_from, best_validation_loss
            )
        )

    for i in range(epoch + 1, nb_epochs):
        print("-----EPOCH{}-----".format(i + 1))
        model.train()
        for batch in train_loader:
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
            loss += current_loss.item()

            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print(
                    "Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(
                        count_iter, time2 - time1, loss / printEvery
                    )
                )
                losses.append(loss)
                loss = 0

        model.eval()
        val_loss = 0
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop("input_ids")
            attention_mask = batch.attention_mask
            batch.pop("attention_mask")
            graph_batch = batch
            x_graph, x_text = model(
                graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
            )
            current_loss = contrastive_loss(x_graph, x_text)
            val_loss += current_loss.item()
        best_validation_loss = min(best_validation_loss, val_loss)
        print(
            "-----EPOCH" + str(i + 1) + "----- done.  Validation loss: ",
            str(val_loss / len(val_loader)),
        )
        if best_validation_loss == val_loss:
            print("validation loss improoved saving checkpoint...")
            save_path = os.path.join("./outputs/", "model" + str(i) + ".pt")
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_accuracy": val_loss,
                    "loss": loss,
                },
                save_path,
            )
            print("checkpoint saved to: {}".format(save_path))

    return save_path
