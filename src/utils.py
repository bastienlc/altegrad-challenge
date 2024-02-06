import os
import time
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.nn import summary

from metrics import Metrics

# Remove warning when using the tokenizer to preprocess the data
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def process_batch(batch, model, optimizer, device, loss):
    input_ids = batch.input_ids
    batch.pop("input_ids")
    attention_mask = batch.attention_mask
    batch.pop("attention_mask")
    graph_batch = batch

    x_graph, x_text = model(
        graph_batch.to(device), input_ids.to(device), attention_mask.to(device)
    )

    current_loss = loss(x_graph, x_text)
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
    metrics: Metrics = Metrics(),
    initial_freeze: int = 0,
    print_every: int = 50,
    load_optimizer: bool = True,
    validate_every: int = 1,
):
    writer = SummaryWriter(log_dir="runs/" + save_name)
    epoch = 0
    loss = 0
    loss_averager = 0
    losses = []
    time1 = time.time()
    running_time = 0
    best_validation_loss = 1e100
    best_validation_score = 0

    model_summary = summary(
        model.graph_encoder,
        next(iter(train_loader)).to(device),
        max_depth=10,
    )
    print(model_summary)
    with open("./outputs/parameters.txt", "w") as f:
        f.write(model_summary)

    if load_from is not None:
        checkpoint = torch.load(load_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        try:
            best_validation_loss = checkpoint["val_loss"]
        except:
            print("No validation loss in checkpoint")
        try:
            best_validation_score = checkpoint["val_score"]
        except:
            print("No validation score in checkpoint")
        try:
            epoch = checkpoint["epoch"]
        except:
            print("No epoch in checkpoint")
        try:
            running_time = checkpoint["time"]
        except:
            print("No time in checkpoint")
        if load_optimizer:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except:
                print("No optimizer in checkpoint")
        print(
            "Loaded model from {}, best validation score={}, best validation loss={}".format(
                load_from, best_validation_score, best_validation_loss
            )
        )

    # Freeze the text encoder for the first epochs so that we don't degrade the pretrained weights
    if initial_freeze > 0:
        model.text_encoder.requires_grad_(False)

    for e in range(epoch + 1, nb_epochs):
        print(f"------------EPOCH {e:^4}------------")
        if initial_freeze > 0 and e == initial_freeze + 1:
            print("[UNFREEZING] text encoder weights")
            model.text_encoder.requires_grad_(True)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss += process_batch(batch, model, optimizer, device, metrics.loss)
            loss_averager += 1

            if batch_idx % print_every == 0 and batch_idx > 0:
                loss /= loss_averager
                print(
                    f"Batch {batch_idx:<3} | {time.strftime('%H:%M:%S', time.gmtime(int(running_time + time.time() - time1)))} | Loss {loss:<6.4f}"
                )
                losses.append(loss)
                writer.add_scalar("Loss/train", loss, e * len(train_loader) + batch_idx)
                loss = 0
                loss_averager = 0

        if e % validate_every == 0:
            step = (e + 1) * len(train_loader)

            validation_metrics = metrics.get(model, val_loader, device=device)
            val_loss = validation_metrics["loss"]
            val_score = validation_metrics["score"]
            writer.add_scalar("Loss/val", val_loss, step)
            writer.add_scalar("Score/val", val_score, step)

            writer.flush()

            if scheduler is not None:
                scheduler.step()

            print(f"[LOSS] {val_loss:<6.4f} | [SCORE] {val_score:<6.4f}")
            print(f"[LR] {optimizer.param_groups[0]['lr']:<8.2E}")

            best_validation_loss = min(best_validation_loss, val_loss)
            best_validation_score = max(best_validation_score, val_score)

            if best_validation_score == val_score:
                save_path = os.path.join("./outputs/", save_name + str(e) + ".pt")
                torch.save(
                    {
                        "epoch": e,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_score": val_score,
                        "time": running_time + time.time() - time1,
                    },
                    save_path,
                )
                if e > 1:
                    try:
                        os.remove(last_save_path)
                    except:
                        pass

                last_save_path = save_path
                print(f"[SAVED] at {save_path}")

    writer.close()
    return save_path, best_validation_loss, best_validation_score


def get_test_embeddings(
    graph_encoder: nn.Module,
    text_encoder: nn.Module,
    test_loader: DataLoader,
    test_text_loader: DataLoader,
    device: torch.device,
    low_memory: bool = False,
):
    if low_memory:
        text_encoder.to("cpu")
        graph_encoder.to(device)

    graph_encoder.eval()
    text_encoder.eval()
    graph_embeddings = []
    for batch in test_loader:
        for output in graph_encoder(batch.to(device)):
            graph_embeddings.append(output.tolist())

    if low_memory:
        graph_encoder.to("cpu")
        text_encoder.to(device)

    text_embeddings = []
    for batch in test_text_loader:
        for output in text_encoder(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ):
            text_embeddings.append(output.tolist())

    if low_memory:
        graph_encoder.to(device)
        text_encoder.to(device)

    return torch.Tensor(np.array(graph_embeddings)), torch.Tensor(
        np.array(text_embeddings)
    )


def get_train_embeddings(
    graph_encoder: nn.Module,
    text_encoder: nn.Module,
    loader: GeometricDataLoader,
    device: torch.device,
    low_memory: bool = False,
):
    # Make sure your data loader is not shuffling the data
    if low_memory:
        text_encoder.to("cpu")
        graph_encoder.to(device)

    graph_encoder.eval()
    text_encoder.eval()
    graph_embeddings = []
    for batch in loader:
        batch.pop("input_ids")
        batch.pop("attention_mask")
        for output in graph_encoder(batch.to(device)):
            graph_embeddings.append(output.tolist())

    if low_memory:
        graph_encoder.to("cpu")
        text_encoder.to(device)

    text_embeddings = []
    for batch in loader:
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        for output in text_encoder(
            input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ):
            text_embeddings.append(output.tolist())

    if low_memory:
        graph_encoder.to(device)
        text_encoder.to(device)

    return torch.Tensor(np.array(graph_embeddings)), torch.Tensor(
        np.array(text_embeddings)
    )
