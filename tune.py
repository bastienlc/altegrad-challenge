import os
from functools import partial

import torch
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from torch import optim
from transformers import AutoTokenizer

from load import load_dataset
from models.baseline import BaselineModel
from utils import get_metrics, process_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_wrapper(config):
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = BaselineModel(
        model_name=model_name,
        num_node_features=300,
        nout=768,
        nhid=config["nhid"],
        graph_hidden_channels=config["graph_hidden_channels"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.999), weight_decay=0.01
    )

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        epoch = 0

    TUNE_ORIG_WORKING_DIR = os.environ.get("TUNE_ORIG_WORKING_DIR")
    train_loader, val_loader = load_dataset(tokenizer, root=TUNE_ORIG_WORKING_DIR)

    loss = 0
    print_every = 50
    for e in range(epoch, config["max_epochs"]):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss += process_batch(batch, model, optimizer, device)

            if batch_idx % print_every == 0:
                checkpoint_data = {
                    "epoch": e,
                    "batch_idx": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                checkpoint = Checkpoint.from_dict(checkpoint_data)

                val_loss, val_score = get_metrics(model, val_loader)
                session.report(
                    {
                        "train_loss": loss / print_every,
                        "val_loss": val_loss,
                        "val_score": val_score,
                    },
                    checkpoint=checkpoint,
                )
                loss = 0


config = {
    "nhid": tune.choice([150, 300, 450]),
    "graph_hidden_channels": tune.choice([150, 300, 450]),
    "lr": tune.loguniform(5e-6, 1e-4),
    "max_epochs": 2,
}

scheduler = ASHAScheduler(
    metric="val_score",
    mode="max",
    max_t=10,
    grace_period=1,
    reduction_factor=2,
)

result = tune.run(
    partial(train_wrapper),
    resources_per_trial={"cpu": 20, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
)

best_trial = result.get_best_trial("val_score", "max", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
