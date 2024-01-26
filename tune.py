import os
from functools import partial

import torch
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from torch import optim
from transformers import AutoTokenizer

from load import load_dataset
from models.torch_gat import GATModel
from utils import process_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_wrapper(config):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    d_linear_layers = []
    for i in range(config["d_linear_layers"]):
        d_linear_layers.append(config[f"d_linear_layers_{i}"])

    model = GATModel(
        model_name=model_name,
        num_node_features=300,
        nout=384,
        d_hidden_dim=config["d_hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_linear_layers=d_linear_layers,
        dropout=config["dropout"],
        activation=config["activation"],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"],
    )

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        epoch = 1

    TUNE_ORIG_WORKING_DIR = os.environ.get("TUNE_ORIG_WORKING_DIR")
    train_loader, _ = load_dataset(tokenizer, root=TUNE_ORIG_WORKING_DIR, batch_size=32)

    loss = 0
    loss_counter = 0
    print_every = 100
    for e in range(epoch, 5):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            loss += process_batch(batch, model, optimizer, device)
            loss_counter += 1

            if loss_counter % print_every == 0:
                checkpoint_data = {
                    "epoch": e,
                    "batch_idx": batch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                checkpoint = Checkpoint.from_dict(checkpoint_data)

                session.report(
                    {
                        "loss": loss / loss_counter,
                    },
                    checkpoint=checkpoint,
                )
                loss = 0
                loss_counter = 0


config = {
    "d_hidden_dim": tune.lograndint(100, 2000),
    "num_layers": tune.lograndint(1, 20),
    "num_heads": tune.lograndint(1, 20),
    "d_linear_layers": tune.choice([1, 2, 3, 4]),
    "d_linear_layers_0": tune.lograndint(100, 2000),
    "d_linear_layers_1": tune.lograndint(100, 2000),
    "d_linear_layers_2": tune.lograndint(100, 2000),
    "d_linear_layers_3": tune.lograndint(100, 2000),
    "dropout": tune.uniform(0.01, 0.2),
    "activation": tune.choice(["ReLU", "LeakyReLU", "Sigmoid", "Tanh"]),
    "lr": tune.loguniform(5e-6, 1e-4),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
}

scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=100,
    grace_period=3,
    reduction_factor=2,
)

result = tune.run(
    partial(train_wrapper),
    resources_per_trial={"cpu": 20, "gpu": 1},
    config=config,
    num_samples=1000,
    scheduler=scheduler,
    resume=False,
    local_dir="./tune_altegrad",
)

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final loss: {best_trial.last_result['loss']}")
