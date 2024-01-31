import torch
from torch import optim
from transformers import AutoTokenizer

from load import load_dataset
from metrics import Metrics
from models.diffpool import DiffPoolModel
from utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_dim = 384
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = load_dataset(tokenizer, batch_size=64, num_workers=8)

model = DiffPoolModel(
    model_name=model_name,
    num_node_features=300,
    nout=embeddings_dim,
).to(device)

optimizer = optim.AdamW(
    model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0001
)

scheduler = optim.lr_scheduler.MultiplicativeLR(
    optimizer,
    lr_lambda=lambda epoch: 0.9333,  # At this rate we go from 1e-4 to 1e-7 in 100 epochs
    verbose=False,
)

save_path, _, _ = train(
    model,
    optimizer,
    train_loader,
    val_loader,
    scheduler=scheduler,
    metrics=Metrics("circle", m=0.1, gamma=1),
    nb_epochs=100,
    device=device,
    initial_freeze=3,
    print_every=50,
)
