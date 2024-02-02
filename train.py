import torch
from torch import optim
from torch.optim.lr_scheduler import MultiplicativeLR, SequentialLR
from transformers import AutoTokenizer

from load import load_dataset
from metrics import Metrics
from models.diffpool import DiffPoolModel
from utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_dim = 384
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = load_dataset(tokenizer, batch_size=64, num_workers=4)

model = DiffPoolModel(
    model_name=model_name,
    num_node_features=300,
    nout=embeddings_dim,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

# Constant learning rate for 10 epochs then decay by 0.95 every epoch until 100 epochs then constant
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1),
        MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.95, last_epoch=100),
    ],
    milestones=[9],
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
