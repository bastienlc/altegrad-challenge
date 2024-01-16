import torch
from torch import optim
from transformers import AutoTokenizer

from load import load_dataset
from models.baseline import BaselineModel
from training_utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = load_dataset(tokenizer)

model = BaselineModel(
    model_name=model_name,
    num_node_features=300,
    nout=768,
    nhid=300,
    graph_hidden_channels=450,
).to(device)

optimizer = optim.AdamW(
    model.parameters(), lr=5e-5, betas=(0.9, 0.999), weight_decay=0.01
)

save_path, _, _ = train(
    model,
    optimizer,
    train_loader,
    val_loader,
    nb_epochs=10,
    device=device,
)
