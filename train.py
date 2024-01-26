import torch
from torch import optim
from transformers import AutoTokenizer

from load import load_dataset
from models.torch_gat import GATModel
from utils import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_dim = 384
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_loader, val_loader = load_dataset(tokenizer)

model = GATModel(
    model_name=model_name,
    num_node_features=300,
    nout=embeddings_dim,
    d_hidden_dim=1000,
    num_layers=5,
    num_heads=2,
    d_linear_layers=[1000],
).to(device)

optimizer = optim.AdamW(
    model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01
)

save_path, _, _ = train(
    model,
    optimizer,
    train_loader,
    val_loader,
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, verbose=True, factor=0.7
    ),
    custom_loss="contrastive",
    nb_epochs=50,
    device=device,
)
