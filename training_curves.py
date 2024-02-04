import matplotlib.pyplot as plt
from tbparse import SummaryReader

log_dir = "./outputs/saved/diffpool-base"

reader = SummaryReader(log_dir)
df = reader.scalars

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

df[df["tag"] == "Loss/train"].plot(x="step", y="value", ax=ax[0], label="train_loss")
df[df["tag"] == "Loss/val"].plot(x="step", y="value", ax=ax[0], label="val_loss")
df[df["tag"] == "Score/val"].plot(x="step", y="value", ax=ax[1], label="val_score")

ax[0].set_title("Loss")
ax[1].set_title("Score")
plt.show()
