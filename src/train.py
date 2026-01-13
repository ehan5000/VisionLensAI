import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset import VisionDataset
from model import VisionNet

def main():
    # Ensure dataset exists
    if not os.path.exists("vision_data.npz"):
        raise FileNotFoundError("vision_data.npz sory :c not found. Run: python src/generate_data.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VisionDataset("vision_data.npz")
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = VisionNet(in_dim=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    epochs = 25
    train_losses = []
    val_losses = []

    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            preds = model(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * X.size(0)

        train_loss = running / n_train
        train_losses.append(train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = criterion(preds, y)
                running_val += loss.item() * X.size(0)

        val_loss = running_val / n_val
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "vision_model.pt")

    print(f"Saved best model to vision_model.pt (best val loss: {best_val:.3f})")

    # Save a loss curve for the README / recruiter credibility
    plt.figure()
    plt.plot(np.arange(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(np.arange(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved training_loss.png")

if __name__ == "__main__":
    main()
