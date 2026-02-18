"""Train the CRF prediction DNN with early stopping and wandb logging."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, TensorDataset

import config
import utils
from model import CRFPredictor

log = utils.setup_logging("train")


def load_data():
    """Load train and validation CSVs, return DataLoaders."""
    train_df = pd.read_parquet(config.DATASET_DIR / "train.parquet")
    val_df = pd.read_parquet(config.DATASET_DIR / "val.parquet")

    # Input columns: everything except metadata and label
    meta_cols = {"crf_class", "video_name", "segment_name", "crf", "vmaf"}
    input_cols = [c for c in train_df.columns if c not in meta_cols]

    X_train = torch.tensor(train_df[input_cols].values, dtype=torch.float32)
    y_train = torch.tensor(train_df["crf_class"].values, dtype=torch.long)
    X_val = torch.tensor(val_df[input_cols].values, dtype=torch.float32)
    y_val = torch.tensor(val_df["crf_class"].values, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    log.info(f"Train: {len(train_ds)} samples, {len(input_cols)} features")
    log.info(f"Val:   {len(val_ds)} samples")

    return train_loader, val_loader, len(input_cols)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def plot_curves(history: dict, out_dir: Path):
    """Plot training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs, history["train_acc"], label="Train")
    ax1.plot(epochs, history["val_acc"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, history["train_loss"], label="Train")
    ax2.plot(epochs, history["val_loss"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()
    log.info(f"Saved training curves to {out_dir / 'training_curves.png'}")


def main():
    parser = argparse.ArgumentParser(description="Train CRF prediction DNN")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE)
    parser.add_argument("--wandb-project", type=str, default="crf-estimation",
                        help="Weights & Biases project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="Weights & Biases run name")
    args = parser.parse_args()

    utils.ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    train_loader, val_loader, input_dim = load_data()

    model = CRFPredictor(input_dim=input_dim, num_classes=config.NUM_CRF_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {num_params:,}")

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            "architecture": "DNN-5layer",
            "hidden_layers": [100, 68, 50, 100, 50],
            "input_dim": input_dim,
            "num_classes": config.NUM_CRF_CLASSES,
            "crf_values": config.CRF_VALUES,
            "learning_rate": args.lr,
            "batch_size": config.BATCH_SIZE,
            "max_epochs": args.epochs,
            "early_stop_patience": args.patience,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "vmaf_cap": config.VMAF_CAP,
            "codec": "SVT-AV1",
            "svt_preset": config.SVT_PRESET,
            "num_parameters": num_params,
            "device": str(device),
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        },
    )
    wandb.watch(model, log="all", log_freq=100)

    # Early stopping state
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else val_loss,
        })

        log.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    # Log final summary to wandb
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val_loss"] = best_val_loss
    wandb.summary["best_val_accuracy"] = history["val_acc"][best_epoch - 1]
    wandb.summary["final_train_accuracy"] = history["train_acc"][-1]

    # Save model
    model_path = config.MODELS_DIR / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": config.NUM_CRF_CLASSES,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }, model_path)
    log.info(f"Saved model to {model_path}")

    # Log model artifact to wandb
    artifact = wandb.Artifact("crf-predictor", type="model")
    artifact.add_file(str(model_path))
    wandb.log_artifact(artifact)

    # Save history
    with open(config.RESULTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f)

    plot_curves(history, config.RESULTS_DIR)

    wandb.finish()


if __name__ == "__main__":
    main()
