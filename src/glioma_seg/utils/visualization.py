"""Plotting helpers for raw / processed volumes and training history."""

import os
import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from glioma_seg.data.io import DataExtractor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Visualizer:
    """Inspect raw and preprocessed MRI volumes via mid-slice plots."""

    def __init__(self, seed):
        self.seed = seed

    def plot_raw(self, raw_dir, num_samples=5):
        """Plot mid-axial slices from raw NIfTI images for random subjects."""
        if self.seed is not None:
            random.seed(self.seed)

        subjects = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
        samples = random.sample(subjects, num_samples)

        fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        for i, subj in enumerate(samples):
            flair, t1c, label = DataExtractor(raw_dir).load(subj)
            idx = flair.shape[2] // 2

            axs[i, 0].imshow(flair[:, :, idx], cmap="gray")
            axs[i, 0].set_title(f"{subj} - FLAIR")
            axs[i, 0].axis("off")

            axs[i, 1].imshow(t1c[:, :, idx], cmap="gray")
            axs[i, 1].set_title(f"{subj} - T1c")
            axs[i, 1].axis("off")

            axs[i, 2].imshow(label[:, :, idx], cmap="tab20", interpolation="none")
            axs[i, 2].set_title(f"{subj} - Label")
            axs[i, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def plot_processed(self, processed_dir, num_samples=5):
        """Plot mid-axial slices from preprocessed ``.pt`` files."""
        if self.seed is not None:
            random.seed(self.seed)

        train_dir = os.path.join(processed_dir, "train")
        data_files = sorted(os.listdir(os.path.join(train_dir, "data")))
        label_files = sorted(os.listdir(os.path.join(train_dir, "labels")))

        idxs = random.sample(range(len(data_files)), num_samples)

        fig, axs = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))

        for i, idx in enumerate(idxs):
            data = torch.load(os.path.join(train_dir, "data", data_files[idx]))
            label = torch.load(os.path.join(train_dir, "labels", label_files[idx]))

            sl = data.shape[-1] // 2

            axs[0, i].imshow(data[0, :, :, sl], cmap="gray")
            axs[0, i].set_title(data_files[idx])
            axs[0, i].axis("off")

            axs[1, i].imshow(data[1, :, :, sl], cmap="gray")
            axs[1, i].set_title("Modality 2")
            axs[1, i].axis("off")

            axs[2, i].imshow(label[0, :, :, sl], cmap="viridis")
            axs[2, i].set_title(label_files[idx])
            axs[2, i].axis("off")

        plt.tight_layout()
        plt.show()


class Plotter:
    """Plot training history and qualitative prediction examples."""

    def __init__(self):
        pass

    def plot_history(self, history_path: str):
        """Plot training/validation loss and Dice curves from a saved history dict."""
        logs = torch.load(history_path)
        epochs = logs["epoch"]
        train_losses = logs["training_loss"]
        val_losses = logs["validation_loss"]
        dices = logs["dice_score"]

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(epochs, dices, label="Dice Score (%)", color="tab:green")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Score (%)")
        plt.title("Dice Score")
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def visualize_samples(
        self,
        model,
        loader: DataLoader,
        num_samples: int = 5,
        slice_index: int = None,
    ):
        """Plot random (image, ground truth, prediction) triples from a loader."""
        dataset = loader.dataset
        picks = random.sample(range(len(dataset)), num_samples)
        model.to(DEVICE)

        for idx in picks:
            data, label = dataset[idx]
            x = data.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(x)
                pred = torch.argmax(out, dim=1)

            d_dim = data.shape[-1]
            z = slice_index if slice_index is not None else d_dim // 2

            img = data[0, :, :, z].cpu()
            gt = label[0, :, :, z].cpu()
            pr = pred[0, :, :, z].cpu()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img, cmap="gray")
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(gt, cmap="jet", alpha=0.6)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pr, cmap="jet", alpha=0.6)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()
