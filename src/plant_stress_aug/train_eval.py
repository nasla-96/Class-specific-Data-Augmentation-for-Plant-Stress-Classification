from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from .dataset import ImageFolderCustom
from .model import Net, to_device


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [model.validation_step(device, batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit_one_cycle(chromosome, chromosome_idx, config, augmentation_names, output_dir="outputs"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.empty_cache()

    current_device = torch.device(f"cuda:{chromosome_idx % config['ga']['parallel_workers']}" if torch.cuda.is_available() else "cpu")
    train_dataset_dir = config["data"]["train_dir"]
    val_dataset_dir = config["data"]["val_dir"]

    dataset = ImageFolderCustom(train_dataset_dir, augmentation_names=augmentation_names, chromosome=chromosome)
    test_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_set = ImageFolder(val_dataset_dir, transform=test_transforms)

    train_fraction = config["training"]["train_split"]
    lengths = [int(np.ceil(train_fraction * len(dataset))), int(np.floor((1 - train_fraction) * len(dataset)))]
    train_set, valid_set = data.random_split(dataset, lengths)

    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = to_device(
        Net(
            checkpoint_path=config["model"]["checkpoint_path"],
            num_classes=config["model"]["num_classes"],
        ),
        device=current_device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["optimizer_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        config["training"]["max_lr"],
        epochs=config["training"]["epochs"],
        steps_per_epoch=len(train_loader),
    )

    history = []
    best_acc = 0.0
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.training_step(current_device, batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        result = evaluate(model, val_loader, current_device)
        best_acc = max(best_acc, result["val_acc"])
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["epoch"] = epoch
        model.epoch_end(epoch, result)
        history.append(result)

    y_pred_list = []
    y_true_list = []
    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as tepoch:
            for inp, labels in tepoch:
                inp, labels = inp.to(current_device), labels.to(current_device)
                preds = model(inp)
                _, pred_tags = torch.max(preds, dim=1)
                y_pred_list.append(pred_tags.cpu().numpy())
                y_true_list.append(labels.cpu().numpy())

    flat_pred = [item for batch in y_pred_list for item in batch]
    flat_true = [item for batch in y_true_list for item in batch]
    matrix = confusion_matrix(flat_true, flat_pred)
    mean_per_class = (matrix.diagonal() / matrix.sum(axis=1)).mean()
    per_class = np.round(matrix.diagonal() / matrix.sum(axis=1), 4)

    with open(output_dir / "perclass_accuracies.csv", "a", encoding="utf-8") as fd:
        fd.write(f"{per_class} , {mean_per_class}\n")

    plt.figure()
    sns.heatmap(matrix, annot=True, fmt="").set(
        title="confusion matrix", xlabel="Predicted Label", ylabel="True Label"
    )
    plt.savefig(output_dir / f"matrix_{mean_per_class:.6f}.png")
    plt.close()

    torch.save(model.state_dict(), output_dir / f"model_{mean_per_class:.16f}.pth")
    torch.cuda.empty_cache()

    return history, mean_per_class
