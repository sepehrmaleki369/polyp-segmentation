import os
import torch
import torch.optim as optim
import yaml
import argparse
import numpy as np
import random

from models.transunet import TransUNet
from models.new_transunet import EnhancedTransUNet
from models.unetpp import UNetPP

from utils.data_utils import get_loaders, get_transforms
from utils.metrics import BCEDiceLoss, calculate_metrics

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    batch_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, list):
            loss = sum([loss_fn(o, masks) for o in outputs]) / len(outputs)
        else:
            loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        metrics = calculate_metrics(outputs if not isinstance(outputs, list) else outputs[-1], masks)
        for key in batch_metrics:
            batch_metrics[key] += metrics[key]
        epoch_loss += loss.item()

    epoch_loss /= len(loader)
    for key in batch_metrics:
        batch_metrics[key] /= len(loader)

    return epoch_loss, batch_metrics

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if isinstance(outputs, list):
            loss = sum([loss_fn(o, masks) for o in outputs]) / len(outputs)
        else:
            loss = loss_fn(outputs, masks)

        val_loss += loss.item()
        metrics = calculate_metrics(outputs if not isinstance(outputs, list) else outputs[-1], masks)
        for key in val_metrics:
            val_metrics[key] += metrics[key]

    val_loss /= len(loader)
    for key in val_metrics:
        val_metrics[key] /= len(loader)

    return val_loss, val_metrics

def main(args):
    set_seed(args.seed)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config_name = os.path.splitext(os.path.basename(args.config))[0]
    save_dir = os.path.join('results', config_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms(
        height=config['img_size'], width=config['img_size']
    )
    train_loader, val_loader = get_loaders(
        train_img_dir=config['train_img_dir'],
        train_mask_dir=config['train_mask_dir'],
        val_img_dir=config['val_img_dir'],
        val_mask_dir=config['val_mask_dir'],
        batch_size=config['batch_size'],
        train_transform=train_transform,
        val_transform=val_transform,
        num_workers=config['num_workers']
    )

    model_name = config['model_name']
    if model_name == 'transunet':
        model = TransUNet(
            img_dim=config['img_size'],
            num_channels=3,
            num_classes=1,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        )
    elif model_name == 'new_transunet':
        model = EnhancedTransUNet(
            img_dim=config['img_size'],
            num_channels=3,
            num_classes=1,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads']
        )
    elif model_name == 'unetpp':
        model = UNetPP(
            num_classes=config['num_classes'],
            deep_supervision=config.get('deep_supervision', False)
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    loss_fn = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_dice = 0.0
    train_losses, val_losses = [], []

    print(f"""
{'=' * 80}
Training {config_name} for {config['num_epochs']} epochs
{'=' * 80}
""")

    for epoch in range(1, config['num_epochs'] + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_loss, val_metrics = validate(
            model, val_loader, loss_fn, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch}/{config['num_epochs']} "
            f"➤ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} "
            f"| Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f}"
        )

        scheduler.step(val_loss)

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"► Saved best model to {best_path} (Dice: {best_dice:.4f})")

    final_path = os.path.join(save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"""
Training completed! Models saved to {save_dir}
Best: {best_path}
Final: {final_path}
""")

    np.save(os.path.join(save_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(save_dir, 'val_losses.npy'), np.array(val_losses))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    main(args)


