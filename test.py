import os
import glob
import torch
import argparse
import yaml
import numpy as np

from models.transunet import TransUNet
from models.new_transunet import EnhancedTransUNet
from models.unetpp import UNetPP

from utils.data_utils import get_loaders, get_transforms
from utils.metrics import calculate_metrics
from utils.visualization import visualize_batch, create_error_map

def test_model_from_config(config_path, model_path=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    results_dir = os.path.join('results', config_name)

    if model_path is None:
        model_path = os.path.join(results_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"""
Checkpoint not found for {config_name} at {model_path}, skipping.
""")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"""
Using device: {device}""")
    print(f"""
Testing {config_name}: model {model_path}""")

    arch = config['model_name']
    if arch == 'transunet':
        model = TransUNet(
            img_dim=config['img_size'], num_channels=3, num_classes=1,
            embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads']
        )
    elif arch == 'new_transunet':
        model = EnhancedTransUNet(
            img_dim=config['img_size'], num_channels=3, num_classes=1,
            embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads']
        )
    elif arch == 'unetpp':
        model = UNetPP(
            num_classes=config['num_classes'],
            deep_supervision=config.get('deep_supervision', False)
        )
    else:
        raise ValueError(f"Unknown model architecture: {arch}")

    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    _, val_transform = get_transforms(height=config['img_size'], width=config['img_size'])
    test_loader = get_loaders(
        train_img_dir=config['test_img_dir'],
        train_mask_dir=config['test_mask_dir'],
        val_img_dir=config['test_img_dir'],
        val_mask_dir=config['test_mask_dir'],
        batch_size=config['batch_size'],
        train_transform=val_transform,
        val_transform=val_transform,
        num_workers=config['num_workers']
    )[1]

    all_metrics = {k: [] for k in ['dice', 'iou', 'precision', 'recall', 'f1']}
    all_images, all_masks, all_preds = [], [], []
    with torch.no_grad():
        num = len(test_loader.dataset)
        print(f"""
Testing on {num} images...""")
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            batch_metrics = calculate_metrics(outputs, masks)

            for k in all_metrics:
                all_metrics[k].extend([batch_metrics[k]] * images.size(0))
            all_images.extend(images.cpu())
            all_masks.extend(masks.cpu())
            all_preds.extend(outputs.cpu())

    for k in all_metrics:
        all_metrics[k] = np.array(all_metrics[k])
    avg = {k: float(all_metrics[k].mean()) for k in all_metrics}
    std = {k: float(all_metrics[k].std()) for k in all_metrics}

    print("""
Test Results:""")
    for k in ['dice', 'iou', 'precision', 'recall', 'f1']:
        print(f"{k.capitalize():9s}: {avg[k]:.4f} ± {std[k]:.4f}")

    out_dir = os.path.join(results_dir, 'test_results')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'test_metrics.npy'), all_metrics)
    with open(os.path.join(out_dir, 'test_config.yaml'), 'w') as f:
        yaml.dump({'config': config, 'avg_metrics': avg}, f)

    viz_dir = os.path.join(out_dir, 'visualization')
    os.makedirs(viz_dir, exist_ok=True)
    visualize_batch(all_images[:10], all_masks[:10], all_preds[:10],
                    save_path=os.path.join(viz_dir, 'predictions.png'))
    for i in range(min(5, len(all_images))):
        create_error_map(all_preds[i], all_masks[i],
                         save_path=os.path.join(viz_dir, f'error_map_{i}.png'))

def main():
    parser = argparse.ArgumentParser(description="Test one or all polyp segmentation configs")
    parser.add_argument('--config', type=str, help="Path to config file (or 'all' for every YAML)")
    parser.add_argument('--model_path', type=str, default=None, help="Optional path to model checkpoint")
    args = parser.parse_args()

    if args.config == 'all':
        for cfg in sorted(glob.glob('configs/*.yaml')):
            if cfg.endswith('_temp.yaml'):
                continue
            test_model_from_config(cfg)
    else:
        if not args.config or not os.path.exists(args.config):
            raise ValueError("Please provide a valid --config path or 'all'.")
        test_model_from_config(args.config, args.model_path)

if __name__ == '__main__':
    main()

