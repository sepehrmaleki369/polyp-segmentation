# Polyp Segmentation Project

This project implements deep learning models for polyp segmentation in medical images. It includes implementations of various architectures including U-Net++, TransUNet, and an enhanced version of TransUNet.

## Project Structure

```
polyp-segmentation/
├── configs/              # Configuration files for different models
├── data/                 # Data directory
├── datasets/            # Dataset processing
├── models/              # Model architectures
│   ├── unetpp.py       # U-Net++ implementation
│   ├── transunet.py    # TransUNet implementation
│   └── new_transunet.py # Enhanced TransUNet implementation
├── results/             # Training results and model checkpoints
├── utils/              # Utility functions
├── colab_runner.py     # Google Colab execution script
├── parameter_analysis.py # Parameter analysis utilities
├── prepare_datasets.py # Dataset preparation script
├── test.py            # Testing script
└── train.py           # Training script
```

## Models

The project includes implementations of the following architectures:

1. **U-Net++**: A nested U-Net architecture with dense skip connections
2. **TransUNet**: A hybrid architecture combining Transformer and U-Net
3. **Enhanced TransUNet**: An improved version of TransUNet with additional features

## Requirements

The project requires the following dependencies:
- Python 3.x
- PyTorch
- torchvision
- numpy
- scikit-learn
- matplotlib
- tqdm

## Usage

### Dataset Preparation

```bash
python prepare_datasets.py
```

### Training

To train a model, use the `train.py` script with appropriate configuration:

```bash
python train.py --config configs/transunet_config.yaml
```

Available configurations:
- `transunet_config.yaml`
- `transunet_embed576_config.yaml`
- `transunet_lr5e-5_config.yaml`
- `unetpp_config.yaml`
- `unetpp_deepsup_config.yaml`
- `unetpp_lr5e-5_config.yaml`

### Testing

To evaluate a trained model:

```bash
python test.py --config configs/transunet_config.yaml
```

### Parameter Analysis

For analyzing model parameters:

```bash
python parameter_analysis.py
```

### Google Colab Support

The project includes a `colab_runner.py` script for running experiments on Google Colab.

## Results

Training results and model checkpoints are stored in the `results/` directory. Comparison results between different models are stored in `comparison_results/`.

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add your citation here]
```

## Contact

[Add your contact information here] 