import os
import shutil
import glob
import random
from sklearn.model_selection import train_test_split
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

def create_directory_structure(base_dir):
    """Create the directory structure for datasets"""
    for dataset in ['kvasir_seg', 'cvc_clinicdb']:
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'masks']:
                os.makedirs(os.path.join(base_dir, dataset, split, subdir), exist_ok=True)

def prepare_kvasir_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Prepare Kvasir-SEG dataset"""
    print("Preparing Kvasir-SEG dataset...")

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(source_dir, 'images', '*.jpg')))

    # Split into train, validation, and test sets
    train_files, temp_files = train_test_split(image_files, test_size=(1-train_ratio), random_state=42)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=test_ratio/(test_ratio+val_ratio),
        random_state=42
    )

    # Function to copy files
    def copy_files(files, split):
        for img_path in tqdm(files, desc=f"Copying {split} files"):
            img_filename = os.path.basename(img_path)
            mask_filename = img_filename

            mask_path = os.path.join(source_dir, 'masks', mask_filename)

            # Copy image
            shutil.copy(
                img_path,
                os.path.join(dest_dir, 'kvasir_seg', split, 'images', img_filename)
            )

            # Copy mask
            shutil.copy(
                mask_path,
                os.path.join(dest_dir, 'kvasir_seg', split, 'masks', mask_filename)
            )

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"Kvasir-SEG dataset prepared: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def prepare_cvc_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Prepare CVC-ClinicDB dataset"""
    print("Preparing CVC-ClinicDB dataset...")

    # Get all image files
    image_files = sorted(glob.glob(os.path.join(source_dir, 'Original', '*.tif')))

    # Split into train, validation, and test sets
    train_files, temp_files = train_test_split(image_files, test_size=(1-train_ratio), random_state=42)
    val_files, test_files = train_test_split(
        temp_files,
        test_size=test_ratio/(test_ratio+val_ratio),
        random_state=42
    )

    # Function to copy files
    def copy_files(files, split):
        for img_path in tqdm(files, desc=f"Copying {split} files"):
            img_filename = os.path.basename(img_path)
            mask_filename = img_filename  # Same name but in different directory

            mask_path = os.path.join(source_dir, 'Ground Truth', mask_filename)

            # Convert to jpg and copy
            # Image
            img = Image.open(img_path)
            jpg_img_filename = img_filename.replace('.tif', '.jpg')
            img.save(os.path.join(dest_dir, 'cvc_clinicdb', split, 'images', jpg_img_filename))

            # Mask
            mask = Image.open(mask_path)
            jpg_mask_filename = mask_filename.replace('.tif', '.jpg')
            mask.save(os.path.join(dest_dir, 'cvc_clinicdb', split, 'masks', jpg_mask_filename))

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')

    print(f"CVC-ClinicDB dataset prepared: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

def main(args):
    # Create directory structure
    create_directory_structure(args.dest_dir)

    # Prepare datasets
    if args.kvasir_dir:
        prepare_kvasir_dataset(
            args.kvasir_dir,
            args.dest_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )

    if args.cvc_dir:
        prepare_cvc_dataset(
            args.cvc_dir,
            args.dest_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare polyp segmentation datasets")
    parser.add_argument("--kvasir_dir", type=str, default=None, help="Path to Kvasir-SEG dataset")
    parser.add_argument("--cvc_dir", type=str, default=None, help="Path to CVC-ClinicDB dataset")
    parser.add_argument("--dest_dir", type=str, default="data", help="Destination directory")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio")
    args = parser.parse_args()

    main(args)
