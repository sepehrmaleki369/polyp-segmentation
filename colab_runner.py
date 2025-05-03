import os
import subprocess
import argparse
import glob
import time
import sys

def run_command(cmd, description=None):
    """Run a shell command and print output"""
    if description:
        print(f"""
{'=' * 80}
{description}
{'=' * 80}
""")
    print(f"""Running: {cmd}""")
    proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    return proc.returncode == 0

def extract_with_progress(zip_file, dest_dir):
    """Extract zip file with a simple progress indicator"""
    print(f"""Extracting {zip_file} to {dest_dir}...""")
    os.makedirs(dest_dir, exist_ok=True)
    proc = subprocess.Popen(f"unzip -q {zip_file} -d {dest_dir}", shell=True)
    while proc.poll() is None:
        sys.stdout.write('.')
        sys.stdout.flush()
        time.sleep(1)
    print("""
Extraction complete.""")
    return True

def download_dataset():
    """Download and extract the Kvasir-SEG dataset"""
    print("""Downloading Kvasir-SEG dataset...""")
    os.makedirs("datasets", exist_ok=True)
    run_command(
        "wget https://datasets.simula.no/downloads/kvasir-seg.zip -P datasets/",
        "Downloading Kvasir-SEG dataset"
    )
    extract_with_progress("datasets/kvasir-seg.zip", "datasets/")
    print("""Dataset downloaded and extracted successfully.""")

def prepare_dataset():
    """Prepare the dataset for training"""
    run_command(
        "python prepare_datasets.py --kvasir_dir datasets/Kvasir-SEG --dest_dir data",
        "Preparing dataset"
    )

def train_model(config_path, epochs=None):
    """Train using the given config file, optionally overriding epochs"""
    cfg = config_path
    if epochs is not None:
        with open(cfg, 'r') as f:
            txt = f.read()
        import re
        txt = re.sub(r"num_epochs: \d+", f"num_epochs: {epochs}", txt)
        tmp = cfg.replace('.yaml', '_temp.yaml')
        with open(tmp, 'w') as f:
            f.write(txt)
        cfg = tmp

    run_command(
        f"python train.py --config {cfg} --seed 42",
        f"Training with config {os.path.basename(config_path)}"
    )

    if epochs is not None and cfg.endswith('_temp.yaml'):
        os.remove(cfg)

def test_all_configs():
    """Test all trained models with their matching configs"""
    for cfg in sorted(glob.glob('configs/*.yaml')):
        if cfg.endswith('_temp.yaml') or 'msraformer' in cfg:
            continue
        exp = os.path.splitext(os.path.basename(cfg))[0]
        matches = sorted(glob.glob(f"results/{exp}_*"))
        if not matches:
            print(f"No trained models found for config {exp}, skipping.")
            continue
        model_dir = matches[-1]
        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Checkpoint not found at {model_path}, skipping.")
            continue
        run_command(
            f"python test.py --config configs/{exp}.yaml --model_path {model_path}",
            f"Testing {exp} model"
        )

def compare_models():
    """Compare every pair of configs via parameter_analysis.py"""
    all_dirs = sorted(glob.glob('results/*/test_results'))
    all_names = [os.path.basename(os.path.dirname(d)) for d in all_dirs]

    for i in range(len(all_dirs)):
        for j in range(i+1, len(all_dirs)):
            name1, name2 = all_names[i], all_names[j]
            dir1, dir2   = all_dirs[i], all_dirs[j]
            run_command(
                f"python parameter_analysis.py "
                f"--model1_dir {dir1} --model2_dir {dir2} "
                f"--model1_name {name1} --model2_name {name2} "
                f"--output_dir comparison_{name1}_vs_{name2}",
                f"""Comparing {name1} vs {name2}"""
            )

def main():
    parser = argparse.ArgumentParser(description="Run polyp segmentation experiments in Colab")
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--prepare',  action='store_true')
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--test',     action='store_true')
    parser.add_argument('--compare',  action='store_true')
    parser.add_argument('--epochs',   type=int, default=None)
    parser.add_argument('--all',      action='store_true')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('comparison_results', exist_ok=True)

    if args.all or args.download:
        download_dataset()
    if args.all or args.prepare:
        prepare_dataset()
    if args.all or args.train:
        for cfg in sorted(glob.glob('configs/*.yaml')):
            if cfg.endswith('_temp.yaml') or 'msraformer' in cfg:
                continue
            train_model(cfg, args.epochs)
    if args.all or args.test:
        test_all_configs()
    if args.all or args.compare:
        compare_models()

if __name__ == '__main__':
    main()



