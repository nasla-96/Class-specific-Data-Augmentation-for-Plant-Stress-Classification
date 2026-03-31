# Class-specific Data Augmentation for Plant Stress Classification

Code for class-specific data augmentation for plant stress image classification.

This project searches for a different augmentation policy for each class instead of applying one shared augmentation pipeline to the whole dataset. A genetic algorithm optimizes class-wise augmentation probabilities, and a ResNet-50 classifier is used to evaluate each policy.

## Repository structure

```text
Class-specific-Data-Augmentation-for-Plant-Stress-Classification/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default_config.yaml
├── scripts/
│   ├── run_ga_augmentation.py
│   └── run_ga_augmentation.slurm
└── src/
    └── plant_stress_aug/
        ├── __init__.py
        ├── augmentations.py
        ├── dataset.py
        ├── model.py
        └── train_eval.py
```

## Method overview

The workflow is:

1. Load plant stress images using an `ImageFolder`-style dataset.
2. Represent augmentation probabilities as a chromosome of shape `classes × augmentations`.
3. Apply class-specific augmentations during training.
4. Train and validate a ResNet-50 model for each candidate chromosome.
5. Use the genetic algorithm to maximize mean per-class accuracy.

## Expected data layout

```text
data/
├── train/
│   ├── class_0/
│   ├── class_1/
│   └── ...
└── val/
    ├── class_0/
    ├── class_1/
    └── ...
```

Update the class folder names and paths as needed in `configs/default_config.yaml`.

## Environment setup

Create the environment and install dependencies:

```bash
conda create -n plant_aug python=3.10 -y
conda activate plant_aug
pip install -r requirements.txt
```

If you already have a working CUDA/PyTorch environment on Nova, keep that and install only any missing packages.

## Configuration

Main settings are stored in:

```bash
configs/default_config.yaml
```

Edit at least these fields before running:

- `data.train_dir`
- `data.val_dir`
- `model.num_classes`
- `model.checkpoint_path`

You can also adjust batch size, number of workers, GA population size, and number of generations.

## Run locally

```bash
python scripts/run_ga_augmentation.py --config configs/default_config.yaml --output_dir outputs
```

Outputs such as fitness plots, saved GA state, best solutions, and metrics will be written to `outputs/`.

## Run on Nova with 4 GPUs

A SLURM script is included:

```bash
scripts/run_ga_augmentation.slurm
```

Submit with:

```bash
sbatch scripts/run_ga_augmentation.slurm
```

Before submitting, update the following in the SLURM script:

- account name
- partition name
- conda environment name
- repo path on Nova
- dataset paths in the config file

## Notes

A few practical cleanup steps are still worth doing before making this fully public:

- replace placeholder dataset paths with portable relative paths
- confirm the checkpoint loading path
- save outputs into run-specific folders
- add one small example dataset or example command with expected output
- include a short description of the classes in the dataset

## Example Nova workflow

```bash
# on Nova
cd /work/mech-ai-scratch/nasla
unzip Class-specific-Data-Augmentation-for-Plant-Stress-Classification.zip
cd Class-specific-Data-Augmentation-for-Plant-Stress-Classification

conda activate plant_aug
pip install -r requirements.txt

# edit config if needed
nano configs/default_config.yaml

# run interactively
python scripts/run_ga_augmentation.py --config configs/default_config.yaml --output_dir outputs

# or submit to SLURM
sbatch scripts/run_ga_augmentation.slurm
```

## Git workflow to push changes from Nova

If this repo is already connected to GitHub:

```bash
cd /work/mech-ai-scratch/nasla/Class-specific-Data-Augmentation-for-Plant-Stress-Classification

git status
git add .
git commit -m "Restructure repo, add README, config, and SLURM script"
git push origin main
```

If the remote is not set yet:

```bash
git init
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Class-specific-Data-Augmentation-for-Plant-Stress-Classification.git
git add .
git commit -m "Initial structured project layout"
git push -u origin main
```
