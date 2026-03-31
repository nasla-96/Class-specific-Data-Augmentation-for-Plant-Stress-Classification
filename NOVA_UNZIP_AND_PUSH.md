# Unzip and push changes on Nova

## 1) Copy the zip file to Nova
From your local machine:

```bash
scp Class-specific-Data-Augmentation-for-Plant-Stress-Classification.zip USERNAME@nova:/work/mech-ai-scratch/nasla/
```

## 2) SSH into Nova

```bash
ssh USERNAME@nova
```

## 3) Unzip the repo

```bash
cd /work/mech-ai-scratch/nasla
unzip Class-specific-Data-Augmentation-for-Plant-Stress-Classification.zip
cd Class-specific-Data-Augmentation-for-Plant-Stress-Classification
```

If you need to overwrite an older copy:

```bash
unzip -o Class-specific-Data-Augmentation-for-Plant-Stress-Classification.zip
```

## 4) Check the files

```bash
ls
find . -maxdepth 2 -type f | sort
```

## 5) Push to GitHub

If the repo already has a remote:

```bash
git remote -v
git status
git add .
git commit -m "Restructure repo and add README plus SLURM script"
git push origin main
```

If the remote is missing:

```bash
git init
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Class-specific-Data-Augmentation-for-Plant-Stress-Classification.git
git add .
git commit -m "Initial structured project layout"
git push -u origin main
```

## 6) If Git asks for authentication

Use a GitHub personal access token instead of your password, or set up SSH keys and use the SSH remote URL.
