# PepDiffusion — Custom AMP Fine-tuning

Adaptation of PepDiffusion (Wang et al., Science Advances 2025) for custom AMP dataset.

## Setup
```bash
conda env create -f PepDiffusion.yaml
conda activate pepdiff_env
```

## Pipeline (run in order)
```bash
# Step 1: Encode your dataset to latent vectors
python main.py --work GetMem_nc --vae_model_path ./data/model_299_0.10607143263973876_0.0909070645059858_.pth

# Step 2: Combine latent files
python combine_mem.py

# Step 3a: Fine-tune from pretrained diffusion weights
python main.py --work LatentDiffusion_nocondition --LatentDiffusion_batch_size 64

# Step 3b: OR retrain from scratch (remove --pretrained flag)

# Step 4: Generate sequences
python main.py --work Generate \
  --Generate_VAE_model_path ./data/model_299_0.10607143263973876_0.0909070645059858_.pth
```

## Dataset
Place your dataset CSV in `./dataset/` with column `SEQUENCE`.

## Pretrained Weights
- VAE weights included: `data/model_299_..._.pth` (frozen, do not retrain)
- Diffusion weights: download from https://zenodo.org/records/13762213
  or train from scratch using pipeline above


## Option B — Retrain from scratch: Skip loading pretrained weights
- Find line 583 in main.py and comment it out:
    model.load_state_dict(torch.load('./data/best_model.pth'))

## Original Paper
Wang et al., "Artificial intelligence using a latent diffusion model enables
the generation of diverse and potent antimicrobial peptides"
Science Advances 11, eadp7171 (2025)
https://github.com/Wangyj2023/PepDiffusion