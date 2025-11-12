# Seismic Impedance Inversion with 3D cGAN

This repo contains a clean reference implementation for **seismic impedance inversion** using a **3D conditional GAN** (U‑Net generator + PatchGAN discriminator). The model learns to map seismic amplitude volumes to acoustic impedance, using sparse well‑log impedance samples as supervision. It includes robust SEG‑Y I/O and sliding‑window inference for whole‑volume prediction.

## Features

* 3D U‑Net‑style **Generator** and 3D **PatchGAN** Discriminator
* **Masked L1** (only where well labels exist) + adversarial loss
* Sliding‑window inference with center‑crop blending
* Robust **SEG‑Y** read/write (depth domain assumed)
* Simple CLI & reproducible training

## File overview

* `seismic_inversion_gan.py` — training, inference, SEG‑Y export
* `requirements.txt` — Python package requirements

## Quickstart

### 1) Install

```bash
python -m venv .venv && source .venv/bin/activate  # or conda
pip install -r requirements.txt
```

### 2) Data

* **Training SEG‑Y** cube (depth domain). The sample axis must match the well depths.
* **Well CSV** with columns: `iline,xline,depth,impedance`

  * `iline`/`xline`: inline & crossline identifiers in the same grid as the SEG‑Y
  * `depth`: in the same units/scale as the SEG‑Y sample axis
  * `impedance`: acoustic impedance values

### 3) Train

```bash
python seismic_inversion_gan.py \
  --segy /path/to/train_cube.segy \
  --wells /path/to/well_impedance.csv \
  --out-model ./models/impedance_gen.keras \
  --epochs 50 --batch-size 2 --patch 128 128 128 --stride 64 64 64
```

### 4) Predict & export SEG‑Y

```bash
python seismic_inversion_gan.py \
  --segy /path/to/train_cube.segy \
  --wells /path/to/well_impedance.csv \
  --pred-segy /path/to/predict_cube.segy \
  --out-model ./models/impedance_gen.keras \
  --out-segy ./out/GAN_impedance.sgy \
  --epochs 50 --batch-size 2 --patch 128 128 128 --stride 64 64 64
```

> Tip: Use the same `--patch` and `--stride` at inference that you used in training.

## Notes

* Normalization: robust z‑score for seismic; log‑transform + standardize for impedance.
* Training uses label smoothing and small TV‑like regularization for stability.
* Mixed precision can speed up training on modern GPUs: `tf.keras.mixed_precision.set_global_policy("mixed_float16")`.

## Reproducibility

We set NumPy and TensorFlow seeds; full determinism may still vary across hardware/driver stacks.

## License

MIT (see `LICENSE`).

## Citation

If you find this useful, please cite the repo. A minimal BibTeX entry:

```bibtex
@software{seismic_inversion_gan,
  title        = {Seismic Impedance Inversion with 3D cGAN},
  author       = {Abreham Yacob Abreham},
  year         = {2025},
  url          = {https://github.com/aya505/}
}
```

