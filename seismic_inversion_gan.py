#!/usr/bin/env python3
"""
Seismic Impedance Inversion with a 3D cGAN (U-Net Generator + PatchGAN Discriminator)
-------------------------------------------------------------------------------

This script trains a conditional GAN to predict acoustic impedance volumes from
3D seismic amplitude cubes, using sparse well-log impedance samples as labels.
It supports sliding-window inference and SEG-Y export of the predicted cube.

Key features
- 3D U-Net-style generator, 3D PatchGAN discriminator
- Masked L1 loss (only where well labels exist) + adversarial loss
- Sliding-window prediction with center-crop blending
- Robust SEG-Y I/O with segyio (depth domain assumed)
- CLI with sensible defaults; reproducible training

Usage (example)
---------------
python seismic_inversion_gan.py \
  --segy "/path/to/train_cube.segy" \
  --wells "/path/to/well_impedance.csv" \
  --pred-segy "/path/to/predict_cube.segy" \
  --out-model "/path/to/impedance_gen.keras" \
  --out-segy "/path/to/GANpredicted_impedance.sgy" \
  --epochs 50 --batch-size 2 --patch 128 128 128 --stride 64 64 64

Well CSV requirements
---------------------
CSV must contain at least: `iline`, `xline`, `depth`, `impedance`
(Depth units must match SEG-Y sample axis; see `samples` in the template SEG-Y.)

Notes
-----
- Memory: patch extraction is generator-based to avoid loading all patches at once.
- Normalization: robust z-score for seismic; standardization for log-impedance.
- For production, consider mixed precision (tf.keras.mixed_precision.set_global_policy).

Author: (your name)
License: MIT (or your chosen license)
"""
from __future__ import annotations

import os
import sys
import argparse
import itertools
import math
import numpy as np
import pandas as pd
import segyio
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

# -------------------------
# Reproducibility & logging
# -------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
np.random.seed(42)
tf.random.set_seed(42)

# -------------------------
# Model building blocks
# -------------------------
try:
    from tensorflow_addons.layers import InstanceNormalization as InstNorm
except Exception:  # fallback if tfa not available
    InstNorm = None


def _norm():
    return InstNorm(axis=-1, epsilon=1e-5) if InstNorm else layers.BatchNormalization(momentum=0.9, epsilon=1e-5)


def conv3(x, f, k=3, s=1, act=None, use_norm=True):
    x = layers.Conv3D(f, k, strides=s, padding="same", use_bias=False, kernel_initializer="he_normal")(x)
    if use_norm:
        x = _norm()(x)
    if act == "lrelu":
        x = layers.LeakyReLU(0.2)(x)
    elif act == "relu":
        x = layers.Activation("relu")(x)
    return x


def down(x, f, s=2, use_norm=True):  # strided conv downsample
    return conv3(x, f, k=3, s=s, act="lrelu", use_norm=use_norm)


def up(x, skip, f, dropout=False):  # resize-conv upsample
    x = layers.UpSampling3D(size=(2, 2, 2))(x)
    x = conv3(x, f, k=3, s=1, act="relu", use_norm=True)
    if dropout:
        x = layers.Dropout(0.5)(x)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = conv3(x, f, k=3, s=1, act="relu", use_norm=True)
    return x


def build_generator_3d(input_shape=(128, 128, 128, 1), base_filters=32) -> Model:
    """3D U-Net-like generator with linear head for regression."""
    assert all(d % 128 == 0 for d in input_shape[:3]), "Use dims divisible by 128 for default topology."
    inp = layers.Input(shape=input_shape)

    # Encoder 128→64→32→16→8→4→2→1
    d1 = down(inp, base_filters, s=2, use_norm=False)     # 128→64
    d2 = down(d1, base_filters * 2, s=2)                  # 64→32
    d3 = down(d2, base_filters * 4, s=2)                  # 32→16
    d4 = down(d3, base_filters * 8, s=2)                  # 16→8
    d5 = down(d4, base_filters * 16, s=2)                 # 8→4
    d6 = down(d5, base_filters * 16, s=2)                 # 4→2
    b  = down(d6, base_filters * 32, s=2)                 # 2→1 bottleneck

    # Decoder 1→2→4→8→16→32→64→128
    u1 = up(b,  d6, base_filters * 16, dropout=True)
    u2 = up(u1, d5, base_filters * 16, dropout=True)
    u3 = up(u2, d4, base_filters * 8)
    u4 = up(u3, d3, base_filters * 4)
    u5 = up(u4, d2, base_filters * 2)
    u6 = up(u5, d1, base_filters)

    # Final up (no skip): 64→128
    x  = layers.Conv3DTranspose(base_filters, 4, strides=2, padding="same", use_bias=False)(u6)
    x  = layers.BatchNormalization()(x)
    x  = layers.ReLU()(x)
    x  = conv3(x, base_filters, k=3, s=1, act="relu", use_norm=True)

    out = layers.Conv3D(1, 1, padding="same", activation=None)(x)
    return Model(inp, out, name="Generator3D_128_linear")


def build_discriminator_3d(input_shape, base_filters=32) -> Model:
    """3D PatchGAN discriminator. Outputs logits (use BCE from_logits=True)."""
    cond_inp = layers.Input(shape=input_shape)
    tgt_inp  = layers.Input(shape=input_shape)

    x = layers.Concatenate(axis=-1)([cond_inp, tgt_inp])  # 2 channels
    x = down(x, base_filters,   s=2, use_norm=False)  # 32→16
    x = down(x, base_filters*2, s=2, use_norm=True)   # 16→8
    x = down(x, base_filters*4, s=2, use_norm=True)   # 8→4
    x = down(x, base_filters*8, s=2, use_norm=True)   # 4→2

    x = layers.Conv3D(1, kernel_size=3, strides=1, padding="same")(x)  # logits
    return Model([cond_inp, tgt_inp], x, name="Discriminator3D")


# -------------------------
# Losses & training module
# -------------------------
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def masked_l1(y_true, y_pred, mask, eps=1e-6):
    mask = tf.cast(mask, y_pred.dtype)
    l1 = tf.reduce_sum(tf.abs((y_true - y_pred) * mask))
    denom = tf.reduce_sum(mask) + eps
    return l1 / denom


class ImpedanceCGAN(Model):
    def __init__(self, gen, disc, lambda_l1=100.0, lr_g=2e-4, lr_d=2e-4):
        super().__init__()
        self.gen = gen
        self.disc = disc
        self.lambda_l1 = lambda_l1
        self.g_opt = Adam(lr_g, beta_1=0.5, beta_2=0.999)
        self.d_opt = Adam(lr_d, beta_1=0.5, beta_2=0.999)
        self.train_metrics = {
            "g_total": tf.keras.metrics.Mean(name="g_total"),
            "g_adv":   tf.keras.metrics.Mean(name="g_adv"),
            "g_l1":    tf.keras.metrics.Mean(name="g_l1"),
            "d_total": tf.keras.metrics.Mean(name="d_total"),
        }

    @tf.function
    def train_step(self, batch):
        seis, imp_true, mask = batch  # [B,D,H,W,1]
        mask = tf.cast(mask, seis.dtype)
        bg   = 1.0 - mask

        real_tgt = 0.9  # label smoothing helps stability
        fake_tgt = 0.0

        with tf.GradientTape(persistent=True) as tape:
            imp_pred = self.gen(seis, training=True)

            # Compose real target to keep background identical on both paths
            y_comp = mask * imp_true + bg * imp_pred
            y_comp = tf.stop_gradient(y_comp)

            d_real = self.disc([seis, y_comp],   training=True)
            d_fake = self.disc([seis, imp_pred], training=True)

            d_loss_real = bce(tf.ones_like(d_real) * real_tgt, d_real)
            d_loss_fake = bce(tf.ones_like(d_fake) * fake_tgt, d_fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

            g_adv = bce(tf.ones_like(d_fake) * real_tgt, d_fake)
            g_l1  = masked_l1(imp_true, imp_pred, mask)

            # small regularizers on background for stability
            g_bg = tf.reduce_mean(tf.square(imp_pred * bg))
            dz = tf.abs((imp_pred * bg)[:, 1:, :, :, :] - (imp_pred * bg)[:, :-1, :, :, :])
            dy = tf.abs((imp_pred * bg)[:, :, 1:, :, :] - (imp_pred * bg)[:, :, :-1, :, :])
            dx = tf.abs((imp_pred * bg)[:, :, :, 1:, :] - (imp_pred * bg)[:, :, :, :-1, :])
            g_tv = 0.1 * tf.reduce_mean(dz) + tf.reduce_mean(dy) + tf.reduce_mean(dx)

            g_total = g_adv + self.lambda_l1 * g_l1 + 0.05 * g_bg + 1e-3 * g_tv

        g_grads = tape.gradient(g_total, self.gen.trainable_variables)
        d_grads = tape.gradient(d_loss,  self.disc.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grads, self.gen.trainable_variables))
        self.d_opt.apply_gradients(zip(d_grads,  self.disc.trainable_variables))
        del tape

        self.train_metrics["g_total"].update_state(g_total)
        self.train_metrics["g_adv"].update_state(g_adv)
        self.train_metrics["g_l1"].update_state(g_l1)
        self.train_metrics["d_total"].update_state(d_loss)
        return {k: m.result() for k, m in self.train_metrics.items()}

    def reset_metrics(self):
        for m in self.train_metrics.values():
            m.reset_state()


# -------------------------
# Data utilities
# -------------------------

def normalize_seismic(x: np.ndarray) -> np.ndarray:
    """Robust z-score normalization per volume."""
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def build_sparse_mask_from_csv(well_df: pd.DataFrame,
                               ilines: np.ndarray,
                               xlines: np.ndarray,
                               samples: np.ndarray,
                               volume_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Map well points (iline, xline, depth) to voxel indices and build sparse label & mask.

    Returns:
        imp_vol_raw: (D,H,W) float32 with zeros where unknown and impedance where provided
        mask_vol:    (D,H,W) float32 with 1 at labeled voxels, else 0
    """
    D, H, W = volume_shape
    il_lookup = {v: i for i, v in enumerate(ilines)}  # inline index along W
    xl_lookup = {v: i for i, v in enumerate(xlines)}  # xline index along H

    dz = float(np.mean(np.diff(samples)))  # sample step (depth)
    z0 = float(samples[0])

    imp_vol_raw = np.zeros((D, H, W), np.float32)
    mask_vol    = np.zeros((D, H, W), np.float32)

    required = {"iline", "xline", "depth", "impedance"}
    if not required.issubset(well_df.columns):
        missing = ", ".join(sorted(required - set(well_df.columns)))
        raise ValueError(f"Well CSV is missing required columns: {missing}")

    for _, r in well_df.iterrows():
        il = int(r["iline"])
        xl = int(r["xline"])
        z  = float(r["depth"])  # depth domain
        if il in il_lookup and xl in xl_lookup:
            k = il_lookup[il]  # W axis
            j = xl_lookup[xl]  # H axis
            i = int(round((z - z0) / dz))  # D axis
            if 0 <= i < D:
                imp_vol_raw[i, j, k] = float(r["impedance"])
                mask_vol[i, j, k] = 1.0
    return imp_vol_raw, mask_vol


def extract_patches_3d(arr_shape, patch_size, stride):
    D, H, W = arr_shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride
    for z in range(0, max(D - pD + 1, 1), sD):
        for y in range(0, max(H - pH + 1, 1), sH):
            for x in range(0, max(W - pW + 1, 1), sW):
                if z + pD <= D and y + pH <= H and x + pW <= W:
                    yield z, y, x


def make_training_patches(seis_vol, imp_vol, mask_vol, patch_size, stride, min_labeled_voxels=32):
    for z, y, x in extract_patches_3d(seis_vol.shape, patch_size, stride):
        z2, y2, x2 = z + patch_size[0], y + patch_size[1], x + patch_size[2]
        sp = seis_vol[z:z2, y:y2, x:x2]
        ip = imp_vol[z:z2, y:y2, x:x2]
        mp = mask_vol[z:z2, y:y2, x:x2]
        if np.sum(mp) >= min_labeled_voxels:
            yield sp[..., None], ip[..., None], mp[..., None]  # add channel


def tf_dataset_from_patches(patches_iter, batch_size=2, shuffle=True, buffer=256):
    # Consume generator into lists (moderate size). For very large datasets,
    # consider writing to TFRecords or a streaming generator of tf.Tensors.
    seis_list, imp_list, mask_list = [], [], []
    for sp, ip, mp in patches_iter:
        seis_list.append(sp)
        imp_list.append(ip)
        mask_list.append(mp)
    seis = np.stack(seis_list).astype(np.float32)
    imp  = np.stack(imp_list).astype(np.float32)
    mask = np.stack(mask_list).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((seis, imp, mask))
    if shuffle:
        ds = ds.shuffle(min(len(seis_list), buffer), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -------------------------
# Sliding-window inference
# -------------------------

def _tile_starts(L, P, S):
    starts = list(range(0, max(L - P, 0) + 1, S))
    if not starts or starts[-1] != L - P:
        starts.append(L - P)
    return starts


def sliding_window_predict_centercrop(vol, model, patch_size=(128, 128, 128), stride=(64, 64, 64), crop=4):
    D, H, W = vol.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    z1, y1, x1 = crop, crop, crop
    z2, y2, x2 = pd - crop, ph - crop, pw - crop
    cz, cy, cx = z2 - z1, y2 - y1, x2 - x1

    out = np.zeros((D, H, W), np.float32)
    wgt = np.zeros((D, H, W), np.float32)

    for z in _tile_starts(D, pd, sd):
        for y in _tile_starts(H, ph, sh):
            for x in _tile_starts(W, pw, sw):
                sp = vol[z:z + pd, y:y + ph, x:x + pw][..., None].astype(np.float32)
                pred = model.predict(sp[None, ...], verbose=0)[0, ..., 0]
                oz1, oy1, ox1 = z + z1, y + y1, x + x1
                oz2, oy2, ox2 = oz1 + cz, oy1 + cy, ox1 + cx
                out[oz1:oz2, oy1:oy2, ox1:ox2] += pred[z1:z2, y1:y2, x1:x2]
                wgt[oz1:oz2, oy1:oy2, ox1:ox2] += 1.0
    return out / np.maximum(wgt, 1e-6)


# -------------------------
# SEG-Y Helpers
# -------------------------

def read_segy_cube(path: str):
    with segyio.open(path, mode='r', ignore_geometry=False) as f:
        f.mmap()
        cube = segyio.tools.cube(f)  # (n_ilines, n_xlines, n_samples)
        ilines = np.array(f.ilines)
        xlines = np.array(f.xlines)
        samples = np.array(f.samples)
    # reorder to (D,H,W) = (samples, xlines, inlines)
    vol = np.transpose(cube, (2, 1, 0)).astype(np.float32)
    return vol, ilines, xlines, samples


def export_impedance_to_segy(imp_pred, template_segy_path, out_segy_path, assume_h_is_inline=True):
    if imp_pred.ndim != 3:
        raise ValueError("imp_pred must be (D, H, W).")
    D, H, W = imp_pred.shape

    cube = (np.transpose(imp_pred, (2, 1, 0)) if assume_h_is_inline
            else np.transpose(imp_pred, (1, 2, 0)))
    cube = np.ascontiguousarray(cube, dtype=np.float32)

    with segyio.open(template_segy_path, "r", ignore_geometry=False) as t:
        ilines = list(t.ilines)
        xlines = list(t.xlines)
        n_il, n_xl = len(ilines), len(xlines)

        try:
            ns = int(t.bin[segyio.BinField.Samples])
        except Exception:
            ns = int(segyio.tools.ns(t))
        try:
            dt = int(t.bin[segyio.BinField.Interval])
        except Exception:
            dt = int(segyio.tools.dt(t))
        try:
            sample_format = int(t.bin[segyio.BinField.Format])
        except Exception:
            sample_format = 5  # IEEE float

        sample_axis = np.asarray(t.samples, dtype=np.float32)
        if cube.shape != (n_il, n_xl, ns):
            raise ValueError(f"Shape mismatch: cube {cube.shape} vs template (ilines={n_il}, xlines={n_xl}, samples={ns}).")

        il2i = {il: i for i, il in enumerate(ilines)}
        xl2i = {xl: i for i, xl in enumerate(xlines)}

        spec = segyio.spec()
        try:
            spec.sorting = t.sorting
        except Exception:
            pass
        spec.format = sample_format
        spec.samples = sample_axis
        spec.ilines = ilines
        spec.xlines = xlines

        with segyio.create(out_segy_path, spec) as f:
            try:
                f.text[0] = t.text[0]
            except Exception:
                pass

            # copy a few common bin fields when present
            for name in ("Samples", "Interval", "Format", "Traces", "MeasurementSystem", "RevMajor", "RevMinor"):
                key = getattr(segyio.BinField, name, None)
                if key is not None:
                    try:
                        f.bin[key] = int(t.bin[key])
                    except Exception:
                        pass

            tf = segyio.TraceField
            has_cdp_x = has_cdp_y = True
            try:
                _ = t.attributes(tf.CDP_X)[0]
                _ = t.attributes(tf.CDP_Y)[0]
            except Exception:
                has_cdp_x = has_cdp_y = False

            il_attr = np.asarray(t.attributes(tf.INLINE_3D))
            xl_attr = np.asarray(t.attributes(tf.CROSSLINE_3D))
            if has_cdp_x:
                x_attr = np.asarray(t.attributes(tf.CDP_X))
            if has_cdp_y:
                y_attr = np.asarray(t.attributes(tf.CDP_Y))

            ntr = len(t.trace)
            if len(il_attr) != ntr or len(xl_attr) != ntr:
                raise ValueError(f"Template attributes length mismatch: ntr={ntr}, len(il)={len(il_attr)}, len(xl)={len(xl_attr)}")

            for itr in range(ntr):
                il = int(il_attr[itr])
                xl = int(xl_attr[itr])

                if (il in il2i) and (xl in xl2i):
                    f.trace[itr] = cube[il2i[il], xl2i[xl], :]
                else:
                    f.trace[itr] = np.zeros(ns, dtype=np.float32)

                f.header[itr][tf.INLINE_3D] = il
                f.header[itr][tf.CROSSLINE_3D] = xl
                if has_cdp_x:
                    f.header[itr][tf.CDP_X] = int(x_attr[itr])
                if has_cdp_y:
                    f.header[itr][tf.CDP_Y] = int(y_attr[itr])
            f.flush()


# -------------------------
# Main training / inference
# -------------------------

def main(args: argparse.Namespace) -> int:
    # Load seismic training cube
    seis_vol, ilines, xlines, samples = read_segy_cube(args.segy)
    print("seis_vol shape (D,H,W):", seis_vol.shape)

    # Normalize seismic robustly
    seis_flat = seis_vol.reshape(-1, 1)
    seis_norm = normalize_seismic(seis_flat).astype(np.float32)
    # standard scaler for stability
    from sklearn.preprocessing import StandardScaler
    scaler_x = StandardScaler()
    scaler_x.fit(seis_norm)
    seis_scaled = scaler_x.transform(seis_norm).reshape(seis_vol.shape).astype(np.float32)

    # Build sparse impedance label & mask from wells
    wells = pd.read_csv(args.wells)
    imp_vol_raw, mask_vol = build_sparse_mask_from_csv(wells, ilines, xlines, samples, seis_scaled.shape)

    # Scale labels in log-domain
    imp_vol_log = np.log10(np.clip(imp_vol_raw, 1e-6, None)).astype(np.float32)
    from sklearn.preprocessing import StandardScaler as _Std
    scaler_y = _Std()
    scaler_y.fit(imp_vol_log.reshape(-1, 1))
    imp_vol = scaler_y.transform(imp_vol_log.reshape(-1, 1)).reshape(imp_vol_raw.shape).astype(np.float32)

    # Create patches & dataset
    patch = tuple(args.patch)
    stride = tuple(args.stride)
    train_patches = list(make_training_patches(seis_scaled, imp_vol, mask_vol, patch_size=patch, stride=stride, min_labeled_voxels=args.min_label_vox))
    print("train_patches:", len(train_patches))
    if len(train_patches) == 0:
        print("No training patches found with labels. Check well CSV and thresholds.")
        return 1
    ds = tf_dataset_from_patches(train_patches, batch_size=args.batch_size)

    # Build models
    gen = build_generator_3d(input_shape=patch + (1,), base_filters=args.base_filters)
    disc = build_discriminator_3d(input_shape=patch + (1,), base_filters=args.base_filters)
    gan = ImpedanceCGAN(gen, disc, lambda_l1=args.lambda_l1, lr_g=args.lr_g, lr_d=args.lr_d)

    # Train
    for epoch in range(1, args.epochs + 1):
        gan.reset_metrics()
        for batch in ds:
            logs = gan.train_step(batch)
        print(f"Epoch {epoch:03d} | G_total={logs['g_total']:.4f}  G_adv={logs['g_adv']:.4f}  G_L1={logs['g_l1']:.4f}  D_total={logs['d_total']:.4f}")

    # Save generator
    if args.out_model:
        os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
        gen.save(args.out_model)
        print("Saved generator to:", args.out_model)

    # Inference on prediction cube
    if args.pred_segy and args.out_segy:
        vol_pred, _, _, _ = read_segy_cube(args.pred_segy)
        vol_pred_flat = vol_pred.reshape(-1, 1).astype(np.float32)
        vol_pred_norm = normalize_seismic(vol_pred_flat)
        vol_pred_scaled = scaler_x.transform(vol_pred_norm).reshape(vol_pred.shape).astype(np.float32)

        imp_pred_scaled = sliding_window_predict_centercrop(vol_pred_scaled, gen, patch_size=patch, stride=stride, crop=args.crop)
        imp_pred_log = scaler_y.inverse_transform(imp_pred_scaled.reshape(-1, 1)).ravel()
        imp_pred = np.power(10.0, imp_pred_log).reshape(vol_pred.shape).astype(np.float32)

        os.makedirs(os.path.dirname(args.out_segy), exist_ok=True)
        export_impedance_to_segy(imp_pred, args.pred_segy, args.out_segy, assume_h_is_inline=True)
        print("Exported predicted impedance SEG-Y to:", args.out_segy)

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D cGAN for seismic impedance inversion with SEG-Y IO.")
    p.add_argument("--segy", required=True, help="Training SEG-Y (depth domain)")
    p.add_argument("--wells", required=True, help="CSV with columns: iline,xline,depth,impedance")
    p.add_argument("--pred-segy", default=None, help="SEG-Y to predict (if omitted, only training runs)")
    p.add_argument("--out-model", default=None, help="Path to save generator model (.keras)")
    p.add_argument("--out-segy", default=None, help="Path to save predicted impedance SEG-Y")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--lambda-l1", type=float, default=100.0)
    p.add_argument("--lr-g", type=float, default=2e-4)
    p.add_argument("--lr-d", type=float, default=2e-4)

    p.add_argument("--patch", type=int, nargs=3, default=[128, 128, 128], metavar=("D", "H", "W"))
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64], metavar=("SD", "SH", "SW"))
    p.add_argument("--crop", type=int, default=4, help="Center-crop margin kept from each side during inference")
    p.add_argument("--min-label-vox", type=int, default=32, help="Min labeled voxels per patch")

    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main(parse_args()))
