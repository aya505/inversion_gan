**Seismic Impedance Inversion using conditional GAN**
This project trains a **conditional GAN** to invert seismic reflection amplitudes into (log-)acoustic impedance.

- **Inputs:** 3D seismic reflection amplitude patches.
- **Targets (labels):** well-derived **log-impedance**.
- **Generator (G):** maps seismic → impedance prediction using U-Net.
- **Discriminator (D):** PatchGAN conditioned on the same seismic; scores whether an impedance patch is **real** (from wells) or **generated**, encouraging geologically realistic results.

**Training loss:** adversarial + L1 content loss

**Outputs:** impedance volumes with improved lateral continuity and detail.
