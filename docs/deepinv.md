# fastprox in place of DeepInverse's TV denoisers

[DeepInverse](https://deepinv.github.io)'s `TVDenoiser` and `TVL1Denoiser`
compute the prox of isotropic and anisotropic TV,

    argmin_z  ½ ‖z − y‖² + γ Σ_p ‖(D z)_p‖      (‖·‖₂ for TVDenoiser, ‖·‖₁ for TVL1Denoiser)

with forward differences that are zero past the last entry, on batches of
shape (B, C, H, W) or (B, C, D, H, W), each channel separately. These are the
problems of `fastprox.torch.tv_iso` and `fastprox.torch.tv_aniso` (and of
`fastprox.jax.tv_iso` / `tv_aniso`), which run on the input's device and
agree with DeepInverse's denoisers to their convergence (to 1e-12 and 4e-14 on
test images). [`examples/deepinv_tv.py`](../examples/deepinv_tv.py) checks
everything on this page.

```python
import fastprox.torch as fpt
from deepinv.models import TVDenoiser

z = TVDenoiser(crit=1e-5, n_it_max=1000)(y, ths)       # DeepInverse
z = fpt.tv_iso(y, ths, tol=1e-6)                       # fastprox; tv_aniso for TVL1Denoiser
```

## What maps to what

| DeepInverse | fastprox | Notes |
| --- | --- | --- |
| `ths` (a number, or one per sample) | `gamma` (a number, or a tensor that broadcasts to `y.shape[:-2]`) | one step per sample: `ths` of shape (B,) is `gamma` of shape (B, 1) |
| `crit`: relative change of x between iterations | `tol`: duality gap at most `tol · ‖y‖² / 2` per image | the gap bounds the error, ‖z − z*‖² ≤ 2 gap; a small change between iterations does not |
| `n_it_max` | `max_iter` | |
| warm restart: the denoiser keeps `x2`, `u2` between calls | `warm_start=info.dual` from the previous call (`return_info=True`) | explicit, so one function serves many problems |
| autograd through the iterations | implicit differentiation at the solution (`deriv_tol`) | memory does not grow with the iterations |
| `TVL1Denoiser` | `tv_aniso` | |
| 3D input (B, C, D, H, W) | `ndim=3` | |

fastprox also takes per-axis `weights` (e.g. voxel spacing), and per-pixel
`pixel_weights` (isotropic) or per-edge `edge_weights` (anisotropic), one map
for all images or one per image, all with gradients: a spatially varying
regularisation map, learned or predicted by a network.

## Differences to expect

- **Stopping.** DeepInverse stops when x changes by less than `crit`
  relative to its size. From a warm start this can stop early: in the
  example's inpainting loop, its defaults end a few hundredths away from the
  loop with exact proxes, against about 1e-3 for fastprox at `tol = 1e-6`.
  fastprox stops on the duality gap, which certifies the error of every image.
- **Gradients.** DeepInverse differentiates through its iterations, so the
  backward keeps every iterate: about 10 MiB per iteration for 8 × 256²
  float32 images (4 GiB at 400). fastprox differentiates the optimality
  conditions on the regions where the solution is flat and keeps the
  solution and the dual (6–10 MiB here). Its gradients are those of the
  converged prox, up to the error the gap allows (docs/api.md): in float32
  they level off near 0.065 (isotropic) and 0.04 (anisotropic) relative to
  the exact gradient on these images, which DeepInverse reaches after about
  400 iterations (isotropic).
- **Cost** (Apple M5 Pro, eight 256² images, `bench/compare_deepinv.py`;
  docs/benchmarks.md): to the same accuracy, fastprox's isotropic TV is
  3–16× faster on the GPU (MPS) and 1.3–7× on the CPU, and the backward at
  matched gradient accuracy takes 166 ms against 712; anisotropic TV is
  2.2–12× faster on MPS, with a backward of 45–52 ms against 630. On an
  NVIDIA T4 (CUDA) the forward gains are 6–8× at loose accuracy and
  1.1–1.4× at 1e-3 to 1e-4; the isotropic backward takes 281 ms against 1002
  (forward plus backward 560 against 1381 ms), the anisotropic one 9 ms
  against 843, with the same memory savings.
- **Requirements.** fastprox needs Python 3.12 or later.

## The C kernels from the GPU

`fastprox.torch.prox(fp.TVIso(), y, ths)` (or `fp.TVAniso()`) runs the C
kernels on the host and copies back. With unified memory the round trip is
negligible (0.5 ms for 8 × 256² on an M5 Pro), and the anisotropic C kernel,
with exact 1D TV steps and a polish, reaches 1e-4 in half the time of the
device path. `gamma` is one number there.
