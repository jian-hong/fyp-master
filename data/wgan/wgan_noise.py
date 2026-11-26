"""
1D WGAN-GP for Q.Lin silence-gap noise segments (AE pipeline).

Uses a DCGAN-style ConvTranspose1D generator and Conv1D critic with
spectral normalization, gradient penalty lambda=10, and MMD-based early
stopping (5-epoch smoothed metric, patience=10).

Training data: envelope-threshold silence regions on raw Q.Lin .mat traces
(10% policy pool — disjoint from frozen 90% test file when possible).

Run:
  python data/wgan/wgan_noise.py [--epochs 400] [--no-train]

Outputs:
  results/wgan_noise/best.pt
  docs/wgan_validation.png (after training or with --plot-only load best.pt)
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.parametrizations import spectral_norm
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger("wgan_noise")

PROJECT = Path(__file__).resolve().parents[2]


def envelope_1d(x: np.ndarray, win: int = 128) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=np.float64).ravel())
    if len(x) < win:
        return x
    k = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(x, k, mode="same")


def extract_silence_windows_from_trace(
    trace: np.ndarray,
    window_len: int = 2048,
    stride: int = 512,
    env_percentile: float = 15.0,
) -> list[np.ndarray]:
    env = envelope_1d(trace, win=min(128, max(16, window_len // 16)))
    thr = float(np.percentile(env, env_percentile))
    mask = env < thr
    out: list[np.ndarray] = []
    half = stride // 2
    max_start = len(trace) - window_len + 1
    for start in range(0, max_start, half):
        if bool(np.all(mask[start : start + window_len])):
            w = trace[start : start + window_len].astype(np.float32).copy()
            mu, sg = float(w.mean()), float(w.std() + 1e-8)
            w = (w - mu) / sg
            out.append(w)
    return out


def load_qlin_mat_paths() -> list[Path]:
    qlin_dirs = [
        PROJECT / "Datset from Q.lin",
        Path(r"D:\New folder (2)\Data\FYP_MASTER\Datset from Q.lin"),
    ]
    paths: list[Path] = []
    for d in qlin_dirs:
        if d.exists():
            paths.extend(sorted(d.glob("*.mat")))
    return paths


def build_silence_dataset(
    max_windows: int = 50_000,
    window_len: int = 2048,
    seed: int = 42,
    percent_train: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Silence-gap windows only; subset index for ~10 percent training split."""
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []

    mat_paths = load_qlin_mat_paths()
    if not mat_paths:
        LOGGER.warning("No .mat files found for silence extraction.")
    else:
        try:
            import scipy.io as sio
        except ImportError as e:
            raise RuntimeError("scipy required for Q.Lin .mat loading") from e

        for mat_path in mat_paths:
            if len(windows) >= max_windows:
                break
            mat = sio.loadmat(str(mat_path))
            key = next((k for k in mat if not k.startswith("_") and isinstance(mat[k], np.ndarray)), None)
            if key is None:
                continue
            raw = mat[key].astype(np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            for row in raw:
                if len(windows) >= max_windows:
                    break
                windows.extend(extract_silence_windows_from_trace(row))

    if not windows:
        raise RuntimeError("No silence windows extracted — check .mat paths and envelope settings.")

    rng.shuffle(windows)
    windows = windows[:max_windows]
    x = np.stack(windows, axis=0)
    t = torch.from_numpy(x).float()
    n = len(t)
    n_tr = max(1, int(n * percent_train))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_tr]
    return t, train_idx


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Unbiased MMD^2 with Gaussian kernel; median bandwidth on ||x-y||^2."""
    x = x.float().flatten(1)
    y = y.float().flatten(1)
    n, m = x.size(0), y.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0.0, device=x.device)
    with torch.no_grad():
        d = torch.cdist(x, y, p=2.0) ** 2
        med = d.median()
    sigma2 = torch.clamp(med / 2.0, min=1e-6)
    def k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(a, b, p=2.0) ** 2
        return torch.exp(-d2 / (2.0 * sigma2))
    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)
    kxx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
    kyy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
    kxy = kxy.mean()
    return kxx + kyy - 2.0 * kxy


class Generator1D(nn.Module):
    """DCGAN-style 1D generator: z -> (B,1,2048)."""

    nz_dim: int = 128

    def __init__(self, nz: int = 128, ngf: int = 128) -> None:
        super().__init__()
        self.nz = nz
        self.fc = nn.Linear(nz, ngf * 8 * 32)
        seq: list[nn.Module] = []
        c = ngf * 8
        n_down = [(c, ngf * 4), (ngf * 4, ngf * 4), (ngf * 4, ngf * 4), (ngf * 4, ngf * 2),
                  (ngf * 2, ngf), (ngf, ngf // 2), (ngf // 2, 1)]
        for cin, cout in n_down[:-1]:
            seq += [
                nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(True),
            ]
        cin, cout = n_down[-1]
        seq.append(nn.ConvTranspose1d(cin, cout, kernel_size=4, stride=2, padding=1))
        self.net = nn.Sequential(*seq)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        x = self.fc(z).view(b, -1, 32)
        x = self.net(x)
        if x.shape[-1] != 2048:
            x = F.interpolate(x, size=2048, mode="linear", align_corners=False)
        return x


def _spectral_conv(ci: int, co: int, k: int, s: int, p: int) -> nn.Conv1d:
    return spectral_norm(nn.Conv1d(ci, co, kernel_size=k, stride=s, padding=p))


class Critic1D(nn.Module):
    """Strided Conv1D critic — spectral normalization on every conv."""

    def __init__(self, ndf: int = 96, signal_len: int = 2048) -> None:
        super().__init__()
        l = signal_len
        for _ in range(4):
            l = (l + 2 * 1 - 4) // 2 + 1
        ks = max(3, min(l, signal_len))

        self.main = nn.Sequential(
            _spectral_conv(1, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(ndf, ndf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(ndf * 2, ndf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(ndf * 4, ndf * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            _spectral_conv(ndf * 8, 1, ks, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(x.size(0), -1)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    fake = fake.detach()
    eps = torch.rand(real.size(0), 1, 1, device=real.device, dtype=real.dtype)
    interp = eps * real + (1 - eps) * fake
    interp = interp.requires_grad_(True)
    out = critic(interp)
    if out.dim() > 2:
        out = out.squeeze(-1).squeeze(-1)
    grads = torch.autograd.grad(
        outputs=out.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return ((grads.reshape(grads.size(0), -1).norm(2, dim=1) - 1.0) ** 2).mean()


def infinite_loader(loader: DataLoader, device: torch.device) -> Iterator[torch.Tensor]:
    while True:
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield batch[0].to(device).unsqueeze(1)
            else:
                yield batch.to(device).unsqueeze(1)


def train_wgan_noise(
    epochs_max: int = 400,
    lr: float = 1e-4,
    n_critic: int = 5,
    gp_lambda: float = 10.0,
    batch_size: int = 32,
    device_str: str = "cuda",
    out_root: Path | None = None,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    out_root = out_root or (PROJECT / "results" / "wgan_noise")
    out_root.mkdir(parents=True, exist_ok=True)

    data_t, idx = build_silence_dataset()
    train_x = data_t[idx].clone()
    ds = TensorDataset(train_x)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = infinite_loader(loader, device)

    G = Generator1D().to(device)
    D = Critic1D().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.99))

    smoother: deque[float] = deque(maxlen=5)
    best_smooth = float("inf")
    pat = 0
    best_epoch = -1

    for epoch in range(epochs_max):
        G.train()
        D.train()
        for _ in range(max(1, len(loader))):
            for _step in range(n_critic):
                real = next(it)
                z = torch.randn(real.size(0), G.nz, device=device)

                fake = G(z)

                crit_real = D(real).view(-1)
                crit_fake = D(fake.detach()).view(-1)
                gp = gradient_penalty(D, real, fake)
                loss_d = crit_fake.mean() - crit_real.mean() + gp_lambda * gp
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            z = torch.randn(real.size(0), G.nz, device=device)
            gen = G(z)
            loss_g = -D(gen).view(-1).mean()
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            opt_g.step()

        G.eval()
        with torch.no_grad():
            mmd_b = min(512, train_x.shape[0])
            idx = torch.randint(0, train_x.shape[0], (mmd_b,))
            rz = train_x[idx].to(device)
            fz = torch.randn(rz.shape[0], G.nz, device=device)
            fake_e = G(fz)

            mm = float(rbf_mmd2(rz, fake_e.squeeze(1)).detach().cpu())

        smoother.append(mm)
        sm = float(sum(smoother) / len(smoother))
        tag = "warm" if len(smoother) < 5 else "ok"
        LOGGER.info("Epoch %d MMD_smooth(%s)=%.6f inst=%.6f", epoch + 1, tag, sm, mm)

        if sm < best_smooth - 1e-6:
            best_smooth = sm
            pat = 0
            best_epoch = epoch + 1
            ckpt_path = out_root / "best.pt"
            torch.save(
                {"G": G.state_dict(), "D": D.state_dict(), "epoch": epoch, "mmd_smooth": sm},
                ckpt_path,
            )
            LOGGER.info("Saved %s", ckpt_path)
        elif len(smoother) >= 5:
            pat += 1
            if pat >= 10:
                LOGGER.info("Early stop: MMD smoothed plateau (patience %d)", pat)
                break

    LOGGER.info("Best epoch=%d best_mmd_smooth=%.6f", best_epoch, best_smooth)


def mean_psd(windows: torch.Tensor, fs: float = 1e6) -> tuple[np.ndarray, np.ndarray]:
    x = windows.numpy()
    psds = []
    for i in range(x.shape[0]):
        xi = np.asarray(x[i], dtype=np.float64)
        xi = xi - xi.mean()
        spec = np.fft.rfft(xi)
        freqs = np.fft.rfftfreq(len(xi), d=1.0 / fs)
        psds.append(np.abs(spec) ** 2 / len(xi))
    return freqs, np.mean(np.stack(psds, axis=0), axis=0)


def plot_validation(_g_path_unused: Path, data_t: torch.Tensor, seed: int = 42) -> None:
    ckpt_path = PROJECT / "results" / "wgan_noise" / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing WGAN checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    nz = Generator1D.nz_dim
    gen = Generator1D()
    gen.load_state_dict(payload["G"])
    gen.eval()

    rng = torch.Generator().manual_seed(seed)
    n_fake = min(1000, max(512, len(data_t)))
    with torch.no_grad():
        zs = torch.randn(n_fake, nz, generator=rng)
        fake_w = gen(zs).squeeze(1).detach().cpu()

    n_real = min(1000, len(data_t))
    perm = torch.randperm(len(data_t), generator=rng)[:n_real]
    real_w = data_t[perm]

    fq, psd_real = mean_psd(real_w)
    _, psd_fake = mean_psd(fake_w)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].semilogy(fq, psd_real + 1e-20, label="silence-gap real", lw=1.2)
    axes[0].semilogy(fq, psd_fake + 1e-20, label="WGAN synthetic", lw=1.2, alpha=0.85)
    axes[0].set_xlim(0, min(350e3, fq[-1]))
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Mean PSD")
    axes[0].legend()
    axes[0].grid(True, ls=":", alpha=0.4)
    axes[0].set_title("WGAN silence noise — mean PSD overlap")

    axes[1].axis("off")
    axes[1].text(0.02, 0.9, "Check: overlays should broadly match AE band shaping.", fontsize=11)
    fig.tight_layout()
    doc = PROJECT / "docs"
    doc.mkdir(parents=True, exist_ok=True)
    out_png = doc / "wgan_validation.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", out_png)


def load_training_defaults() -> dict:
    p = PROJECT / "config" / "training_config.yaml"
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-train", action="store_true", help="Only plot PSD from saved best.pt")
    ap.add_argument("--plot-only", action="store_true", help="Same as --no-train")
    args = ap.parse_args()

    tcfg = load_training_defaults()
    bs = int(tcfg.get("wgan_noise_batch_size", tcfg.get("batch_size", 32)))
    data_t, _idx = build_silence_dataset()

    if not (args.no_train or args.plot_only):
        train_wgan_noise(
            epochs_max=args.epochs,
            batch_size=bs,
            device_str=args.device,
        )

    try:
        plot_validation(PROJECT / "results" / "wgan_noise" / "best.pt", data_t)
    except FileNotFoundError as e:
        LOGGER.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
