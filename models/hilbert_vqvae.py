"""
hilbert_vqvae.py — Local VQ-VAE for EEG microstate tokenisation with a
pluggable window descriptor.

Pipeline
--------
    chunk (B, C, L_w)
      -> SincHilbertEncoder (Sinc -> Hilbert -> descriptor)  -> (B, d_desc)
      -> proj_down (B, D)                                     z_e
      -> EMA codebook lookup (B, D)                           z_q, token k
      -> descriptor decoder (B, d_desc)                       reconstruction

The descriptor module (descriptors.py) sets the feature dimension d_desc and
whether the features are non-negative. The decoder reads the non-negativity flag
to choose its output activation (Softplus if non-negative, else linear), so
swapping descriptors requires no other change.

Notation
--------
B  batch size | C channels | L_w window length | N_filt filters
Filt_dim filter length | d_desc descriptor dimension | K codebook size
D codebook dimension
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from models.sinc_hilbert_encoder import SincHilbertEncoder


# ======================================================================
# EMA Vector Quantizer
# ======================================================================

class EMAVectorQuantizer(nn.Module):
    """
    EMA-updated nearest-neighbour quantizer with raw Euclidean distance.

    - Raw squared Euclidean distance (no L2 normalisation).
    - Codebook initialised at std=0.02.
    - EMA codebook updates; codebook is a buffer, not a gradient parameter.
    - Dual dead-entry reset: count < dead_threshold OR usage < 1/(2K).
    - Loss = recon + beta * commitment (no codebook loss term).
    """

    def __init__(self, codebook_size, codebook_dim,
                 gamma=0.99, beta=0.25, dead_threshold=2.0):
        super().__init__()
        self.codebook_size      = codebook_size
        self.codebook_dim       = codebook_dim
        self.gamma              = gamma
        self.beta               = beta
        self.dead_threshold     = dead_threshold
        self.min_usage_fraction = 1.0 / (2.0 * codebook_size)

        self.register_buffer('codebook',
            torch.randn(codebook_size, codebook_dim) * 0.02)
        self.register_buffer('ema_count', torch.ones(codebook_size))
        self.register_buffer('ema_sum',   self.codebook.clone())

    def forward(self, z_e):
        distances = (
            torch.sum(z_e ** 2, dim=1, keepdim=True)
            - 2 * z_e @ self.codebook.t()
            + torch.sum(self.codebook ** 2, dim=1)
        )
        tokens = distances.argmin(dim=-1)
        z_q    = self.codebook[tokens]

        if self.training:
            one_hot     = F.one_hot(tokens, self.codebook_size).float()
            batch_count = one_hot.sum(dim=0)
            batch_sum   = one_hot.t() @ z_e.detach()

            self.ema_count = self.gamma * self.ema_count + (1 - self.gamma) * batch_count
            self.ema_sum   = self.gamma * self.ema_sum   + (1 - self.gamma) * batch_sum
            self.codebook  = self.ema_sum / self.ema_count.unsqueeze(1).clamp(min=1e-5)

            total          = self.ema_count.sum().clamp(min=1e-5)
            usage_fraction = self.ema_count / total
            dead_mask = (
                (self.ema_count   < self.dead_threshold)
                | (usage_fraction < self.min_usage_fraction)
            )
            n_dead = dead_mask.sum().item()
            if n_dead > 0:
                idx = torch.randint(0, z_e.shape[0], (n_dead,), device=z_e.device)
                self.codebook[dead_mask]  = z_e.detach()[idx]
                self.ema_count[dead_mask] = self.dead_threshold
                self.ema_sum[dead_mask]   = z_e.detach()[idx] * self.dead_threshold

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, tokens, commitment_loss

    def codebook_diagnostics(self):
        total      = self.ema_count.sum().clamp(min=1e-5)
        usage_frac = self.ema_count / total
        active     = (self.ema_count > self.dead_threshold).sum().item()

        # cosine similarity — informative for signed embeddings, but for
        # non-negative descriptors (amplitude, coherence) all vectors sit in the
        # positive orthant and cosine is high by default, so it is NOT a reliable
        # collapse signal. Reported for display only.
        cb_norm = F.normalize(self.codebook, dim=-1)
        cos     = cb_norm @ cb_norm.t()
        mask    = ~torch.eye(self.codebook_size, dtype=torch.bool, device=cos.device)
        mean_cos = cos[mask].mean().item()

        # mean pairwise Euclidean distance, normalised by the mean vector norm.
        # This is sign-independent: genuine collapse drives it toward 0, whereas
        # a healthy spread keeps it order 1. Use this to judge true diversity.
        norms   = self.codebook.norm(dim=-1)
        scale   = norms.mean().clamp(min=1e-8)
        d2      = torch.cdist(self.codebook, self.codebook)   # (K, K)
        mean_dist = (d2[mask].mean() / scale).item()

        return {
            "active_entries":  active,
            "active_fraction": active / self.codebook_size,
            "mean_cosine_sim": mean_cos,     # display only
            "mean_norm_dist":  mean_dist,    # collapse signal (sign-independent)
            "ema_count":       self.ema_count.cpu(),
            "usage_fraction":  usage_frac.cpu(),
        }


# ======================================================================
# Descriptor decoder (MLP) — activation depends on descriptor sign
# ======================================================================

class DescriptorDecoder(nn.Module):
    """
    MLP decoder from a codebook vector to the descriptor vector.

        Linear(D -> hidden) -> GELU -> Linear(hidden -> out_dim) -> [activation]

    activation is Softplus when the descriptor is non-negative (amplitude,
    coherence) and identity (linear) when it is signed (cross-spectral density,
    complex correlation), so the output range matches the target range.
    """

    def __init__(self, codebook_dim, out_dim, nonnegative, hidden_dim=128):
        super().__init__()
        layers = [
            nn.Linear(codebook_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        ]
        if nonnegative:
            layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z_q_st):
        return self.net(z_q_st)


# ======================================================================
# Hilbert VQ-VAE
# ======================================================================

class HilbertVQVAE(nn.Module):
    """
    Local VQ-VAE with a pluggable Hilbert-based window descriptor.

    Parameters
    ----------
    num_EEG_Channels : int   — C.
    N_filt           : int   — number of Sinc filters.
    Filt_dim         : int   — filter kernel length (free; frequency resolution).
    window_samples   : int   — L_w.
    descriptor       : str   — 'amplitude' | 'cross_spectral' |
                               'complex_correlation' | 'coherence'.
    fs, cutoff       : sampling frequency and max filter frequency (Hz).
    freq_low         : minimum allowed filter low-cutoff (Hz), hard constraint.
    freq_high        : maximum allowed filter high-cutoff (Hz); defaults to cutoff.
    min_band         : minimum filter bandwidth (Hz).
    fixed_bands      : list[(low,high)] or None — freeze the filterbank to these
                       exact bands (not trained); N_filt becomes len(fixed_bands).
    codebook_size    : int   — K.
    codebook_dim     : int   — D.
    decoder_hidden   : int   — decoder MLP hidden width.
    gamma, beta, dead_threshold : EMA quantizer hyperparameters.
    device           : torch.device or None.
    """

    def __init__(self,
                 num_EEG_Channels,
                 N_filt,
                 Filt_dim,
                 window_samples,
                 descriptor="amplitude",
                 fs=100.0,
                 cutoff=50.0,
                 freq_low=1.0,
                 freq_high=None,
                 min_band=2.0,
                 fixed_bands=None,
                 codebook_size=16,
                 codebook_dim=128,
                 decoder_hidden=128,
                 gamma=0.99,
                 beta=0.25,
                 dead_threshold=2.0,
                 device=None):
        super().__init__()

        self.num_EEG_Channels = num_EEG_Channels
        self.N_filt           = N_filt
        self.window_samples   = window_samples
        self.codebook_size    = codebook_size
        self.descriptor_name  = descriptor
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = SincHilbertEncoder(
            num_EEG_Channels=num_EEG_Channels,
            N_filt=N_filt,
            Filt_dim=Filt_dim,
            descriptor=descriptor,
            fs=fs,
            cutoff=cutoff,
            window_samples=window_samples,
            freq_low=freq_low,
            freq_high=freq_high,
            min_band=min_band,
            fixed_bands=fixed_bands,
            device=device,
        )
        # N_filt may have been overridden by fixed_bands
        self.N_filt = self.encoder.N_filt
        self.descriptor_dim = self.encoder.embedding_dim
        self.nonnegative    = self.encoder.nonnegative
        self.T_out          = self.encoder.T_out

        self.proj_down = nn.Linear(self.descriptor_dim, codebook_dim)

        self.quantizer = EMAVectorQuantizer(
            codebook_size=codebook_size, codebook_dim=codebook_dim,
            gamma=gamma, beta=beta, dead_threshold=dead_threshold,
        )

        self.decoder = DescriptorDecoder(
            codebook_dim=codebook_dim,
            out_dim=self.descriptor_dim,
            nonnegative=self.nonnegative,
            hidden_dim=decoder_hidden,
        )

    def encode(self, x):
        """x: (B, C, L_w) -> z_e (B, D), target (B, d_desc), filters, freqs"""
        embedding, target, filters, freqs = self.encoder(x)
        z_e = self.proj_down(embedding)
        return z_e, target, filters, freqs

    def forward(self, x):
        """
        x: (B, C, L_w)
        Returns desc_hat (B, d_desc), desc_target (B, d_desc), tokens (B,), losses.
        """
        z_e, desc_target, _, _ = self.encode(x)
        z_q_st, tokens, commitment_loss = self.quantizer(z_e)
        desc_hat = self.decoder(z_q_st)

        recon_loss = F.mse_loss(desc_hat, desc_target)
        total_loss = recon_loss + self.quantizer.beta * commitment_loss

        return desc_hat, desc_target, tokens, {
            "total":      total_loss,
            "recon":      recon_loss,
            "commitment": commitment_loss,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def tokenise_recording(self, signal, extract_overlap_pct=50, device=None):
        """
        Tokenise a continuous (C, T) recording with sliding windows.
        extract_overlap_pct: 50 = 50% overlap; 75 = finer; 0 = none.
        Returns tokens (N_windows,), offsets (N_windows,).
        """
        device = device or self.device
        self.eval()
        if not isinstance(signal, torch.Tensor):
            signal = torch.from_numpy(signal).float()
        signal = signal.to(device)

        ws     = self.window_samples
        stride = max(1, int(ws * (1 - extract_overlap_pct / 100)))
        C, T   = signal.shape
        starts = list(range(0, T - ws + 1, stride))
        windows = torch.stack([signal[:, s:s + ws] for s in starts])

        all_tokens = []
        for i in range(0, len(windows), 64):
            batch = windows[i:i + 64]
            z_e, _, _, _ = self.encode(batch)
            d = (torch.sum(z_e ** 2, dim=1, keepdim=True)
                 - 2 * z_e @ self.quantizer.codebook.t()
                 + torch.sum(self.quantizer.codebook ** 2, dim=1))
            all_tokens.append(d.argmin(dim=-1).cpu())
        return torch.cat(all_tokens), np.array(starts)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_codebook_descriptors(self, dataloader, device=None):
        """
        For each token k, the mean descriptor vector across assigned windows.
        Returns descriptors (K, d_desc), counts (K,).

        The descriptor layout depends on the chosen module; use the helper
        unpack_* functions in descriptors-aware notebook code to reshape it
        (e.g. amplitude -> (N_filt, C)).
        """
        device = device or self.device
        K = self.codebook_size
        desc_accum  = torch.zeros(K, self.descriptor_dim)
        count_accum = torch.zeros(K)
        self.eval()
        for batch in dataloader:
            batch = batch.to(device)
            z_e, target, _, _ = self.encode(batch)
            _, tokens, _ = self.quantizer(z_e)
            for b in range(batch.shape[0]):
                k = tokens[b].item()
                desc_accum[k]  += target[b].cpu()
                count_accum[k] += 1
        safe = count_accum.clamp(min=1).view(K, 1)
        return desc_accum / safe, count_accum


# ======================================================================
# Utility
# ======================================================================

def relative_recon_error(desc_hat, desc_target):
    """mean|desc_hat - desc_target| / mean|desc_target| — scale-independent."""
    mae  = (desc_hat - desc_target).abs().mean()
    mag  = desc_target.abs().mean().clamp(min=1e-8)
    return (mae / mag).item()
