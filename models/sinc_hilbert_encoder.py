"""
sinc_hilbert_encoder.py — SincNet + Hilbert encoder with a pluggable descriptor.

    x: (B, C, L_w)
      -> InstanceNorm
      -> SincConv (N_filt, Filt_dim)   -> (B, N_filt, C, T_out)
      -> Hilbert transform             -> (B, N_filt, C, T_out) complex
      -> descriptor module             -> (B, out_dim), target (B, out_dim)

The descriptor module (see descriptors.py) decides how the analytic signal
becomes a feature vector. The encoder itself is agnostic to that choice; it
exposes the descriptor's out_dim as self.embedding_dim and its non-negativity
flag as self.nonnegative.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from models.sinc_convolution import SincConvolution
from models.descriptors import build_descriptor


class SincHilbertEncoder(nn.Module):
    """
    Parameters
    ----------
    num_EEG_Channels : int   — C.
    N_filt           : int   — number of Sinc filters.
    Filt_dim         : int   — filter kernel length (free; frequency resolution).
    descriptor       : str   — descriptor name (see descriptors.DESCRIPTOR_REGISTRY).
    fs               : float — sampling frequency (Hz).
    cutoff           : float — max filter frequency (Hz).
    window_samples   : int   — L_w.
    freq_low         : float — minimum allowed filter low-cutoff (Hz), hard constraint.
    freq_high        : float — maximum allowed filter high-cutoff (Hz); defaults to cutoff.
    min_band         : float — minimum filter bandwidth (Hz).
    fixed_bands      : list[(low,high)] or None — if given, the filterbank is
                       frozen to these exact bands (not trained) and N_filt is
                       set to len(fixed_bands).
    device           : torch.device or None.
    """

    def __init__(self,
                 num_EEG_Channels=2,
                 N_filt=20,
                 Filt_dim=145,
                 descriptor="amplitude",
                 fs=100.0,
                 cutoff=50,
                 window_samples=400,
                 freq_low=1.0,
                 freq_high=None,
                 min_band=2.0,
                 fixed_bands=None,
                 device=None):
        super().__init__()

        self.num_EEG_Channels = num_EEG_Channels
        self.N_filt           = N_filt
        self.Filt_dim         = Filt_dim
        self.fs               = fs
        self.cutoff           = cutoff
        self.window_samples   = window_samples
        self.descriptor_name  = descriptor

        # informational only — descriptors collapse the time axis
        self.T_out = window_samples - Filt_dim + 1

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.instance_norm = nn.InstanceNorm1d(num_EEG_Channels, affine=False)
        self.Sinc_Conv = SincConvolution(
            N_filt=N_filt, Filt_dim=Filt_dim, fs=fs, cutoff=cutoff, device=device,
            freq_low=freq_low, freq_high=freq_high, min_band=min_band,
            fixed_bands=fixed_bands,
        )
        # if fixed bands were given, N_filt is set by the band list
        self.N_filt = self.Sinc_Conv.N_filt
        # expose resolved bounds for downstream use (e.g. plotting)
        self.freq_low  = self.Sinc_Conv.freq_low
        self.freq_high = self.Sinc_Conv.freq_high
        self.min_band  = self.Sinc_Conv.min_band

        # pluggable descriptor module
        self.descriptor = build_descriptor(descriptor, self.N_filt, num_EEG_Channels)

        # exposed for the bottleneck/decoder
        self.embedding_dim = self.descriptor.out_dim
        self.nonnegative   = self.descriptor.nonnegative

    def hilbert_transform(self, x):
        """Analytic signal of a real input along the last axis."""
        x = torch.fft.rfft(x, dim=-1)
        x = torch.view_as_real(x)
        x = x * 2
        x[..., 0, :] = x[..., 0, :] / 2.
        x = F.pad(x, [0, 0, 0, x.shape[-2] - 2])
        x = torch.view_as_complex(x)
        x = torch.fft.ifft(x, norm=None, dim=-1)
        return x

    def forward(self, x):
        """
        x: (B, C, L_w)

        Returns:
            embedding : (B, out_dim)   descriptor feature vector
            target    : (B, out_dim)   reconstruction target (== embedding)
            filters   : learned filter kernels
            freqs     : (filt_beg_freq, filt_end_freq)
        """
        x = self.instance_norm(x)
        x, filters, freqs = self.Sinc_Conv(x)        # (B, N_filt, C, T_out)
        analytic = self.hilbert_transform(x)         # (B, N_filt, C, T_out) complex

        embedding, target = self.descriptor(analytic)
        return embedding, target, filters, freqs
