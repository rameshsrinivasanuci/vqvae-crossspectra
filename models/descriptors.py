"""
descriptors.py — pluggable window-descriptor modules.

A descriptor module turns the per-window Hilbert analytic signal
    analytic : (B, N_filt, C, T_out)   complex
into a fixed-size real feature vector
    descriptor : (B, out_dim)
and an identically-shaped reconstruction target (the descriptor is its own
target — the VQ-VAE reconstructs whatever the descriptor produces).

To add a new descriptor, subclass WindowDescriptor, set self.out_dim and
self.nonnegative in __init__, and implement forward(). Nothing downstream
(encoder bottleneck, quantizer, decoder, notebook, training loop) needs to
change — they read out_dim and nonnegative from the module.

Modules provided
----------------
All four modules are read-outs of the same per-band complex cross matrix
    R_f = (1/T) z_f z_f^H     (C x C, Hermitian),
where z_f is the analytic signal of band f. This unifies them: amplitude is the
(square-rooted) diagonal, the others use the upper triangle.

HilbertAmplitude              sqrt of diag(R_f) = RMS amplitude per channel
                               out_dim = N_filt * C            nonnegative
HilbertCrossSpectralDensity   upper triangle of R_f incl. real diagonal,
                               real + imaginary parts
                               out_dim = N_filt * C^2          signed
HilbertComplexCorrelation     R_f normalised to unit diagonal, strict upper
                               triangle, real + imaginary parts (diag==1 dropped)
                               out_dim = N_filt * C*(C-1)      signed
HilbertCoherence              magnitude of the normalised R_f, strict upper
                               triangle, real only
                               out_dim = N_filt * C*(C-1)/2    nonnegative
"""

import torch
import torch.nn as nn


# ======================================================================
# Base class
# ======================================================================

class WindowDescriptor(nn.Module):
    """
    Base class for window-descriptor modules.

    Subclasses must set, in __init__:
        self.out_dim     : int   — length of the flattened descriptor vector
        self.nonnegative : bool  — True if every descriptor entry is >= 0
                                   (controls the decoder's output activation:
                                    Softplus if True, linear if False)

    and implement:
        forward(analytic) -> (descriptor, target)
            analytic   : (B, N_filt, C, T_out) complex
            descriptor : (B, out_dim) real
            target     : (B, out_dim) real   (here, identical to descriptor)
    """

    def __init__(self, n_filt, n_channels):
        super().__init__()
        self.n_filt     = n_filt
        self.n_channels = n_channels
        self.out_dim    = None        # set by subclass
        self.nonnegative = None       # set by subclass

    def forward(self, analytic):
        raise NotImplementedError

    # -- helpers shared by the correlation-style modules --------------

    @staticmethod
    def _cross_matrix(analytic):
        """
        Per-band complex cross-spectral density matrix.

        analytic : (B, N_filt, C, T) complex
        returns  : (B, N_filt, C, C) complex,  R[...,i,j] = (1/T) sum_t z_i z_j^*
        """
        T = analytic.shape[-1]
        # (B, N_filt, C, T) @ (B, N_filt, T, C) -> (B, N_filt, C, C)
        R = analytic @ analytic.conj().transpose(-1, -2)
        return R / T

    @staticmethod
    def _triu_indices(C, offset, device):
        """Row/col indices of the upper triangle with the given diagonal offset."""
        return torch.triu_indices(C, C, offset=offset, device=device)


# ======================================================================
# 1. Hilbert amplitude (mean over time)
# ======================================================================

class HilbertAmplitude(WindowDescriptor):
    """
    RMS Hilbert amplitude per (filter, channel) --- the square root of the
    diagonal of the per-band cross matrix R_f.

    The diagonal of R_f = (1/T) z_f z_f^H is
        R_f[i,i] = (1/T) sum_t |z_{f,i,t}|^2  =  mean band power on channel i,
    so its square root is the RMS amplitude:
        descriptor[b] = flatten_(f,c) sqrt( R_f[c,c] ).

    This unifies the amplitude descriptor with the correlation-style modules:
    all four are read-outs of the same object R_f. (RMS amplitude differs
    slightly from the mean envelope mean_t|z|, but is the more natural quantity
    here --- it is the square root of band power, the standard EEG feature.)

    out_dim     = N_filt * C
    nonnegative = True
    """

    def __init__(self, n_filt, n_channels):
        super().__init__(n_filt, n_channels)
        self.out_dim     = n_filt * n_channels
        self.nonnegative = True

    def forward(self, analytic):
        R = self._cross_matrix(analytic)                  # (B, F, C, C) complex
        C = analytic.shape[2]
        diag_idx = torch.arange(C, device=analytic.device)
        power = R[..., diag_idx, diag_idx].real           # (B, F, C) band power
        rms   = power.clamp(min=0).sqrt()                 # RMS amplitude
        desc  = rms.flatten(start_dim=1)                  # (B, N_filt*C)
        return desc, desc


# ======================================================================
# 2. Cross-spectral density (upper triangle, real + imag)
# ======================================================================

class HilbertCrossSpectralDensity(WindowDescriptor):
    """
    Per-band complex cross-spectral density R_f = (1/T) z_f z_f^H.

    Keeps the upper triangle INCLUDING the diagonal. The diagonal entries are
    real (auto-spectral power), so their imaginary parts (structurally zero)
    are not stored. Off-diagonal entries contribute both real and imaginary.

    Packing per band: C real diagonal values + C(C-1)/2 complex off-diagonal
    values (each real+imag) = C + C(C-1) = C^2 numbers.

    out_dim     = N_filt * C^2
    nonnegative = False   (real/imag parts may be negative)
    """

    def __init__(self, n_filt, n_channels):
        super().__init__(n_filt, n_channels)
        C = n_channels
        self.out_dim     = n_filt * C * C
        self.nonnegative = False

    def forward(self, analytic):
        B, F, C, _ = analytic.shape
        R = self._cross_matrix(analytic)                  # (B, F, C, C) complex

        diag_idx = torch.arange(C, device=analytic.device)
        diag = R[..., diag_idx, diag_idx].real            # (B, F, C)

        ui = self._triu_indices(C, offset=1, device=analytic.device)
        off = R[..., ui[0], ui[1]]                         # (B, F, C(C-1)/2) complex

        desc = torch.cat([
            diag.flatten(start_dim=1),
            off.real.flatten(start_dim=1),
            off.imag.flatten(start_dim=1),
        ], dim=1)                                          # (B, N_filt*C^2)
        return desc, desc


# ======================================================================
# 3. Complex correlation coefficient (strict upper triangle, real + imag)
# ======================================================================

class HilbertComplexCorrelation(WindowDescriptor):
    """
    Complex correlation coefficient
        R~_f[i,j] = R_f[i,j] / sqrt(R_f[i,i] R_f[j,j]).

    The diagonal is identically 1 and carries no information, so it is dropped.
    Keeps the STRICT upper triangle, real and imaginary parts.

    Packing per band: C(C-1)/2 complex off-diagonal values (real+imag)
    = C(C-1) numbers.

    out_dim     = N_filt * C*(C-1)
    nonnegative = False
    """

    def __init__(self, n_filt, n_channels):
        super().__init__(n_filt, n_channels)
        C = n_channels
        self.out_dim     = n_filt * C * (C - 1)
        self.nonnegative = False

    def forward(self, analytic):
        B, F, C, _ = analytic.shape
        R = self._cross_matrix(analytic)                  # (B, F, C, C) complex

        diag_idx = torch.arange(C, device=analytic.device)
        d = R[..., diag_idx, diag_idx].real.clamp(min=1e-10).sqrt()  # (B, F, C)
        denom = d.unsqueeze(-1) * d.unsqueeze(-2)          # (B, F, C, C)
        Rn = R / denom                                     # normalised

        ui = self._triu_indices(C, offset=1, device=analytic.device)
        off = Rn[..., ui[0], ui[1]]                        # (B, F, C(C-1)/2) complex

        desc = torch.cat([
            off.real.flatten(start_dim=1),
            off.imag.flatten(start_dim=1),
        ], dim=1)                                          # (B, N_filt*C*(C-1))
        return desc, desc


# ======================================================================
# 4. Coherence (magnitude of complex correlation, strict upper triangle)
# ======================================================================

class HilbertCoherence(WindowDescriptor):
    """
    Coherence = magnitude of the complex correlation coefficient,
        coh_f[i,j] = | R_f[i,j] | / sqrt(R_f[i,i] R_f[j,j]).

    Real, non-negative, in [0, 1]. Diagonal is identically 1 and dropped.
    Keeps the STRICT upper triangle, real-valued.

    Packing per band: C(C-1)/2 real values.

    out_dim     = N_filt * C*(C-1)/2
    nonnegative = True
    """

    def __init__(self, n_filt, n_channels):
        super().__init__(n_filt, n_channels)
        C = n_channels
        self.out_dim     = n_filt * (C * (C - 1)) // 2
        self.nonnegative = True

    def forward(self, analytic):
        B, F, C, _ = analytic.shape
        R = self._cross_matrix(analytic)                  # (B, F, C, C) complex

        diag_idx = torch.arange(C, device=analytic.device)
        d = R[..., diag_idx, diag_idx].real.clamp(min=1e-10).sqrt()
        denom = d.unsqueeze(-1) * d.unsqueeze(-2)
        coh = (R.abs() / denom)                            # (B, F, C, C) real

        ui = self._triu_indices(C, offset=1, device=analytic.device)
        off = coh[..., ui[0], ui[1]]                       # (B, F, C(C-1)/2)

        desc = off.flatten(start_dim=1)                    # (B, N_filt*C*(C-1)/2)
        return desc, desc


# ======================================================================
# Registry — name -> class, for config-driven selection
# ======================================================================

DESCRIPTOR_REGISTRY = {
    "amplitude":            HilbertAmplitude,
    "cross_spectral":       HilbertCrossSpectralDensity,
    "complex_correlation":  HilbertComplexCorrelation,
    "coherence":            HilbertCoherence,
}


def build_descriptor(name, n_filt, n_channels):
    """
    Instantiate a descriptor by name.

    name in {'amplitude', 'cross_spectral', 'complex_correlation', 'coherence'}.
    Raises KeyError with the valid options if the name is unknown.
    """
    if name not in DESCRIPTOR_REGISTRY:
        raise KeyError(
            f"Unknown descriptor '{name}'. "
            f"Options: {list(DESCRIPTOR_REGISTRY.keys())}"
        )
    return DESCRIPTOR_REGISTRY[name](n_filt, n_channels)


# ======================================================================
# Advisory: minimum low-frequency cutoff per descriptor
# ======================================================================

# Minimum number of cycles within one window for a frequency estimate to be
# trustworthy under each descriptor. Amplitude needs essentially one cycle;
# coherence/correlation average a cross term over the window and need several
# cycles or they report spuriously high coupling at low frequencies.
MIN_CYCLES_PER_DESCRIPTOR = {
    "amplitude":           1.0,   # band power per channel — no cross averaging
    "cross_spectral":      5.0,   # cross term averaged over the window
    "complex_correlation": 5.0,   # normalised cross term
    "coherence":           7.0,   # magnitude of normalised cross term — most demanding
}


def suggest_freq_low(window_samples, fs, descriptor, min_cycles=None):
    """
    Advisory minimum filter low-cutoff (Hz) for a given window and descriptor.

    A frequency f completes  f * (window_samples / fs)  cycles within one window.
    Coherence and correlation estimates average a cross term over the window and
    are unreliable when that count is small: a frequency with only ~1 cycle per
    window will show spuriously high coherence regardless of true coupling. The
    suggested floor is

        f_low_min = min_cycles / (window_samples / fs)
                  = min_cycles * fs / window_samples.

    This is ADVISORY only — it prints guidance; the user still sets freq_low.

    Parameters
    ----------
    window_samples : int    — L_w.
    fs             : float  — sampling frequency (Hz).
    descriptor     : str    — descriptor name (selects the default min_cycles).
    min_cycles     : float or None
        Override the per-descriptor default number of cycles.

    Returns
    -------
    dict with keys:
        freq_low_min : float — suggested minimum low-cutoff (Hz)
        min_cycles   : float — cycles assumed
        window_sec   : float — window duration (s)
    """
    if min_cycles is None:
        min_cycles = MIN_CYCLES_PER_DESCRIPTOR.get(descriptor, 5.0)
    window_sec   = window_samples / fs
    freq_low_min = min_cycles / window_sec
    return {
        "freq_low_min": freq_low_min,
        "min_cycles":   min_cycles,
        "window_sec":   window_sec,
    }
