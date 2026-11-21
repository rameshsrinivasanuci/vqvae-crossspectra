#!/usr/bin/env python3
"""
fit_vqvae.py — end-to-end fit/finalise of the Hilbert VQ-VAE from a notebook
handoff bundle.

Workflow
--------
1. Run steps 1-3 of hilbert_microstate_vqvae7.ipynb (config, segment, codebook
   sweep). Its final cell writes a handoff bundle (<prefix>_sweep.pt) holding the
   full config and the trained sweep model for every K.
2. Inspect the elbow, pick K, then run this script:

       python fit_vqvae.py --sweep run_sweep.pt --codebook-size 24

   By default it LOADS the already-trained sweep model for that K (trained for
   SWEEP_EPOCHS) — no retraining, no re-specification of parameters. Pass
   --retrain to fit a fresh model at K for FULL_EPOCHS instead.

The script re-segments the data from the saved config (deterministic), so the
original DATA_PATH must be reachable. It saves the four output files exactly as
the notebook did, plus only three figures: the final training curves, the
codebook-pruning usage plot, and the Sinc filter bands.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")                 # file output only, no display
import matplotlib.pyplot as plt

from models.hilbert_vqvae import HilbertVQVAE, relative_recon_error
from models.descriptors import suggest_freq_low
from models.early_stopping import EarlyStopping
from models.utils import (UnsupervisedEEGDataset, segment_records,
                          split_segments)
from torch.utils.data import DataLoader


# ======================================================================
# model build / train  (mirrors notebook cells 7, 11)
# ======================================================================

def build_model(cfg, codebook_size, device):
    return HilbertVQVAE(
        num_EEG_Channels=cfg["C"], N_filt=cfg["N_FILT"], Filt_dim=cfg["FILT_DIM"],
        window_samples=cfg["SEGMENT_SAMPLES"], descriptor=cfg["DESCRIPTOR"],
        fs=cfg["FS"], cutoff=cfg["CUTOFF"],
        freq_low=cfg["FREQ_LOW"], freq_high=cfg["FREQ_HIGH"], min_band=cfg["MIN_BAND"],
        fixed_bands=cfg["FIXED_BANDS"],
        codebook_size=codebook_size,
        codebook_dim=cfg["CODEBOOK_DIM"], decoder_hidden=cfg["DECODER_HIDDEN"],
        encoder_hidden=cfg["ENCODER_HIDDEN"],
    ).to(device)


def train_model(model, train_loader, val_loader, cfg, n_epochs, device,
                checkpoint_path="ckpt.pt", collapse_active_frac=0.25,
                collapse_dist=0.05, collapse_patience=5, verbose=True):
    """Train with val early stopping and a sign-independent collapse auto-stop.
    Returns (history, stop_reason). Identical logic to the notebook."""
    optimizer = optim.Adam(model.parameters(), lr=cfg["LR"])
    stopper = EarlyStopping(patience=cfg["PATIENCE"], path=checkpoint_path)
    history = {k: [] for k in
               ["train_loss", "val_loss", "train_recon", "val_recon",
                "val_err", "active", "dist"]}
    collapse_count, stop_reason = 0, "max_epochs"

    for epoch in range(n_epochs):
        model.train()
        tl = tr = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            _, _, _, losses = model(batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
            tl += losses["total"].item()
            tr += losses["recon"].item()
        n_tr = len(train_loader)

        model.eval()
        vl = vr = ve = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                dh, dt, _, losses = model(batch)
                vl += losses["total"].item()
                vr += losses["recon"].item()
                ve += relative_recon_error(dh, dt)
        n_va = len(val_loader)

        diag = model.quantizer.codebook_diagnostics()
        history["train_loss"].append(tl / n_tr)
        history["val_loss"].append(vl / n_va)
        history["train_recon"].append(tr / n_tr)
        history["val_recon"].append(vr / n_va)
        history["val_err"].append(ve / n_va)
        history["active"].append(diag["active_entries"])
        history["dist"].append(diag["mean_norm_dist"])

        if verbose and epoch % 10 == 0:
            print(f'  epoch {epoch:3d} | train {history["train_loss"][-1]:.4f} | '
                  f'val {history["val_loss"][-1]:.4f} | '
                  f'err {history["val_err"][-1]*100:.2f}% | '
                  f'active {diag["active_entries"]}/{model.codebook_size} | '
                  f'dist {diag["mean_norm_dist"]:.3f}')

        if (diag["active_fraction"] < collapse_active_frac
                and diag["mean_norm_dist"] < collapse_dist):
            collapse_count += 1
            if collapse_count >= collapse_patience:
                print(f"  collapsed at epoch {epoch}")
                stop_reason = "collapse"
                break
        else:
            collapse_count = 0

        stopper(history["val_loss"][-1], model)
        if stopper.early_stop:
            stop_reason = "val_loss"
            break

    if stop_reason != "collapse":
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history, stop_reason


# ======================================================================
# connectivity-matrix reconstruction  (mirrors notebook cell 22)
# ======================================================================

def token_band_matrices(vec, descriptor, n_filt, C):
    """Real (n_filt, C, C) per-band connectivity matrices for one token
    descriptor vector. Packing mirrors descriptors.py exactly."""
    vec = np.asarray(vec)
    M = C * (C - 1) // 2
    iu = np.triu_indices(C, k=1)
    mats = np.zeros((n_filt, C, C), dtype=float)

    if descriptor == "coherence":
        block = vec.reshape(n_filt, M)
        for f in range(n_filt):
            m = np.zeros((C, C))
            m[iu] = block[f]
            m = m + m.T
            np.fill_diagonal(m, 1.0)
            mats[f] = m
        return mats, "coherence (0-1)"

    if descriptor == "complex_correlation":
        re = vec[:n_filt * M].reshape(n_filt, M)
        im = vec[n_filt * M:].reshape(n_filt, M)
        for f in range(n_filt):
            mag = np.zeros((C, C))
            mag[iu] = np.hypot(re[f], im[f])
            mag = mag + mag.T
            np.fill_diagonal(mag, 1.0)
            mats[f] = mag
        return mats, "|complex correlation| (0-1)"

    if descriptor == "cross_spectral":
        diag = vec[:n_filt * C].reshape(n_filt, C)
        re = vec[n_filt * C: n_filt * C + n_filt * M].reshape(n_filt, M)
        im = vec[n_filt * C + n_filt * M:].reshape(n_filt, M)
        for f in range(n_filt):
            mag = np.zeros((C, C))
            mag[iu] = np.hypot(re[f], im[f])
            mag = mag + mag.T
            np.fill_diagonal(mag, diag[f])
            mats[f] = mag
        return mats, "|cross-spectral density|"

    raise ValueError(f"{descriptor} has no connectivity-matrix view.")


# ======================================================================
# data loading / segmentation  (mirrors notebook cells 3, 5)
# ======================================================================

def load_records(cfg):
    """Re-load the continuous record(s) from the saved DATA_PATH, returning the
    records object in the same normalised form the notebook used."""
    raw = np.load(cfg["DATA_PATH"], allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        records = [np.asarray(r) for r in raw]
    elif raw.ndim == 2:
        records = raw
    elif raw.ndim == 3:
        records = raw
    else:
        raise ValueError(f"unexpected data shape {raw.shape}")
    return records


def record_list(records):
    """Normalise records to a list of (C, T) arrays."""
    if isinstance(records, list):
        return records
    if getattr(records, "ndim", 2) == 2:
        return [records]
    return [records[i] for i in range(records.shape[0])]


# ======================================================================
# main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Finalise the Hilbert VQ-VAE from a notebook sweep bundle.")
    ap.add_argument("--sweep", required=True,
                    help="handoff bundle (<prefix>_sweep.pt) written by the notebook")
    ap.add_argument("--codebook-size", type=int, required=True,
                    help="chosen K (must be one of the swept sizes unless --retrain)")
    ap.add_argument("--retrain", action="store_true",
                    help="fit a fresh model at K for FULL_EPOCHS instead of "
                         "loading the sweep's model")
    ap.add_argument("--full-epochs", type=int, default=None,
                    help="override FULL_EPOCHS when retraining (default: from config)")
    ap.add_argument("--out-dir", default=".",
                    help="directory for saved outputs (default: current dir)")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load the handoff bundle -------------------------------------
    bundle = torch.load(args.sweep, map_location=device, weights_only=False)
    cfg = bundle["config"]
    K = args.codebook_size
    print(f"Loaded sweep bundle: {args.sweep}")
    print(f"  descriptor={cfg['DESCRIPTOR']}  segment={cfg['SEGMENT_SEC']}s "
          f"({cfg['SEGMENT_SAMPLES']} samples)  swept K={cfg['CODEBOOK_SIZES']}")

    # ---- re-segment the data from the saved config -------------------
    records = load_records(cfg)
    segments, seg_starts, seg_record_id = segment_records(
        records, cfg["SEGMENT_SAMPLES"], step_samples=None)
    tr_mask, val_mask = split_segments(
        seg_record_id, val_fraction=cfg["VAL_FRACTION"], seed=42)
    X_train, X_val = segments[tr_mask], segments[val_mask]
    print(f"Re-segmented: {segments.shape[0]} segments "
          f"(train {len(X_train)} | val {len(X_val)})")

    train_loader = DataLoader(UnsupervisedEEGDataset(X_train),
                              batch_size=cfg["BATCH_SIZE"], shuffle=True)
    val_loader = DataLoader(UnsupervisedEEGDataset(X_val),
                            batch_size=cfg["BATCH_SIZE"], shuffle=False)
    full_loader = DataLoader(UnsupervisedEEGDataset(segments),
                             batch_size=cfg["BATCH_SIZE"], shuffle=False)

    # ---- get the model: load sweep model (default) or retrain --------
    if args.retrain:
        n_epochs = args.full_epochs if args.full_epochs is not None else cfg["FULL_EPOCHS"]
        print(f"Retraining K={K} from scratch for up to {n_epochs} epochs...")
        model = build_model(cfg, K, device)
        final_history, reason = train_model(
            model, train_loader, val_loader, cfg, n_epochs, device,
            checkpoint_path=os.path.join(args.out_dir, f"ckpt_final_K{K}.pt"),
            verbose=True)
        print(f"stop: {reason}")
    else:
        if str(K) not in bundle["sweep_state"] and K not in bundle["sweep_state"]:
            raise SystemExit(
                f"K={K} not in the sweep ({list(bundle['sweep_state'].keys())}). "
                f"Re-run with --retrain to fit it, or pick a swept K.")
        key = K if K in bundle["sweep_state"] else str(K)
        print(f"Loading sweep model for K={K} (trained {cfg['SWEEP_EPOCHS']} epochs)...")
        model = build_model(cfg, K, device)
        model.load_state_dict(bundle["sweep_state"][key])
        final_history = bundle["sweep_histories"][key]

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # ---- prune rarely-used codebook entries  (notebook cell 13) ------
    PRUNE_PCT = cfg["PRUNE_PCT"]

    @torch.no_grad()
    def token_usage(m, loader):
        counts = torch.zeros(m.codebook_size)
        m.eval()
        for batch in loader:
            z_e, _, _, _ = m.encode(batch.to(device))
            _, tokens, _ = m.quantizer(z_e)
            counts += torch.bincount(tokens, minlength=m.codebook_size).float().cpu()
        return counts

    usage_counts = token_usage(model, full_loader)
    total_windows = usage_counts.sum().clamp(min=1)
    usage_pct = 100.0 * usage_counts / total_windows
    order = torch.argsort(usage_pct, descending=True)
    survivors = usage_pct >= PRUNE_PCT

    if PRUNE_PCT and PRUNE_PCT > 0:
        cb_device = model.quantizer.codebook.device
        dead_idx = torch.where(~survivors)[0].to(cb_device)
        surv_idx = torch.where(survivors)[0].to(cb_device)
        if len(surv_idx) == 0:
            print("No entry meets the threshold; skipping prune.")
            pruned_tokens = []
        elif len(dead_idx) == 0:
            print(f"All {model.codebook_size} entries used >= {PRUNE_PCT}%; nothing to prune.")
            pruned_tokens = []
        else:
            model.quantizer.codebook[dead_idx] = 1e9
            model.quantizer.ema_count[dead_idx] = 0.0
            pruned_tokens = dead_idx.cpu().tolist()
            print(f"Pruned {len(pruned_tokens)} / {model.codebook_size} entries "
                  f"(used < {PRUNE_PCT}% of {int(total_windows)} windows).")
        after = token_usage(model, full_loader)
        assert int(after.sum()) == int(total_windows), "window count changed!"
        assert after[torch.where(~survivors)[0].cpu()].sum() == 0, "a pruned entry still used!"
        usage_counts = after
        print(f"Active tokens after prune: {int((after > 0).sum())}")
    else:
        pruned_tokens = []
        print("Pruning disabled (PRUNE_PCT = 0).")

    prune_info = {
        "prune_pct": PRUNE_PCT,
        "pruned_tokens": pruned_tokens,
        "surviving_tokens": torch.where(survivors)[0].tolist(),
        "usage_counts": usage_counts.int().tolist(),
    }

    # ---- codebook descriptors + band labels  (notebook cell 17) ------
    cb_desc, cb_counts = model.compute_codebook_descriptors(full_loader, device=device)
    if cfg["FIXED_BANDS"] is not None:
        low_f = np.array([lo for (lo, hi) in cfg["FIXED_BANDS"]])
        high_f = np.array([hi for (lo, hi) in cfg["FIXED_BANDS"]])
        freq_labels = [f"{lo:.0f}-{hi:.0f}Hz" for (lo, hi) in cfg["FIXED_BANDS"]]
    else:
        from models.sinc_convolution import constrained_bandpass
        b1 = model.encoder.Sinc_Conv.filt_b1.detach().cpu()
        band = model.encoder.Sinc_Conv.filt_band.detach().cpu()
        low_f, high_f = constrained_bandpass(
            b1, band, fs=cfg["FS"], min_freq=cfg["FREQ_LOW"],
            min_band=cfg["MIN_BAND"], cutoff=cfg["CUTOFF"], freq_high=cfg["FREQ_HIGH"])
        low_f, high_f = low_f.numpy(), high_f.numpy()
        centers = (low_f + high_f) / 2
        freq_labels = [f"{f:.1f}Hz" for f in centers]

    # ---- tokenise the continuous record(s)  (notebook cell 26) -------
    step = cfg["TOKENISE_STEP_SAMPLES"]
    all_seqs, all_offsets = [], []
    for rec in record_list(records):
        toks, offs = model.tokenise_continuous(rec, step_samples=step, device=device)
        all_seqs.append(toks.numpy().astype(np.int64))
        all_offsets.append(offs.astype(np.int64))
    N_records = len(all_seqs)
    seq_lengths = np.array([len(t) for t in all_seqs])
    print(f"Tokenised {N_records} record(s): total {seq_lengths.sum()} tokens "
          f"(one per segment).")

    # =================================================================
    # SAVE OUTPUTS  (notebook cells 30-33, unchanged formats)
    # =================================================================
    C = cfg["C"]
    EFFECTIVE_N_FILT = cfg["EFFECTIVE_N_FILT"]
    SAVE_PREFIX = f"{cfg['subid']}_{cfg['DESCRIPTOR']}_K{K}"
    SAVE_DIR = args.out_dir

    # 1. model (.pt)
    model_config = {
        "descriptor": cfg["DESCRIPTOR"], "codebook_size": K,
        "num_EEG_Channels": C, "N_filt": EFFECTIVE_N_FILT, "Filt_dim": cfg["FILT_DIM"],
        "window_samples": cfg["SEGMENT_SAMPLES"], "fs": cfg["FS"], "cutoff": cfg["CUTOFF"],
        "freq_low": cfg["FREQ_LOW"], "freq_high": cfg["FREQ_HIGH"], "min_band": cfg["MIN_BAND"],
        "fixed_bands": cfg["FIXED_BANDS"], "codebook_dim": cfg["CODEBOOK_DIM"],
        "decoder_hidden": cfg["DECODER_HIDDEN"], "encoder_hidden": cfg["ENCODER_HIDDEN"],
        "segment_sec": cfg["SEGMENT_SEC"],
        "tokenise_step_samples": cfg["TOKENISE_STEP_SAMPLES"],
    }
    model_path = os.path.join(SAVE_DIR, SAVE_PREFIX + "_model.pt")
    torch.save({"state_dict": model.state_dict(), "config": model_config}, model_path)
    print("saved", model_path)

    # 2. statistics (.npz)
    @torch.no_grad()
    def split_metrics(loader):
        model.eval()
        err = tot = rec = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(device)
            dh, dt, _, losses = model(batch)
            err += relative_recon_error(dh, dt)
            tot += losses["total"].item()
            rec += losses["recon"].item()
            n += 1
        return err / n, tot / n, rec / n

    train_err, train_tot, train_rec = split_metrics(train_loader)
    val_err, val_tot, val_rec = split_metrics(val_loader)
    diag = model.quantizer.codebook_diagnostics()
    print(f"TRAIN rel_err={train_err*100:.2f}%  VAL rel_err={val_err*100:.2f}%  "
          f"active={diag['active_entries']}/{K}")

    stats_path = os.path.join(SAVE_DIR, SAVE_PREFIX + "_stats.npz")
    np.savez(
        stats_path,
        hist_train_loss=np.array(final_history["train_loss"]),
        hist_val_loss=np.array(final_history["val_loss"]),
        hist_train_recon=np.array(final_history["train_recon"]),
        hist_val_recon=np.array(final_history["val_recon"]),
        hist_val_err=np.array(final_history["val_err"]),
        hist_active=np.array(final_history["active"]),
        hist_dist=np.array(final_history["dist"]),
        final_train_rel_err=train_err, final_val_rel_err=val_err,
        final_train_total=train_tot, final_val_total=val_tot,
        final_train_recon=train_rec, final_val_recon=val_rec,
        active_entries=diag["active_entries"], mean_norm_dist=diag["mean_norm_dist"],
        mean_cosine_sim=diag["mean_cosine_sim"],
        ema_count=diag["ema_count"].numpy(), usage_fraction=diag["usage_fraction"].numpy(),
        sweep_K=np.array(cfg["CODEBOOK_SIZES"]),
        sweep_best_err=np.array(bundle["sweep_best_err"]),
        descriptor=cfg["DESCRIPTOR"], codebook_size=K,
        fs=cfg["FS"], n_channels=C, n_filt=EFFECTIVE_N_FILT,
    )
    print("saved", stats_path)

    # 3. codebook analysis (.npz)
    cb_desc_np = cb_desc.numpy()
    cb_counts_np = cb_counts.numpy()
    cb_path = os.path.join(SAVE_DIR, SAVE_PREFIX + "_codebook.npz")
    save_kwargs = dict(
        descriptor=cfg["DESCRIPTOR"], codebook_size=K,
        n_filt=EFFECTIVE_N_FILT, n_channels=C,
        counts=cb_counts_np, descriptor_raw=cb_desc_np,
        band_low=np.array(low_f, dtype=float), band_high=np.array(high_f, dtype=float),
        band_labels=np.array(freq_labels),
        codebook_vectors=model.quantizer.codebook.detach().cpu().numpy(),
        prune_pct=prune_info["prune_pct"],
        pruned_tokens=np.array(prune_info["pruned_tokens"], dtype=int),
        surviving_tokens=np.array(prune_info["surviving_tokens"], dtype=int),
        usage_counts=np.array(prune_info["usage_counts"], dtype=int),
    )
    if len(prune_info["pruned_tokens"]) > 0:
        save_kwargs["codebook_vectors"][np.array(prune_info["pruned_tokens"], dtype=int)] = 0.0

    if cfg["DESCRIPTOR"] == "amplitude":
        save_kwargs["amplitude_maps"] = cb_desc_np.reshape(K, EFFECTIVE_N_FILT, C)
    else:
        mats_all = np.zeros((K, EFFECTIVE_N_FILT, C, C), dtype=float)
        for k in range(K):
            if cb_counts_np[k] > 0:
                m, _ = token_band_matrices(cb_desc_np[k], cfg["DESCRIPTOR"],
                                           EFFECTIVE_N_FILT, C)
                mats_all[k] = m
        save_kwargs["connectivity_matrices"] = mats_all
    np.savez(cb_path, **save_kwargs)
    print("saved", cb_path)

    # 4. token representation (.npz)
    tokens_obj = np.empty(N_records, dtype=object)
    offsets_obj = np.empty(N_records, dtype=object)
    for i in range(N_records):
        tokens_obj[i] = all_seqs[i]
        offsets_obj[i] = all_offsets[i]
    flat_tokens = np.concatenate(all_seqs) if N_records else np.array([], np.int64)
    flat_offsets = np.concatenate(all_offsets) if N_records else np.array([], np.int64)
    flat_record = np.concatenate(
        [np.full(len(all_seqs[i]), i, np.int64) for i in range(N_records)]
    ) if N_records else np.array([], np.int64)
    tok_path = os.path.join(SAVE_DIR, SAVE_PREFIX + "_tokens.npz")
    np.savez(
        tok_path,
        tokens_per_record=tokens_obj, offsets_per_record=offsets_obj,
        seq_lengths=seq_lengths,
        flat_tokens=flat_tokens, flat_offsets=flat_offsets, flat_record_id=flat_record,
        codebook_size=K, fs=cfg["FS"],
        segment_sec=cfg["SEGMENT_SEC"], segment_samples=cfg["SEGMENT_SAMPLES"],
        tokenise_step_samples=(step if step is not None else -1),
        surviving_tokens=np.array(prune_info["surviving_tokens"], dtype=int),
        prune_pct=prune_info["prune_pct"],
    )
    print("saved", tok_path)

    # =================================================================
    # FIGURES — only the three final fitting visualisations
    # =================================================================
    # (a) training curves  (notebook cell 15)
    h = final_history
    epochs = range(1, len(h["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, h["train_loss"], label="train loss")
    axes[0].plot(epochs, h["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title(f"Training vs validation loss (K={K}, {cfg['DESCRIPTOR']})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    best_ep = int(np.argmin(h["val_loss"])) + 1
    axes[0].axvline(best_ep, color="black", ls=":", alpha=0.6)
    axes[1].plot(epochs, [e * 100 for e in h["val_err"]], color="seagreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val relative error (%)")
    axes[1].set_title("Reconstruction error")
    axes[1].grid(True, alpha=0.3)
    ax3 = axes[2]
    ax3b = ax3.twinx()
    l1 = ax3.plot(epochs, h["active"], color="steelblue", label="active entries")
    l2 = ax3b.plot(epochs, h["dist"], color="tomato", label="mean norm dist")
    ax3.axhline(K, color="steelblue", ls=":", alpha=0.4)
    ax3b.axhline(0.05, color="tomato", ls=":", alpha=0.4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Active entries", color="steelblue")
    ax3b.set_ylabel("Mean norm dist", color="tomato")
    ax3.set_title("Codebook health")
    ax3.legend(l1 + l2, [x.get_label() for x in l1 + l2], loc="center right", fontsize=8)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_train = os.path.join(SAVE_DIR, SAVE_PREFIX + "_training.png")
    plt.savefig(fig_train, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", fig_train)

    # (b) codebook pruning usage  (notebook cell 13 bar plot)
    nz = order[usage_counts[order] > 0] if PRUNE_PCT else order[usage_counts[order] > 0]
    plt.figure(figsize=(10, 3.5))
    pcts = (100.0 * usage_counts[nz] / usage_counts.sum().clamp(min=1)).cpu().numpy()
    colors = ["steelblue" if p >= PRUNE_PCT else "indianred" for p in pcts]
    plt.bar(range(len(nz)), pcts, color=colors)
    plt.axhline(PRUNE_PCT, color="black", ls="--", lw=1, label=f"PRUNE_PCT = {PRUNE_PCT}%")
    plt.xlabel("codebook entry (sorted by usage)")
    plt.ylabel("% of segments")
    plt.title("Token usage distribution (blue = kept, red = pruned)")
    plt.legend()
    plt.tight_layout()
    fig_prune = os.path.join(SAVE_DIR, SAVE_PREFIX + "_pruning.png")
    plt.savefig(fig_prune, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", fig_prune)

    # (c) Sinc filter bands  (notebook cell 28)
    plt.figure(figsize=(8, 5))
    for i, (lo, hi) in enumerate(zip(low_f, high_f)):
        plt.plot([float(lo), float(hi)], [i, i], color="red", lw=2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Filter index")
    plt.title("Filter bands (fixed)" if cfg["FIXED_BANDS"] is not None
              else "Learned Sinc filter bands")
    plt.xticks(np.arange(0, cfg["CUTOFF"] + 5, 5))
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig_bands = os.path.join(SAVE_DIR, SAVE_PREFIX + "_filterbands.png")
    plt.savefig(fig_bands, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", fig_bands)

    print(f"\nAll outputs written with prefix: {SAVE_PREFIX}")


if __name__ == "__main__":
    main()
