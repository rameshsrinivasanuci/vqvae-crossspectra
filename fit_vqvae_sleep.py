#!/usr/bin/env python3
"""
fit_vqvae_sleep.py — end-to-end fit/finalise of the Hilbert VQ-VAE from a notebook
handoff bundle, for sleep recordings with a hypnogram.

Same as fit_vqvae.py, except that
  * the data come from an alldata/<subject>_200Hz.npz file (written by
    epctl_zips_to_200Hz_npz.ipynb), using the PICK_CHANNELS and DATA_SCALE saved
    in the bundle, and
  * every segment (instance) and every token window gets a sleep-stage label by
    majority vote over the hypnogram, saved in <prefix>_sleep_staging.npz and
    added to <prefix>_tokens.npz.

Workflow
--------
1. Run steps 1-3 of hilbert_microstate_sweep_sleep.ipynb (config, segment,
   codebook sweep). Its final cell writes a handoff bundle (<prefix>_sweep.pt) holding the
   full config and the trained sweep model for every K.
2. Inspect the elbow, pick K, then run this script:

       python fit_vqvae_sleep.py --sweep run_sweep.pt --codebook-size 24

   By default it LOADS the already-trained sweep model for that K (trained for
   SWEEP_EPOCHS) — no retraining, no re-specification of parameters. Pass
   --retrain to fit a fresh model at K for FULL_EPOCHS instead.

The script re-segments the data from the saved config (deterministic), so the
original DATA_PATH must be reachable (a relative path is resolved from the
directory you run the script in). It saves the four output files exactly as
fit_vqvae.py does, plus <prefix>_sleep_staging.npz, and the same three figures:
the final training curves, the codebook-pruning usage plot, and the Sinc filter bands.

Sleep-stage labels
------------------
Each window [start, start + SEGMENT_SAMPLES) takes the hypnogram stage covering
the most of its samples (samples outside the hypnogram count as '?'). With
30 s segments starting at 0 the labels equal the hypnogram exactly. On a tie,
the window takes the label of the previous window (in time) if that label is
one of the tied stages, otherwise the tied stage occurring first in the window;
every tie is reported with a warning and flagged in the saved files.
Codes: W=0, N1=1, N2=2, N3=3, R=4; L (lights), '?' and any other label = -1.
"""

import os
import json
import warnings
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
    """Load the sleep recording from the npz at DATA_PATH, with the channel
    selection and scaling recorded in the bundle (exactly as the notebook did).
    Returns the (C, T) record and the hypnogram arrays."""
    sleep = np.load(cfg["DATA_PATH"])
    fs_file = int(round(float(sleep["sfreq"])))
    if fs_file != cfg["FS"]:
        raise ValueError(f"FS in {cfg['DATA_PATH']} is {fs_file}, bundle says {cfg['FS']}")

    ch_names_all = sleep["ch_names"].tolist()
    picks = cfg.get("PICK_CHANNELS")
    if picks is None:
        pick_idx = np.arange(len(ch_names_all))
    else:
        pick_idx = np.array([ch_names_all.index(name) for name in picks])

    records = sleep["data"][pick_idx].astype(np.float32)
    records = records * np.float32(cfg.get("DATA_SCALE", 1.0))
    if records.shape[0] != cfg["C"]:
        raise ValueError(f"{records.shape[0]} channels loaded, bundle says C={cfg['C']}")

    hyp = {
        "labels": sleep["stage_labels"],
        "onset": sleep["stage_onset_sample"],
        "duration": sleep["stage_duration_samples"],
        "ch_names": [ch_names_all[i] for i in pick_idx],
    }
    return records, hyp


# ======================================================================
# sleep-stage labels for windows (majority vote over the hypnogram)
# ======================================================================

STAGE_CODE_NAMES = np.array(["W", "N1", "N2", "N3", "R"])
STAGE_TO_CODE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def sample_stage_labels(hyp, n_samples):
    """Per-sample stage index into label_names; samples outside the hypnogram are '?'."""
    label_names = np.array(sorted(set(hyp["labels"].tolist())) + ["?"])
    label_index = {name: i for i, name in enumerate(label_names)}
    sample_label = np.full(n_samples, label_index["?"], dtype=np.int16)
    for name, onset, duration in zip(hyp["labels"], hyp["onset"], hyp["duration"]):
        start = int(onset)
        stop = min(int(onset + duration), n_samples)
        sample_label[start:stop] = label_index[name]
    return sample_label, label_names


def stage_labels_for_windows(window_starts, window_samples, sample_label,
                             label_names, fs, what):
    """Majority-vote stage for each window [start, start + window_samples),
    counted in samples. Tie -> label of the previous window (in time) if it is one
    of the tied stages, otherwise the tied stage occurring first in the window.
    Every tie is reported with a warning. Returns (labels, codes, is_tie)."""
    window_starts = np.asarray(window_starts)
    n_windows = len(window_starts)
    n_labels = len(label_names)
    order = np.argsort(window_starts, kind="stable")

    chosen = np.zeros(n_windows, dtype=np.int16)
    is_tie = np.zeros(n_windows, dtype=bool)
    tie_details = []
    previous = None
    for w in order:
        start = int(window_starts[w])
        window = sample_label[start:start + window_samples]
        counts = np.bincount(window, minlength=n_labels)
        tied = np.flatnonzero(counts == counts.max())
        if len(tied) == 1:
            chosen[w] = tied[0]
        else:
            is_tie[w] = True
            if previous is not None and previous in tied:
                chosen[w] = previous
                rule = "previous window"
            else:
                first_positions = [int(np.argmax(window == t)) for t in tied]
                chosen[w] = tied[int(np.argmin(first_positions))]
                rule = "first in window"
            tie_details.append((int(w), start, label_names[tied].tolist(),
                                label_names[chosen[w]], rule))
        previous = chosen[w]

    labels = label_names[chosen]
    codes = np.array([STAGE_TO_CODE.get(lab, -1) for lab in labels], dtype=np.int8)

    n_ties = len(tie_details)
    if n_ties > 0:
        warnings.warn(f"{what}: {n_ties} of {n_windows} windows had a tied majority "
                      f"vote and were labelled by the tie rule (see list below).")
        for w, start, tied_names, chosen_name, rule in tie_details[:20]:
            t0 = start / fs
            print(f"  tie: {what} {w} at {t0:.1f} s, tied {tied_names} -> "
                  f"{chosen_name} ({rule})")
        if n_ties > 20:
            print(f"  ... {n_ties - 20} more")
    return labels, codes, is_tie


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
    records, hyp = load_records(cfg)
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

    # ---- sleep-stage label for each segment --------------------------
    total_T = records.shape[1]
    sample_label, label_names = sample_stage_labels(hyp, total_T)
    seg_starts_np = np.asarray(seg_starts)
    seg_stage_labels, seg_stage_codes, seg_stage_tie = stage_labels_for_windows(
        seg_starts_np, cfg["SEGMENT_SAMPLES"], sample_label, label_names,
        cfg["FS"], "segment")
    names, counts = np.unique(seg_stage_labels, return_counts=True)
    print("Segment stages: " + ", ".join(f"{n}={c}" for n, c in zip(names, counts))
          + f"  (ties: {int(seg_stage_tie.sum())})")
    if cfg["SEGMENT_SAMPLES"] == int(hyp["duration"][0]):
        n_compare = min(len(seg_starts_np), len(hyp["labels"]))
        aligned = np.array_equal(seg_starts_np[:n_compare], hyp["onset"][:n_compare])
        identical = np.array_equal(seg_stage_labels[:n_compare], hyp["labels"][:n_compare])
        print(f"Segment length = epoch length: starts aligned = {aligned}, "
              f"labels identical to hypnogram = {identical}")

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

    # ---- sleep-stage label for each token window ----------------------
    tok_stage_labels, tok_stage_codes, tok_stage_tie = stage_labels_for_windows(
        all_offsets[0], cfg["SEGMENT_SAMPLES"], sample_label, label_names,
        cfg["FS"], "token")
    print(f"Token windows labelled by the tie rule: {int(tok_stage_tie.sum())}")

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
        stage_labels=tok_stage_labels, stage_codes=tok_stage_codes,
        stage_tie=tok_stage_tie, stage_code_names=STAGE_CODE_NAMES,
    )
    print("saved", tok_path)

    # 5. sleep staging matched to the segmentation (.npz)
    staging_path = os.path.join(SAVE_DIR, SAVE_PREFIX + "_sleep_staging.npz")
    np.savez(
        staging_path,
        # one entry per instance (segment), same order as the segments
        segment_starts=seg_starts_np,
        segment_samples=cfg["SEGMENT_SAMPLES"], segment_sec=cfg["SEGMENT_SEC"],
        segment_stage_labels=seg_stage_labels,
        segment_stage_codes=seg_stage_codes,
        segment_stage_tie=seg_stage_tie,
        segment_is_train=np.asarray(tr_mask), segment_is_val=np.asarray(val_mask),
        # one entry per token window
        token_offsets=all_offsets[0],
        token_stage_labels=tok_stage_labels,
        token_stage_codes=tok_stage_codes,
        token_stage_tie=tok_stage_tie,
        # code meaning and the original hypnogram
        stage_code_names=STAGE_CODE_NAMES,
        hypnogram_labels=hyp["labels"],
        hypnogram_onset_sample=hyp["onset"],
        hypnogram_duration_samples=hyp["duration"],
        tie_rule=np.array("majority vote in samples; tie -> previous window label if "
                          "tied, else tied stage occurring first in the window"),
        fs=cfg["FS"], subid=np.array(cfg["subid"]),
        source_file=np.array(cfg["DATA_PATH"]), ch_names=np.array(hyp["ch_names"]),
    )
    print("saved", staging_path)

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
