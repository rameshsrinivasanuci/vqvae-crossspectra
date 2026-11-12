from torch.utils.data import Dataset
import torch
import numpy as np


class UnsupervisedEEGDataset(Dataset):
    def __init__(self, X): # X: Input data of shape (B, C, T)

        # lets just keep it as numpy for now lol
        self.X = X

    def __len__(self): # to obtain the number of x in X.
        return len(self.X)

    def __getitem__(self, idx): # to index an x in X.
        return torch.from_numpy(self.X[idx]).float()


# ======================================================================
# Continuous-record segmentation (chunk = instance)
# ======================================================================
#
# These replace the earlier "pre-cut (N, C, T) + sub-window each instance"
# front end. A continuous recording is cut into non-overlapping segments of a
# single chosen length, and each whole segment is one training instance (one
# window) — there is no sub-chunking. The segment length is the only scale
# parameter, so sweeping it lets you compare the model at different timescales.

def segment_record(record, segment_samples, step_samples=None):
    """
    Cut one continuous recording into fixed-length segments.

    Parameters
    ----------
    record          : array (C, T) — one continuous multichannel recording.
    segment_samples : int — segment length in samples (== the model window L_w).
    step_samples    : int or None — hop between segment starts. None gives
                      non-overlapping segments (step == segment_samples).

    Returns
    -------
    segments : array (N_seg, C, segment_samples) — one whole segment per row.
    starts   : array (N_seg,) — sample index where each segment begins. This is
               the timing map you keep for downstream (hypnogram / Markov) work.

    The ragged tail shorter than one segment is dropped.
    """
    record = np.asarray(record)
    C, T   = record.shape
    step   = int(segment_samples if step_samples is None else step_samples)
    starts = list(range(0, T - segment_samples + 1, step))
    segments = np.stack([record[:, s:s + segment_samples] for s in starts]) \
               if starts else np.empty((0, C, segment_samples), record.dtype)
    return segments, np.asarray(starts, dtype=np.int64)


def segment_records(records, segment_samples, step_samples=None):
    """
    Segment one recording or a list of recordings into instances.

    Parameters
    ----------
    records : array (C, T), or array (N, C, T), or list of (C, T) arrays.
              A single 2-D array is treated as one recording; a 3-D array or a
              list is treated as several recordings.
    segment_samples, step_samples : see segment_record.

    Returns
    -------
    segments  : array (N_seg_total, C, segment_samples) — all segments stacked.
    starts    : array (N_seg_total,) — segment start sample WITHIN its own record.
    record_id : array (N_seg_total,) — which recording each segment came from.

    record_id lets the split be done by recording (no leakage across records);
    with a single recording it is all zeros and the split falls back to a
    contiguous block (see split_segments).
    """
    # normalise input to a list of (C, T) recordings
    if isinstance(records, np.ndarray) and records.ndim == 2:
        rec_list = [records]
    elif isinstance(records, np.ndarray) and records.ndim == 3:
        rec_list = [records[i] for i in range(records.shape[0])]
    else:
        rec_list = list(records)

    seg_parts, start_parts, id_parts = [], [], []
    for r, rec in enumerate(rec_list):
        segs, starts = segment_record(rec, segment_samples, step_samples)
        if len(segs) == 0:
            continue
        seg_parts.append(segs)
        start_parts.append(starts)
        id_parts.append(np.full(len(segs), r, dtype=np.int64))

    if not seg_parts:
        C = rec_list[0].shape[0] if rec_list else 0
        empty = np.empty((0, C, segment_samples))
        return empty, np.array([], np.int64), np.array([], np.int64)

    segments  = np.concatenate(seg_parts, axis=0)
    starts    = np.concatenate(start_parts, axis=0)
    record_id = np.concatenate(id_parts, axis=0)
    return segments, starts, record_id


def split_segments(record_id, val_fraction=0.2, seed=42):
    """
    Leakage-safe train/validation split over segments.

    If there is more than one recording, whole recordings are held out for
    validation (split by record). If there is only one recording, a single
    CONTIGUOUS tail block of segments is held out, because adjacent segments of
    one record are autocorrelated and a random split would leak.

    Returns
    -------
    train_mask, val_mask : boolean arrays over the segments.
    """
    record_id = np.asarray(record_id)
    n_seg     = len(record_id)
    uniq      = np.unique(record_id)

    if len(uniq) > 1:
        rng       = np.random.default_rng(seed)
        perm      = rng.permutation(uniq)
        n_val_rec = max(1, int(round(len(uniq) * val_fraction)))
        val_recs  = set(perm[:n_val_rec].tolist())
        val_mask  = np.array([rid in val_recs for rid in record_id])
    else:
        # single record: hold out a contiguous tail block
        n_val    = max(1, int(round(n_seg * val_fraction)))
        val_mask = np.zeros(n_seg, dtype=bool)
        val_mask[n_seg - n_val:] = True

    return ~val_mask, val_mask