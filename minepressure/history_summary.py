"""Paper algorithm components; no experiment execution or stored observations."""
import warnings
import numpy as np
PARAMS = ['coal_stress_1', 'coal_stress_2', 'microseismic_freq', 'roof_sep_1', 'roof_sep_2', 'surround_deform_1', 'surround_deform_2', 'surround_deform_3', 'surround_deform_4', 'support_resistance_1', 'support_resistance_2']
CONTROLS = ['advance_speed', 'support_reinforcement', 'pressure_relief', 'roof_reinforcement', 'advance_speed_reduction', 'inspection_or_maintenance']

def features(w, c, engineering_limits):
    limits = np.asarray(engineering_limits, dtype=np.float32)
    """w=(B,24,6,11), c=(B,24,6,6); no labels or future controls."""
    finite = np.isfinite(w)
    last_idx = np.where(finite, np.arange(24)[None, :, None, None], -1).max(1)
    last = np.take_along_axis(w, np.maximum(last_idx, 0)[:, None], axis=1)[:, 0]
    last = np.where(last_idx >= 0, last, np.nan)
    recent = w[:, -8:]
    mask = np.isfinite(recent)
    t = np.arange(-7, 1, dtype=np.float32)[None, :, None, None]
    count = mask.sum(1)
    sx = np.where(mask, recent, 0).sum(1, dtype=np.float64)
    st = np.where(mask, t, 0).sum(1)
    stt = np.where(mask, t * t, 0).sum(1)
    stx = np.where(mask, recent * t, 0).sum(1, dtype=np.float64)
    den = count * stt - st * st
    slope = np.divide(count * stx - st * sx, den, out=np.full(last.shape, np.nan), where=(count >= 4) & (den > 0)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        blocks = [last, w[:, -1], np.nanmean(w[:, -4:], 1), np.nanstd(w[:, -4:], 1), np.nanmean(w, 1), np.nanstd(w, 1), np.nanmin(w, 1), np.nanmax(w, 1), slope, np.nanmean(w[:, -4:], 1) - np.nanmean(w[:, :4], 1), finite.mean(1).astype(np.float32)]
        shared = np.concatenate([np.nanmean(w[:, -1], 1), np.nanmax(w[:, -1], 1)], 1)
    blocks = [b / limits if j < 10 else b for j, b in enumerate(blocks)]
    context = np.concatenate([c[:, -1], c[:, -4:].mean(1), c.mean(1), c.max(1)], 2)
    x = np.concatenate(blocks + [context, np.repeat((shared / np.tile(limits, 2))[:, None], 6, 1)], 2)
    assert x.shape[2] == 167
    return (x.astype(np.float32), last, slope, last_idx)
