"""Historical array preparation; no data files or target values are read."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from .history_summary import features


def prepare_history_arrays(values, controls, *, engineering_limits, advance_scale):
    """values: B,24,6,11; controls: B,24,6,6. Input already obeys upload cutoff."""
    values = np.asarray(values, np.float32)
    controls = np.asarray(controls, np.float32)
    assert values.shape[1:] == (24, 6, 11)
    assert controls.shape == (*values.shape[:3], 6)
    limits = np.asarray(engineering_limits, dtype=np.float32)
    summary, anchor, _, _ = features(values, controls, limits)
    normalized = values.transpose(0, 2, 1, 3)/limits
    normalized[..., 3:9] -= (anchor[..., 3:9]/limits[3:9])[:, :, None, :]
    mask = np.isfinite(values).transpose(0, 2, 1, 3)
    context = controls.transpose(0, 2, 1, 3).copy()
    context[..., 0] /= advance_scale
    sequence = np.concatenate([np.nan_to_num(normalized, nan=0), mask, context], -1)
    assert sequence.shape[1:] == (6, 24, 28) and np.isfinite(sequence).all()
    return dict(sequence=sequence.astype(np.float32), summary_raw=summary,
                anchor_raw=anchor)


class FixedSchemaImputer:
    def __init__(self, medians, indicator_columns):
        self.medians = medians
        self.indicator_columns = indicator_columns
        self.n_features_in_ = 167

    def transform(self, x):
        return np.concatenate([np.where(np.isnan(x), self.medians, x),
                               np.isnan(x[:, self.indicator_columns])], axis=-1)


class FixedSchemaPreprocessor:
    """Train-only medians/scaling with an explicitly fixed indicator schema."""
    def fit(self, training_raw, indicator_columns, *, missing_column_values):
        x = np.asarray(training_raw).reshape(-1, 167)
        self.indicator_columns = np.asarray(indicator_columns, dtype=np.int64)
        assert len(self.indicator_columns) == 110
        observed_columns = np.isfinite(x).any(axis=0)
        self.medians = np.broadcast_to(missing_column_values, (x.shape[1],)).astype(float).copy()
        self.medians[observed_columns] = np.nanmedian(x[:, observed_columns], axis=0)
        self.imputer = FixedSchemaImputer(self.medians, self.indicator_columns)
        self.scaler = StandardScaler().fit(self.imputer.transform(x))
        return self

    def transform(self, raw):
        shape = raw.shape[:-1]
        out = self.scaler.transform(self.imputer.transform(np.asarray(raw).reshape(-1, 167)))
        return out.astype(np.float32).reshape(*shape, 277)

    def as_pipeline_preprocessing(self):
        return dict(imputer=self.imputer, scaler=self.scaler)
