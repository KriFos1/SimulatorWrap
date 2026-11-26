"""PET's labelled seeds for scalar simulator summaries; columns are independent objectives."""

import numpy as np


def scalar_seeds(weights, reports, datatypes):
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("VJP report and datatype labels must be unique")
    cells, vectors = [], []
    for label, row in weights.iterrows():
        for key, value in row.items():
            if value is None:
                continue
            if label not in reports or key not in datatypes:
                raise ValueError(f"Unknown VJP observation {(label, key)}")
            vector = np.atleast_1d(np.asarray(value, dtype=float))
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise ValueError(f"Expected finite scalar-observation seeds at {(label, key)}")
            cells.append((label, key))
            vectors.append(vector)
    if not cells:
        raise ValueError("VJP needs at least one observed cell")
    return cells, np.stack(vectors)
