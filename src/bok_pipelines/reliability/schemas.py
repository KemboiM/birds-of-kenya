"""Table contracts checked with Pandera.

A table that breaks its contract either becomes a data-check issue (metadata) or stops the
pipeline (tables the pipelines write themselves, where a break means a bug).
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

PROBABILITY = Check.in_range(0.0, 1.0)

# One row for each training image: where it came from and under which license
METADATA = DataFrameSchema(
    {
        "file": Column(str, unique=True, nullable=False),
        "species": Column(str, nullable=False),
        "license": Column(str, nullable=False),
    },
    strict=False, coerce=True,
)

# One row for each label in an evaluation report
PER_LABEL = DataFrameSchema(
    {
        "label": Column(str, unique=True, nullable=False),
        "test_images": Column(int, Check.ge(0)),
        "precision": Column(float, PROBABILITY),
        "recall": Column(float, PROBABILITY),
        "f1": Column(float, PROBABILITY),
    },
    strict=False, coerce=True,
)

# One row for each bird found in a field photo by identify.py
FIELD_PREDICTIONS = DataFrameSchema(
    {
        "photo": Column(str, nullable=False),
        "bird_id": Column(int, Check.ge(1)),
        "predicted": Column(str, nullable=False),
        "identified": Column(bool),
        "confidence": Column(float, PROBABILITY),
        "detector_score": Column(float, PROBABILITY, nullable=True),
    },
    unique=["photo", "bird_id"], strict=False, coerce=True,
)


def schema_issues(schema, df, check_name):
    """Validate without stopping; returns (validated or original df, issue rows for data checks)."""
    try:
        return schema.validate(df, lazy=True), []
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        rows = []
        for r in fc.head(500).itertuples():
            where = f"row {int(r.index)}" if pd.notna(r.index) else "table"
            rows.append(("", "", f"{check_name}: column {r.column}: {r.check} failed at {where} "
                                 f"(value {r.failure_case!r})"))
        if len(fc) > 500:
            rows.append(("", "", f"{check_name}: {len(fc) - 500} more failures not listed"))
        return df, rows
