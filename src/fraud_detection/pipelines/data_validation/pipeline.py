"""data_validation pipeline -- not implemented yet.

Planned nodes:
- pandera schema checks on int_train / int_test (types, ranges, null rates per column group)
- train/test column parity and TransactionDT ordering checks
- fail fast with a readable report in 08_reporting/data_validation.json
"""

from kedro.pipeline import Pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([])
