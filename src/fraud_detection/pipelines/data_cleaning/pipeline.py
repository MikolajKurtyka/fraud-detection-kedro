"""data_cleaning: 02_intermediate -> 03_primary.

1. Time-split int_train into train (earlier days) and valid (later days).
2. Fit the column filter on train only -> column_filter.json.
3. Apply the same filter to train, valid and test.
"""

from kedro.pipeline import Pipeline, node

from .nodes import apply_column_filter, fit_column_filter, split_by_time


def create_pipeline(**kwargs) -> Pipeline:
    apply_nodes = [
        node(
            apply_column_filter,
            inputs=[source, "column_filter"],
            outputs=f"prm_{split}",
            name=f"apply_column_filter_{split}",
        )
        for split, source in (
            ("train", "train_split"),
            ("valid", "valid_split"),
            ("test", "int_test"),
        )
    ]
    return Pipeline(
        [
            node(
                split_by_time,
                inputs=["int_train", "params:data_cleaning", "params:target"],
                outputs=["train_split", "valid_split"],
                name="split_by_time",
            ),
            node(
                fit_column_filter,
                inputs=["train_split", "params:data_cleaning", "params:target"],
                outputs="column_filter",
                name="fit_column_filter",
            ),
            *apply_nodes,
        ],
        tags="cleaning",
    )
