"""data_ingestion: 01_raw CSVs -> 02_intermediate parquet (one table per split).

Per split: type each raw table separately against the schema in
parameters_data_ingestion.yml, then left-join identity onto transactions.
The same nodes run for train and test; the target column is optional.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    merge_transaction_identity,
    normalize_identity_columns,
    type_identity,
    type_transaction,
)


def _split_nodes(split: str) -> list:
    return [
        node(
            type_transaction,
            inputs=[f"raw_{split}_transaction", "params:data_ingestion", "params:target"],
            outputs=f"{split}_transaction_typed",
            name=f"type_{split}_transaction",
        ),
        node(
            normalize_identity_columns,
            inputs=f"raw_{split}_identity",
            outputs=f"{split}_identity_normalized",
            name=f"normalize_{split}_identity",
        ),
        node(
            type_identity,
            inputs=[f"{split}_identity_normalized", "params:data_ingestion"],
            outputs=f"{split}_identity_typed",
            name=f"type_{split}_identity",
        ),
        node(
            merge_transaction_identity,
            inputs=[
                f"{split}_transaction_typed",
                f"{split}_identity_typed",
                "params:data_ingestion",
            ],
            outputs=f"int_{split}",
            name=f"merge_{split}",
        ),
    ]


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(_split_nodes("train") + _split_nodes("test"), tags="ingestion")
