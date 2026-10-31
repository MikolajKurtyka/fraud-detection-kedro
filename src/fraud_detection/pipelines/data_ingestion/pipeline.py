"""data_ingestion: 01_raw CSVs -> 02_intermediate parquet (one table per split)."""

from kedro.pipeline import Pipeline, node

from .nodes import merge_transaction_identity, normalize_identity_columns, reduce_memory


def _split_nodes(split: str) -> list:
    return [
        node(
            normalize_identity_columns,
            inputs=f"raw_{split}_identity",
            outputs=f"{split}_identity_normalized",
            name=f"normalize_{split}_identity",
        ),
        node(
            merge_transaction_identity,
            inputs=[
                f"raw_{split}_transaction",
                f"{split}_identity_normalized",
                "params:data_ingestion",
            ],
            outputs=f"{split}_merged",
            name=f"merge_{split}",
        ),
        node(
            reduce_memory,
            inputs=[f"{split}_merged", "params:data_ingestion"],
            outputs=f"int_{split}",
            name=f"reduce_memory_{split}",
        ),
    ]


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(_split_nodes("train") + _split_nodes("test"), tags="ingestion")
