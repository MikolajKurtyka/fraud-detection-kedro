"""data_validation: 02_intermediate -> 08_reporting/data_validation.json.

pandera checks on int_train / int_test (types, IDs, value ranges, required columns, time
order, fraud rate) and train/test parity (columns, dtypes, test after train, null rates).
With ``fail_on_error`` the run stops on the first failed validation, listing every failed
check. In ``kedro run`` it runs as soon as both tables exist, before the column filter is
fitted (which needs the time split first).
"""

from kedro.pipeline import Pipeline, node

from .nodes import validate_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                validate_data,
                inputs=[
                    "int_train",
                    "int_test",
                    "params:data_ingestion",
                    "params:data_validation",
                    "params:target",
                ],
                outputs="data_validation_report",
                name="validate_data",
            )
        ],
        tags="validation",
    )
