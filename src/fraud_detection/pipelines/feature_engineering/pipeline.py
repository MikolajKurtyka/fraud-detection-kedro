"""feature_engineering: 03_primary -> 04_feature -> 05_model_input.

1. Stateless features for every split (time, normalized D, amount, email parts, uid).
2. Fit encoders (group stats, frequencies, category encodings) on train only.
3. Apply the same encoders to train, valid and test, then select the model columns.

The inference path for a single API request is the same chain:
add_stateless_features -> apply_feature_encoders -> build_model_input.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    add_stateless_features,
    apply_feature_encoders,
    build_model_input,
    fit_feature_encoders,
)

SPLITS = ("train", "valid", "test")


def create_pipeline(**kwargs) -> Pipeline:
    nodes = []
    for split in SPLITS:
        nodes += [
            node(
                add_stateless_features,
                inputs=[f"prm_{split}", "params:feature_engineering"],
                outputs=f"{split}_stateless",
                name=f"add_stateless_features_{split}",
            ),
            node(
                apply_feature_encoders,
                inputs=[f"{split}_stateless", "feature_encoders"],
                outputs=f"feat_{split}",
                name=f"apply_feature_encoders_{split}",
            ),
            node(
                build_model_input,
                inputs=[
                    f"feat_{split}",
                    "feature_encoders",
                    "params:feature_engineering",
                    "params:target",
                ],
                outputs=f"mdl_in_{split}",
                name=f"build_model_input_{split}",
            ),
        ]
    nodes.append(
        node(
            fit_feature_encoders,
            inputs=["train_stateless", "params:feature_engineering", "params:target"],
            outputs="feature_encoders",
            name="fit_feature_encoders",
        )
    )
    return Pipeline(nodes, tags="features")
