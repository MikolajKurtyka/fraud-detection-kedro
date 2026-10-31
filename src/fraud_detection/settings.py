"""Kedro project settings. kedro-mlflow hooks register themselves via entry points."""

from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc

CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # Allows ${oc.env:VAR,default} in mlflow.yml so Docker/CI can point at another server.
    "custom_resolvers": {"oc.env": oc.env},
}
