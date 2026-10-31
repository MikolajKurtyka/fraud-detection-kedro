"""Stateless features + encoders fitted on train, applied identically to every split."""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
