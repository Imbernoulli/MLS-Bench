"""Lightweight generation package shim for the task-local d2Cache subset."""

from .klass import klass_generate
from .vanilla import vanilla_generate

__all__ = ["klass_generate", "vanilla_generate"]
