"""
Dataset adapters: abstract interface + adapter registry.

Each dataset source (MeshCoder, Infinigen, Objaverse, ShapeNet, etc.)
implements a concrete adapter that knows how to:
  1. Load raw data from its native format
  2. Produce an iterable of SFTSample records

The registry allows config-driven instantiation:
    adapter_cls = AdapterRegistry.get("MeshCoderAdapter")
    adapter = adapter_cls(config)
    samples = adapter.load()
"""
from __future__ import annotations

import abc
import logging
from typing import Iterator

from core.models import SFTSample

logger = logging.getLogger(__name__)


class BaseAdapter(abc.ABC):
    """
    Abstract base class for all dataset adapters.
    
    Subclasses must implement:
        - load() -> Iterator[SFTSample]
        - name property
    
    Optionally override:
        - validate_config() for source-specific config checks
        - estimate_count() for progress bars
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.validate_config()
    
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier for this adapter (e.g., 'meshcoder')."""
        ...
    
    @abc.abstractmethod
    def load(self) -> Iterator[SFTSample]:
        """
        Yield SFTSample records from this data source.
        
        At minimum, each sample must have:
            - object_id, category, code, gt_mesh_path
            - metrics (at least execution_success and rlvr_reward)
        
        image_paths and num_views are set later by the ViewSampler
        if the adapter doesn't provide pre-rendered views.
        """
        ...
    
    def validate_config(self):
        """Override to check source-specific config requirements."""
        pass
    
    def estimate_count(self) -> int | None:
        """Return estimated number of samples, or None if unknown."""
        return None


class AdapterRegistry:
    """
    Simple registry mapping adapter class names to classes.
    
    Usage:
        @AdapterRegistry.register
        class MeshCoderAdapter(BaseAdapter):
            ...
        
        cls = AdapterRegistry.get("MeshCoderAdapter")
    """
    _registry: dict[str, type[BaseAdapter]] = {}
    
    @classmethod
    def register(cls, adapter_cls: type[BaseAdapter]) -> type[BaseAdapter]:
        cls._registry[adapter_cls.__name__] = adapter_cls
        logger.debug(f"Registered adapter: {adapter_cls.__name__}")
        return adapter_cls
    
    @classmethod
    def get(cls, name: str) -> type[BaseAdapter]:
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise KeyError(
                f"Unknown adapter '{name}'. Available: {available}"
            )
        return cls._registry[name]
    
    @classmethod
    def list_adapters(cls) -> list[str]:
        return list(cls._registry.keys())
