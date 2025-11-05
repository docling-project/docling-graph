"""
Deterministic node ID registry for cross-batch consistency.

Ensures that the same entity always gets the same node ID,
even when extracted in different batches.
"""

import hashlib
import json
from typing import Dict, Optional, Set

from pydantic import BaseModel


class NodeIDRegistry:
    """
    Global registry that maps entity fingerprints to stable node IDs.

    This ensures deterministic, globally-consistent node IDs across:
    - Multiple batch extractions
    - Model merging operations
    - Graph conversion
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        # Map: entity_fingerprint → node_id
        self.fingerprint_to_id: Dict[str, str] = {}

        # Map: node_id → entity_fingerprint (reverse lookup)
        self.id_to_fingerprint: Dict[str, str] = {}

        # Track seen nodes for collision detection
        self.seen_classes: Dict[str, Set[str]] = {}

    def _generate_fingerprint(self, model_instance: BaseModel) -> str:
        """
        Generate a fingerprint (content hash) for an entity.

        This identifies WHAT the entity is (independent of when it was created).

        For materials: hash of (name, category, chemical_formula)
        For measurements: hash of (name, numeric_value, unit)
        For processes: hash of (step_type, name, sequence_order)
        """
        # Extract identity fields that uniquely identify this entity
        model_config = model_instance.model_config

        # Get graph_id_fields from config
        id_fields = []
        if hasattr(model_config, "get"):
            id_fields = model_config.get("graph_id_fields", [])
        else:
            id_fields = getattr(model_config, "graph_id_fields", [])

        # Build fingerprint from identity fields
        fingerprint_data = {}

        if id_fields:
            for field in id_fields:
                if hasattr(model_instance, field):
                    value = getattr(model_instance, field)

                    # Normalize lists to sorted tuples
                    if isinstance(value, list):
                        try:
                            value = tuple(sorted(set(value)))
                        except TypeError:
                            value = tuple(value)

                    fingerprint_data[field] = value
        else:
            # No identity fields specified; use all non-empty fields
            for field_name, field_value in model_instance:
                if field_value and not isinstance(field_value, list | dict | BaseModel):
                    fingerprint_data[field_name] = field_value

        # Create deterministic hash
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, default=str)
        fingerprint = hashlib.blake2b(fingerprint_str.encode()).hexdigest()[:16]

        return fingerprint

    def get_node_id(
        self,
        model_instance: BaseModel,
        auto_register: bool = True,
    ) -> str:
        """
        Get or create a deterministic node ID for a model instance.

        Args:
            model_instance: The Pydantic model instance
            auto_register: If True, automatically register unknown entities

        Returns:
            Stable node ID in format: ClassName_fingerprint
        """
        fingerprint = self._generate_fingerprint(model_instance)
        class_name = model_instance.__class__.__name__

        # Check if we've seen this entity before
        if fingerprint in self.fingerprint_to_id:
            existing_id = self.fingerprint_to_id[fingerprint]
            # Verify class name matches (detect collisions)
            if not existing_id.startswith(class_name):
                raise ValueError(
                    f"Node ID collision: fingerprint {fingerprint} maps to both "
                    f"{existing_id} (old) and {class_name} (new)"
                )
            return existing_id

        # Create new node ID
        if class_name not in self.seen_classes:
            self.seen_classes[class_name] = set()

        node_id = f"{class_name}_{fingerprint}"

        if auto_register:
            self.fingerprint_to_id[fingerprint] = node_id
            self.id_to_fingerprint[node_id] = fingerprint
            self.seen_classes[class_name].add(fingerprint)

        return node_id

    def register_batch(self, models: list[BaseModel]) -> None:
        """
        Register all models in a batch to pre-populate the registry.

        Call this BEFORE converting models to graph to ensure consistent IDs.
        """
        for model in models:
            self.get_node_id(model, auto_register=True)

    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_entities": len(self.fingerprint_to_id),
            "classes": {cls: len(fps) for cls, fps in self.seen_classes.items()},
        }
