"""Central registry of detection/segmentation classes.

Each entry maps a CLI flag (dict key) to its internal name (`internal_name`).
The internal name is the load-bearing identifier: it names the PyTorch output
submodule (and therefore the state_dict / ONNX output), the wandb metric, the
loss criterion, the dataset key used by the sampler, and the LHS of
`zettaset_lookup`. Flag and internal name can differ freely.

Adding a new task (e.g., "psd_v2", "ribosome") is a one-line edit:

    'ribo': ClassSpec(internal_name='ribosome', binarize=True),

The matching `--ribo <weight>` CLI flag (train) and `--ribo` switch (test)
become available automatically; no edits to option.py needed.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class ClassSpec:
    """Specification for a detection/segmentation class.

    Attributes:
        internal_name: The single load-bearing identifier for this class.
            Used as the PyTorch output submodule name (and state_dict /
            ONNX key), the wandb metric name, the loss criterion key, the
            dataset key consumed by the sampler, and the LHS of
            `zettaset_lookup`. Distinct from the CLI flag (the REGISTRY
            dict key).
        channels: Output channels. Either an int, or a callable
            opt -> int for classes whose width depends on another flag
            (e.g. blood_vessel takes opt.blv_num_channels).
        binarize: Whether the dataset should binarize the annotation.
            Bool or callable opt -> bool.
        semantic_id: If set, this class participates in semantic_mapping
            when --sem is on (label ID used by the combined-semantic GT).
    """
    internal_name: str
    channels: Union[int, Callable] = 1
    binarize: Union[bool, Callable] = False
    semantic_id: Optional[int] = None

    def resolve_channels(self, opt) -> int:
        return self.channels(opt) if callable(self.channels) else self.channels

    def resolve_binarize(self, opt) -> bool:
        return self.binarize(opt) if callable(self.binarize) else self.binarize


REGISTRY: dict[str, ClassSpec] = {
    # Affinity / boundary
    'aff':  ClassSpec(internal_name='affinity',   channels=3),
    'long': ClassSpec(internal_name='long_range', channels=lambda opt: len(opt.edges)),
    'bdr':  ClassSpec(internal_name='boundary'),

    # Detection (binary)
    'syn':  ClassSpec(internal_name='synapse',          binarize=True),
    'psd':  ClassSpec(internal_name='synapse',          binarize=True),
    'mit':  ClassSpec(internal_name='mitochondria',     binarize=True),
    'mye':  ClassSpec(internal_name='myelin',           binarize=True),
    'fld':  ClassSpec(internal_name='fold',             binarize=True),
    'glia': ClassSpec(internal_name='glia',             binarize=True, semantic_id=5),
    'img':  ClassSpec(internal_name='image'),
    'mito_to_cell': ClassSpec(internal_name='mitochondria_to_cell'),

    # Multi-channel blood vessel: binarize only when collapsed to 1 channel
    'blv': ClassSpec(
        internal_name='blood_vessel',
        channels=lambda opt: opt.blv_num_channels,
        binarize=lambda opt: opt.blv_num_channels == 1,
        semantic_id=7,
    ),

    # Semantic-segmentation classes (binarize only when --sem is off)
    'soma':  ClassSpec(internal_name='soma',                binarize=True, semantic_id=3),
    'dend':  ClassSpec(internal_name='dendrite',            semantic_id=1),
    'axon':  ClassSpec(internal_name='axon',                semantic_id=2),
    'nucl':  ClassSpec(internal_name='nucleus',             semantic_id=4),
    'ecs':   ClassSpec(internal_name='extracellular_space', semantic_id=6),
    'other': ClassSpec(internal_name='other_class',         semantic_id=10),

    # Embedding (consumed by MeanLoss / metric learning)
    'vec':      ClassSpec(internal_name='embedding',              channels=lambda opt: opt.embed_dim),
    'mito_emb': ClassSpec(internal_name='mitochondria_embedding', channels=lambda opt: opt.mito_emb_dim),
}


def semantic_mapping() -> dict[str, int]:
    """Class-name -> label-ID dict, derived from semantic_id fields."""
    return {
        spec.internal_name: spec.semantic_id
        for spec in REGISTRY.values()
        if spec.semantic_id is not None
    }


def requires_binarize(opt) -> list[str]:
    """Output names whose dataset annotation must be binarized.

    Mirrors the legacy hand-maintained list: includes classes with
    binarize=True, then drops any that are taken over by semantic_mapping
    when --sem is on.
    """
    names = {
        spec.internal_name
        for spec in REGISTRY.values()
        if spec.resolve_binarize(opt)
    }
    if getattr(opt, 'sem', False):
        names -= set(semantic_mapping().keys())
    return sorted(names)
