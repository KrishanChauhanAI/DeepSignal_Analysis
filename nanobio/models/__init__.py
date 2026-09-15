"""Model architectures: classical, baseline DL, Prototypical Network, VAE."""

from nanobio.models.classical import build_classical_models
from nanobio.models.protonet import (
    AttentionAggregator,
    HierarchicalProtoNet,
    HybridFusion,
    ProtoNetOutput,
    SegmentEncoder,
)
from nanobio.models.resnet1d import (
    TCN,
    CNNBiLSTM,
    CNNEncoder,
    InceptionTime,
    ProjectionHead,
    ResidualBlock1D,
    ResNet1D,
    ResNetEncoder,
    # NEW: Research encoders
    TCNEncoder,
    TransformerClassifier,
    build_ssl_encoder,
)

__all__ = [
    # Classical
    "build_classical_models",
    # Baseline DL
    "ResidualBlock1D", "ResNet1D", "TCN", "InceptionTime",
    "CNNBiLSTM", "TransformerClassifier",
    # ProtoNet
    "SegmentEncoder", "AttentionAggregator", "HybridFusion",
    "HierarchicalProtoNet", "ProtoNetOutput",
    # Research encoders (SSL / few-shot)
    "TCNEncoder", "CNNEncoder", "ResNetEncoder",
    "ProjectionHead", "build_ssl_encoder",
]