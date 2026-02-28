from signalflow.nn.encoder.conv1d import Conv1dEncoder
from signalflow.nn.encoder.convtran import ConvTranEncoder
from signalflow.nn.encoder.gmlp import gMLPEncoder
from signalflow.nn.encoder.gru import GRUEncoder
from signalflow.nn.encoder.inception import InceptionTimeEncoder
from signalflow.nn.encoder.itransformer import iTransformerEncoder
from signalflow.nn.encoder.lstm import LSTMEncoder
from signalflow.nn.encoder.mamba import MambaEncoder
from signalflow.nn.encoder.omniscale import OmniScaleCNNEncoder
from signalflow.nn.encoder.patchtst import PatchTSTEncoder
from signalflow.nn.encoder.resnet1d import ResNet1dEncoder
from signalflow.nn.encoder.tcn import TCNEncoder
from signalflow.nn.encoder.transformer import TransformerEncoder
from signalflow.nn.encoder.tsmixer import TSMixerEncoder
from signalflow.nn.encoder.xception import XceptionTimeEncoder
from signalflow.nn.encoder.xcm import XCMEncoder

__all__ = [
    # CNN-based
    "Conv1dEncoder",
    "ConvTranEncoder",
    "GRUEncoder",
    "InceptionTimeEncoder",
    # RNN-based
    "LSTMEncoder",
    # State Space Models
    "MambaEncoder",  # Mamba SSM (O(T) complexity)
    "OmniScaleCNNEncoder",
    "PatchTSTEncoder",
    "ResNet1dEncoder",
    "TCNEncoder",
    # Mixer-based
    "TSMixerEncoder",
    # Transformer-based
    "TransformerEncoder",
    "XCMEncoder",
    "XceptionTimeEncoder",
    "gMLPEncoder",
    "iTransformerEncoder",  # Inverted Transformer (ICLR 2024)
]
