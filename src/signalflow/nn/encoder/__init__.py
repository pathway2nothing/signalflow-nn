from signalflow.nn.encoder.lstm import LSTMEncoder
from signalflow.nn.encoder.gru import GRUEncoder
from signalflow.nn.encoder.conv1d import Conv1dEncoder
from signalflow.nn.encoder.tcn import TCNEncoder
from signalflow.nn.encoder.inception import InceptionTimeEncoder
from signalflow.nn.encoder.transformer import TransformerEncoder
from signalflow.nn.encoder.patchtst import PatchTSTEncoder
from signalflow.nn.encoder.resnet1d import ResNet1dEncoder
from signalflow.nn.encoder.tsmixer import TSMixerEncoder
from signalflow.nn.encoder.xception import XceptionTimeEncoder
from signalflow.nn.encoder.xcm import XCMEncoder
from signalflow.nn.encoder.gmlp import gMLPEncoder
from signalflow.nn.encoder.omniscale import OmniScaleCNNEncoder
from signalflow.nn.encoder.convtran import ConvTranEncoder
from signalflow.nn.encoder.mamba import MambaEncoder
from signalflow.nn.encoder.itransformer import iTransformerEncoder

__all__ = [
    # RNN-based
    "LSTMEncoder",
    "GRUEncoder",
    # CNN-based
    "Conv1dEncoder",
    "TCNEncoder",
    "InceptionTimeEncoder",
    "ResNet1dEncoder",
    "XceptionTimeEncoder",
    "OmniScaleCNNEncoder",
    # Transformer-based
    "TransformerEncoder",
    "PatchTSTEncoder",
    "ConvTranEncoder",
    "iTransformerEncoder",  # Inverted Transformer (ICLR 2024)
    # Mixer-based
    "TSMixerEncoder",
    "gMLPEncoder",
    "XCMEncoder",
    # State Space Models
    "MambaEncoder",  # Mamba SSM (O(T) complexity)
]
