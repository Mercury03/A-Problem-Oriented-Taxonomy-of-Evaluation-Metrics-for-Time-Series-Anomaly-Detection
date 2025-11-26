import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.embed import DataEmbedding
from .AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .FourierCorrelation import FourierBlock, FourierCrossAttention
from .MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from .SelfAttention_Family import FullAttention, ProbAttention
from .Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.output_v = configs.output_v
        self.LIN = configs.LIN
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.norm = nn.LayerNorm(self.seq_len, eps=0, elementwise_affine=False)

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        self.enc_embedding = DataEmbedding(self.output_v, self.d_model, self.dropout, position=False)
        self.dec_embedding = DataEmbedding(self.output_v, self.d_model, self.dropout, position=True)
        
        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    self.output_v,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, self.output_v, bias=True)
            
        )

    def forward(self, x_enc, drop=0):
            # Add local normalization and data cleaning
            if hasattr(self, 'LIN') and self.LIN:
                x_enc[:, :, :self.output_v] = self.norm(x_enc[:, :, :self.output_v].permute(0, 2, 1)).transpose(1, 2)
            if drop:
                x_enc[:, -2 ** (drop - 1) - 1:-1, :self.output_v] = 0
            x_enc[torch.isinf(x_enc)] = 0
            x_enc[torch.isnan(x_enc)] = 0
            
            # Extract ground truth
            gt = x_enc[:, -1:, :].clone()
            gt = gt[:, :, :self.output_v]
            
            # Separate input sequence
            x_input = x_enc[:, :-1, :self.output_v].clone()
            season = torch.zeros(x_input.shape[0], 1, self.output_v).to(x_input.device)
            
            # Decomposition initialization
            mean = torch.mean(x_input, dim=1).unsqueeze(1)
            season_init, trend_init = self.decomp(x_input)
            
            # Decoder input preparation
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = torch.cat([season_init[:, -self.label_len:, :], season], dim=1)
            
            # Encoding process
            enc_out = self.enc_embedding(x_input)
            enc_out, _ = self.encoder(enc_out, attn_mask=None)
            
            # Decoding process
            dec_out = self.dec_embedding(seasonal_init)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                   trend=trend_init)
            
            # Final output
            output = trend_part + seasonal_part
            return output[:, -1:, :], gt


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        output_v = 7
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'softmax'
        seq_len = 96
        label_len = 48
        pred_len = 1
        output_attention = False
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        activation = 'gelu'
        wavelet = 0
        LIN = False

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    out = model.forward(enc)
    print(out[0].shape)