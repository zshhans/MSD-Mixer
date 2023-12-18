import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
# from .utils import get_activation
from torchsummary import summary

def get_activation(activ: str):
    if activ == "gelu":
        return nn.GELU()
    elif activ == "sigmoid":
        return nn.Sigmoid()
    elif activ == "tanh":
        return nn.Tanh()
    elif activ == "relu":
        return nn.ReLU()

    raise RuntimeError("activation should not be {}".format(activ))

class MLPBlock(nn.Module):

    def __init__(
        self,
        dim,
        in_features: int,
        hid_features: int,
        out_features: int,
        activ="gelu",
        drop: float = 0.0,
        jump_conn='trunc',
    ):
        super().__init__()
        self.dim = dim
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hid_features),
            get_activation(activ),
            nn.Linear(hid_features, out_features),
            DropPath(drop))
        if jump_conn == "trunc":
            self.jump_net = nn.Identity()
        elif jump_conn == 'proj':
            self.jump_net = nn.Linear(in_features, out_features)
        else:
            raise ValueError(f"jump_conn:{jump_conn}")

    def forward(self, x):
        x = torch.transpose(x, self.dim, -1)
        x = self.jump_net(x)[..., :self.out_features] + self.net(x)
        x = torch.transpose(x, self.dim, -1)
        return x


class PatchEncoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        channel_wise_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        inter_patch_mlp = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ,
                         drop)

        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(patch_size, 1)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c (l1 l2) -> b c l1 l2", l2=patch_size))
        self.net.append(norm_class(in_chn))
        self.net.append(channel_wise_mlp)
        self.net.append(norm_class(out_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(out_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(linear)
        self.net.append(Rearrange("b c l1 1 -> b c l1"))

    def forward(self, x):
        # b,c,l
        return self.net(x)


class PatchDecoder(nn.Module):

    def __init__(
        self,
        in_len: int,
        hid_len: int,
        in_chn: int,
        hid_chn: int,
        out_chn,
        patch_size: int,
        hid_pch: int,
        norm=None,
        activ="gelu",
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential()
        inter_patch_mlp = MLPBlock(2, in_len // patch_size, hid_len, in_len // patch_size, activ,
                         drop)
        channel_wise_mlp = MLPBlock(1, in_chn, hid_chn, out_chn, activ, drop)
        if norm == 'bn':
            norm_class = nn.BatchNorm2d
        elif norm == 'in':
            norm_class = nn.InstanceNorm2d
        else:
            norm_class = nn.Identity
        linear = nn.Linear(1, patch_size)
        intra_patch_mlp = MLPBlock(3, patch_size, hid_pch, patch_size, activ, drop)
        self.net.append(Rearrange("b c l1 -> b c l1 1"))
        self.net.append(linear)
        self.net.append(norm_class(in_chn))
        self.net.append(intra_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(inter_patch_mlp)
        self.net.append(norm_class(in_chn))
        self.net.append(channel_wise_mlp)
        self.net.append(Rearrange("b c l1 l2 -> b c (l1 l2)"))

    def forward(self, x):
        # b,c,l
        return self.net(x)


class PredictionHead(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 hid_len,
                 in_chn,
                 out_chn,
                 hid_chn,
                 activ,
                 drop=0.0) -> None:
        super().__init__()
        self.net = nn.Sequential()
        if in_chn != out_chn:
            c_jump_conn = "proj"
        else:
            c_jump_conn = "trunc"
        self.net.append(
            MLPBlock(1,
                in_chn,
                hid_chn,
                out_chn,
                activ=activ,
                drop=drop,
                jump_conn=c_jump_conn))
        self.net.append(
            MLPBlock(2,
                in_len,
                hid_len,
                out_len,
                activ=activ,
                drop=drop,
                jump_conn='proj'))

    def forward(self, x):
        return self.net(x)


class MSDMixer(nn.Module):

    def __init__(self,
                 in_len,
                 out_len,
                 in_chn,
                 ex_chn,
                 out_chn,
                 patch_sizes,
                 hid_len,
                 hid_chn,
                 hid_pch,
                 hid_pred,
                 norm,
                 last_norm,
                 activ,
                 drop,
                 reduction="sum") -> None:
        super().__init__()
        self.in_len = in_len
        self.out_len = out_len
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.last_norm = last_norm
        self.reduction = reduction
        self.patch_encoders = nn.ModuleList()
        self.patch_decoders = nn.ModuleList()
        self.pred_heads = nn.ModuleList()
        self.patch_sizes = patch_sizes
        self.paddings = []
        all_chn = in_chn + ex_chn
        for i, patch_size in enumerate(patch_sizes):
            res = in_len % patch_size
            padding = (patch_size - res) % patch_size
            self.paddings.append(padding)
            padded_len = in_len + padding
            self.patch_encoders.append(
                PatchEncoder(padded_len, hid_len, all_chn, hid_chn,
                          in_chn, patch_size, hid_pch, norm, activ, drop))
            self.patch_decoders.append(
                PatchDecoder(padded_len, hid_len, in_chn, hid_chn, in_chn,
                        patch_size, hid_pch, norm, activ, drop))
            if out_len != 0 and out_chn != 0:
                self.pred_heads.append(
                    PredictionHead(padded_len // patch_size, out_len, hid_pred,
                                    in_chn, out_chn, hid_chn, activ, drop))
            else:
                self.pred_heads.append(nn.Identity())

    def forward(self, x, x_mark=None, x_mask=None):
        x = rearrange(x, "b l c -> b c l")
        if x_mark is not None:
            x_mark = rearrange(x_mark, "b l c -> b c l")
        if x_mask is not None:
            x_mask = rearrange(x_mask, "b l c -> b c l")
        if self.last_norm:
            x_last = x[:, :, [-1]].detach()
            x = x - x_last
            if x_mark is not None:
                x_mark_last = x_mark[:, :, [-1]].detach()
                x_mark = x_mark - x_mark_last
        y_pred = []
        for i in range(len(self.patch_sizes)):
            x_in = x
            if x_mark is not None:
                x_in = torch.cat((x, x_mark), 1)
            x_in = F.pad(x_in, (self.paddings[i], 0), "constant", 0)
            emb = self.patch_encoders[i](x_in)
            comp = self.patch_decoders[i](emb)[:, :, self.paddings[i]:]
            pred = self.pred_heads[i](emb)
            if x_mask is not None:
                comp = comp * x_mask
            x = x - comp
            if self.out_len != 0 and self.out_chn != 0:
                y_pred.append(pred)

        if self.out_len != 0 and self.out_chn != 0:
            y_pred = reduce(torch.stack(y_pred, 0), "h b c l -> b c l",
                            self.reduction)
            if self.last_norm and self.out_chn == self.in_chn:
                y_pred += x_last
            y_pred = rearrange(y_pred, "b c l -> b l c")
            return y_pred, x
        else:
            return None, x




if __name__ =="__main__":
    input_len = 96
    h_dim = 512
    input_chn = 11
    out_chn = 7
    patch_size = 24

    model_ptch_encoder = PatchEncoder(input_len, h_dim, input_chn, h_dim, out_chn, patch_size, h_dim, norm=None)
    model_ptch_encoder.to('cuda:0')
    summary(model_ptch_encoder, input_size=(input_chn,input_len,))


    model_ptch_decoder = PatchDecoder(input_len, h_dim, out_chn, h_dim,out_chn, patch_size, h_dim, norm=None)
    model_ptch_decoder.to('cuda:0')
    summary(model_ptch_decoder, input_size=(out_chn,input_len//patch_size))

    model_predict_head = PredictionHead(input_len//patch_size, input_len, h_dim, out_chn, out_chn, h_dim, activ="gelu")
    model_predict_head.to('cuda:0')
    summary(model_predict_head, input_size=(out_chn,input_len//patch_size))