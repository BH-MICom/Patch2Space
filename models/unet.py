import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.utils import optional_import
einops, _ = optional_import("einops")

class Convnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, stride) -> None:
        super().__init__()
        convnd = nn.Conv3d if num_dims == 3 else nn.Conv2d
        normnd = nn.InstanceNorm3d if num_dims == 3 else nn.InstanceNorm2d
        self.conv = convnd(in_channels, out_channels, 3, stride, 1)
        self.norm = normnd(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.norm(x))

class StackedConvnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, first_stride) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Convnd(num_dims, in_channels, out_channels, first_stride),
            Convnd(num_dims, out_channels, out_channels, 1)
        )
    
    def forward(self, x):
        return self.blocks(x)


class UNet(nn.Module):
    def __init__(
        self, 
        config,
        hidden_size: int = 768,
        num_heads: int = 12,
        cls_dim: int = 8,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
        qkv_bias: bool = False,
        ) -> None:
        super().__init__()
        self.num_dims = config.MODEL.NUM_DIMS
        assert self.num_dims in [2, 3], 'only 2d or 3d inputs are supported'
        self.num_classes = config.DATASET.NUM_CLASSES

        self.extra = config.MODEL.EXTRA
        self.enc_channels = self.extra.ENC_CHANNELS
        self.dec_channels = self.extra.DEC_CHANNELS
        
        # encoder
        self.enc = nn.ModuleList()
        prev_channels = config.DATASET.NUM_MODALS_MR
        for i, channels in enumerate(self.enc_channels):
            # we do not perform downsampling at first convolution layer
            first_stride = 2 if i != 0 else 1
            self.enc.append(StackedConvnd(self.num_dims, prev_channels, channels, first_stride))
            prev_channels = channels

        self.fc_mr = nn.Linear(512 * 4 * 4 * 4, 768)
        self.fc_ct = nn.Linear(512 * 4 * 4 * 4, 768)
        
        self.fc_mri_back = nn.Linear(768, 512 * 4 * 4 * 4)

        # decoder
        self.dec_up = nn.ModuleList()
        self.dec_cat = nn.ModuleList()
        prev_channels = channels
        deconvnd = nn.ConvTranspose3d if self.num_dims == 3 else nn.ConvTranspose2d
        for channels in self.dec_channels:
            self.dec_up.append(deconvnd(prev_channels, channels, 2, 2, bias=False))
            self.dec_cat.append(
                nn.Sequential(
                    StackedConvnd(self.num_dims, channels*2, channels, 1),  # Concat skip features
                    StackedConvnd(self.num_dims, channels, channels, 1)
                )
            )
            prev_channels = channels

        # encoder
        self.enc_ct = nn.ModuleList()
        prev_channels_ct = config.DATASET.NUM_MODALS_CT
        for i, channels in enumerate(self.enc_channels):
            # we do not perform downsampling at first convolution layer
            first_stride = 2 if i != 0 else 1
            self.enc_ct.append(StackedConvnd(self.num_dims, prev_channels_ct, channels, first_stride))
            prev_channels_ct = channels

        # outputs
        convnd = nn.Conv3d if self.num_dims == 3 else nn.Conv2d
        self.conv_segs = nn.ModuleList()
        for channels in self.dec_channels[1:]:
            self.conv_segs.append(convnd(channels, self.num_classes, 1, 1, 0, bias=False))

        self.fusion_layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attention_mr': SelfAttention(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate),
                'self_attention_ct': SelfAttention(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate),
                'spatial_attention': SpatialAttention(cls_dim=cls_dim),
                'fusion_mlp_mr': MLP(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate),
                'fusion_mlp_ct': MLP(hidden_size=hidden_size, mlp_dim=mlp_dim, dropout_rate=dropout_rate),
                'cross_attention': CrossAttention(hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate, qkv_bias=qkv_bias),
            }) for _ in range(6)
        ])

    def forward(self, x):
        skips, outs = [], []
        # mr_data = x

        mr_data, ct_data = x[0], x[1]
        
        for layer in self.enc[:-1]:
            mr_data = layer(mr_data)
            skips.append(mr_data)
        mr_data = self.enc[-1](mr_data)
        # x = self.enc[-1](mr_data)

        mr_data_flat = mr_data.view(mr_data.size(0), -1)
        f_mr = self.fc_mr(mr_data_flat)
        f_mr = f_mr.unsqueeze(0)

        for layer in self.enc_ct[:-1]:
            ct_data = layer(ct_data)
        ct_data = self.enc_ct[-1](ct_data)

        ct_data_flat = ct_data.view(ct_data.size(0), -1)
        f_ct = self.fc_ct(ct_data_flat)
        f_ct = f_ct.unsqueeze(0)

        for layer in self.fusion_layers:
            F_mr = layer['self_attention_mr'](f_mr)
            F_ct= layer['self_attention_ct'](f_ct)
            F_mri2 = layer['spatial_attention'](PE[0], PE[1], F_ct) + F_mr
            F_mri3 = layer['fusion_mlp_mr'](F_mri2)
            # F_mri3 = layer['fusion_mlp_mr'](F_mr)
            F_ct3 = layer['fusion_mlp_ct'](F_ct)
            f_mr = f_mr + layer['cross_attention'](F_mri3, F_ct3)
            # f_mr = f_mr + F_mri3
            
        f_mr = self.fc_mri_back(f_mr)
        f_mr = f_mr.squeeze(0)
        
        x = f_mr.view(mr_data.shape[0], mr_data.shape[1], mr_data.shape[2], mr_data.shape[3], mr_data.shape[4])

        for i, (layer_up, layer_cat) in enumerate(zip(self.dec_up, self.dec_cat)):
            x = layer_up(x)
            x = torch.cat([x, skips[-(i+1)]], dim=1)
            x = layer_cat(x)
            if i >= 1:
                outs.append(self.conv_segs[i-1](x))
        outs.reverse()
        return outs

class SelfAttention(nn.Module):
    """
    A self-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.

        """

        super(SelfAttention, self).__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        q, k, v = einops.rearrange(self.qkv(x), "b h (qkv l d) -> qkv b l h d", qkv=3, l=self.num_heads)
        att_mat = (torch.einsum("blxd,blyd->blxy", q, k) * self.scale).softmax(dim=-1)
        att_mat = self.drop_weights(att_mat)
        x = torch.einsum("bhxy,bhyd->bhxd", att_mat, v)
        x = einops.rearrange(x, "b h l d -> b l (h d)")
        x = self.out_proj(x)
        x = self.drop_output(x)
        return x

class CrossAttention(nn.Module):
    """
    A cross-attention block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        """
        Args:
            hidden_size (int): dimension of hidden layer.
            num_heads (int): number of attention heads.
            dropout_rate (float): fraction of the input units to drop.
            qkv_bias (bool, optional): bias term for the qkv linear layer. Defaults to False.
        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.drop_output = nn.Dropout(dropout_rate)
        self.drop_weights = nn.Dropout(dropout_rate)

    def forward(self, x, context):
        """
        Args:
            x (torch.Tensor): input tensor. B x S x C
            context (torch.Tensor, optional): context tensor. B x S x C

        Returns:
            torch.Tensor: B x S x C
        """
        b, s, c = x.size()  # batch size, sequence length, hidden size

        # Calculate query, key, and value
        q = self.to_q(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, S, head_dim)
        kv = context if context is not None else x
        k = self.to_k(kv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        att_mat = (torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale).softmax(dim=-1)  # Attention scores
        att_mat = self.drop_weights(att_mat)

        x = torch.einsum("bhqk,bhvd->bhqd", att_mat, v)  # Attention output
        x = x.transpose(1, 2).contiguous().view(b, s, -1)  # Reshape to (B, S, hidden_size)
        x = self.out_proj(x)
        x = self.drop_output(x)

        return x


class MLP(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            dropout_rate: faction of the input units to drop.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class SpatialAttention(nn.Module):
    """
    A class to calculate the Attention Matrix based on the provided formula.
    """

    def __init__(self, cls_dim: int):
        """
        Args:
            cls_dim: The dimension for scaling the attention scores.
        """
        super().__init__()
        self.cls_dim = cls_dim

    def forward(self, pe_mri: torch.Tensor, pe_ct: torch.Tensor, F_ct: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Attention Matrix.

        Args:
            pe_mri (torch.Tensor): The MRI features, shape (B, N, C).
            pe_ct (torch.Tensor): The CT features, shape (B, M, C).

        Returns:
            torch.Tensor: The Attention Matrix, shape (B, N, M).
        """
        # Perform tensor multiplication
        attention_scores = torch.bmm(pe_mri, pe_ct.transpose(1, 2)) / (self.cls_dim ** 0.5)

        # Apply softmax to obtain attention weights
        attention_matrix = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum of CT features based on the attention matrix (B, N, C)
        fused_features = torch.bmm(attention_matrix, F_ct)

        return fused_features
