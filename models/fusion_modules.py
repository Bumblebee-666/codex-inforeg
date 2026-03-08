import torch
import torch.nn as nn
import torch.nn.functional as F
# 666


class SumFusion(nn.Module):
    def __init__(self, input_dim=512, output_dim=100):
        super(SumFusion, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim)
        self.fc_y = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = self.fc_x(x) + self.fc_y(y)
        return x, y, output


class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.audio_out = nn.Linear(512, output_dim)
        self.visual_out = nn.Linear(512, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        x_output = self.audio_out(x)
        y_output = self.visual_out(y)
        return x,y,x_output, y_output, output


class FiLM(nn.Module):
    """
    FiLM: Visual Reasoning with a General Conditioning Layer,
    https://arxiv.org/pdf/1709.07871.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_film=True):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_film = x_film

    def forward(self, x, y):

        if self.x_film:
            film = x
            to_be_film = y
        else:
            film = y
            to_be_film = x

        gamma, beta = torch.split(self.fc(film), self.dim, 1)

        output = gamma * to_be_film + beta
        output = self.fc_out(output)

        return x, y, output


class GatedFusion(nn.Module):
    """
    Efficient Large-Scale Multi-Modal Classification,
    https://arxiv.org/pdf/1802.02892.pdf.
    """

    def __init__(self, input_dim=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()

        self.fc_x = nn.Linear(input_dim, dim)
        self.fc_y = nn.Linear(input_dim, dim)
        self.fc_out = nn.Linear(dim, output_dim)

        self.x_gate = x_gate  # whether to choose the x to obtain the gate

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output


class LargeKernelMod(nn.Module):
    """LKM-style modulation for feature maps."""
    def __init__(self, channels, dropout=0.0, kernel_size=7, dilation=3):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(1, channels)
        self.dw_conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=False
        )
        self.dw_dilated = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels, bias=False
        )
        self.pw_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = self.dw_conv(x) + self.dw_dilated(x)
        x = self.pw_conv(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class CrossGate2(nn.Module):
    """Cross-gated modulation for two modalities (feature maps, LKM-style)."""
    def __init__(
        self,
        dim=512,
        dropout=0.0,
        strength=1.0,
        strength_a=None,
        strength_v=None,
        gate_temperature=1.0,
    ):
        super().__init__()
        self.lkm_a = LargeKernelMod(dim, dropout=dropout)
        self.lkm_v = LargeKernelMod(dim, dropout=dropout)
        self.gate_a = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gate_v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.scale_a = nn.Parameter(torch.ones(1))
        self.scale_v = nn.Parameter(torch.ones(1))
        self.strength_a = strength if strength_a is None else strength_a
        self.strength_v = strength if strength_v is None else strength_v
        self.gate_temperature = max(1e-6, gate_temperature)

    def forward(self, a, v):
        # a: [B, C, H, W], v: [B*T, C, H, W]
        b = a.size(0)
        v_batch = v.size(0)
        c = v.size(1)
        v_h = v.size(2)
        v_w = v.size(3)

        if v_batch % b != 0:
            raise ValueError("CrossGate2 expects visual batch to be a multiple of audio batch.")
        t = v_batch // b

        a_feat = self.lkm_a(a)
        v_feat = self.lkm_v(v)
        a_h = a_feat.size(2)
        a_w = a_feat.size(3)

        if t > 1:
            v_feat_mean = v_feat.reshape(b, t, c, v_h, v_w).mean(dim=1)
        else:
            v_feat_mean = v_feat

        # Audio and visual branches can have different spatial sizes (e.g., 9x6 vs 7x7).
        if (v_h, v_w) != (a_h, a_w):
            v_for_a = F.adaptive_avg_pool2d(v_feat_mean, output_size=(a_h, a_w))
        else:
            v_for_a = v_feat_mean
        gate_a = torch.sigmoid(self.gate_a(v_for_a) / self.gate_temperature)

        if (a_h, a_w) != (v_h, v_w):
            a_for_v = F.interpolate(a_feat, size=(v_h, v_w), mode='bilinear', align_corners=False)
        else:
            a_for_v = a_feat
        gate_v = torch.sigmoid(self.gate_v(a_for_v) / self.gate_temperature)

        if t > 1:
            gate_v = gate_v.unsqueeze(1).expand(-1, t, -1, -1, -1).contiguous().view(v_batch, c, v_h, v_w)

        a_out = a + self.strength_a * (self.scale_a * (a_feat * gate_a))
        v_out = v + self.strength_v * (self.scale_v * (v_feat * gate_v))
        return a_out, v_out


class CrossGate3(nn.Module):
    """Cross-gated modulation for three modalities (pairwise delta gating, strengthened)."""
    def __init__(self, dim=256, hidden_mult=2.0, dropout=0.0, strength=1.0):
        super().__init__()
        hidden_dim = max(1, int(dim * hidden_mult))
        self.norm_t = nn.LayerNorm(dim)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.proj_t = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.proj_a = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.proj_v = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate_t = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate_a = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate_v = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.scale_t = nn.Parameter(torch.ones(1))
        self.scale_a = nn.Parameter(torch.ones(1))
        self.scale_v = nn.Parameter(torch.ones(1))
        self.strength = strength

    def forward(self, t, a, v):
        t_norm = self.norm_t(t)
        a_norm = self.norm_a(a)
        v_norm = self.norm_v(v)

        proj_t = self.proj_t(t_norm)
        proj_a = self.proj_a(a_norm)
        proj_v = self.proj_v(v_norm)

        delta_t = proj_t - t
        delta_a = proj_a - a
        delta_v = proj_v - v

        gate_t = torch.sigmoid(self.gate_t(delta_a + delta_v + a_norm + v_norm))
        gate_a = torch.sigmoid(self.gate_a(delta_t + delta_v + t_norm + v_norm))
        gate_v = torch.sigmoid(self.gate_v(delta_t + delta_a + t_norm + a_norm))

        t_out = t + self.strength * (self.scale_t * (proj_t * gate_t))
        a_out = a + self.strength * (self.scale_a * (proj_a * gate_a))
        v_out = v + self.strength * (self.scale_v * (proj_v * gate_v))
        return t_out, a_out, v_out
