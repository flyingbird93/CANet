import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
import trilinear
import sys
from resnet_lut import *
from einops import rearrange
from efficientnet_lut import EfficientNet as create_model


def Bilinear_pooling(x, y):
    # two matrix trans to 512, 512, then sqrt, then than 0, final normilize [0,1]
    N = x.size()[0]
    x = x.view(N, 512, 1)
    y = y.view(N, 512, 1)
    x = torch.bmm(x, torch.transpose(y, 1, 2)) / 1 ** 2  # Bilinear # three dims matrix multiply
    assert x.size() == (N, 512, 512)
    x = x.view(N, 512 ** 2) # bn, 512*512
    x_sign = torch.sign(x) # bn, 512, 512
    x = torch.sqrt(torch.abs(x + 1e-5)) # bn, 512, 512
    x = x * x_sign
    x = torch.nn.functional.normalize(x)

    return x


class BiAttn(nn.Module):
    def __init__(self, in_channels, act_ratio=0.25, act_fn=nn.GELU, gate_fn=nn.Sigmoid):  #
        super().__init__()
        reduce_channels = int(in_channels * act_ratio)
        self.norm = nn.LayerNorm(in_channels)
        self.global_reduce = nn.Linear(in_channels, reduce_channels)
        self.local_reduce = nn.Linear(in_channels, reduce_channels)
        self.act_fn = act_fn()
        self.channel_select = nn.Linear(reduce_channels, in_channels)
        self.spatial_select = nn.Linear(reduce_channels * 2, 1)
        self.gate_fn = gate_fn()

    def forward(self, x):
        ori_x = x                                               # bn, 4, 49, 512
        x = self.norm(x)                                        # bn, 4, 49, 512
        x_global = x.mean(1, keepdim=True)                      # bn, 1, 49, 512
        x_global = self.act_fn(self.global_reduce(x_global))    # bn, 1, 49, 128
        x_local = self.act_fn(self.local_reduce(x))             # bn, 4, 49, 128

        c_attn = self.channel_select(x_global)                  # bn, 1, 49, 512
        c_attn = self.gate_fn(c_attn)  # [B, 1, C]              # bn, 1, 49, 512
        expand_global = x_global.expand(-1, x.shape[1], -1, -1)
        fusion_local_and_global = torch.cat([x_local, expand_global], dim=-1)
        s_attn = self.spatial_select(fusion_local_and_global)   # bn, 4, 49, 512
        s_attn = self.gate_fn(s_attn)  # [B, N, 1]              # bn, 4, 49, 1

        attn = c_attn * s_attn  # [B, N, C]                     # bn, 4, 49, 512
        return ori_x * attn


class BiAttnMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.attn = BiAttn(out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)             # bn, 4, 49, 512
        x = self.attn(x)            # bn, 4, 49, 512
        x = self.drop(x)            # bn, 4, 49, 512
        return x


def Conv_BN_ReLU(inp, oup, kernel, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, BiAttnMlp(dim, mlp_dim)) # , dropout=dropout
                # PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, BiAtt in self.layers:
            x = attn(x) + x

            # change traditional mlp to BiAtt FFN
            x = BiAtt(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv_BN_ReLU(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = Conv_BN_ReLU(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape      # bn, 512, 14, 14
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw) # bn, 4, 49, 512

        # change light vit to traditional transfomer
        x = self.transformer(x)

        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4):
        super(MV2Block, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.pad = nn.ZeroPad2d(4)
        self.net1 = create_model.from_pretrained('efficientnet-b0')
        self.net2 = Network_ResNet()
        self.conv = nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 3)
        # self.out_fc = nn.Linear(512*512, 3)
        self.out_fc = nn.Linear(512 * 2, 3)

        # add params
        inp = 3
        num_heads = 2
        LUT_dim = 33
        drop_path_rate = 0.
        self.LUT_num = 5
        self.num_heads = 1
        self.scale = (inp // num_heads) ** -0.5
        self.k = nn.Linear(LUT_dim, inp)
        self.q = nn.Linear(LUT_dim, inp)
        self.fuse = nn.Conv2d(inp, inp, 1, 1, 0, bias=False)
        self.proj_k = nn.Sequential(nn.Linear(49, 49),
                                  nn.Dropout(drop_path_rate))
        self.proj_v = nn.Sequential(nn.Linear(49, 49),
                                  nn.Dropout(drop_path_rate))
        self.layer_norm = nn.LayerNorm([512, 49])

        # mobilenet
        channels = [16, 64, 128, 256, 512, 512, 512, 128, 160, 160, 640]
        expansion = 4
        self.conv1 = Conv_BN_ReLU(3, channels[0], kernel=3, stride=2)
        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 2, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 2, expansion))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[4], channels[5], 2, expansion))   # 512, 512, 2, 4
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))

        # mvit
        kernel_size=3
        patch_size=(1, 1)
        # dims = [144, 192, 240]
        dims = [512, 1028, 380]
        L = [1, 4, 3]
        self.mvit = nn.ModuleList([])
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[4], kernel_size, patch_size, int(dims[0] * 2)))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[4], kernel_size, patch_size, int(dims[1] * 4)))
        # self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)))

    def forward(self, img, lut1, lut2, lut3): #, lut4, lut5):
        # image feature and LUT
        img = self.upsample(img)                                # 1, 3, 256, 256
        bs, C, H, W = img.shape
        bn = 3
        # LUT
        lut = torch.cat([lut1, lut2, lut3], dim=0) #, lut4, lut5], dim=0)  # 5, 3, 216, 216

        # -------------- img local ------------------
        img_out = self.conv1(img)                                       # 1, 16, 112, 128
        img_out = self.mv2[0](img_out)                                  # 1, 64, 56, 56
        img_out = self.mv2[1](img_out)                                  # 1, 128, 56, 56
        img_out = self.mv2[2](img_out)                                  # 1, 256, 28, 28
        img_out = self.mv2[3](img_out)                                  # 1, 256, 14, 14
        result_mv_output = self.mv2[4](img_out)                         # 1, 512, 7, 7


        # -------------- img to LUT attention ------------------  # tokens bn, T, Ct -->  T, bn, Ct
        lut = self.pad(lut)                                     # 5, 3, 224, 224
        lut = self.net2(lut)                                    # 5, 512, 7, 7
        lut = lut.reshape(bn, 512, -1)                          # 5, 512, 49
        t = lut.permute(0, 2, 1)
        k = result_mv_output.view(bs, 512, -1)                  # 1, 512, 196
        attn = (k @ t) * self.scale                             # bs x T x HW     5, 512, 512

        attn_out = attn.softmax(dim=-1)                         # bs x T x HW     5, 512, 512
        attn_out = (attn_out @ k)   # k.permute(0, 2, 1)        # bs x T x C      5, 512, 196
                                                                # note here: k=v without transform
        lut = lut + attn_out                                    # self.drop_path(t)
        lut = self.layer_norm(lut)                              # 5, 512, 49

        # -------------- LUT feature transformer ------------------
        transformer_input = lut.reshape(bn, 512, 7, 7)            # bn, 512, 7, 7
        lut_mvit_output = self.mvit[0](transformer_input)         # bn, 512, 14, 14


        # -------------- global to local attention ------------------
        reshape_lut_output = lut_mvit_output.reshape(bn, 512, -1) # bn, 512, 49
        bs, c, h, w = result_mv_output.shape                      # bn, 512, 7, 7

        # Q, K, V
        q_1 = result_mv_output.view(bs, 512, -1)                  # bs, 512, 49

        k_1 = self.proj_k(reshape_lut_output).permute(0, 2, 1)  #.view(bs, self.num_heads, -1, self.LUT_num)  # from T x bs x Ct -> bs x C x T -> bs x N x C/N x T
        attn = (q_1 @ k_1) * self.scale                         # bs x N x HW x T    bn, 512, 512
        attn_out = attn.softmax(dim=-1)                         # bs x N x HW x T    bn, 512, 512

        v_1 = self.proj_v(reshape_lut_output)                   # bs x N x C/N x T   bn, 512, 49
        attn_out = (attn_out @ v_1)                             # bs x N x HW x C/N  bn, 512, 49

        g_a = attn_out.reshape(bn, 512, 7, 7)                   # bs x C x HW
        lut_out = result_mv_output + g_a  # self.drop_path(g_sum)   # bn, 512, 7, 7

        # fusion
        result  = self.avgpool(result_mv_output)                # bn, 512, 1, 1
        lut_out = self.avgpool(lut_out).mean(dim=0)             # 512, 1, 1
        feat_concat = torch.cat((result, lut_out.unsqueeze(0)), dim=1).reshape(bs, -1)
        out = self.out_fc(feat_concat)                          # bn, 5

        return out


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ContextModulate(nn.Module):

    def __init__(self):
        super(ContextModulate, self).__init__()
        self.conv_1 = nn.Conv2d(27, 128, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        nn_Unfold = nn.Unfold(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        output_img = nn_Unfold(x)
        transform_img = output_img.view(x.shape[0], 27, x.shape[2], x.shape[3])
        out1 = self.relu(self.conv_1(transform_img))
        out2 = self.sigmoid(self.conv_2(out1))

        return out2


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1)]
    layers.append(nn.LeakyReLU(0.2))
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        #layers.append(nn.BatchNorm2d(out_filters))

    return layers


class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')

        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):

        super(Generator3DLUT_identity, self).__init__()

        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 36:
            file = open("/media/flyingbird/Work/Work/ImageEnhancement/Code/Image-Adaptive-3DLUT-master/DualBLN-master/code/IdentityLUT36.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k

                    x = lines[n].split()

                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])

        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=36):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = nn.init.kaiming_normal_(torch.zeros(3, dim, dim, dim, dtype=torch.float), mode="fan_in",
                                           nonlinearity="relu")

        # self.LUT = nn.Parameter(torch.tensor(self.LUT))
        self.LUT = nn.Parameter(self.LUT.clone().detach())

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):

        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(lut, x, output, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut, x.permute(1, 0, 2, 3).contiguous(), output, dim, shift, binsize, W, H,
                                          batch)

            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(x, x_grad, lut_grad, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            assert 1 == trilinear.backward(x.permute(1, 0, 2, 3).contiguous(), x_grad.permute(1, 0, 2, 3).contiguous(),
                                           lut_grad, dim, shift, binsize, W, H, batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=36):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


if __name__ == '__main__':

    import torch

    input_feat = torch.rand(1, 3, 224, 224).cuda()

    lut_dim = 36
    LUT1 = Generator3DLUT_identity(dim=lut_dim).cuda()
    LUT2 = Generator3DLUT_zero(dim=lut_dim).cuda()
    LUT3 = Generator3DLUT_zero(dim=lut_dim).cuda()

    lut1 = LUT1.state_dict()['LUT'].unsqueeze(dim=0).reshape(3, 216, 216).unsqueeze(dim=0)
    lut2 = LUT2.state_dict()['LUT'].unsqueeze(dim=0).reshape(3, 216, 216).unsqueeze(dim=0)
    lut3 = LUT3.state_dict()['LUT'].unsqueeze(dim=0).reshape(3, 216, 216).unsqueeze(dim=0)

    model = Model().cuda()
    output = model(input_feat, lut1, lut2, lut3)
    print(output.shape)