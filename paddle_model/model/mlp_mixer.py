import math
from functools import partial

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant,Normal

from model.helpers import named_apply
from model.layers.patch_embed import PatchEmbed
from model.layers.mlp import Mlp
from model.layers.drop import DropPath
from model.layers.weight_init import lecun_normal_
from model.layers.helpers import to_2tuple
from model.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    mixer_b16_224=_cfg(
        url='/home/aistudio/check/ckpt/mixer_b16_224.pdparams',
    ),
    mixer_b16_224_in21k=_cfg(
        url='/home/aistudio/check/ckpt/mixer_b16_224_in21k.pdparams',
        num_classes=21843
    )
)
class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

normal_ = Normal(std=1e-6)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

class MixerBlock(nn.Layer):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose((0,2,1))).transpose((0,2,1)))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x

class MlpMixer(nn.Layer):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else Identity()

        self.init_weights(nlhb=nlhb)

    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(axis=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _init_weights(module: nn.Layer, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            zeros_(module.weight)
            constant_ = Constant(value=head_bias)
            constant_(module.bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.initializer.XavierUniform(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        normal_(module.bias)
                    else:
                        zeros_(module.bias)
    elif isinstance(module, nn.Conv2D):
        lecun_normal_(module.weight)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2D, nn.GroupNorm)):
        ones_(module.weight)
        zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()

def _create_mixer(variant, pretrained=False, **kwargs):
    num_classes=kwargs.get('num_classes', None) # 外面传的类别
    cfg=default_cfgs[variant]
    is_fintuned=True if num_classes is not None and num_classes!=cfg['num_classes'] else False
    if num_classes is None:kwargs.setdefault('num_classes',cfg['num_classes'])
    model = MlpMixer(**kwargs)
    if pretrained and cfg['url'].find('pdparams')!=-1:
        state_dict=paddle.load(cfg['url'])
        if is_fintuned: # 自己传了类别，且与预训练权重的类别不符，修改head的权重和偏置
            state_dict['head.weight']=model.head.weight
            state_dict['head.bias']=model.head.bias
        model.set_dict(state_dict)
    return model


@register_model
def mixer_b16_224(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def mixer_b16_224_in21k(pretrained=False, **kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224_in21k', pretrained=pretrained, **model_args)
    return model
