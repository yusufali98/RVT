import torch
import torch.nn as nn

import sys
sys.path.append('/workspace/VMamba')

from classification.models import vmamba
from collections import OrderedDict


class VSSM_RVT(vmamba.VSSM):
    '''
    Modified VSMM class which removes patch_embed and classifier layer
    '''

    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        # =========================
        posembed=False,
        imgsize=224,
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=vmamba.LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        # _make_patch_embed = dict(
        #     v1=self._make_patch_embed, 
        #     v2=self._make_patch_embed_v2,
        # ).get(patchembed_version, None)
        # self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=vmamba.PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            
            if downsample_version != "none":
                downsample = _make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=norm_layer,
                    channel_first=self.channel_first,
                ) if (i_layer < self.num_layers - 1) else nn.Identity()
            else:
                downsample = nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        # self.classifier = nn.Sequential(OrderedDict(
        #     norm=norm_layer(self.num_features), # B,H,W,C
        #     permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
        #     avgpool=nn.AdaptiveAvgPool2d(1),
        #     flatten=nn.Flatten(1),
        #     head=nn.Linear(self.num_features, num_classes),
        # ))

        self.apply(self._init_weights)
    
    def forward(self, x: torch.Tensor):
        
        # x = self.patch_embed(x)
        # print("patch emb shape: ", x.shape)

        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':

    model = VSSM_RVT(drop_path_rate=0.6,
                     dims=512, depths=[8],
                     ssm_d_state=16,
                     ssm_ratio=2.0,
                     ssm_conv_bias=False,
                     forward_type="v05_noz",
                     downsample_version="v3",
                     patchembed_version="v2",
                     norm_layer="ln2d").to('cuda')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params: ", total_params)

    a = torch.randn((15,128,30,30)).to('cuda')

    print("res shape: ", model(a).shape)