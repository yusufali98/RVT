# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import copy
import torch

from torch import nn

import rvt.mvt.utils as mvt_utils

from rvt.mvt.mvt_single import MVT as MVTSingle
from rvt.mvt.config import get_cfg_defaults
from rvt.mvt.renderer import BoxRenderer

from rvt.mvt.mvt_mamba import MVT_Mamba as MVTSingleMamba

class MVT(nn.Module):
    def __init__(
        self,
        depth,
        img_size,
        add_proprio,
        proprio_dim,
        add_lang,
        lang_dim,
        lang_len,
        img_feat_dim,
        feat_dim,
        im_channels,
        attn_dim,
        attn_heads,
        attn_dim_head,
        activation,
        weight_tie_layers,
        attn_dropout,
        decoder_dropout,
        img_patch_size,
        final_dim,
        self_cross_ver,
        add_corr,
        add_pixel_loc,
        add_depth,
        pe_fix,
        use_mamba,
        mamba_d_model,
        mamba_bidirectional,
        mamba_d_state,
        mamba_use_pos_enc,
        renderer_device="cuda:0",
    ):
        """MultiView Transfomer"""
        super().__init__()

        # creating a dictonary of all the input parameters
        args = copy.deepcopy(locals())
        del args["self"]
        del args["__class__"]

        # for verifying the input
        self.img_feat_dim = img_feat_dim
        self.add_proprio = add_proprio
        self.proprio_dim = proprio_dim
        self.add_lang = add_lang
        if add_lang:
            lang_emb_dim, lang_max_seq_len = lang_dim, lang_len
        else:
            lang_emb_dim, lang_max_seq_len = 0, 0
        self.lang_emb_dim = lang_emb_dim
        self.lang_max_seq_len = lang_max_seq_len

        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(img_size, img_size),
            with_depth=add_depth,
        )
        self.num_img = self.renderer.num_img
        self.proprio_dim = proprio_dim
        self.img_size = img_size

        if use_mamba:
            print("Using MVT Mamba....")
            self.mvt1 = MVTSingleMamba(**args, renderer=self.renderer)

        else:
            del args["mamba_d_model"]
            del args["use_mamba"]
            del args["mamba_bidirectional"]
            del args["mamba_d_state"]
            del args["mamba_use_pos_enc"]

            print("Using MVT TrF....")
            self.mvt1 = MVTSingle(**args, renderer=self.renderer)
        
        total_params = sum(p.numel() for p in self.mvt1.parameters() if p.requires_grad)
        print("Total params: ", total_params)

    def get_pt_loc_on_img(self, pt, dyn_cam_info, out=None):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, np, x = pt.shape
        assert x == 3
        assert out is None
        out = self.mvt1.get_pt_loc_on_img(pt, dyn_cam_info)

        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """
        wpt = self.mvt1.get_wpt(out, dyn_cam_info, y_q)
        return wpt

    def render(self, pc, img_feat, img_aug, dyn_cam_info):
        mvt = self.mvt1

        # print("img_feat: ", len(img_feat), img_feat[0].shape)
        # print("pc shape: ", len(pc), pc[0].shape)

        with torch.no_grad():
            if dyn_cam_info is None:
                dyn_cam_info_itr = (None,) * len(pc)
            else:
                dyn_cam_info_itr = dyn_cam_info

            if mvt.add_corr:
                img = [
                    self.renderer(
                        _pc,
                        torch.cat((_pc, _img_feat), dim=-1),
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]
            else:
                img = [
                    self.renderer(
                        _pc,
                        _img_feat,
                        fix_cam=True,
                        dyn_cam_info=(_dyn_cam_info,)
                        if not (_dyn_cam_info is None)
                        else None,
                    ).unsqueeze(0)
                    for (_pc, _img_feat, _dyn_cam_info) in zip(
                        pc, img_feat, dyn_cam_info_itr
                    )
                ]

            img = torch.cat(img, 0)
            img = img.permute(0, 1, 4, 2, 3)

            # print("img_feat after render: ", img.shape)

            # for visualization purposes
            if mvt.add_corr:

                # import numpy as np
                # from torchvision.utils import save_image
                # import cv2

                mvt.img = img[:, :, 3:].clone().detach()

                # print("mvt img: ", type(mvt.img), mvt.img.shape, mvt.img.dtype, mvt.img.max(), mvt.img.min())
            
                # # Loop through each image in the batch and save as PNG
                # for i in range(mvt.img.shape[1]):
                #     image = mvt.img[0,i,:3]  # Get the image tensor
                #     # Convert from tensor to numpy array and transpose dimensions

                #     print("vis img: ", image.shape)
                #     image_np = np.transpose(image.detach().cpu().numpy(), (1, 2, 0))
                #     # Normalize pixel values to range [0, 255] (assuming values are in [0, 1])
                #     image_np = (image_np * 255).astype(np.uint8)
                #     # Save as PNG using torchvision
                #     cv2.imwrite(f'image_{i}.png', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            else:
                mvt.img = img.clone().detach()

            # image augmentation
            if img_aug != 0:
                stdv = img_aug * torch.rand(1, device=img.device)
                # values in [-stdv, stdv]
                noise = stdv * ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                img = torch.clamp(img + noise, -1, 1)

            if mvt.add_pixel_loc:
                bs = img.shape[0]
                pixel_loc = mvt.pixel_loc.to(img.device)
                img = torch.cat(
                    (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                )
            
            # print("img_feat after add pixel loc: ", img.shape)
        
        return img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
    ):
        if not self.training:
            # no img_aug when not training
            assert img_aug == 0

        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.img_feat_dim

        if self.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

        if self.add_lang:
            bs4, x4, x5 = lang_emb.shape
            assert bs == bs4
            assert (
                x4 == self.lang_max_seq_len
            ), "Does not support lang_emb of shape {lang_emb.shape}"
            assert (
                x5 == self.lang_emb_dim
            ), "Does not support lang_emb of shape {lang_emb.shape}"
        else:
            assert (lang_emb is None) or (
                torch.all(lang_emb == 0)
            ), f"Invalid input for lang={lang}"

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        import time

        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug)
        # print("img feat type: ", type(img_feat), len(img_feat), img_feat[0].shape, img_feat[1].shape, img_feat[2].shape)

        t_start = time.time()
        img = self.render(
            pc,
            img_feat,
            img_aug,
            dyn_cam_info=None,
        )
        t_end = time.time()
        # print("rendering time. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

        t_start = time.time()
        out = self.mvt1(img=img, proprio=proprio, lang_emb=lang_emb, **kwargs)
        t_end = time.time()
        # print("DNN forward pass. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

        return out

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    mvt = MVT(**cfg)
    breakpoint()
