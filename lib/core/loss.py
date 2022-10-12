import torch
import torch.nn as nn
import numpy as np
import math
import cv2
import torch.nn.functional as F
from lib.models.smpl import SMPL
from lib.utils.geometry_utils import *


def ohkm(loss, topk):
    topk *= 2
    ohkm_loss = 0.
    for i in range(loss.size()[0]):
        sub_loss = loss[i]
        topk_val, topk_idx = torch.topk(
            sub_loss, k=topk, dim=0, sorted=False
        )
        tmp_loss = torch.gather(sub_loss, 0, topk_idx)
        ohkm_loss += torch.sum(tmp_loss) / topk
        # ohkm_loss /= loss.size()[0]
    return ohkm_loss


class HANetLoss(nn.Module):
    def __init__(self, w_decoder, lamada, smpl_model_dir, smpl):
        super().__init__()
        self.w_decoder = w_decoder
        self.lamada = lamada
        self.smpl_model_dir = smpl_model_dir
        self.smpl = smpl
        self.mse_criterion = nn.MSELoss(reduction='sum')

    def mask_lr1_loss(self, inputs, mask, targets, teach, mask_teach):
        Bs, C, L = inputs.shape

        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)

        inputs_list = inputs.split(1, 1)
        not_mask_list = not_mask.split(1, 1)
        targets_list = targets.split(1, 1)
        loss = []

        not_mask_teach = 1 - mask_teach.int()
        not_mask_teach = not_mask_teach.unsqueeze(1).repeat(1, C, 1).float()
        teach_list = teach.split(1, 1)
        not_mask_teach_list = not_mask_teach.split(1, 1)

        for i in range(C):
            pred = inputs_list[i].squeeze()
            not_mask_s = not_mask_list[i].squeeze()
            gt = targets_list[i].squeeze()
            pred_t = teach_list[i].squeeze()
            not_mask_t = not_mask_teach_list[i].squeeze()

            l_s = F.l1_loss(pred * not_mask_s, gt
                            * not_mask_s, reduction='sum') / N
            l_t = F.l1_loss(pred_t * not_mask_t, gt
                            * not_mask_t, reduction='sum') / N

            if l_s > l_t:  # learn from teach
                l_oml = F.l1_loss(
                    pred * not_mask_t, pred_t * not_mask_t, reduction='sum') / N
            else:
                l_oml = F.l1_loss(
                    pred_t * not_mask_t, pred * not_mask_t, reduction='sum') / N

            loss.append(l_s * self.lamada + l_t *
                        self.w_decoder + l_oml * self.w_decoder)
        return loss

    def mask_lr2_loss(self, inputs, mask, targets, teach, mask_teach):
        Bs, C, L = inputs.shape

        not_mask = 1 - mask.int()
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)

        inputs_list = inputs.split(1, 1)
        not_mask_list = not_mask.split(1, 1)
        targets_list = targets.split(1, 1)
        loss = []

        not_mask_teach = 1 - mask_teach.int()
        not_mask_teach = not_mask_teach.unsqueeze(1).repeat(1, C, 1).float()
        teach_list = teach.split(1, 1)
        not_mask_teach_list = not_mask_teach.split(1, 1)

        for i in range(C):
            pred = inputs_list[i].squeeze()
            not_mask_s = not_mask_list[i].squeeze()
            gt = targets_list[i].squeeze()
            pred_t = teach_list[i].squeeze()
            not_mask_t = not_mask_teach_list[i].squeeze()

            l_s = self.mse_criterion(
                pred * not_mask_s, gt * not_mask_s) / N
            l_t = self.mse_criterion(
                pred_t * not_mask_t, gt * not_mask_t) / N

            if l_s > l_t:  # learn from teach
                l_oml = self.mse_criterion(
                    pred * not_mask_t, pred_t * not_mask_t) / N
            else:
                l_oml = self.mse_criterion(
                    pred_t * not_mask_t, pred * not_mask_t) / N  # decoder mask all free

            loss.append(l_s * self.lamada + l_t *
                        self.w_decoder + l_oml * self.w_decoder)
        return loss

    def forward(self,
                hierarchical_encoder,
                decoder,
                gt,
                mask_src,
                mask_pad,
                use_smpl_loss=False):
        B, L, C = hierarchical_encoder.shape

        if use_smpl_loss and self.smpl:
            return self.forward_smpl(hierarchical_encoder, decoder, gt, mask_src, mask_pad)
        else:
            loss_1 = self.forward_lr1(hierarchical_encoder, decoder, gt, mask_src, mask_pad)
            loss_2 = self.forward_lr2(hierarchical_encoder, decoder, gt, mask_src, mask_pad)
            # print(len(loss_1), len(loss_2), loss_1[0].shape)
            loss_1 = [l.unsqueeze(0) for l in loss_1]
            # loss_1 = ohkm(torch.stack(loss_1, dim=1), C//4)
            loss_2 = [l.unsqueeze(0) for l in loss_2]
            loss_2 = ohkm(torch.stack(loss_2, dim=1), C//4)

            # final_loss = ohkm(loss_1, C//4) + ohkm(loss_2, C//4)
            return {"l1_loss": torch.sum(torch.stack(loss_1, dim=1)),
                    # "l2_loss": loss_2,
                    'final_loss': torch.sum(torch.stack(loss_1, dim=1)) + ohkm(torch.stack(loss_1, dim=1), C//4)}

    def forward_lr1(self, hierarchical_encoder, decoder, gt, mask_src, mask_pad):
        B, L, C = hierarchical_encoder.shape
        hierarchical_encoder = hierarchical_encoder.permute(0, 2, 1)
        decoder = decoder.permute(0, 2, 1)  # [b,c,t]
        gt = gt.permute(0, 2, 1)

        weighted_loss = self.mask_lr1_loss(
            hierarchical_encoder, mask_pad, gt, decoder, mask_src)
        # weighted_loss = self.w_decoder * loss_decoder + self.lamada * loss_pose

        return weighted_loss

    def forward_lr2(self, hierarchical_encoder, decoder, gt, mask_src, mask_pad):
        B, L, C = hierarchical_encoder.shape
        hierarchical_encoder = hierarchical_encoder.permute(0, 2, 1)
        decoder = decoder.permute(0, 2, 1)  # [b,c,t]
        gt = gt.permute(0, 2, 1)

        weighted_loss = self.mask_lr2_loss(
            hierarchical_encoder, mask_pad, gt, decoder, mask_src)
        # weighted_loss = self.w_decoder * loss_decoder + self.lamada * loss_pose

        return weighted_loss

    def forward_smpl(self, hierarchical_encoder, decoder, gt, mask_src, mask_pad):
        SMPL_TO_J14 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 38]
        B, L, C = hierarchical_encoder.shape

        hierarchical_encoder = rot6D_to_axis(hierarchical_encoder.reshape(-1, 6)).reshape(-1, 24 * 3)
        decoder = rot6D_to_axis(decoder.reshape(-1, 6)).reshape(-1, 24 * 3)
        gt = rot6D_to_axis(gt.reshape(-1, 6)).reshape(-1, 24 * 3)

        device = hierarchical_encoder.device
        smpl = SMPL(model_path=self.smpl_model_dir,
                    gender="neutral",
                    batch_size=1).to(device)

        gt_smpl_joints = smpl.forward(
            global_orient=gt[:, 0:3].to(torch.float32),
            body_pose=gt[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        decoder_smpl_joints = smpl.forward(
            global_orient=decoder[:, 0:3].to(torch.float32),
            body_pose=decoder[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        hierarchical_encoder_smpl_joints = smpl.forward(
            global_orient=hierarchical_encoder[:, 0:3].to(torch.float32),
            body_pose=hierarchical_encoder[:, 3:].to(torch.float32),
        ).joints[:, SMPL_TO_J14]

        gt_smpl_joints = gt_smpl_joints.reshape(B, L, -1).permute(0, 2, 1)
        decoder_smpl_joints = decoder_smpl_joints.reshape(B, L,
                                                          -1).permute(0, 2, 1)
        hierarchical_encoder_smpl_joints = hierarchical_encoder_smpl_joints.reshape(B, L,
                                                          -1).permute(0, 2, 1)

        loss_decoder = self.mask_lr1_loss(decoder_smpl_joints, mask_src,
                                          gt_smpl_joints)  # mask:[b, t]
        loss_pose = self.mask_lr1_loss(hierarchical_encoder_smpl_joints, mask_pad,
                                       gt_smpl_joints)

        weighted_loss = self.w_decoder * loss_decoder + self.lamada * loss_pose

        return weighted_loss
