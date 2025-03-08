########################################################
#                                                      #
#       author: omitted for anonymous submission       #
#                                                      #
#     credits and copyright coming upon publication    #
#                                                      #
########################################################


import os
import sys

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from utils.lovasz_losses import lovasz_softmax


def build_lovasz(wce=False, lovasz=True, num_class=20, ignore_label=0, weight=1):
    weights = torch.ones(num_class+1).cuda()
    weights[-1] = weight
    loss_funs = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_label)

    if wce and lovasz:
        return loss_funs, lovasz_softmax
    elif wce and not lovasz:
        return wce
    elif not wce and lovasz:
        return lovasz_softmax
    else:
        raise NotImplementedError


class ContrastiveLoss(nn.Module):
    def __init__(self, n_classes=18):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, emb_k, emb_q, labels, coor_ori, epoch):
        """
        emb_k: the feature bank with the aggregated embeddings over the iterations
        emb_q: the embeddings for the current iteration
        labels: the correspondent class labels for each sample in emb_q
        """
        if epoch or emb_k is not None:
            total_loss = torch.tensor(0.0).cuda()
            assert (
                    emb_q.shape[0] == labels.shape[0]
            ), "mismatch on emb_q and labels shapes!"
            emb_k = F.normalize(emb_k, dim=-1)
            emb_q = F.normalize(emb_q, dim=1)

            labels = labels.long().cuda() - 1
            labels[labels < 0] = 66

            for i, emb in enumerate(emb_q):
                batch_valid_idx = torch.where(coor_ori[:, 0] == i)
                batch_contents_coor = coor_ori[batch_valid_idx]
                batch_valid_label = labels[batch_contents_coor.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                batch_valid_feat = emb_q.permute(0, 2, 3, 4, 1)
                batch_valid_feat = batch_valid_feat[batch_contents_coor.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()

                if not (66 in batch_valid_label.unique() and len(batch_valid_label.unique()) == 1):
                    batch_valid_label[batch_valid_label == 66] = self.n_classes

                    label_sq = torch.unique(batch_valid_label, return_inverse=True)[1]
                    oh_label = (F.one_hot(label_sq)).unsqueeze(-2)  # one hot labels
                    count = oh_label.view(-1, oh_label.shape[-1]).sum(
                        dim=0
                    )  # num of voxels per class
                    pred = batch_valid_feat.unsqueeze(-1)
                    oh_pred = (
                            pred * oh_label
                    )  # (N, K, Ncp) Ncp num classes present in the batch_valid_label
                    res_raw = oh_pred.sum(dim=0) / count  # avg feat per class

                    res_new = (res_raw[~res_raw.isnan()]).view(
                        -1, self.n_classes
                    )  # filter out nans given by intermediate classes (present because of oh)

                    label_list = batch_valid_label.unique()
                    if self.n_classes in label_list:
                        label_list = label_list[:-1]
                        res_new = res_new[:-1, :]

                    # temperature-scaled cosine similarity
                    final = (res_new.cuda() @ emb_k.T.cuda()) / 0.1

                    loss = F.cross_entropy(final, label_list)
                    total_loss += loss

            return total_loss / emb_q.shape[0]

        return torch.tensor(0).cuda()


class CenterLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.count = torch.zeros(self.n_classes, dtype=torch.int64).cuda()  # count for class
        # mean features from every class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        self.criterionl2 = torch.nn.MSELoss(reduction="none")

        self.previous_features = None
        self.previous_count = None

    @torch.no_grad()
    def cumulate(self, logits: torch.Tensor, sem_gt: torch.Tensor):
        sem_pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        gt_labels = torch.unique(sem_gt).tolist()
        for label in gt_labels:
            if label == 66:
                continue
            sem_gt_current = sem_gt == label
            sem_pred_current = sem_pred == label
            tps_current = torch.logical_and(sem_gt_current, sem_pred_current)
            if tps_current.sum() == 0:
                continue
            
            logits_tps = logits[torch.where(tps_current == 1)]
            avg_mav = torch.mean(logits_tps, dim=0)
            n_tps = logits_tps.shape[0]

            # features is running mean for mav
            self.features[label] = (
                    self.features[label] * self.count[label] + avg_mav * n_tps
            )

            self.count[label] += n_tps
            self.features[label] /= self.count[label] + 1e-8

    def forward(
            self, logits: torch.Tensor, sem_gt: torch.Tensor, is_train: torch.bool
    ) -> torch.Tensor:
        if is_train:
            # update mav only at training time
            sem_gt = sem_gt.type(torch.uint8)
            self.cumulate(logits, sem_gt)
        if self.previous_features is None:
            return torch.tensor(0.0).cuda()
        gt_labels = torch.unique(sem_gt).tolist()

        acc_loss = torch.tensor(0.0).cuda()
        for label in gt_labels[:-1]:
            mav = self.features[label]
            logs = logits[torch.where(sem_gt == label)]
            
            mav = mav.expand(logs.shape[0], -1)
            if self.count[label] > 0:
                ew_l2 = 0.05 * self.criterionl2(logs, mav)
                acc_loss += ew_l2.mean()

        return acc_loss

    def update(self):
        self.previous_features = self.features
        self.previous_count = self.count

        # resetting for next epoch
        self.count = torch.zeros(self.n_classes, dtype=torch.int64).cuda()  # count for class
        self.features = {
            i: torch.zeros(self.n_classes).cuda() for i in range(self.n_classes)
        }

        return self.previous_features

    def read(self):
        mav_tensor = torch.zeros(self.n_classes, self.n_classes)
        for key in self.previous_features.keys():
            mav_tensor[key] = self.previous_features[key]
        return mav_tensor


class ObjectosphereLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, logits, sem_gt):
        logits_unk = logits[torch.where(sem_gt == -1)]
        logits_kn = logits[torch.where((sem_gt != 66) & (sem_gt != -1))]

        if len(logits_unk):
            loss_unk = torch.linalg.norm(logits_unk, dim=1).mean()
        else:
            loss_unk = torch.tensor(0)
        if len(logits_kn):
            loss_kn = F.relu(self.sigma - torch.linalg.norm(logits_kn, dim=1)).mean()
        else:
            loss_kn = torch.tensor(0)
        
        loss = 10 * loss_unk + loss_kn
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight, device):
        super(CrossEntropyLoss, self).__init__()
        self.weight = torch.tensor(weight).to(device)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2 ** 8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction="none",
            ignore_index=-1,
        )
        self.ce_loss.to(device)

    def forward(self, inputs, targets):
        losses = []
        targets_m = targets.clone()
        if targets_m.sum() == 0:
            import ipdb;
            ipdb.set_trace()  # fmt: skip
        targets_m -= 1
        loss_all = self.ce_loss(inputs, targets_m.long())
        number_of_pixels_per_class = torch.bincount(
            targets.flatten().type(self.dtype), minlength=self.num_classes
        )
        divisor_weighted_pixel_sum = torch.sum(
            number_of_pixels_per_class[1:] * self.weight
        )  # without void
        if divisor_weighted_pixel_sum > 0:
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
        else:
            losses.append(torch.tensor(0.0).cuda())

        return losses