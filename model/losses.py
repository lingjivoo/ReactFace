import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()

    def __repr__(self):
        return "KLLoss()"



class NeighbourLoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(NeighbourLoss, self).__init__()
        self.rec_loss = nn.SmoothL1Loss(reduction ='none')
        # self.rec_loss = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_3dmm_neighbours, pred_3dmm, distribution):

        pred_3dmm = pred_3dmm.unsqueeze(1).expand_as(gt_3dmm_neighbours)

        rec_3dmm_exp =  (torch.min(self.rec_loss(pred_3dmm[:, :, :, :52], gt_3dmm_neighbours[:,:, :, :52]).mean(dim=(2, 3)), dim=1)[0]).mean()
        rec_3dmm_pose = (torch.min(self.rec_loss(pred_3dmm[:, :, :, 52:], gt_3dmm_neighbours[:,:, :, 52:]).mean(dim=(2, 3)), dim=1)[0]).mean()

        rec_loss = rec_3dmm_exp + 10 * rec_3dmm_pose

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_3dmm_neighbours.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_3dmm_neighbours.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss

        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "NeighbourLoss()"




class VAELoss(nn.Module):
    def __init__(self, kl_p=0.0002):
        super(VAELoss, self).__init__()
        self.rec_loss = nn.SmoothL1Loss(reduce=True, size_average=True)
        # self.rec_loss = nn.MSELoss(reduce=True, size_average=True)
        self.kl_loss = KLLoss()
        self.kl_p = kl_p

    def forward(self, gt_3dmm, pred_3dmm, distribution):
        rec_3dmm_exp =  self.rec_loss(pred_3dmm[:, :, :52], gt_3dmm[:,:, :52]).mean()
        rec_3dmm_pose = self.rec_loss(pred_3dmm[:, :, 52:], gt_3dmm[:,:,  52:]).mean()

        rec_loss = rec_3dmm_exp + 10 * rec_3dmm_pose

        mu_ref = torch.zeros_like(distribution[0].loc).to(gt_3dmm.get_device())
        scale_ref = torch.ones_like(distribution[0].scale).to(gt_3dmm.get_device())
        distribution_ref = torch.distributions.Normal(mu_ref, scale_ref)

        kld_loss = 0
        for t in range(len(distribution)):
            kld_loss += self.kl_loss(distribution[t], distribution_ref)
        kld_loss = kld_loss / len(distribution)

        loss = rec_loss + self.kl_p * kld_loss

        return loss, rec_loss, kld_loss

    def __repr__(self):
        return "VAELoss()"


class DivLoss(nn.Module):
    def __init__(self):
        super(DivLoss, self).__init__()

    def forward(self, Y_1, Y_2):
        loss = 0.0
        b, t, c = Y_1.shape
        Y_g = torch.cat([Y_1.view(b, 1, -1), Y_2.view(b, 1, -1)], dim=1)
        for Y in Y_g:
            dist = F.pdist(Y, 2) ** 2
            loss += (-dist / 100).exp().mean()
        loss /= b
        return loss

    def __repr__(self):
        return "DivLoss()"





class SmoothLoss(nn.Module):
    def __init__(self, k =0.1):
        super(SmoothLoss, self).__init__()
        self.sml1 = nn.SmoothL1Loss(reduce=True, size_average=True)
        self.k = k

    def forward(self, x):
        loss = self.sml1((x[:, 2:, 52:] - x[:, 1:-1, 52:]),
                     (x[:, 1:-1, 52:] - x[:, :-2, 52:])) + \
                self.k * self.sml1(
                    (x[:, 2:, :52] - x[:, 1:-1, :52]),
                    (x[:, 1:-1, :52] - x[:, :-2, :52]))
        return loss

    def __repr__(self):
        return "SmoothLoss"
