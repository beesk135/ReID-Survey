import torch
from torch import nn
import torch.nn.functional as F

# def convert_label_to_similarity(normed_feature: Tensor, \
#                                 label: Tensor) \
#                                 -> Tuple[Tensor, Tensor]:
#     similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
#     label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
#
#     positive_matrix = label_matrix.triu(diagonal=1)
#     negative_matrix = label_matrix.logical_not().triu(diagonal=1)
#
#     similarity_matrix = similarity_matrix.view(-1)
#     positive_matrix = positive_matrix.view(-1)
#     negative_matrix = negative_matrix.view(-1)
#     return similarity_matrix[positive_matrix], \
#             similarity_matrix[negative_matrix]
#
# class CircleLoss(nn.Module):
#     def __init__(self, m: float, gamma: float) -> None:
#         super(CircleLoss, self).__init__()
#         self.m = m
#         self.gamma = gamma
#         self.soft_plus = nn.Softplus()
#
#     def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
#         ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
#         an = torch.clamp_min(sn.detach() + self.m, min=0.)
#
#         delta_p = 1 - self.m
#         delta_n = self.m
#
#         logit_p = -ap * (sp - delta_p) * self.gamma
#         logit_n = an * (sn - delta_n) * self.gamma
#
#         loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + \
#                 torch.logsumexp(logit_p, dim=0))
#         return loss


class CircleLoss(nn.Module):
    def __init__(self, m=0.25, s=96):
        super(CircleLoss, self).__init__()
        self.m = m
        self.s = s
        self.soft_plus = nn.Softplus()

    def forward(self, feat1, label1, feat2_t, label2_t):
        sim_mat = torch.mm(feat1, feat2_t)

        N, M = sim_mat.size()

        is_pos = label1.view(N, 1).expand(N, M).eq(label2_t.expand(N, M)).float()

        same_indx = torch.eye(N, N, device='cuda')
        remain_indx = torch.zeros(N, M - N, device='cuda')
        same_indx = torch.cat((same_indx, remain_indx), dim=1)
        is_pos = is_pos - same_indx

        is_neg = label1.view(N, 1).expand(N, M).ne(label2_t.expand(N, M)).float()

        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg

        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -self.s * alpha_p * (s_p - delta_p)
        logit_n = self.s * alpha_n * (s_n - delta_n)

        loss = F.softplus(torch.logsumexp(logit_p, dim=1) \
            + torch.logsumexp(logit_n, dim=1)).mean()

        return loss
