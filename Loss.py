import torch.nn as nn
import torch


class My_Loss(nn.Module):
    def __init__(self, tao, lamda, device, batch_size):
        super(My_Loss, self).__init__()
        self.tao = tao
        self.device = device
        self.lamda = lamda
        self.batch_size = batch_size

    def forward(self, vector1, vector2):
        return self.contrastive_learning_loss(vector1, vector2) + self.lamda * self.alignment_loss(vector1, vector2) + self.lamda * self.uniformity_loss(vector1, vector2)

    def contrastive_learning_loss(self, v1, v2):
        numerator = torch.exp(self.sim1(v1, v2)).to(self.device)
        v2_all_sample = torch.Tensor(self.batch_size, self.batch_size, 768)
        v2_all_sample.copy_(v2)
        v2_all_sample = v2_all_sample.permute(1, 0, 2)
        denominator = torch.sum(torch.exp(self.sim2(v1, v2_all_sample)), 1).to(self.device)
        return -1 * torch.mean(torch.log(numerator / denominator))

    def alignment_loss(self, v1, v2):
        v = v2 - v1
        return torch.sum(torch.norm(v, dim=1), 0) / v1.shape[0]     # / self.tao

    def uniformity_loss(self, v1, v2):
        v2_all_sample = torch.Tensor(self.batch_size, self.batch_size, 768)
        v2_all_sample.copy_(v2)
        v2_all_sample = v2_all_sample.permute(1, 0, 2)
        v_all_sample = v2_all_sample - v1
        # tmp_v = torch.clamp(torch.exp(torch.sum(v1 * v2_all_sample, 2) / self.tao), min=1e-7, max=3e38)
        tmp_v = torch.exp(-2 * torch.norm(v_all_sample, dim=2))       #/ self.tao)
        return torch.sum(torch.log(torch.sum(tmp_v - tmp_v * torch.eye(v1.shape[0]), 1) / (v1.shape[0] - 1)), 0) / v1.shape[0]

    def sim1(self, v1, v2):
        return torch.sum(v1 * v2, 1) / (torch.sqrt(torch.sum(v1 * v1, 1)) * torch.sqrt(torch.sum(v2 * v2, 1)) * self.tao)

    def sim2(self, v1, v2):
        return torch.sum(v1 * v2, 2) / (torch.sqrt(torch.sum(v1 * v1, 1)) * torch.sqrt(torch.sum(v2 * v2, 2)) * self.tao)


class My_Cross_Loss(nn.Module):
    def __init__(self, tao, ita, device):
        super(My_Cross_Loss, self).__init__()
        self.tao = tao
        self.ita = ita
        self.device = device

    def forward(self, vector1, vector2, vector3):
        numerator = torch.exp(self.sim(vector1, vector2)).to(self.device)
        denominator = torch.exp(self.sim(vector1, vector2)).to(self.device) + torch.exp(self.sim(vector1, vector3)).to(self.device)
        return -1 * torch.mean(torch.log(numerator / denominator))
