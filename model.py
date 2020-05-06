import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

NUM_POINTS = 2048
NUM_EVECS = 150

class FMNet(nn.Module):
    def __init__(self):
        super(FMNet, self).__init__()
        self.feat_refiner = FeatRefineLayer(in_channels=352)
        self.corres = CorresLayer()

    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        """

        :param feat_x: B * 2048 * 352, handcrafted point-wise feature of x
        :param feat_y: B * 2048 * 352, handcrafted point-wise feature of y
        :param evecs_x: B * 2048 * 150, Laplace basis of x, each column is a basis vector
        :param evecs_y: B * 2048 * 150, Laplace basis of y, each column is a basis vector
        :return: Q(2048*2048) is soft point-wise correspondence, C(150*150) functional map
        """
        feat_x, feat_y = self.feat_refiner(feat_x), self.feat_refiner(feat_y)
        Q, C = self.corres(feat_x, feat_y, evecs_x, evecs_y)
        return Q, C


class FeatRefineLayer(nn.Module):
    def __init__(self, in_channels=352, out_channels_list=None):
        super(FeatRefineLayer, self).__init__()
        self.in_channels = in_channels
        if out_channels_list is None:
            out_channels_list = [352, 352, 352, 352, 352, 352, 352]
        self.out_channels_list = out_channels_list
        self.res_layers = nn.ModuleList()
        for out_channels in self.out_channels_list:
            self.res_layers.append(ResLayer(in_channels, out_channels, out_channels))
            in_channels = out_channels

    def forward(self, x):
        """

        :param x: B * 2048 * 352, handcrafted point-wise feature of x
        :return: B * 2048 * 352, refined point-wise feature of x
        """
        for res_layer in self.res_layers:
            x = res_layer(x)
        return x


class ResLayer(nn.ModuleList):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(ResLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm1d(num_features=mid_channels, eps=1e-3, momentum=1e-3)
        self.fc2 = nn.Linear(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(num_features=out_channels, eps=1e-3, momentum=1e-3)
        self.fc3 = None
        if in_channels != out_channels:
            self.fc3 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """

        :param x: B * 2048 * 352, refining point-wise feature of x
        :return: B * 2048 * 352, refining point-wise feature of x
        """
        x_res = F.relu(self.bn1(self.fc1(x).transpose(1, 2)).transpose(1, 2))
        x_res = self.bn2(self.fc2(x_res).transpose(1, 2)).transpose(1, 2)
        if self.in_channels != self.out_channels:
            x = self.fc3(x)
        x_res += x
        return x_res



class CorresLayer(nn.Module):
    def __init__(self):
        super(CorresLayer, self).__init__()

    def forward(self, feat_x, feat_y, evecs_x, evecs_y):
        """

        :param feat_x: B * 2048 * 352, refined point-wise feature of x
        :param feat_y: B * 2048 * 352, refined point-wise feature of y
        :param evecs_x: B * 2048 * 150, Laplace basis of x, each column is a basis vector
        :param evecs_y: B * 2048 * 150, Laplace basis of y, each column is a basis vector
        :return: Q and C
        """
        # solve ls C*A=B, i.e. A.T*C.T=B.T
        batch_size = feat_x.size(0)
        A = torch.bmm(evecs_x.transpose(2, 1), feat_x)
        B = torch.bmm(evecs_y.transpose(2, 1), feat_y)
        A, B = A.transpose(2, 1), B.transpose(2, 1)
        for i in range(batch_size):
            # C_i, _ = torch.gels(B[i], A[i])
            C_i = torch.inverse(A[i].transpose(1, 0)@ A[i]) @ A[i].transpose(1, 0) @ B[i]  # C=(A.T*A)^{-1}*A.T*B
            if i == 0:
                C = C_i.unsqueeze(0)[:, :NUM_EVECS, :]
            else:
                C = torch.cat((C, C_i.unsqueeze(0)[:, :NUM_EVECS, :]), dim=0)
        C = C.transpose(2, 1)
        # function map-> point2point map
        P = abs(torch.bmm(torch.bmm(evecs_y, C), evecs_x.transpose(2, 1)))
        Q = F.normalize(P, 2, 1) ** 2
        return Q, C

if __name__ == '__main__':
    feat_x = torch.rand(8, 2048, 352)
    feat_y = torch.rand(8, 2048, 352)
    evecs_x = torch.rand(8, 2048, 150)
    evecs_y = torch.rand(8, 2048, 150)
    net = FMNet()
    Q, C = net(feat_x, feat_y, evecs_x, evecs_y)
    print(torch.sum(Q, 1))
    print(C.shape)








