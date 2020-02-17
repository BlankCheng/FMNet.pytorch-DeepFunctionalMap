import torch
import torch.nn as nn
import torch.nn.functional as F

class GeodesicLoss(nn.Module):
    """
    Geodesic Loss. Calculate total geodesic distance difference between estimated corresponding pairs.
    """
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, Q, dist_x, dist_y):
        """

        :param Q: B * 2048 * 2048, soft correspondence matrix
        :param dist_x: B * 2048 * 2048, geodesic distances of x
        :param dist_y: B * 2048 * 2048, geodesic distances of y
        :return: a scalar as loss
        """
        criterion = nn.MSELoss(reduce=True)
        loss = criterion(dist_x, torch.bmm(Q.transpose(2, 1), torch.bmm(dist_y, Q)))
        return loss
