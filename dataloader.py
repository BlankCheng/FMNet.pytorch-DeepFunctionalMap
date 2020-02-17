import os
import torch
import torch.utils.data as DT
import scipy.io as sio

class FAUSTDataset(DT.Dataset):
    def __init__(self, args):
        super(FAUSTDataset, self).__init__()
        self.data_path = os.path.join(args.root_path, 'data', args.dataset,  args.phase)
        self.pcds = []
        self.evecs = []
        self.feats = []
        self.dists = []  # not input into network, only for loss
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                f = sio.loadmat(os.path.join(root, file))
                self.pcds.append(f['pcd'])
                self.evecs.append(f['evecs'])
                self.feats.append(f['feat'])
                self.dists.append(f['dist'])

    def __getitem__(self, item):
        """

        :param item: index, index*2 are chosen as pairs
        :return: (pcd_x, pcd_y, evecs_x, evecs_y, feat_x, feat_y, keypoints_x, keypoints_y, dist_x, dist_y)
        """
        x, y = self.pcds[item], self.pcds[item*2]
        evecs_x, evecs_y = self.evecs[item], self.evecs[item*2]  # not rotational invariant
        feat_x, feat_y = self.feats[item], self.feats[item*2] # not rotational invatriant
        dist_x, dist_y = self.dists[item], self.dists[item*2]
        return torch.tensor(x).float(), torch.tensor(y).float(),\
               torch.tensor(evecs_x).float(), torch.tensor(evecs_y).float(), \
               torch.tensor(feat_x).float(), torch.tensor(feat_y).float(), \
               torch.tensor(dist_x).float(), torch.tensor(dist_y).float()

    def __len__(self):
        return len(self.pcds) // 2


class KeyPointDataset(DT.Dataset):
    def __init__(self, args):
        super(KeyPointDataset, self).__init__()
        self.data_path = os.path.join(args.root_path, 'data', args.dataset, args.phase, args.category)
        self.pcds = []
        self.evecs = []
        self.feats = []
        self.keypoints = []  # not input into network, only for loss
        self.dists = []  # not input into network, only for loss
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                f = sio.loadmat(os.path.join(root, file))
                self.pcds.append(f['pcd'])
                self.evecs.append(f['evecs'])
                self.feats.append(f['feat'])
                self.keypoints.append(f['keypoints'])
                self.dists.append(f['dist'])

    def __getitem__(self, item):
        """

        :param item: index, index*2 are chosen as pairs
        :return: (pcd_x, pcd_y, evecs_x, evecs_y, feat_x, feat_y, keypoints_x, keypoints_y, dist_x, dist_y)
        """
        x, y = self.pcds[item], self.pcds[item*2]
        evecs_x, evecs_y = self.evecs[item], self.evecs[item*2]  # related with rotation and translation
        feat_x, feat_y = self.feats[item], self.feats[item*2] # related with rotation and translation
        keypoints_x, keypoints_y = self.keypoints[item], self.keypoints[item*2]
        dist_x, dist_y = self.dists[item], self.dists[item*2]
        return torch.tensor(x).float(), torch.tensor(y).float(),\
               torch.tensor(evecs_x).float(), torch.tensor(evecs_y).float(), \
               torch.tensor(feat_x).float(), torch.tensor(feat_y).float(), \
               torch.tensor(keypoints_x).int(), torch.tensor(keypoints_y).int(), \
               torch.tensor(dist_x).float(), torch.tensor(dist_y).float()

    def __len__(self):
        return len(self.pcds) // 2

# DATA DIR STRUCTURE
# --FMNet.pytorch
#     --data
#         --KeypointNet
#               --train
#                 --03001627
#                     --1.mat
#                     --2.mat
#                     --3.mat
#                 --02691159
#                     --1.mat
#                     --2.mat
#                     --3.mat
#         --FAUST
#               --train
#                 --1.mat
#               --test
#                 --1.mat
