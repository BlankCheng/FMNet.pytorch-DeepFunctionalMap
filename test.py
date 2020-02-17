import argparse
from tqdm import tqdm
import os
import scipy.io as sio
import torch
import torch.nn.parallel
import torch.utils.data as DT

from dataloader import FAUSTDataset, KeyPointDataset
from model import FMNet
from utils.loss import GeodesicLoss


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['faust', 'keypointnet'])
    parser.add_argument('--category', type=str, default='02691156')  # only for keypointnet
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # init
    args = arg_parse()
    if args.seed != -1:
        torch.manual_seed(args.seed)

    # load test data
    if args.dataset == 'faust':
        test_dataset = FAUSTDataset(args=args)
    elif args.dataset == 'keypointnet':
        test_dataset = KeyPointDataset(args=args)
    else:
        raise NotImplementedError("Dataset not implemented now.")
    test_dataloader = DT.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # network config
    model_path = os.path.join(args.root_path, 'checkpoints', args.dataset, args.model_name)
    if not os.path.exists(model_path):
        print("No trained model in {}".format(model_path))
        exit(-1)
    print(model_path)
    net = FMNet()
    net.load_state_dict(torch.load(model_path))
    net = net.cuda().float()
    criterion = GeodesicLoss()

    # train
    i = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            i += 1
            x, y, evecs_x, evecs_y, feat_x, feat_y, dist_x, dist_y = data  # FAUST has no keypoints
            x, y, evecs_x, evecs_y, feat_x, feat_y, dist_x, dist_y = \
            x.cuda(), y.cuda(), evecs_x.cuda(), evecs_y.cuda(), feat_x.cuda(), feat_y.cuda(), dist_x.cuda(), dist_y.cuda()
            net = net.eval()
            Q, C = net(feat_x, feat_y, evecs_x, evecs_y)
            loss = criterion(Q, dist_x, dist_y)
            print("loss:{}".format(loss))
            sio.savemat(os.path.join(args.root_path, 'results', args.dataset, 'pair{}.mat'.format(i)),  # note #pair is not related to #ply
                        {'x': x.cpu().detach().numpy().squeeze(),
                         'y': y.cpu().detach().numpy().squeeze(),
                         'Q': Q.cpu().detach().numpy().squeeze(),
                         })






