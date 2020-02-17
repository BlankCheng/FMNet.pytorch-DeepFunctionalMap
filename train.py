import argparse
from tqdm import tqdm
import os
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as DT

from dataloader import FAUSTDataset, KeyPointDataset
from model import FMNet
from utils.loss import GeodesicLoss


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['faust', 'keypointnet'])
    parser.add_argument('--category', type=str, default='02691156')  # only for keypointnet
    parser.add_argument('--root_path', type=str, default='./')
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=str, default='1e-3')
    parser.add_argument('--max_epochs', type=int, default=300)
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
    save_dir = os.path.join(args.root_path, 'checkpoints', args.dataset)

    # load train data
    if args.dataset == 'faust':
        train_dataset = FAUSTDataset(args=args)
    elif args.dataset == 'keypointnet':
        train_dataset = KeyPointDataset(args=args)
    else:
        raise NotImplementedError("Dataset not implemented now.")
    train_dataloader = DT.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    # network config
    net = FMNet()
    optimizer = optim.Adam(net.parameters(), lr=eval(args.lr), betas=(0.9, 0.999))
    schedular = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = GeodesicLoss()
    net = net.cuda().float()

    # train
    for epoch in range(1, args.max_epochs+1):
        schedular.step()
        i = 0
        for data in tqdm(train_dataloader):
            x, y, evecs_x, evecs_y, feat_x, feat_y, dist_x, dist_y = data  # FAUST has no keypoints
            x, y, evecs_x, evecs_y, feat_x, feat_y, dist_x, dist_y = \
            x.cuda(), y.cuda(), evecs_x.cuda(), evecs_y.cuda(), feat_x.cuda(), feat_y.cuda(), dist_x.cuda(), dist_y.cuda()
            optimizer.zero_grad()
            net = net.train()
            Q, C = net(feat_x, feat_y, evecs_x, evecs_y)
            loss = criterion(Q, dist_x, dist_y)
            loss.backward()
            optimizer.step()
            i += 1
            print("#epoch:{}, #batch:{}, loss:{}".format(epoch, i+1, loss))

        if epoch % 20 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))





