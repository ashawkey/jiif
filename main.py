import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import *
from datasets import *
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='jiif')
parser.add_argument('--model', type=str, default='JIIF')
parser.add_argument('--loss', type=str, default='L1')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='NYU')
parser.add_argument('--data_root', type=str, default='./data/nyu_labeled/')
parser.add_argument('--train_batch', type=int, default=1)
parser.add_argument('--test_batch', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--epoch', default=100, type=int, help='max epoch')
parser.add_argument('--eval_interval',  default=10, type=int, help='eval interval')
parser.add_argument('--checkpoint',  default='scratch', type=str, help='checkpoint to use')
parser.add_argument('--scale',  default=8, type=int, help='scale')
parser.add_argument('--interpolation',  default='bicubic', type=str, help='interpolation method to generate lr depth')
parser.add_argument('--lr',  default=0.0001, type=float, help='learning rate')
parser.add_argument('--lr_step',  default=40, type=float, help='learning rate decay step')
parser.add_argument('--lr_gamma',  default=0.2, type=float, help='learning rate decay gamma')
parser.add_argument('--input_size',  default=None, type=int, help='crop size for hr image')
parser.add_argument('--sample_q',  default=30720, type=int, help='sampled pixels per hr depth')
parser.add_argument('--noisy',  action='store_true', help='add noise to train dataset')
parser.add_argument('--test',  action='store_true', help='test mode')
parser.add_argument('--report_per_image',  action='store_true', help='report RMSE of each image')
parser.add_argument('--save',  action='store_true', help='save results')
parser.add_argument('--batched_eval',  action='store_true', help='batched evaluation to avoid OOM for large image resolution')

args = parser.parse_args()

seed_everything(args.seed)

# model
if args.model == 'DKN':
    model = DKN(kernel_size=3, filter_size=15, residual=True)
elif args.model == 'FDKN':
    model = FDKN(kernel_size=3, filter_size=15, residual=True)
elif args.model == 'DJF':
    model = DJF(residual=True)
elif args.model == 'JIIF':
    model = JIIF(args, 128, 128)
else:
    raise NotImplementedError(f'Model {args.model} not found')

# loss
if args.loss == 'L1':
    criterion = nn.L1Loss()
elif args.loss == 'L2':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError(f'Loss {args.loss} not found')

# dataset
if args.dataset == 'NYU':
    dataset = NYUDataset
elif args.dataset == 'Lu':
    dataset = LuDataset
elif args.dataset == 'Middlebury':
    dataset = MiddleburyDataset
elif args.dataset == 'NoisyMiddlebury':
    dataset = NoisyMiddleburyDataset
else:
    raise NotImplementedError(f'Dataset {args.loss} not found')

if args.model in ['JIIF']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation, augment=True, to_pixel=True, sample_q=args.sample_q, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation, augment=False, to_pixel=True, sample_q=None) # full image
elif args.model in ['DJF', 'DKN', 'FDKN']:
    if not args.test:
        train_dataset = dataset(root=args.data_root, split='train', scale=args.scale, downsample=args.interpolation, augment=True, pre_upsample=True, input_size=args.input_size, noisy=args.noisy)
    test_dataset = dataset(root=args.data_root, split='test', scale=args.scale, downsample=args.interpolation, augment=False, pre_upsample=True)
else:
    raise NotImplementedError(f'Dataset for model type {args.model} not found')

# dataloader
if not args.test:
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, pin_memory=True, drop_last=False, shuffle=True, num_workers=args.num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch, pin_memory=True, drop_last=False, shuffle=False, num_workers=args.num_workers)

# trainer
if not args.test:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    trainer = Trainer(args, args.name, model, objective=criterion, optimizer=optimizer, lr_scheduler=scheduler, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint, eval_interval=args.eval_interval)
else:
    trainer = Trainer(args, args.name, model, objective=criterion, metrics=[RMSEMeter(args)], device='cuda', use_checkpoint=args.checkpoint)

# main
if not args.test:
    trainer.train(train_loader, test_loader, args.epoch)

if args.save:
    # save results (doesn't need GT)
    trainer.test(test_loader)
else:
    # evaluate (needs GT)
    trainer.evaluate(test_loader)
