import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
import numpy as np
import models

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
					help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						'batch size of all GPUs on the current node when '
						'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint-path', default='wideresnet_pruned_model.pth.tar',
					type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
					help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						'N processes per node, which has N GPUs. This is the '
						'fastest way to use PyTorch for either single node or '
						'multi node data parallel training')
parser.add_argument('--wideresnet', action='store_true',
					help='fine tuning WideResNet, architecture flag get ignored')

best_acc1 = 0

def validate_checkpoint(chpt):
    val_checkpoint_state = {}
    for key in chpt:
        val_checkpoint_state[key[7:]] = chpt[key]
    return val_checkpoint_state

def main():
	args = parser.parse_args()
	args.evaluate = True
	args.wideresnet = True

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
					'This will turn on the CUDNN deterministic setting, '
					'which can slow down your training considerably! '
					'You may see unexpected behavior when restarting '
					'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
					'disable data parallelism.')

	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	args.distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	print("Executed the main function")
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function
		mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
		# mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, weight_mask_arr))
	else:
		# Simply call main_worker function
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
# def main_worker(gpu, ngpus_per_node, args, weight_mask_arr):
	global best_acc1
	args.gpu = gpu

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	if args.distributed:
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
								world_size=args.world_size, rank=args.rank)
	# create model
	if args.wideresnet:
		model_cfg = getattr(models, 'WideResNet28x10')
		num_classes = 100
		model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

	if args.distributed:
		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int(args.workers / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		else:
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if torch.cuda.is_available():
			model = torch.nn.DataParallel(model).cuda()

	# define loss function (criterion) and optimizer
	if torch.cuda.is_available():
		criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	else:
		criterion = nn.CrossEntropyLoss()

	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)

	# optionally resume from a checkpoint
	if args.checkpoint_path:
		if os.path.isfile(args.checkpoint_path):
			print("=> loading checkpoint '{}'".format(args.checkpoint_path))
			if torch.cuda.is_available():
				checkpoint = torch.load(args.checkpoint_path)
				model.load_state_dict(checkpoint['state_dict'])
			else:
				checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
				model.load_state_dict(validate_checkpoint(checkpoint['state_dict']))
			optimizer.load_state_dict(checkpoint['optimizer'])
			print("=> loaded checkpoint '{}' (epoch {})"
				.format(args.checkpoint_path, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.checkpoint_path))

	cudnn.benchmark = True

	# Data loading code
	dataset = 'CIFAR100'
	print("Loading dataset {} from {}".format(dataset, args.data))
	ds = getattr(datasets, dataset)
	path = os.path.join(args.data, dataset.lower())
	train_dataset = ds(path, train=True, download=True, transform=model_cfg.transform_train)
	val_dataset = ds(path, train=False, download=True, transform=model_cfg.transform_test)

	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		shuffle=(train_sampler is None),
		num_workers=args.workers,
		pin_memory=True,
		sampler=train_sampler)

	val_loader = torch.utils.data.DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=True)

	if args.evaluate:
		validate(val_loader, model, criterion, args)
		return

def validate(val_loader, model, criterion, args):
	batch_time = AverageMeter('Time', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
							prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			if args.gpu is not None:
				if torch.cuda.is_available():
					input = input.cuda(args.gpu, non_blocking=True)
			if torch.cuda.is_available():
				target = target.cuda(args.gpu, non_blocking=True)

			# compute output
			output = model(input)
			loss = criterion(output, target)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(acc1[0], input.size(0))
			top5.update(acc5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.print(i)

		# TODO: this should also be done with the ProgressMeter
		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			.format(top1=top1, top5=top5))

	return top1.avg


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, *meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def print(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res


if __name__ == '__main__':
	main()
