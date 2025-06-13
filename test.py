import argparse
import os
import random
import time
import warnings
import copy
import numpy as np

import torch
import torch.nn as nn  ### MODIFIED ###: 导入nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF  ### MODIFIED ###: 导入functional
from utils import AverageMeter, ProgressMeter, ToTensor, accuracy, normalize_image, parse_gpus
from report_acc_regime import init_acc_regime, update_acc_regime
# ### MODIFIED ###: 以下两个导入不再需要，因为我们不再计算分类准确率
# from report_acc_regime import init_acc_regime, update_acc_regime
# from loss import BinaryCrossEntropy
from checkpoint import save_checkpoint, load_checkpoint
from thop import profile
from networks import create_net
from loss import BinaryCrossEntropy

# ... (parser 定义部分保持不变)
parser = argparse.ArgumentParser(description='PredRNet-RAISE-Fusion for Image Reconstruction')
# dataset settings
parser.add_argument('--dataset-dir', default='/home/scxhc1/AVR-PredRNet/data/datasets',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='RAVEN',
                    help='dataset name')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--image-size', default=80, type=int,
                    help='image size')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
# network settings
parser.add_argument('-a', '--arch', metavar='ARCH', default='predrnet_raven',
                    help='model architecture')
parser.add_argument('--num-extra-stages', default=0, type=int,
                    help='number of extra normal residue blocks or predictive coding blocks')
parser.add_argument('--classifier-hidreduce', default=4, type=int,
                    help='classifier hidden dimension scale')
parser.add_argument('--block-drop', default=0.0, type=float,
                    help="dropout within each block")
parser.add_argument('--classifier-drop', default=0.0, type=float,
                    help="dropout within classifier block")
parser.add_argument('--num-filters', default=32, type=int,
                    help="basic filters for backbone network")
parser.add_argument('--in-channels', default=1, type=int,
                    help="input image channels")
# training settings
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# others settings
parser.add_argument("--ckpt", default="ckpts/test_raven_raise_",
                    help="folder to output checkpoints")
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default="0",
                    help='GPU id to use.')
parser.add_argument('-p', '--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument("--fp16", action='store_true',
                    help="whether to use fp16 for training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--show-detail', action='store_true',
                    help="whether to show detail accuracy on all sub-types")
parser.add_argument('--subset', default='None', type=str,
                    help='subset selection for dataset')
parser.add_argument("--pic-dir", default='/home/scxhc1/AVR-PredRNet/reconstructed_images', type=str)

# ... (seed_worker, get_data_loader 函数保持不变)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loader(args, data_split='train', transform=None, subset=None):
    if args.dataset_name in ('RAVEN', 'RAVEN-FAIR', 'I-RAVEN'):
        from data import RAVEN as create_dataset
    elif 'PGM' in args.dataset_name:
        from data import PGM as create_dataset
    elif 'Analogy' in args.dataset_name:
        from data import Analogy as create_dataset
    elif 'CLEVR-Matrix' in args.dataset_name:
        from data import CLEVR_MATRIX as create_dataset
    else:
        raise ValueError(
            "not supported dataset_name = {}".format(args.dataset_name)
        )
    dataset = create_dataset(
        args.dataset_dir, data_split=data_split, image_size=args.image_size,
        transform=transform, subset=subset
    )
    if args.seed is not None:
        g = torch.Generator()
        g.manual_seed(args.seed)
    else:
        g = None
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(data_split == "train"),
        num_workers=args.workers, pin_memory=True, sampler=None,
        generator=g, worker_init_fn=seed_worker,
    )
    return data_loader


best_loss = float('inf')  ### MODIFIED ###: 从 best_acc1 改为 best_loss, 越小越好


def main():
    # ... (main 函数的前半部分基本不变)
    args = parser.parse_args()
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name)
    args.ckpt += args.dataset_name
    args.ckpt += "-" + args.arch
    if "pred" in args.arch:
        args.ckpt += "-prb" + str(args.num_extra_stages)
    else:
        args.ckpt += "-ext" + str(args.num_extra_stages)
    if args.block_drop > 0.0 or args.classifier_drop > 0.0:
        args.ckpt += "-b" + str(args.block_drop) + "c" + str(args.classifier_drop)
    args.ckpt += "-imsz" + str(args.image_size)
    args.ckpt += "-wd" + str(args.weight_decay)
    args.ckpt += "-ep" + str(args.epochs)
    args.gpu = parse_gpus(args.gpu)
    if args.gpu is not None:
        args.device = torch.device("cuda:{}".format(args.gpu[0]))
    else:
        args.device = torch.device("cpu")
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        args.ckpt += '-seed' + str(args.seed)
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    main_worker(args)


def main_worker(args):
    global best_loss  ### MODIFIED ###: 使用 best_loss

    # create model
    model = create_net(args)

    log_path = os.path.join(args.ckpt, "log.txt")
    log_file = open(log_path, mode="a" if os.path.exists(log_path) else "w")

    for key, value in vars(args).items():
        log_file.write('{0}: {1}\n'.format(key, value))
    args.log_file = log_file

    # ... (FLOPs 计算部分不变)
    model_flops = copy.deepcopy(model)
    if "Analogy" in args.dataset_name:
        x = torch.randn(1, 9, args.image_size, args.image_size)
    elif "CLEVR-Matrix" in args.dataset_name:
        x = torch.randn(1, 16, 3, args.image_size, args.image_size)
    else:
        # ### MODIFIED ###: 模型的forward签名变了，需要提供两个参数来计算FLOPs
        images = torch.randn(1, 16, args.image_size, args.image_size)
        target = torch.tensor([0])
    # flops, params = profile(model_flops, inputs=(x,)) # 旧的调用方式
    flops, params = profile(model_flops, inputs=(images, target))  # 新的调用方式

    print("model [%s] - params: %.6fM" % (args.arch, params / 1e6))
    print("model [%s] - FLOPs: %.6fG" % (args.arch, flops / 1e9))
    args.log_file.write("--------------------------------------------------\n")
    args.log_file.write("Network - " + args.arch + "\n")
    args.log_file.write("Params - %.6fM" % (params / 1e6) + "\n")
    args.log_file.write("FLOPs - %.6fG" % (flops / 1e9) + "\n")

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.device)
        model = model.to(args.gpu[0])
        model = torch.nn.DataParallel(model, args.gpu)

    # ### MODIFIED ###: 定义新的损失函数和优化器
    criterion1 = nn.MSELoss().to(args.device)
    criterion2 = BinaryCrossEntropy().cuda(args.device)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        # 注意: 加载checkpoint时，best_acc需要处理成best_loss
        model, optimizer, best_acc, start_epoch = load_checkpoint(args, model, optimizer)
        best_loss = 1.0 - best_acc  # 简单转换，或从checkpoint中保存loss
        args.start_epoch = start_epoch

    # ... (数据加载部分不变)
    tr_transform = transforms.Compose([ToTensor()])  # 移除了RandomFlip，重建任务中通常不用
    ts_transform = transforms.Compose([ToTensor()])
    tr_loader = get_data_loader(args, data_split='train', transform=tr_transform, subset=args.subset)
    vl_loader = get_data_loader(args, data_split='val', transform=ts_transform, subset=args.subset)
    ts_loader = get_data_loader(args, data_split='test', transform=ts_transform, subset=args.subset)
    args.log_file.write(f"Number of training samples: {len(tr_loader.dataset)}\n")
    args.log_file.write(f"Number of validation samples: {len(vl_loader.dataset)}\n")
    args.log_file.write(f"Number of testing samples: {len(ts_loader.dataset)}\n")
    args.log_file.write("--------------------------------------------------\n")
    args.log_file.close()

    if args.evaluate:
        validate(ts_loader, model, criterion1, criterion2, args, valid_set="Test")
        return

    if args.fp16:
        args.scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.start_epoch, args.epochs):
        args.log_file = open(log_path, mode="a")

        train(tr_loader, model, criterion1, criterion2, optimizer, epoch, args)
        val_loss = validate(vl_loader, model, criterion1, criterion2, args, valid_set="Valid")

        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_best:
            test_loss = validate(ts_loader, model, criterion1, criterion2, args, valid_set="Test")
            best_test_loss = test_loss
        else:
            best_test_loss = -1  # Or keep the last best

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_loss": best_loss,  ### MODIFIED ###
            "optimizer": optimizer.state_dict(),
        }, is_best, epoch, save_path=args.ckpt)

        epoch_msg = (
            f"----------- Best Loss at epoch {epoch}: Valid {best_loss:.6f} Test {best_test_loss:.6f} -----------")
        print(epoch_msg)
        args.log_file.write(epoch_msg + "\n")
        args.log_file.close()


def train(data_loader, model, criterion1, criterion2, optimizer, epoch, args):
    # ### MODIFIED ###: 重写train函数以适应新任务
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')  # 调整精度
    top1 = AverageMeter('Acc', ':6.2f')

    progress = ProgressMeter(len(data_loader), [batch_time, data_time, losses, top1], prefix=f"Epoch: [{epoch}]")

    curr_lr = optimizer.param_groups[0]["lr"]
    model.train()
    end = time.time()

    for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        images = normalize_image(images)

        # 前向传播
        reconstructed, output = model(images, target)

        # 准备Ground Truth
        gt_panels = model.module.ground_truth_panels if isinstance(model,
                                                                   nn.DataParallel) else model.ground_truth_panels
        b, num_panels, h, w = gt_panels.shape
        gt_panels = gt_panels.view(b * num_panels, 1, h, w)

        # 尺寸对齐
        gt_panels_resized = TF.resize(gt_panels, size=reconstructed.shape[-2:])

        # 计算损失
        loss = criterion1(reconstructed, gt_panels_resized) + criterion2(output, target)

        # 反向传播
        if args.fp16:
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        # losses.update(loss.item(), images.size(0))
        top1.update(acc1[0][0], images.size(0))

        if i % args.print_freq == 0 or i == len(data_loader) - 1:
            epoch_msg = progress.get_message(i + 1)
            epoch_msg += (f"\tLr  {curr_lr:.6f}")
            print(epoch_msg)
            args.log_file.write(epoch_msg + "\n")



def validate(data_loader, model, criterion1, criterion2, args, valid_set='Valid'):
    if 'RAVEN' in args.dataset_name:
        acc_regime = init_acc_regime(args.dataset_name)
    else:
        acc_regime = None

    # ### MODIFIED ###: 重写validate函数
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(len(data_loader), [batch_time, losses, top1], prefix=f'{valid_set}: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, meta_target, structure_encoded, data_file) in enumerate(data_loader):
            images = images.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)
            images = normalize_image(images)

            # 计算输出
            reconstructed, output = model(images, target)

            # 准备Ground Truth
            gt_panels = model.module.ground_truth_panels if isinstance(model,
                                                                       nn.DataParallel) else model.ground_truth_panels
            b, num_panels, h, w = gt_panels.shape
            gt_panels = gt_panels.view(b * num_panels, 1, h, w)
            gt_panels_resized = TF.resize(gt_panels, size=reconstructed.shape[-2:])

            loss = criterion1(reconstructed, gt_panels_resized) + criterion2(output, target)
            losses.update(loss.item(), images.size(0))

            # measure accuracy and record loss
            acc1 = accuracy(output, target)

            top1.update(acc1[0][0], images.size(0))

            if acc_regime is not None:
                update_acc_regime(args.dataset_name, acc_regime, output, target, structure_encoded, data_file)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(data_loader) - 1:
                epoch_msg = progress.get_message(i + 1)
                print(epoch_msg)


        if acc_regime is not None:
            for key in acc_regime.keys():
                if acc_regime[key] is not None:
                    if acc_regime[key][1] > 0:
                        acc_regime[key] = float(acc_regime[key][0]) / acc_regime[key][1] * 100
                    else:
                        acc_regime[key] = None

            mean_acc = 0
            for key, val in acc_regime.items():
                mean_acc += val
            mean_acc /= len(acc_regime.keys())
        else:
            mean_acc = top1.avg
        epoch_msg = '----------- {valid_set} Acc {mean_acc:.3f}, Avg Loss {losses:.6} -----------'.format(
            valid_set=valid_set, mean_acc=mean_acc, losses=losses.avg
        )
        print(epoch_msg)

        if not args.evaluate:
            args.log_file.write(epoch_msg + "\n")

        if args.show_detail:
            if acc_regime is not None:
                for key, val in acc_regime.items():
                     print("configuration [{}] Acc {:.3f}".format(key, val))
        else:
            print(f"Mean Accuracy: {mean_acc:.2f}%")

    return mean_acc


if __name__ == '__main__':
    main()