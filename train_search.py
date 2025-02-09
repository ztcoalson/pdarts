import os
import sys
import time
import glob
import numpy as np
import torch
import pdarts_utils as pdarts_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import copy
from model_search import Network
from genotypes import PRIMITIVES
from genotypes import Genotype

sys.path.append("../poisons/")
from poisons import LabelFlippingPoisoningDataset, CleanLabelPoisoningDataset
from poisons_utils import imshow

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=2, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=25, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment path')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--tmp_data_dir', type=str, default='/tmp/cache/', help='temp data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')
parser.add_argument('--dropout_rate', action='append', default=[], help='dropout rate of skip connect')
parser.add_argument('--add_width', action='append', default=['0'], help='add channels')
parser.add_argument('--add_layers', action='append', default=['0'], help='add layers')
parser.add_argument('--dset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist', 'svhn', 'fashion_mnist'], help='search with cifar100 dataset')
parser.add_argument('--save_full_model', action='store_true', default=False, help='save the entire supernet (used for crafting poisons)')
parser.add_argument('--anti_search', action='store_true', default=False, help='negate searching objective for arch. params')
parser.add_argument('--track_grads', action='store_true', default=False, help='track gradients of arch. and network params')

### new attack args
parser.add_argument('--poisons_type', type=str, choices=['label_flip', 'clean_label', 'none', 'diffusion_denoise'], default='none')
parser.add_argument('--poisons_path', type=str, default=None)

args = parser.parse_args()

if args.dset == 'cifar100':
    TASK_CLASSES = 100
else:
    TASK_CLASSES = 10

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

if args.track_grads:
    full_arch_grad_history = []
    full_param_grad_history = []
else:
    full_arch_grad_history = None
    full_param_grad_history = None

def main():
    if not torch.cuda.is_available():
        print('No GPU device available')
        sys.exit(1)
    
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    
    if args.dset == 'cifar100':
        train_transform, _ = pdarts_utils._data_transforms_cifar100(args)
    elif args.dset == 'mnist':
        train_transform, _ = pdarts_utils._data_transforms_mnist(args)
    elif args.dset == 'fashion_mnist':
        train_transform, _ = pdarts_utils._data_transforms_fashion_mnist(args)
    elif args.dset == 'svhn':
        train_transform, _ = pdarts_utils._data_transforms_svhn(args)
    else:
        train_transform, _ = pdarts_utils._data_transforms_cifar10(args)

    #  prepare dataset
    if args.poisons_type == 'none':
        if args.dset == 'cifar100':
            train_data = dset.CIFAR100(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        elif args.dset == 'mnist':
            train_data = dset.MNIST(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        elif args.dset == 'fashion_mnist':
            train_data = dset.FashionMNIST(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        elif args.dset == 'svhn':
            train_data = dset.SVHN(root=args.tmp_data_dir, split='train', download=True, transform=train_transform)
        else:
            train_data = dset.CIFAR10(root=args.tmp_data_dir, train=True, download=True, transform=train_transform)
        
        n_poisons = 0
    else:
        if args.dset != 'cifar10':
            raise ValueError('CIFAR10 and MNIST not supported for poisoning')

        train_kwargs = {
            'root': args.tmp_data_dir,
            'train': True,
            'download': True,
            'transform': None
        }

        # Build poisoned dataset
        if args.poisons_type == 'label_flip':
            train_data = LabelFlippingPoisoningDataset(args.poisons_path, train_transform, train_kwargs)
            n_poisons = train_data.get_num_poisons()
        elif args.poisons_type == 'clean_label':
            train_data = CleanLabelPoisoningDataset(args.poisons_path, train_transform, train_kwargs)
            n_poisons = train_data.get_num_poisons()
        elif args.poisons_type == 'diffusion_denoise':
            sys.path.append('../diffusion_denoise')
            from utils.datasets import _load_denoised_cifar
            train_transform, _ = pdarts_utils._data_transforms_denoised_diffusion(augment=True, normalize=True)
            train_data, _ = _load_denoised_cifar(
                augment=True, 
                datpath=args.poisons_path,
                train_transform=train_transform
            )

            n_poisons = 0

            # img = train_data[10][0]
            # imshow(img, "TEST_PLS", denormalize=True)
            # exit()
        else:
            raise ValueError('Unknown poisons type: {}'.format(args.poisons_type))

    # make save directory
    if n_poisons > 0:  
        args.save = '{}search-{}-{:.1f}%-{}'.format(args.save, args.note, 
                                                    n_poisons / len(train_data) * 100, 
                                                    time.strftime("%Y%m%d-%H%M%S"))
    else:
        args.save = '{}search-{}-{}'.format(args.save, args.note, 
                                            time.strftime("%Y%m%d-%H%M%S"))
    pdarts_utils.create_exp_dir(args.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", args)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)

    # build Network
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    switches = []
    for i in range(14):
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_normal = copy.deepcopy(switches)
    switches_reduce = copy.deepcopy(switches)
    # To be moved to args
    num_to_keep = [5, 3, 1]
    num_to_drop = [3, 2, 2]
    if len(args.add_width) == 3:
        add_width = args.add_width
    else:
        add_width = [0, 0, 0]
    if len(args.add_layers) == 3:
        add_layers = args.add_layers
    else:
        add_layers = [0, 6, 12]
    if len(args.dropout_rate) ==3:
        drop_rate = args.dropout_rate
    else:
        drop_rate = [0.0, 0.0, 0.0]
    eps_no_archs = [10, 10, 10]
    for sp in range(len(num_to_keep)):
        model = Network(args.init_channels + int(add_width[sp]), TASK_CLASSES, args.layers + int(add_layers[sp]), criterion, switches_normal=switches_normal, switches_reduce=switches_reduce, p=float(drop_rate[sp]))
        model = nn.DataParallel(model)
        model = model.cuda()
        logging.info("param size = %fMB", pdarts_utils.count_parameters_in_MB(model))
        network_params = []
        for k, v in model.named_parameters():
            if not (k.endswith('alphas_normal') or k.endswith('alphas_reduce')):
                network_params.append(v)       
        optimizer = torch.optim.SGD(
                network_params,
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        optimizer_a = torch.optim.Adam(model.module.arch_parameters(),
                    lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)
        sm_dim = -1
        epochs = args.epochs
        eps_no_arch = eps_no_archs[sp]
        scale_factor = 0.2
        for epoch in range(epochs):
            lr = scheduler.get_last_lr()[0]
            logging.info('Epoch: %d lr: %e', epoch, lr)
            epoch_start = time.time()
            # training
            if epoch < eps_no_arch:
                model.module.p = float(drop_rate[sp]) * (epochs - epoch - 1) / epochs
                model.module.update_p()
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=False, anti_search=args.anti_search)
            else:
                model.module.p = float(drop_rate[sp]) * np.exp(-(epoch - eps_no_arch) * scale_factor) 
                model.module.update_p()                
                train_acc, train_obj = train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True, anti_search=args.anti_search)
            scheduler.step()
            
            logging.info('Train_acc %f', train_acc)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds', epoch_duration)
            # validation
            if epochs - epoch < 5:
                valid_acc, valid_obj = infer(valid_queue, model, criterion)
                logging.info('Valid_acc %f', valid_acc)
        pdarts_utils.save(model, os.path.join(args.save, 'weights.pt'))

        if args.save_full_model:
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save, os.path.join(args.save, f'model-{sp}.pt'))

        print('------Dropping %d paths------' % num_to_drop[sp])
        # Save switches info for s-c refinement. 
        if sp == len(num_to_keep) - 1:
            switches_normal_2 = copy.deepcopy(switches_normal)
            switches_reduce_2 = copy.deepcopy(switches_reduce)
        # drop operations with low architecture weights
        arch_param = model.module.arch_parameters()
        normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()        
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_normal[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                # for the last stage, drop all Zero operations
                drop = get_min_k_no_zero(normal_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(normal_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_normal[i][idxs[idx]] = False
        reduce_prob = F.softmax(arch_param[1], dim=-1).data.cpu().numpy()
        for i in range(14):
            idxs = []
            for j in range(len(PRIMITIVES)):
                if switches_reduce[i][j]:
                    idxs.append(j)
            if sp == len(num_to_keep) - 1:
                drop = get_min_k_no_zero(reduce_prob[i, :], idxs, num_to_drop[sp])
            else:
                drop = get_min_k(reduce_prob[i, :], num_to_drop[sp])
            for idx in drop:
                switches_reduce[i][idxs[idx]] = False
        logging.info('switches_normal = %s', switches_normal)
        logging_switches(switches_normal)
        logging.info('switches_reduce = %s', switches_reduce)
        logging_switches(switches_reduce)
        
        if sp == len(num_to_keep) - 1:
            arch_param = model.module.arch_parameters()
            normal_prob = F.softmax(arch_param[0], dim=sm_dim).data.cpu().numpy()
            reduce_prob = F.softmax(arch_param[1], dim=sm_dim).data.cpu().numpy()
            normal_final = [0 for idx in range(14)]
            reduce_final = [0 for idx in range(14)]
            # remove all Zero operations
            for i in range(14):
                if switches_normal_2[i][0] == True:
                    normal_prob[i][0] = 0
                normal_final[i] = max(normal_prob[i])
                if switches_reduce_2[i][0] == True:
                    reduce_prob[i][0] = 0
                reduce_final[i] = max(reduce_prob[i])                
            # Generate Architecture, similar to DARTS
            keep_normal = [0, 1]
            keep_reduce = [0, 1]
            n = 3
            start = 2
            for i in range(3):
                end = start + n
                tbsn = normal_final[start:end]
                tbsr = reduce_final[start:end]
                edge_n = sorted(range(n), key=lambda x: tbsn[x])
                keep_normal.append(edge_n[-1] + start)
                keep_normal.append(edge_n[-2] + start)
                edge_r = sorted(range(n), key=lambda x: tbsr[x])
                keep_reduce.append(edge_r[-1] + start)
                keep_reduce.append(edge_r[-2] + start)
                start = end
                n = n + 1
            # set switches according the ranking of arch parameters
            for i in range(14):
                if not i in keep_normal:
                    for j in range(len(PRIMITIVES)):
                        switches_normal[i][j] = False
                if not i in keep_reduce:
                    for j in range(len(PRIMITIVES)):
                        switches_reduce[i][j] = False
            # translate switches into genotype
            genotype = parse_network(switches_normal, switches_reduce)
            logging.info(genotype)
            ## restrict skipconnect (normal cell only)
            logging.info('Restricting skipconnect...')
            # generating genotypes with different numbers of skip-connect operations
            for sks in range(0, 9):
                max_sk = 8 - sks                
                num_sk = check_sk_number(switches_normal)               
                if not num_sk > max_sk:
                    continue
                while num_sk > max_sk:
                    normal_prob = delete_min_sk_prob(switches_normal, switches_normal_2, normal_prob)
                    switches_normal = keep_1_on(switches_normal_2, normal_prob)
                    switches_normal = keep_2_branches(switches_normal, normal_prob)
                    num_sk = check_sk_number(switches_normal)
                logging.info('Number of skip-connect: %d', max_sk)
                genotype = parse_network(switches_normal, switches_reduce)
                logging.info(genotype)              

def train(train_queue, valid_queue, model, network_params, criterion, optimizer, optimizer_a, lr, train_arch=True, anti_search=False):
    objs = pdarts_utils.AvgrageMeter()
    top1 = pdarts_utils.AvgrageMeter()
    top5 = pdarts_utils.AvgrageMeter()
    
    if args.track_grads:
        arch_grad_history = []
        param_grad_history = []

        for param in model.module.arch_parameters():
            arch_grad_history.append(torch.zeros_like(param.data, device='cpu', requires_grad=False))
        
        for param in network_params:
            param_grad_history.append(torch.zeros_like(param.data, device='cpu', requires_grad=False))

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        if train_arch:
            # In the original implementation of DARTS, it is input_search, target_search = next(iter(valid_queue), which slows down
            # the training when using PyTorch 0.4 and above. 
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda()
            target_search = target_search.cuda(non_blocking=True)
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)

            if anti_search:
                loss_a *= -1

            loss_a.backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

            # add arch grads to history
            if args.track_grads:
                for i, param in enumerate(model.module.arch_parameters()):
                    arch_grad_history[i] += param.grad.data.detach().clone().cpu()

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(network_params, args.grad_clip)
        optimizer.step()

        # add param grads to history
        if args.track_grads:
            for i, param in enumerate(network_params):
                param_grad_history[i] += param.grad.data.detach().clone().cpu()

        prec1, prec5 = pdarts_utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, objs.avg, top1.avg, top5.avg)

    # average arch params and save
    if args.track_grads:
        for i in range(len(arch_grad_history)):
            arch_grad_history[i] /= len(train_queue)
        full_arch_grad_history.append(arch_grad_history)
        torch.save(full_arch_grad_history, os.path.join(args.save, 'arch_grads.pt'))

        for i in range(len(param_grad_history)):
            param_grad_history[i] /= len(train_queue)
        full_param_grad_history.append(param_grad_history)
        torch.save(full_param_grad_history, os.path.join(args.save, 'param_grads.pt'))

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = pdarts_utils.AvgrageMeter()
    top1 = pdarts_utils.AvgrageMeter()
    top5 = pdarts_utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = pdarts_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def parse_network(switches_normal, switches_reduce):

    def _parse_switches(switches):
        n = 2
        start = 0
        gene = []
        step = 4
        for i in range(step):
            end = start + n
            for j in range(start, end):
                for k in range(len(switches[j])):
                    if switches[j][k]:
                        gene.append((PRIMITIVES[k], j - start))
            start = end
            n = n + 1
        return gene
    gene_normal = _parse_switches(switches_normal)
    gene_reduce = _parse_switches(switches_reduce)
    
    concat = range(2, 6)
    
    genotype = Genotype(
        normal=gene_normal, normal_concat=concat, 
        reduce=gene_reduce, reduce_concat=concat
    )
    
    return genotype

def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1
    
    return index
def get_min_k_no_zero(w_in, idxs, k):
    w = copy.deepcopy(w_in)
    index = []
    if 0 in idxs:
        zf = True 
    else:
        zf = False
    if zf:
        w = w[1:]
        index.append(0)
        k = k - 1
    for i in range(k):
        idx = np.argmin(w)
        w[idx] = 1
        if zf:
            idx = idx + 1
        index.append(idx)
    return index
        
def logging_switches(switches):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)
        
def check_sk_number(switches):
    count = 0
    for i in range(len(switches)):
        if switches[i][3]:
            count = count + 1
    
    return count

def delete_min_sk_prob(switches_in, switches_bk, probs_in):
    def _get_sk_idx(switches_in, switches_bk, k):
        if not switches_in[k][3]:
            idx = -1
        else:
            idx = 0
            for i in range(3):
                if switches_bk[k][i]:
                    idx = idx + 1
        return idx
    probs_out = copy.deepcopy(probs_in)
    sk_prob = [1.0 for i in range(len(switches_bk))]
    for i in range(len(switches_in)):
        idx = _get_sk_idx(switches_in, switches_bk, i)
        if not idx == -1:
            sk_prob[i] = probs_out[i][idx]
    d_idx = np.argmin(sk_prob)
    idx = _get_sk_idx(switches_in, switches_bk, d_idx)
    probs_out[d_idx][idx] = 0.0
    
    return probs_out

def keep_1_on(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    for i in range(len(switches)):
        idxs = []
        for j in range(len(PRIMITIVES)):
            if switches[i][j]:
                idxs.append(j)
        drop = get_min_k_no_zero(probs[i, :], idxs, 2)
        for idx in drop:
            switches[i][idxs[idx]] = False            
    return switches

def keep_2_branches(switches_in, probs):
    switches = copy.deepcopy(switches_in)
    final_prob = [0.0 for i in range(len(switches))]
    for i in range(len(switches)):
        final_prob[i] = max(probs[i])
    keep = [0, 1]
    n = 3
    start = 2
    for i in range(3):
        end = start + n
        tb = final_prob[start:end]
        edge = sorted(range(n), key=lambda x: tb[x])
        keep.append(edge[-1] + start)
        keep.append(edge[-2] + start)
        start = end
        n = n + 1
    for i in range(len(switches)):
        if not i in keep:
            for j in range(len(PRIMITIVES)):
                switches[i][j] = False  
    return switches  

if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total searching time: %ds', duration)
