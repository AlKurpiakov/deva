import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import datetime
import gc
import random
import signal
import sys
import time
import torch
import os

from torch.utils.data import DataLoader, Dataset
from Dataloaders.dataset_loader_stanford import Dataset
# from Dataloaders.dataset_loader_matterport import Dataset
# from Dataloaders.dataset_loader_structured3d import Dataset

from loss import BerhuLoss, Cossim, GradLoss, NormalL1Loss
from models.hush import HUSH
from tqdm import tqdm
from metric import *
from utils import *


LAST_CKPT = 'last.pth'
BEST_MODEL = 'best_model_SF2D3D.pth'


def log(msg):
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{stamp}] {msg}', flush=True)


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------
def save_atomic(obj, path):
    """Write to a temporary file first, so a job killed while saving
    never leaves a truncated checkpoint behind."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # weights_only=True (the default since PyTorch 2.6) cannot restore the
    # numpy RNG state stored in the checkpoint.
    return torch.load(path, map_location='cpu', weights_only=False)


def get_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda'])


def resolve_resume_path(args):
    if args.resume == 'none':
        return None
    if args.resume == 'auto':
        path = os.path.join(args.ckpt_dir, LAST_CKPT)
        return path if os.path.isfile(path) else None
    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f'Checkpoint not found: {args.resume}')
    return args.resume


# ----------------------------------------------------------------------------
# Graceful stop: SIGTERM / SIGUSR1 sent by SLURM before the time limit.
# The current epoch is not saved (only complete epochs are), the job then
# resumes from the last finished epoch.
# ----------------------------------------------------------------------------
STOP_REQUESTED = False


def _request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log(f'Received signal {signum}, stopping after the current step.')


def main(args):
    input_dir = args.rootpath
    train_file = args.trainfile
    val_file = args.valfile
    batch_size = args.bs
    epochs = args.epoch
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # In batch jobs tqdm would flood the log with carriage returns.
    tqdm_kwargs = {} if sys.stderr.isatty() else {'mininterval': 60}

    train_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = True, flip = True, root_path=input_dir, path_to_img_list=train_file),
                                                batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last = True, pin_memory=True )
    val_loader = torch.utils.data.DataLoader(dataset = Dataset(rotate = False, flip = False, root_path=input_dir, path_to_img_list=val_file),
                                                batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last = True, pin_memory=True)


    Net = HUSH(min_val=1e-3, max_val=10, degree=10).cuda()
    optimizer = optim.AdamW(list(Net.parameters()), lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6, last_epoch=-1)

    criterion_depth = BerhuLoss()
    criterion_normal = Cossim()
    criterion_grad = GradLoss()
    best_abs = np.inf
    best_rmse = np.inf
    start_epoch = 0
    history = []

    resume_path = resolve_resume_path(args)
    if resume_path is not None:
        ckpt = load_checkpoint(resume_path)
        # Free each part right after it is restored: the checkpoint holds the
        # model and two AdamW buffers (~2.6 GB) in host memory.
        Net.load_state_dict(ckpt.pop('model'))
        optimizer.load_state_dict(ckpt.pop('optimizer'))
        gc.collect()
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_rmse = ckpt['best_rmse']
        history = ckpt.get('history', [])
        set_rng_state(ckpt['rng'])
        log(f'Resumed from {resume_path}: finished epoch {ckpt["epoch"]}, '
            f'best rmse {best_rmse:.4f}, continuing with epoch {start_epoch}')
        del ckpt
        gc.collect()
    else:
        log('Starting training from scratch')

    if start_epoch >= epochs:
        log(f'All {epochs} epochs are already done, nothing to do.')
        return

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        loss_sum, n_iter = 0.0, 0
        Net.train()
        for batch, (rgb, depth, depth_mask) in tqdm(enumerate(train_loader), **tqdm_kwargs):
            if STOP_REQUESTED:
                log(f'Stopped during epoch {epoch}; it will be repeated after restart.')
                return
            optimizer.zero_grad()

            rgb, depth, depth_mask  = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            normal = compute_normals_full_accelerate(depth)
            pred_depth, pred_normal = Net(rgb)

            depth_loss = criterion_depth(pred_depth, depth, mask=depth_mask.to(torch.bool))
            normal_loss = criterion_normal(pred_normal, normal, mask=depth_mask.to(torch.bool))
            grad_loss = criterion_grad(pred_depth, depth, mask=depth_mask.to(torch.bool))
            loss = depth_loss + normal_loss + 0.5*grad_loss

            if batch % 20 == 0 and batch > 0:
                print('[Epoch %d--Iter %d]Total loss %.4f' % (epoch, batch, loss), flush=True)

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            n_iter += 1

        ## Validation per 1 epoch
        Net.eval()
        val_performance = validate_HUSH(Net, val_loader, tqdm_kwargs)
        val_performance = {k: float(v) for k, v in val_performance.items()}

        if val_performance['rmse'] < best_rmse:
            save_atomic(Net.state_dict(), os.path.join(args.ckpt_dir, BEST_MODEL))
            best_rmse = val_performance['rmse']
            log(f'New best rmse {best_rmse:.4f}, saved {BEST_MODEL}')

        history.append({'epoch': epoch, 'train_loss': loss_sum / max(n_iter, 1),
                        'lr': optimizer.param_groups[0]['lr'], **val_performance})

        # Full training state after every epoch. The RNG state is stored last,
        # so a resumed run continues with exactly the same random stream.
        state = {
            'epoch': epoch,
            'model': Net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_rmse': best_rmse,
            'history': history,
            'args': vars(args),
            'rng': get_rng_state(),
        }
        save_atomic(state, os.path.join(args.ckpt_dir, LAST_CKPT))
        if args.keep_all:
            save_atomic(state, os.path.join(args.ckpt_dir, 'epoch_%03d.pth' % epoch))
        del state
        log(f'Epoch {epoch}/{epochs - 1} done in {(time.time() - epoch_start) / 60:.1f} min, '
            f'train loss {history[-1]["train_loss"]:.4f}, val rmse {val_performance["rmse"]:.4f}, '
            f'checkpoint saved to {os.path.join(args.ckpt_dir, LAST_CKPT)}')

        if STOP_REQUESTED:
            log('Stop requested, exiting after a complete epoch.')
            return

    log('Training finished.')


def validate_HUSH(Net, val_loader, tqdm_kwargs=None):
    val_metrics = ValScore_depth()
    with torch.no_grad():
        for batch, (rgb, depth, depth_mask) in tqdm(enumerate(val_loader), **(tqdm_kwargs or {})):
            rgb, depth, depth_mask = rgb.cuda(), depth.cuda(), depth_mask.cuda()
            pred_depth, _ = Net(rgb)

            pred_depth = F.interpolate(pred_depth, depth.shape[-2:], mode='bilinear', align_corners=True)
            performance, N = compute_depth_errors(pred_depth, depth, depth_mask)
            val_metrics.update(performance, N)
        performances = val_metrics.average()
        print("Validation score :", performances, flush=True)

    return performances


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default='/home/SF2D3D/')
    parser.add_argument('--trainfile', type=str, default='./filenames/train_stanford2d3d.txt')
    parser.add_argument('--valfile', type=str, default='./filenames/test_stanford2d3d.txt')

    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('-g', '--gpu_id', default=None, type=int,
                        help='GPU id to use. By default the GPUs given by CUDA_VISIBLE_DEVICES '
                             '(e.g. allocated by SLURM) are used.')
    parser.add_argument('--num_workers', type=int,
                        default=min(16, int(os.environ.get('SLURM_CPUS_PER_TASK', 16))),
                        help='DataLoader workers (default: 16, or SLURM_CPUS_PER_TASK if smaller)')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints',
                        help='Directory for last.pth (saved every epoch) and the best model')
    parser.add_argument('--resume', type=str, default='auto',
                        help="'auto': continue from <ckpt_dir>/last.pth if it exists (default); "
                             "'none': always start from scratch; or a path to a checkpoint")
    parser.add_argument('--keep_all', action='store_true',
                        help='Also keep a copy of the checkpoint of every epoch (about 2.6 GB each)')
    args = parser.parse_args()

    # GPU ID setting. Do not override CUDA_VISIBLE_DEVICES set by SLURM unless asked to.
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    for sig in ('SIGTERM', 'SIGUSR1'):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _request_stop)

    # Fix seed
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    main(args)
