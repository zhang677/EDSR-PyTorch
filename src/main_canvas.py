import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import gc
import itertools
import random
import logging
import ptflops
import oss2
import datetime
import json
import math
import os
from copy import deepcopy
from Canvas import canvas

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            total_timer = utility.timer()
            total_timer.tic()
            while not t.terminate():
                t.train()
                t.test()
            total_timer.hold()
            print("Total training time: {:.1f}s".format(total_timer.release()))
            checkpoint.done()

def get_model(model):
    # The output channel of EDSR is args.n_colors=3
    # Input shape is [16, 3, 48, 48] = [B, C, H, W]
    # Input shape is [16, 3, 96, 96] = [B, C, H, W] (patch_size=96)
    pass
logging.basicConfig(level=logging.DEBUG)
_exp_logger = logging.getLogger()
_exp_logger.setLevel(logging.INFO)
oss_try_times = 10

def get_logger():
    global _exp_logger
    return _exp_logger

def get_oss_bucket():
    return oss2.Bucket(oss2.Auth('LTAI5tCx79brCnGXxKGTsAst', 'F0IVmA99YzX2x8LWkGrp8WBjVH9qsa'),
                       'oss-cn-hangzhou.aliyuncs.com', 'canvas-imagenet', connect_timeout=5)

def save_log(args, kernel_pack, train_metrics, eval_metrics, extra, ckp):
    logger = get_logger()
    assert args.canvas_log_dir
    if kernel_pack:
        logger.info(f'Saving kernel {kernel_pack.name} into {args.canvas_log_dir} ...')
    else:
        logger.info(f'Saving exceptions into {args.canvas_log_dir} ...')
    
    # Make directory (may overwrite).
    if os.path.exists(args.canvas_log_dir):
        assert os.path.isdir(args.canvas_log_dir), 'Canvas logging path must be a directory'
    if 'exception' in extra:
        exception_info = extra['exception']
        if 'memory' in exception_info or 'NaN' in exception_info or 'Pruned' in exception_info:
            # Do not record these types.
            return
        else:
            error_type = 'Error'
        dir_name = f'Canvas_{error_type}_'
        dir_name = dir_name + (f'{kernel_pack.name}' if kernel_pack else f'{time.time_ns()}')
    else:
        assert len(eval_metrics) > 0
        max_score = math.floor(max([item['top1'] for item in eval_metrics]) * 100)
        score_str = ('0' * max(0, 5 - len(f'{max_score}'))) + f'{max_score}'
        dir_name = f'Canvas_{score_str}_{kernel_pack.name}'
    path = os.path.join(args.canvas_log_dir, dir_name)
    if os.path.exists(path):
        logger.info('Overwriting results ...')
    os.makedirs(path, exist_ok=True)

    # Save code, graphviz, args, and results.
    if kernel_pack:
        kernel_pack.save_torch_code(os.path.join(path, kernel_pack.name + '.py'))
        kernel_pack.save_graphviz_code(os.path.join(path, kernel_pack.name + '.dot'))
    else:
        kernel_name = 'exception'
    with open(os.path.join(path, kernel_pack.name + '.json'), 'w') as file:
        json.dump({'args': vars(args), 'timestamp': kernel_pack.timestamp if kernel_pack else None,
                    'train_metrics': train_metrics, 'eval_metrics': eval_metrics,
                    'extra': extra},
                    fp=file, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)

    # Save to OSS buckets.
    if args.canvas_oss_bucket:
        logger.info(f'Uploading into OSS bucket {args.canvas_oss_bucket}')
        prefix = args.canvas_oss_bucket + '/' + dir_name + '/'
        for filename in os.listdir(path):
            global oss_try_times
            success = False
            for i in range(oss_try_times):
                # noinspection PyBroadException
                try:
                    logger.info(f'Uploading {filename} ...')
                    get_oss_bucket().put_object_from_file(prefix + filename, os.path.join(path, filename))
                    success = True
                    break
                except Exception as ex:
                    logger.info(f'Failed to upload, try {i + 1} time(s)')
                    continue
            if not success:
                logger.info(f'Uploading failed for {oss_try_times} time(s)')

def train_canvas(args, t: Trainer, search_mode: bool = False, proxy_mode: bool = False):
    # Single machine
    # Create a logger.
    logger = get_logger()
    logger.info('Begin training ...')

    # Resume from checkpoint is handled initialization of trainer
    # if not args.save:
    #   args.save = now
    # If args.load is given, trainer will load optimizer from given folder
    # if self.args.load != '':
    #   self.optimizer.load(ckp.dir, epoch=len(ckp.log))
    # Therefore, we **must** assgin args.save with no value.
    start_epoch = len(t.ckp.log)
    sched_epochs = t.args.epochs # scheduler_class = lrs.MultiStepLR. Not cycle based schedulers https://github.com/rwightman/pytorch-image-models/blob/main/timm/scheduler/scheduler_factory.py#L196


    # Checkpoint saver.
    # Trainer has a "save" function in the "test" function, so there's no need to create a saver here
    # Trainer will record the best result and the corresponding model in self.ckp.log
    best_metric, best_epoch = None, None
    
    
    # Pruning after epochs.
    overall_pruning_milestones = None
    if args.canvas_epoch_pruning_milestone:
        with open(args.canvas_epoch_pruning_milestone) as f:
            overall_pruning_milestones = json.load(f)
            logger.info(f'Milestones (overall epochs) loaded: {overall_pruning_milestones}')
    
    # Iterate over epochs.
    all_train_metrics, all_eval_metrics = [], []
    for epoch in range(start_epoch, sched_epochs):
        if hasattr(t.loader_train.sampler, 'set_epoch'):
            t.loader_train.sampler.set_epoch(epoch)
        
        # Pruner.
        in_epoch_pruning_milestones = dict()
        if epoch == 0 and search_mode and not proxy_mode and args.canvas_first_epoch_pruning_milestone:
            with open(args.canvas_first_epoch_pruning_milestone) as f:
                in_epoch_pruning_milestones = json.load(f)
                logger.info(f'Milestones (first-epoch loss) loaded: {in_epoch_pruning_milestones}')
        
        # Train.
        
        


        
    pass
    




def canvas_main():
    global _model
    logger = get_logger()
    if args.data_test == ['video']:
        raise NotImplementedError
    else:
        if checkpoint.ok:
            
            _model = model.Model(args, checkpoint)
            canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)
            cpu_clone = deepcopy(_model.model).cpu()

            def restore_model_params_and_replace(pack=None):
                global _model
                _model.model = None
                gc.collect()
                _model.model = deepcopy(cpu_clone).to(_model.device)
                if pack is not None:
                    canvas.replace(model, pack.module, _model.device)
            
            current_best_score = 0
            round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
            logger.info(
                f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)'
            )
            for i in round_range:
                logger.info('Sampling a new kernel ...')
                g_macs, m_flops = 0, 0
                try:
                    kernel_pack = canvas.sample(_model.model,
                                        force_bmm_possibility=args.canvas_bmm_pct,
                                        min_receptive_size=args.canvas_min_receptive_size,
                                        num_primitive_range=(5, 40),
                                        workers=args.canvas_sampling_workers)
                    restore_model_params_and_replace(kernel_pack)
                    g_macs, m_params = ptflops.get_model_complexity_info(_model.model, args.input_size,
                                                                 as_strings=False, print_per_layer_stat=False)
                    g_macs, m_params = g_macs / 1e9, m_params / 1e6
                except RuntimeError as ex:
                    logger.info(f'Exception: {ex}')
                    save_log(args, None, None, None, {'exception': f'{ex}'}, checkpoint)
                    continue
                logger.info(f'Sampled kernel hash: {hash(kernel_pack)}')
                logger.info(f'MACs: {g_macs} G, params: {m_params} M')
                macs_not_satisfied = (args.canvas_min_macs > 0 or args.canvas_max_macs > 0) and \
                             (g_macs < args.canvas_min_macs or g_macs > args.canvas_max_macs)
                params_not_satisfied = (args.canvas_min_params > 0 or args.canvas_max_params > 0) and \
                                    (m_params < args.canvas_min_params or m_params > args.canvas_max_params)
                if macs_not_satisfied or params_not_satisfied:
                    logger.info(f'MACs ({args.canvas_min_macs}, {args.canvas_max_macs}) or '
                                f'params ({args.canvas_min_params}, {args.canvas_max_params}) '
                                f'requirements do not satisfy')
                    continue
                
                # Train.
                loader = data.Data(args)
                _loss = loss.Loss(args, checkpoint) if not args.test_only else None
                proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
                try:
                    if loader.loader_train and loader.loader_test:
                        logger.info('Training on proxy dataset ...')
                        t = Trainer(args, loader, _model, _loss, checkpoint)
                except RuntimeError as ex:
                    exception_info = f'{ex}'
                    logger.warning(f'Exception: {exception_info}')


            total_timer = utility.timer()
            total_timer.tic()
            while not t.terminate():
                t.train()
                t.test()
            total_timer.hold()
            print("Total training time: {:.1f}s".format(total_timer.release()))
            checkpoint.done()


if __name__ == '__main__':
    main()
