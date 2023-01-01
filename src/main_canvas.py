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
import time
from copy import deepcopy
import canvas

torch.manual_seed(args.seed)
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

def get_model(_model, search_mode = False):
    # The output channel of EDSR is args.n_colors=3
    # Input shape is [16, 3, 48, 48] = [B, C, H, W]
    # Input shape is [16, 3, 96, 96] = [B, C, H, W] (patch_size=96)
    logger = get_logger()
    example_input = torch.rand((1, ) + args.input_size).to(_model.device)
    canvas.get_placeholders(_model.model, example_input)

    # Replace kernel.
    if not search_mode and args.canvas_kernel:
        logger.info(f'Replacing kernel from {args.canvas_kernel}')
        pack = canvas.KernelPack.load(args.canvas_kernel)
        _model.model = canvas.replace(_model.model, pack.module, _model.device)
    
    macs, params = ptflops.get_model_complexity_info(_model.model, args.input_size, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
    logger.info(f'MACs: {macs}, params: {params}')

    return _model

def save_log(kernel_pack, train_metrics, eval_metrics, extra):
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
        max_score = math.floor(max([item['psnr'] for item in eval_metrics]) * 100)
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

def canvas_main():
    global _model
    logger = get_logger()
    if args.data_test == ['video']:
        raise NotImplementedError
    else:
        checkpoint = utility.checkpoint(args)          
        _model = get_model(model.Model(args, checkpoint, placeholder=True), search_mode=True)
        example_input = torch.rand((1, ) + args.input_size).to(_model.device)
        canvas.seed(random.SystemRandom().randint(0, 0x7fffffff) if args.canvas_seed == 'pure' else args.seed)
        cpu_clone = deepcopy(_model.model).cpu()

        def restore_model_params_and_replace(pack=None):
            global _model
            _model.model = None
            gc.collect()
            _model.model = deepcopy(cpu_clone).to(_model.device)
            if pack is not None:
                _model.model = canvas.replace(_model.model, pack.module, _model.device)
        
        current_best_score = 0
        round_range = range(args.canvas_rounds) if args.canvas_rounds > 0 else itertools.count()
        logger.info(
            f'Start Canvas kernel search ({args.canvas_rounds if args.canvas_rounds else "infinite"} rounds)'
        )
        for i in round_range:
            logger.info('Sampling a new kernel ...')
            g_macs = 0
            try:
                kernel_pack = canvas.sample(_model.model,
                                    example_input=example_input,
                                    force_bmm_possibility=args.canvas_bmm_pct,
                                    min_receptive_size=args.canvas_min_receptive_size,
                                    num_primitive_range=(5, 40),
                                    workers=args.canvas_sampling_workers,
                                    ensure_spatial_invariance=False)
                restore_model_params_and_replace(kernel_pack)
                g_macs, m_params = ptflops.get_model_complexity_info(_model.model, args.input_size,
                                                                as_strings=False, print_per_layer_stat=False)
                g_macs, m_params = g_macs / 1e9, m_params / 1e6
            except RuntimeError as ex:
                logger.info(f'Exception: {ex}')
                save_log(None, None, None, {'exception': f'{ex}'})
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
            
            loader = data.Data(args)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            # Train.
            proxy_score, train_metrics, eval_metrics, exception_info = 0, None, None, None
            proxy_train_loader = None #loader.loader_train
            proxy_eval_loader = None #loader.loader_test
            try:
                if proxy_train_loader and proxy_eval_loader:
                    logger.info('Training on proxy dataset ...')
                    t = Trainer(args, loader, _model, _loss, checkpoint)
                    _, proxy_eval_metrics = \
                        t.train_canvas(logger, search_mode=True, proxy_mode=True)
                    best_epoch = 0
                    for e in range(1, len(proxy_eval_loader)):
                        if proxy_eval_metrics[e]['psnr'] > proxy_eval_metrics[best_epoch]['psnr']:
                            best_epoch = e
                    proxy_score = proxy_eval_metrics[best_epoch]['psnr']
                    restore_model_params_and_replace(kernel_pack)
                    logger.info(f'Proxy dataset score: {proxy_score}')
                    if proxy_score < args.canvas_proxy_threshold:
                        logger.info(f'Under proxy threshold {args.canvas_proxy_threshold}, skip main dataset training')
                        continue
                    # No scale
                t = Trainer(args, loader, _model, _loss, checkpoint)
                logger.info('Training on main dataset ...')
                train_metrics, eval_metrics = \
                    t.train_canvas(logger, search_mode=True, proxy_mode=False)
                score = max([item['psnr'] for item in eval_metrics])
                logger.info(f'Solution score: {score}')
                if score > current_best_score:
                    current_best_score = score
                    logger.info(f'Current best score: {current_best_score}')
                    if args.canvas_weight_sharing:
                        try:
                            cpu_clone = deepcopy(model).cpu()
                            logger.info(f'Weight successfully shared')
                        except Exception as ex:
                            logger.warning(f'Failed to make weight shared: {ex}')

            except RuntimeError as ex:
                exception_info = f'{ex}'
                logger.warning(f'Exception: {exception_info}')


            # Save into logging directory.
            extra = {'proxy_score': proxy_score, 'g_macs': g_macs, 'm_params': m_params}
            if exception_info:
                extra['exception'] = exception_info
            save_log(kernel_pack, train_metrics, eval_metrics, extra)
            checkpoint.done()
            checkpoint = utility.checkpoint(args)
        checkpoint.done()


if __name__ == '__main__':
    canvas_main()
