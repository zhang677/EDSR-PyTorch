import os
import math
import json
from decimal import Decimal
from collections import OrderedDict
import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        # Train one epoch
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

    def save_canvas(self, dir):
        epoch = self.optimizer.get_last_epoch()
        best = self.ckp.log.max(0)
        self.ckp.save(self, epoch, (best[1][0, 0] + 1 == epoch))

    def train_one_epoch(self, logger, pruning_milestones):
        # second order is False for MultiStepLR
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        num_updates = epoch * len(self.loader_train)
        last_idx = len(self.loader_train) - 1

        timer_data, timer_batch = utility.timer(), utility.timer()
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_batch.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_batch.hold()

            num_updates += 1

            if batch == last_idx or (batch + 1) % self.args.print_every == 0:
                lr = self.optimizer.get_lr()
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {}  '
                    'Time: {:.1f}s '
                    'LR: {:.3e}  '
                    'Data: {:.1f}s'.format(
                        epoch,
                        batch, len(self.loader_train),
                        100. * batch / last_idx,
                        self.loss.display_loss(batch),
                        timer_batch.release(),
                        lr,
                        timer_data.release()))
                loss_float = self.loss.get_loss(batch, 'L1')
                progress = '{:.0f}'.format(100. * batch / last_idx)
                if pruning_milestones and progress in pruning_milestones and \
                        loss_float > pruning_milestones[progress]:
                    raise RuntimeError(f'Pruned by milestone settings at progress {progress}%')
                if math.isnan(loss_float):
                    break
            timer_data.tic()
                      
        # No lookahead Wrapper in timm
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()
        
        return OrderedDict([('loss', loss_float)])

    def validate(self, logger):
        torch.set_grad_enabled(False)
        # eval_loader is the same as test_loader
        # Metric is PSNR. No top1 or top5
        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        last_idx = len(self.loader_test) - 1
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale): # Only x2 scale
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr) # lr: low resolution; hr: high resolution
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    
                    # save the picture
                    if self.args.sava_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)
                
                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                best_psnr = best[0][idx_data, idx_scale]
                if idx_data == last_idx or (idx_data + 1) % self.args.print_every== 0:
                    logger.info(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {} Time: {:.2f})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best_psnr,
                            best[1][idx_data, idx_scale] + 1,
                            timer_test.toc(restart=True)
                        )
                    )
        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        torch.set_grad_enabled(True)

        return OrderedDict([('psnr', best_psnr)])

    def train_canvas(self, logger, search_mode = False, proxy_mode = False):
        # Single machine
        # Create a logger.
        logger.info('Begin training ...')

        # Resume from checkpoint is handled initialization of trainer
        # if not args.save:
        #   args.save = now
        # If args.load is given, trainer will load optimizer from given folder
        # if self.args.load != '':
        #   self.optimizer.load(ckp.dir, epoch=len(ckp.log))
        # Therefore, we **must** assgin args.save with no value.
        start_epoch = len(self.ckp.log)
        sched_epochs = self.args.epochs # scheduler_class = lrs.MultiStepLR. Not cycle based schedulers https://github.com/rwightman/pytorch-image-models/blob/main/timm/scheduler/scheduler_factory.py#L196


        # Checkpoint saver.
        # Trainer has a "save" function in the "test" function, so there's no need to create a saver here
        # Trainer will record the best result and the corresponding model in self.ckp.log
        best_metric, best_epoch = None, None
        
        
        # Pruning after epochs.
        overall_pruning_milestones = None
        if self.args.canvas_epoch_pruning_milestone:
            with open(self.args.canvas_epoch_pruning_milestone) as f:
                overall_pruning_milestones = json.load(f)
                logger.info(f'Milestones (overall epochs) loaded: {overall_pruning_milestones}')
        
        # Iterate over epochs.
        all_train_metrics, all_eval_metrics = [], []
        for epoch in range(start_epoch, sched_epochs):
            if hasattr(self.loader_train.sampler, 'set_epoch'):
                self.loader_train.sampler.set_epoch(epoch)
            
            # Pruner.
            in_epoch_pruning_milestones = dict()
            if epoch == 0 and search_mode and not proxy_mode and self.args.canvas_first_epoch_pruning_milestone:
                with open(self.args.canvas_first_epoch_pruning_milestone) as f:
                    in_epoch_pruning_milestones = json.load(f)
                    logger.info(f'Milestones (first-epoch loss) loaded: {in_epoch_pruning_milestones}')
            
            # Train.
            train_metrics = self.train_one_epoch(logger, in_epoch_pruning_milestones)
            all_train_metrics.append(train_metrics)

            # Check NaN errors.
            if math.isnan(train_metrics['loss']):
                raise RuntimeError('NaN occurs during training')

            # Not distributed, no normalize
            # Distributed training and loss are handled by args.n_GPUs in Loss and Model
            
            # Evaluate.
            eval_metrics = self.validate(logger)
            all_eval_metrics.append(eval_metrics)

            # LR Scheduler is updated in train_one_epoch

            # Check NaN errors.
            if math.isnan(eval_metrics['loss']) and self.args.forbid_eval_nan:
                raise RuntimeError('NaN occurs during validation')
            
            # Summary and checkpoint are saved in validate function
            
            # Pruning by epoch accuracy.
            if f'{epoch}' in overall_pruning_milestones and overall_pruning_milestones[f'{epoch}'] > eval_metrics['psnr']:
                logger.info(f'Early pruned '
                            f'({eval_metrics["psnr"]} < {overall_pruning_milestones[f"{epoch}"]}) at epoch {epoch}')
                break
        if best_metric is not None:
            logger.info(f'Best metric: {best_metric} (epoch {best_epoch})')

        return all_train_metrics, all_eval_metrics    
                    

    