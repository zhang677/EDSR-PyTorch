import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

import ptflops

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
            macs, params = ptflops.get_model_complexity_info(_model.model, args.input_size, as_strings=True,
                                                         print_per_layer_stat=False, verbose=False)
            print(f'MACs: {macs}, params: {params}')                                           
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

if __name__ == '__main__':
    main()
