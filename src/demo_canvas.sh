# main.py for main_canvas
python main_canvas.py --model EDSR --scale 2 --patch_size 96 --reset --ext sep --epochs 5 \
--canvas-log-dir /home/nfs_data/zhanggh/EDSR-PyTorch/experiment/Canvas \
--canvas-epoch-pruning-milestone /home/nfs_data/zhanggh/EDSR-PyTorch/src/edsr_div2k_bs16_debug.json \
--data_range 1-800/801-810 \
--canvas-replace-block \
--canvas-proxy-threshold 10
