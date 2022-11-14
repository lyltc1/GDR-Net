""" usage: python run_model.py --config-file /home/lyltc/git/GDR-Net/configs/symn/ycbv/symn_net_config.py"""
import sys
sys.path.append("../..")
import torch
from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.symn.models import SymNet
from thop import profile



if __name__ == "__main__":
    parser = my_default_argument_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)
    iprint(f"Used model name: {cfg.MODEL.NAME}")
    model, optimizer = eval(cfg.MODEL.NAME).build_model_optimizer(cfg)
    device = "cpu"
    # summary(model.to(device), [(3, 256, 256), (3, 3), (4,)])
    iprint("Model:\n{}".format(model))
    inputs = [torch.randn(32, 3, 256, 256), torch.randn(32, 3, 3), torch.randn(32, 4)]
    flops, params = profile(model, inputs=inputs)
    print("flops:", flops / 1e9, "G")
    print("params:", params / 1e6, "M")
