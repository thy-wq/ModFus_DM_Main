import argparse
import yaml
import os
import torch as th
import torch
import numpy as np
import random
from diffsuion1d.denoising_diffusion_pytorch_1d import (Unet1D,
                                                        Trainer1D, Dataset1D)
from runner.schedule import Schedule

def args_and_config():
    parser = argparse.ArgumentParser()
    # 数据集
    parser.add_argument("--dataset",type=str,default="RML2016a",choices=['RML2016a','RML2016b',"RML2018","RML2022","RML2018v","REAL"])

    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--diffusion_step",type=int,default=100)
    parser.add_argument("--runner", type=str, default='training',
                       help="Choose the mode of runner")
    parser.add_argument("--config", type=str, default='ddpm11a.yml',help="Choose the config file")
    # parser.add_argument("--config", type=str, default='ddpm18v.yml',help="Choose the config file")
    # parser.add_argument("--config", type=str, default='ddpm10b.yml',help="Choose the config file")
                         # 参数配置文件位置
    parser.add_argument("--model", type=str, default='DDPM_piped',
                        choices=['DDPM_piped',"EDM"])
    parser.add_argument("--disc", type=str, default='MCLDNN',
                        help="Choose the discriminator model's structure (ResNet18, PyramidNet)")
    parser.add_argument("--method", type=str, default='PNDM4',
                        help="Choose the numerical methods (DDIM, PNDM2, PNDM4, NDM1, NDM4)")
    parser.add_argument("--sample_step", type=int, default=50,
                        help="Control the total generation step")
    parser.add_argument("--device", type=str, default='cuda',
    # parser.add_argument("--device", type=str, default='cpu',
                        help="Choose the device to use")
    parser.add_argument("--image_path", type=str, default='debug/sample',
                        help="Choose the path to save images")
    parser.add_argument("--restart", action="store_true",
                        help="Restart a previous training process")
    parser.add_argument("--train_path", type=str, default='temp/train/base_multi',
                        help="Choose the path to save training status")
    parser.add_argument("--repeat_size", type=int, default=4,
                        help="Set the model's name")
    # 当模型不是从头开始训练
    # args.resume == ""时从头开始训练，否则加载这里面的参数从该参数继续训练
    parser.add_argument("--resume",type=str,default= "")

    # pred_v
    parser.add_argument("--pred_type", default="pred_noise",type=str,choices=['pred_v',"pred_noise","pred_x0"])

    parser.add_argument("--snr",type=int,default=100,help="snr dB used to train the model")

    parser.add_argument("--augment",type=str,default='rotation',choices=['','rotation'])

    args = parser.parse_args()

    rank = os.environ.get("LOCAL_RANK")
    rank = 0 if rank is None else int(rank)
    world_size = os.environ.get("WORLD_SIZE")
    world_size = 1 if world_size is None else int(world_size)

    # assert args.model.lower() in args.config
    work_dir = os.getcwd()
    with open(f'{work_dir}/{args.config}', 'r') as f:
        config = yaml.safe_load(f)
    config['Schedule']['diffusion_step'] = args.diffusion_step
    return args, config


def check_config(config):
    # assert config['Dataset']['batch_size'] == config['Train']['batch_size'] // 2
    pass


if __name__ == "__main__":
    from datetime import datetime
    now = datetime.now().isoformat()[-4:-1]
    seed = int(now)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # 在如args和config配置
    args, config = args_and_config()
    check_config(config)

    device = th.device(args.device)
    schedule = Schedule(args, config['Schedule'])
    diffusion_step = config["Schedule"]["diffusion_step"]
    assert diffusion_step == args.diffusion_step
    sample_step = args.sample_step
    args.train_path = f"temp-{args.dataset}-{args.model}-{diffusion_step}step-{args.pred_type}/train/base_multi"
    args.image_path = f"temp-{args.dataset}-{args.model}-{diffusion_step}step-{args.pred_type}/sa1mple"
    print(args.train_path)

    assert config["Dataset"]["num_classes"] == len(config["Dataset"]["known"])
    assert config['Model']['struc'] == args.model

    # Load diffusion model 加载DM模型
    if config['Model']['struc'] == 'DDPM_piped':
        print("Use DDPM diffusion model")
        import thop
        from diffsuion1d.denoising_diffusion_pytorch_1d import GaussianDiffusion1D
        if args.dataset == "RML2016a" or args.dataset == "RML2016b"or args.dataset == "RML2022":
            assert config['Dataset']['name'] == args.dataset
            num_knownclass = len(config['Dataset']['known'])
            model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=2, num_classes=num_knownclass).cuda()
            # diffusion = GaussianDiffusion1D(model, seq_length=128, timesteps=diffusion_step, objective='pred_v').cuda()
            diffusion = GaussianDiffusion1D(model, seq_length=128, timesteps=diffusion_step, objective=args.pred_type).cuda()
            # signal = torch.randn([2,2,128]).cuda()
        elif args.dataset == "RML2018":
            assert config['Dataset']['name'] == args.dataset
            assert config['Dataset']['image_size2'] == 1024
            num_knownclass = len(config['Dataset']['known'])
            model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=2, num_classes=num_knownclass).cuda()
            diffusion = GaussianDiffusion1D(model, seq_length=1024, timesteps=diffusion_step, objective=args.pred_type).cuda()
            signal = torch.randn([2, 2, 1024]).cuda()
        elif args.dataset == "RML2018v":
            config['Dataset']['image_size2'] = args.length
            assert config['Dataset']['name'] == args.dataset
            assert config['Dataset']['image_size2'] == args.length
            num_knownclass = len(config['Dataset']['known'])
            model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=2, num_classes=num_knownclass).cuda()
            diffusion = GaussianDiffusion1D(model, seq_length=args.length, timesteps=diffusion_step,
                                            objective=args.pred_type).cuda()
            signal = torch.randn([2, 2, args.length]).cuda()
    elif config['Model']['struc'] == 'EDM':
        print("Use EDM diffusion model")
        import thop
        from diffsuion1d.EDM import EDM
        if args.dataset == "RML2016a" or args.dataset == "RML2016b":
            assert config['Dataset']['name'] == args.dataset
            num_knownclass = len(config['Dataset']['known'])
            model = Unet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=2, num_classes=num_knownclass).cuda()
            diffusion = EDM(model, seq_length=128, timesteps=diffusion_step,
                                            objective=args.pred_type).cuda()
            signal = torch.randn([2, 2, 128]).cuda()
    else:
        print("输入正确的扩散模型参数：config['Model']['struc']")
    # Load runner
    if args.runner == 'training':
        from runner.runner import Runner
        runner = Runner(args, config, schedule, diffusion)
        runner.train_loop()
        print(f'Do not find the runner {args.runner}')



