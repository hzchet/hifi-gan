import argparse
import collections
import warnings
import itertools

import numpy as np
import torch

import src.loss as module_loss
import src.model as module_arch
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    gen_model = config.init_obj(config["gen_arch"], module_arch)
    logger.info(gen_model)
    mpd_model = config.init_obj(config["mpd_arch"], module_arch)
    logger.info(mpd_model)
    msd_model = config.init_obj(config["msd_arch"], module_arch)
    logger.info(msd_model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    gen_model = gen_model.to(device)
    mpd_model = mpd_model.to(device)
    msd_model = msd_model.to(device)
    if len(device_ids) > 1:
        gen_model = torch.nn.DataParallel(gen_model, device_ids=device_ids)
        mpd_model = torch.nn.DataParallel(mpd_model, device_ids=device_ids)
        msd_model = torch.nn.DataParallel(msd_model, device_ids=device_ids)

    # get function handles of loss and metrics
    gen_loss_module = config.init_obj(config["gen_loss"], module_loss).to(device)
    disc_loss_module = config.init_obj(config["disc_loss"], module_loss).to(device)
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    gen_trainable_params = filter(lambda p: p.requires_grad, gen_model.parameters())
    gen_optimizer = config.init_obj(config["gen_optimizer"], torch.optim, gen_trainable_params)
    
    disc_trainable_params = filter(lambda p: p.requires_grad, itertools.chain(
        mpd_model.parameters(),
        msd_model.parameters()
    ))
    disc_optimizer = config.init_obj(config["disc_optimizer"], torch.optim, disc_trainable_params)
    
    gen_lr_scheduler = config.init_obj(config["gen_lr_scheduler"], torch.optim.lr_scheduler, gen_optimizer)
    disc_lr_scheduler = config.init_obj(config["disc_lr_scheduler"], torch.optim.lr_scheduler, disc_optimizer)
    
    trainer = Trainer(
        gen_model,
        mpd_model,
        msd_model,
        gen_loss_module,
        disc_loss_module,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler,
        disc_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-f",
        "--finetune",
        default=None,
        type=str,
        help="path to pretrained checkpoint (default: None)"
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)