import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
import soundfile as sf

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture
    gen_model = config.init_obj(config["gen_arch"], module_model)
    logger.info(gen_model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["gen_state_dict"]
    if config["n_gpu"] > 1:
        gen_model = torch.nn.DataParallel(gen_model)
    gen_model.load_state_dict(state_dict)

    sr = config["preprocessing"].get("sr")
    # prepare model for testing
    gen_model = gen_model.to(device)
    gen_model.eval()
    gen_model.remove_weight_norm()
    
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        key = 'final_test'
        logger.info(f'{key}:')
        i = 1
        for batch in tqdm(dataloaders[key]):
            batch = Trainer.move_batch_to_device(batch, device)
            audio_gen = gen_model(**batch)["audio_gen"].detach().cpu().squeeze(1).numpy()
            
            for audio in audio_gen:
                sf.write(f'{out_dir}/{i}.wav', audio.T, sr)
                i += 1
            

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
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--test_dir",
        default=str(ROOT_PATH / "audio_dir"),
        type=str,
        help="path to a folder containing test audios",
    )
    args.add_argument(
        "-o",
        "--output_dir",
        default=str(ROOT_PATH / "out_dir"),
        type=str,
        help="Dir to save results",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))
    
    assert os.path.exists(args.test_dir), f'{args.test_dir} does not exist'
    
    config.config["data"] = {
        "final_test": {
            "batch_size": 1,
            "datasets": [
                {
                    "type": "InferenceDataset",
                    "args": {
                        "path_to_audios": args.test_dir
                    },
                }
            ],
        }
    }

    main(config, args.output_dir)
