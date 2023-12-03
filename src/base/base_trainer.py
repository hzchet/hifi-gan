from abc import abstractmethod

import torch
from numpy import inf

from src.base.base_model import BaseModel
from src.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self, 
        gen_model: BaseModel,
        mpd_model: BaseModel,
        msd_model: BaseModel,
        gen_criterion,
        disc_criterion,
        gen_optimizer,
        disc_optimizer,
        gen_lr_scheduler,
        disc_lr_scheduler,
        config,
        device
    ):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.gen_model = gen_model
        self.mpd_model = mpd_model
        self.msd_model = msd_model
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_lr_scheduler = gen_lr_scheduler
        self.disc_lr_scheduler = disc_lr_scheduler

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(
            config, self.logger, cfg_trainer["visualize"]
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)
            
        if config.finetune is not None:
            self._load_pretrained_model(config.finetune)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        gen_arch = type(self.gen_model).__name__
        mpd_arch = type(self.mpd_model).__name__
        msd_arch = type(self.msd_model).__name__
        state = {
            "gen_arch": gen_arch,
            "mpd_arch": mpd_arch,
            "msd_arch": msd_arch,
            "epoch": epoch,
            "gen_state_dict": self.gen_model.state_dict(),
            "mpd_state_dict": self.mpd_model.state_dict(),
            "msd_state_dict": self.msd_model.state_dict(),
            "gen_optimizer": self.gen_optimizer.state_dict(),
            "gen_lr_scheduler": self.gen_lr_scheduler.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
            "disc_lr_scheduler": self.disc_lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
    
    def _load_pretrained_model(self, ckpt_path):
        """
        Load pretrained model
        
        :param ckpt_path: Checkpoint path to be finetuned
        """
        ckpt_path = str(ckpt_path)
        self.logger.info("Loading checkpoint: {} ...".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, self.device)
        
        # load architecture params from checkpoint.
        if checkpoint["config"]["gen_arch"] != self.config["gen_arch"] or \
            checkpoint["config"]["mpd_arch"] != self.config["mpd_arch"] or \
            checkpoint["config"]["msd_arch"] != self.config["msd_arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.gen_model.load_state_dict(checkpoint["gen_state_dict"])
        self.mpd_model.load_state_dict(checkpoint["mpd_state_dict"])
        self.msd_model.load_state_dict(checkpoint["msd_state_dict"])
        self.logger.info(
            "Checkpoint loaded. Fine-tuning model..."
        )
    
    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["gen_arch"] != self.config["gen_arch"] or \
            checkpoint["config"]["mpd_arch"] != self.config["mpd_arch"] or \
            checkpoint["config"]["msd_arch"] != self.config["msd_arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that "
                "of checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.gen_model.load_state_dict(checkpoint["gen_state_dict"])
        self.mpd_model.load_state_dict(checkpoint["mpd_state_dict"])
        self.msd_model.load_state_dict(checkpoint["msd_state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
                checkpoint["config"]["gen_optimizer"] != self.config["gen_optimizer"] or
                checkpoint["config"]["gen_lr_scheduler"] != self.config["gen_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        else:
            self.gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
            self.gen_lr_scheduler.load_state_dict(checkpoint["gen_lr_scheduler"])
            self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
            self.disc_lr_scheduler.load_state_dict(checkpoint["disc_lr_scheduler"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
