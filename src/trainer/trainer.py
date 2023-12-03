import logging
import random
from pathlib import Path
from random import shuffle
import itertools

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.collate_fn.mels import MelSpectrogram, MelSpectrogramConfig
from src.utils import inf_loop, MetricTracker


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            gen_model,
            mpd_model,
            msd_model,
            gen_criterion,
            disc_criterion,
            gen_optimizer,
            disc_optimizer,
            gen_lr_scheduler,
            disc_lr_scheduler,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(
            gen_model,
            mpd_model,
            msd_model,
            gen_criterion,
            disc_criterion,
            gen_optimizer, 
            disc_optimizer,
            gen_lr_scheduler,
            disc_lr_scheduler,
            config, 
            device
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.log_step = 50

        self.train_metrics_tracker = MetricTracker(
            "adv_loss", "gen_grad_norm", "disc_grad_norm", 
            "feature_loss", "mel_loss", "gen_loss", "disc_loss",
            writer=self.writer
        )
        self.evaluation_metrics_tracker = MetricTracker(
            "mel_loss", writer=self.writer
        )
        
        mel_cfg = MelSpectrogramConfig()
        self.mel_spec = MelSpectrogram(mel_cfg)
        self.mel_spec.mel_spectrogram.to(device)
        self.num_accumulation_iters = self.config["trainer"].get("num_accumulation_iters", 1)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["spectrogram", "audio"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.gen_model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.mpd_model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.msd_model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.gen_model.train()
        self.mpd_model.train()
        self.msd_model.train()
        self.train_metrics_tracker.reset()
        self.writer.mode = 'train'
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics_tracker,
                    batch_idx=batch_idx
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.gen_model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.mpd_model.parameters():
                        if p.grad is not None:
                            del p.grad
                    for p in self.msd_model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} gen_loss: {:.6f}, disc_loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["gen_loss"].item(), batch["disc_loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "gen_learning_rate", self.gen_lr_scheduler.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "disc_learning_rate", self.disc_lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(**batch)
                self._log_scalars(self.train_metrics_tracker)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics_tracker.result()
                self.train_metrics_tracker.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics
        
        if self.gen_lr_scheduler is not None:
            self.gen_lr_scheduler.step()
        if self.disc_lr_scheduler is not None:
            self.disc_lr_scheduler.step()
            
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx: int):
        batch = self.move_batch_to_device(batch, self.device)
        
        audio_gen = self.gen_model(**batch)["audio_gen"]
        spectrogram_gen = self.mel_spec(audio_gen.squeeze(1))
        batch.update({"audio_gen": audio_gen, "spectrogram_gen": spectrogram_gen})
        
        mpd_outputs = self.mpd_model(batch["audio"], batch["audio_gen"].detach())
        msd_outputs = self.msd_model(batch["audio"], batch["audio_gen"].detach())
        batch.update(mpd_outputs)
        batch.update(msd_outputs)
        
        if is_train:
            self.gen_optimizer.zero_grad()
            self.disc_optimizer.zero_grad()
            
            disc_loss = self.disc_criterion(**batch)
            disc_loss.backward()
            metrics.update("disc_grad_norm", self.get_grad_norm(model='disc'))
            metrics.update("disc_loss", disc_loss.detach().cpu())
            self.disc_optimizer.step()
            batch["disc_loss"] = disc_loss
            
            mpd_outputs = self.mpd_model(**batch)
            msd_outputs = self.msd_model(**batch)
            batch.update(mpd_outputs)
            batch.update(msd_outputs)
            
            gen_loss, adv_loss, mel_loss, fm_loss = self.gen_criterion(**batch)
            gen_loss.backward()
            metrics.update("gen_grad_norm", self.get_grad_norm(model='gen'))
            metrics.update("gen_loss", gen_loss.detach().cpu())
            metrics.update("adv_loss", adv_loss.detach().cpu())
            metrics.update("mel_loss", mel_loss.detach().cpu())
            metrics.update("feature_loss", fm_loss.detach().cpu())
            self.gen_optimizer.step()
            batch["gen_loss"] = gen_loss
            batch["adv_loss"] = adv_loss
            batch["feature_loss"] = fm_loss
            batch["mel_loss"] = mel_loss
        else:
            metrics.update(
                "mel_loss", F.l1_loss(batch["spectrogram"], batch["spectrogram_gen"]).detach().cpu()
            )
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.gen_model.eval()
        self.mpd_model.eval()
        self.msd_model.eval()
        self.writer.mode = part
        self.evaluation_metrics_tracker.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics_tracker,
                    batch_idx=batch_idx
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics_tracker)
            self._log_predictions(**batch)

        return self.evaluation_metrics_tracker.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            audio,
            audio_gen,
            spectrogram,
            spectrogram_gen,
            examples_to_log: int = 10,
            *args,
            **kwargs,
    ):
        batch_size = audio.shape[0]
        random_indices = torch.randperm(batch_size)[:examples_to_log]
        
        sr = self.config["preprocessing"].get("sr")
        log_audio_target = audio[random_indices].detach().cpu()
        log_mel_target = spectrogram[random_indices].detach().cpu()
        log_audio_gen = audio_gen[random_indices].detach().cpu()
        log_mel_gen = spectrogram_gen[random_indices].detach().cpu()
        rows = []
        for example_audio_target, example_mel_target, example_audio_gen, example_mel_gen in zip(
            log_audio_target, log_mel_target, log_audio_gen, log_mel_gen
        ):
            mel_target_image = PIL.Image.open(plot_spectrogram_to_buf(
                example_mel_target.detach().cpu()
            ))
            mel_gen_image = PIL.Image.open(plot_spectrogram_to_buf(
                example_mel_gen.detach().cpu()
            ))
            rows.append({
                "audio_target": self.writer.create_audio_entry(example_audio_target, sr),
                "mel_target": self.writer.create_image_entry(ToTensor()(mel_target_image)),
                "audio_gen": self.writer.create_audio_entry(example_audio_gen, sr),
                "mel_gen": self.writer.create_image_entry(ToTensor()(mel_gen_image))
            })
        
        df = pd.DataFrame(rows)
        self.writer.add_table('predictions', df)
    
    @torch.no_grad()
    def get_grad_norm(self, model: str = 'gen', norm_type=2):
        if model == 'gen':
            parameters = self.gen_model.parameters()
        else:
            parameters = itertools.chain(self.mpd_model.parameters(), self.msd_model.parameters())

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))