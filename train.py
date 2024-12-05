import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader
from model import ReactFace
from model.losses import VAELoss, DivLoss, SmoothLoss, NeighbourLoss
from render import Render
from utils import AverageMeter


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    dataset_path: str = "Path/To/Dataset_root"
    resume: str = ""
    batch_size: int = 4
    learning_rate: float = 0.0001
    epochs: int = 100
    num_workers: int = 2
    weight_decay: float = 5e-4
    optimizer_eps: float = 1e-8
    img_size: int = 256
    crop_size: int = 224
    max_seq_len: int = 800
    window_size: int = 8
    clip_length: int = 256
    feature_dim: int = 128
    audio_dim: int = 768
    tdmm_dim: int = 58
    online: bool = False
    momentum: float = 0.99
    outdir: str = "./results"
    device: str = 'cuda'
    gpu_ids: str = '0'
    kl_p: float = 0.0002
    sm_p: float = 10
    div_p: float = 100
    rendering: bool = True


class Trainer:
    """Trainer class for ReactFace model."""

    def __init__(
            self,
            config: TrainingConfig,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: List[nn.Module],
            optimizer: optim.Optimizer,
            render: Optional[Render] = None
    ):
        """Initialize trainer with model and data loaders."""
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.render = render
        self.device = torch.device(config.device)
        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy')
        ).view(1, 1, -1)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self) -> Tuple[float, float, float, float, float, float]:
        """Train for one epoch."""
        self.model.train()
        meters = {
            'loss': AverageMeter(),
            'rec_loss': AverageMeter(),
            'kld_loss': AverageMeter(),
            'speaker_rec_loss': AverageMeter(),
            'div_loss': AverageMeter(),
            'sm_loss': AverageMeter()
        }

        for batch in tqdm(self.train_loader, desc="Training"):
            speaker_video, speaker_audio, speaker_3dmm, _, _, _, listener_3dmm_neighbour, _ = batch
            # Move data to device
            speaker_video = speaker_video.to(self.device)
            speaker_audio = speaker_audio.to(self.device)
            speaker_3dmm = speaker_3dmm.to(self.device)
            listener_3dmm_neighbour = listener_3dmm_neighbour.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            listener_3dmm_out, distribution, speaker_3dmm_out = self.model(
                speaker_video,
                speaker_audio,
                speaker_out=True
            )

            # Calculate losses
            loss, rec_loss, kld_loss = self.criterion[-1](
                listener_3dmm_neighbour,
                listener_3dmm_out,
                distribution
            )

            speaker_rec_loss = self.criterion[-2](speaker_3dmm, speaker_3dmm_out)

            # Generate additional outputs for diversity loss
            listener_out_2, _ = self.model(speaker_video, speaker_audio)
            listener_out_3, _ = self.model(speaker_video, speaker_audio)

            div_loss = (
                    self.criterion[1](listener_out_2, listener_3dmm_out) +
                    self.criterion[1](listener_out_3, listener_3dmm_out) +
                    self.criterion[1](listener_out_2, listener_out_3)
            )

            smooth_loss = self.criterion[2](listener_3dmm_out)

            # Combine losses
            total_loss = (
                    loss +
                    self.config.div_p * div_loss +
                    self.config.sm_p * smooth_loss +
                    speaker_rec_loss
            )

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Update meters
            batch_size = speaker_video.size(0)
            meters['loss'].update(total_loss.item(), batch_size)
            meters['rec_loss'].update(rec_loss.item(), batch_size)
            meters['kld_loss'].update(kld_loss.item(), batch_size)
            meters['speaker_rec_loss'].update(speaker_rec_loss.item(), batch_size)
            meters['div_loss'].update(div_loss.item(), batch_size)
            meters['sm_loss'].update(smooth_loss.item(), batch_size)

        return tuple(meter.avg for meter in meters.values())

    def validate(self, epoch: int) -> Tuple[float, float, float]:
        """Validate the model."""
        self.model.eval()
        self.model.reset_window_size(8)

        meters = {
            'loss': AverageMeter(),
            'rec_loss': AverageMeter(),
            'kld_loss': AverageMeter()
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating")):
                speaker_video, speaker_audio, _, _, listener_3dmm, listener_refs, _, _ = batch


                # Move data to device
                speaker_video = speaker_video.to(self.device)
                speaker_audio = speaker_audio.to(self.device)
                listener_3dmm = listener_3dmm.to(self.device)
                listener_refs = listener_refs.to(self.device)

                # Forward pass
                listener_3dmm_out, distribution = self.model(speaker_video, speaker_audio)

                # Calculate loss
                loss, rec_loss, kld_loss = self.criterion[0](
                    listener_3dmm,
                    listener_3dmm_out,
                    distribution
                )

                # Update meters
                batch_size = speaker_video.size(0)
                meters['loss'].update(loss.item(), batch_size)
                meters['rec_loss'].update(rec_loss.item(), batch_size)
                meters['kld_loss'].update(kld_loss.item(), batch_size)

                # Render validation results if needed
                if self.config.rendering and self.render and (batch_idx % 50) == 0:
                    self._render_validation_batch(
                        epoch,
                        batch_idx,
                        listener_3dmm_out+self.mean_face.to(self.device),
                        speaker_video,
                        listener_refs
                    )

        self.model.reset_window_size(self.config.window_size)
        return tuple(meter.avg for meter in meters.values())

    def _render_validation_batch(
            self,
            epoch: int,
            batch_idx: int,
            listener_3dmm_out: torch.Tensor,
            speaker_video: torch.Tensor,
            listener_refs: torch.Tensor
    ) -> None:
        """Render validation results for visualization."""
        val_path = Path(self.config.outdir) / 'results_videos' / 'val'
        val_path.mkdir(parents=True, exist_ok=True)
        for bs in range(speaker_video.size(0)):
            self.render.rendering_with_speaker_video(
                str(val_path),
                f"e{epoch + 1}_b{batch_idx + 1}_ind{bs + 1}",
                listener_3dmm_out[bs],
                speaker_video[bs],
                listener_refs[bs]
            )

    def save_checkpoint(
            self,
            epoch: int,
            is_best: bool = False,
            name: str = 'checkpoint'
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        save_path = Path(self.config.outdir)
        save_path.mkdir(parents=True, exist_ok=True)

        if is_best:
            torch.save(checkpoint, save_path / 'best_checkpoint.pth')
        else:
            torch.save(checkpoint, save_path / f'{name}_checkpoint.pth')

    def train(self) -> None:
        """Main training loop."""
        start_epoch = 0
        lowest_val_loss = float('inf')

        # Resume from checkpoint if specified
        if self.config.resume:
            start_epoch = self._load_checkpoint(self.config.resume)

        for epoch in range(start_epoch, self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch: {epoch + 1} "
                f"Train Loss: {train_metrics[0]:.5f} "
                f"Rec Loss: {train_metrics[1]:.5f} "
                f"KLD Loss: {train_metrics[2]:.5f} "
                f"Div Loss: {train_metrics[3]:.5f} "
                f"Smooth Loss: {train_metrics[4]:.5f} "
                f"Speaker Rec Loss: {train_metrics[5]:.5f}"
            )

            # Validate every 50 epochs
            if (epoch + 1) % 50 == 0:
                val_metrics = self.validate(epoch)
                self.logger.info(
                    f"Epoch: {epoch + 1} "
                    f"Val Loss: {val_metrics[0]:.5f} "
                    f"Val Rec Loss: {val_metrics[1]:.5f} "
                    f"Val KLD Loss: {val_metrics[2]:.5f}"
                )

                # Save checkpoint
                self.save_checkpoint(epoch, name=str(epoch + 1))

                # Save best model
                if val_metrics[0] < lowest_val_loss:
                    lowest_val_loss = val_metrics[0]
                    self.save_checkpoint(epoch, is_best=True)

            # Save current model
            self.save_checkpoint(epoch, name='cur')

    def _load_checkpoint(self, checkpoint_path: str) -> int:
        """Load model checkpoint."""
        self.logger.info(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch']


def main():
    """Main function."""
    # Parse command line arguments and create config
    import argparse
    parser = argparse.ArgumentParser(description='ReactFace Training')
    parser.add_argument('--window-size', type=int, default=64, help='Window size for inference')
    parser.add_argument('--rendering', action='store_true', help='Enable rendering')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--kl-p', default=0.0002, type=float, help="hyperparameter for kl-loss")
    parser.add_argument('--sm-p', default=10, type=float, help="hyperparameter for smooth-loss")
    parser.add_argument('--div-p', default=100, type=float, help="hyperparameter for diversity-loss")
    parser.add_argument('--resume', default="", type=str, help="checkpoint path")
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('--outdir', default="./results", type=str, help="result dir")

    # Add arguments (same as your original argparse setup)
    args = parser.parse_args()
    config = TrainingConfig(**vars(args))

    # Set up environment
    os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_ids

    # Create data loaders
    train_loader = get_dataloader(
        config,
        "train",
        load_audio=True,
        load_video_s=True,
        load_3dmm_s=True,
        load_3dmm_l=False,
        load_neighbour_matrix=True
    )

    val_loader = get_dataloader(
        config,
        "val",
        load_audio=True,
        load_video_s=True,
        load_3dmm_l=True,
        load_ref=True
    )

    # Create model
    model = ReactFace(
        img_size=config.img_size,
        output_3dmm_dim=config.tdmm_dim,
        feature_dim=config.feature_dim,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        device=config.device
    )

    # Define loss functions
    criterion = [
        VAELoss(config.kl_p).cuda(),
        DivLoss(),
        SmoothLoss(),
        nn.SmoothL1Loss(reduce=True, size_average=True),
        NeighbourLoss(config.kl_p).cuda()
    ]

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.999),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initialize render
    render = Render('cuda' if torch.cuda.is_available() else 'cpu')

    # Move model to device
    device = torch.device(config.device)
    model = model.to(device)

    # Create trainer and start training
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        render=render
    )

    trainer.train()


if __name__ == "__main__":
    main()
