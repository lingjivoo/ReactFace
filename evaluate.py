import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
from dataclasses import dataclass

from dataset import get_dataloader
from model import ReactFace
from render import Render


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    dataset_path: str
    split: str
    resume: str = ""
    batch_size: int = 4
    num_workers: int = 8
    img_size: int = 256
    crop_size: int = 224
    max_seq_len: int = 800
    window_size: int = 8
    clip_length: int = 751
    feature_dim: int = 128
    audio_dim: int = 768
    tdmm_dim: int = 58
    outdir: str = "./results"
    device: str = 'cuda'
    momentum: float = 0.99
    rendering: bool = False

class Evaluator:
    def __init__(
            self,
            config: EvalConfig,
            model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            render: Optional[Render] = None
    ):
        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.render = render
        self.device = torch.device(config.device)

        # Setup output directories
        self.output_dir = Path(config.outdir) / config.split
        self.coeffs_dir = self.output_dir / 'coeffs'
        self.video_dirs = [self.output_dir / f'video{i + 1}' for i in range(10)]

        # Create directories
        self.coeffs_dir.mkdir(parents=True, exist_ok=True)
        for video_dir in self.video_dirs:
            video_dir.mkdir(parents=True, exist_ok=True)

        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy')
        ).view(1, 1, -1)

    def load_checkpoint(self) -> None:
        """Load model checkpoint."""
        if self.config.resume:
            print(f"Loading checkpoint from {self.config.resume}")
            checkpoint = torch.load(
                self.config.resume,
                map_location='cpu'
            )
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> None:
        """Run evaluation."""
        self.load_checkpoint()

        all_listener_3dmm_list = []

        # First generation
        print("Generating first prediction...")
        listener_3dmm_list = []

        for batch_idx, batch in enumerate(tqdm(self.test_loader)):
            speaker_video, speaker_audio, speaker_3dmm, listener_video, _, listener_ref, _,  video_path = batch

            # Move to device and handle sequence length
            speaker_video = speaker_video[:, :750].to(self.device)
            speaker_audio = speaker_audio.to(self.device)

            if self.config.rendering:
                listener_ref = listener_ref.to(self.device)

            # Generate prediction
            past_reaction_3dmm = torch.zeros(
                speaker_video.size(0),
                self.config.window_size,
                self.config.tdmm_dim,
                device=self.device
            )
            past_motion_sample = None

            # Process in windows
            predictions = []
            audio_internal = speaker_audio.shape[1] // speaker_video.shape[1]

            for i in range(0, 750, self.config.window_size):
                end_idx = min(i + self.config.window_size, 750)
                current_video = speaker_video[:, :end_idx]
                current_audio = speaker_audio[:, : end_idx * audio_internal]

                current_3dmm, current_motion_sample = self.model.inference_step(
                    current_video,
                    current_audio,
                    past_reaction_3dmm,
                    past_motion_sample
                )

                predictions.append(current_3dmm)
                past_reaction_3dmm = current_3dmm
                past_motion_sample = current_motion_sample

            listener_3dmm_out = torch.cat(predictions, dim=1)[:, :750]

            # Save first generation video if rendering
            if self.config.rendering:
                video_name = '_'.join(video_path[0].split('/'))
                render_vectors = (listener_3dmm_out + self.mean_face.to(self.device))[0]
                self.render.rendering_2d(
                    str(self.video_dirs[0]),
                    video_name,
                    render_vectors,
                    listener_ref[0]
                )

            listener_3dmm_list.append(listener_3dmm_out.cpu() + self.mean_face)

        # Combine first generation results
        listener_3dmm = torch.cat(listener_3dmm_list, dim=0)
        all_listener_3dmm_list.append(listener_3dmm.unsqueeze(1))

        print("Saving predictions...")
        np.save(
            self.coeffs_dir / 'tdmm_1x.npy',
            listener_3dmm.numpy().astype(np.float32)
        )

        # Generate 9 more times
        print("Generating 9 more predictions...")
        for gen_idx in range(9):
            print(f"----- Generation {gen_idx + 2}/10 -----")
            listener_3dmm_list = []

            for batch_idx, batch in enumerate(tqdm(self.test_loader)):
                speaker_video, speaker_audio = batch[0][:, :750].to(self.device), batch[1].to(self.device)

                # Generate prediction using sliding window
                past_reaction_3dmm = torch.zeros(
                    speaker_video.size(0),
                    self.config.window_size,
                    self.config.tdmm_dim,
                    device=self.device
                )
                past_motion_sample = None

                predictions = []
                audio_internal = speaker_audio.shape[1] // speaker_video.shape[1]
                for i in range(0, 750, self.config.window_size):
                    end_idx = min(i + self.config.window_size, 750)
                    current_video = speaker_video[:, :end_idx]
                    current_audio = speaker_audio[:, : end_idx * audio_internal]

                    current_3dmm, current_motion_sample = self.model.inference_step(
                        current_video,
                        current_audio,
                        past_reaction_3dmm,
                        past_motion_sample
                    )

                    predictions.append(current_3dmm)
                    past_reaction_3dmm = current_3dmm
                    past_motion_sample = current_motion_sample

                listener_3dmm_out = torch.cat(predictions, dim=1)[:, :750]
                listener_3dmm_list.append(listener_3dmm_out.cpu() + self.mean_face)

            # Combine results for this generation
            listener_3dmm = torch.cat(listener_3dmm_list, dim=0)
            all_listener_3dmm_list.append(listener_3dmm.unsqueeze(1))

        # Save all predictions
        all_listener_3dmm = torch.cat(all_listener_3dmm_list, dim=1)

        print("Saving predictions...")
        np.save(
            self.coeffs_dir / 'tdmm_10x.npy',
            all_listener_3dmm.numpy().astype(np.float32)
        )
        print(f"Evaluation complete. Results saved to {self.output_dir}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='ReactFace Evaluation')
    parser.add_argument('--dataset-path', default="Path/To/Dataset_root", type=str, help="dataset path")
    parser.add_argument('--split', required=True, choices=['val', 'test'], help='Dataset split')
    parser.add_argument('--resume', default='', help='Path to checkpoint')
    parser.add_argument('--outdir', default='./results', help='Output directory')
    parser.add_argument('--gpu-ids', default='0', help='GPU IDs to use')
    parser.add_argument('--window-size', type=int, default=8, help='Window size for inference')
    parser.add_argument('--rendering', action='store_true', help='Enable rendering')
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size (default: 4)')

    args = parser.parse_args()

    # Create config
    config = EvalConfig(
        dataset_path=args.dataset_path,
        split=args.split,
        resume=args.resume,
        window_size=args.window_size,
        outdir=args.outdir,
        rendering=args.rendering,
        momentum = args.momentum,
        num_workers = args.num_workers,
        batch_size = args.batch_size
    )

    # Set GPU devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ["NUMEXPR_MAX_THREADS"] = '16'

    # Create data loader based on rendering flag
    if config.rendering:
        test_loader = get_dataloader(
            config,
            config.split,
            load_audio=True,
            load_video_s=True,
            load_video_l=True,
            load_3dmm_l=False,
            load_ref=True
        )
    else:
        test_loader = get_dataloader(
            config,
            config.split,
            load_audio=True,
            load_video_s=True,
            load_video_l=False,
            load_3dmm_l=False,
            load_ref=True
        )

    # Create model
    model = ReactFace(
        img_size=config.img_size,
        output_3dmm_dim=config.tdmm_dim,
        feature_dim=config.feature_dim,
        max_seq_len=config.max_seq_len,
        window_size=config.window_size,
        device=config.device,
    ).to(config.device)
    model.reset_window_size(config.window_size)

    # Initialize render if needed
    render = None
    if config.rendering:
        render = Render('cuda' if torch.cuda.is_available() else 'cpu')

    # Create evaluator and run evaluation
    evaluator = Evaluator(config, model, test_loader, render)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
