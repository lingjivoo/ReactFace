import argparse
import torch
import numpy as np
import cv2
import librosa
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from torchvision import transforms
from PIL import Image
import os

from model import ReactFace
from render import Render


class WindowProcessor:
    def __init__(self, img_size=256, crop_size=224, window_size=8, sample_rate=16000, fps=25):
        self.window_size = window_size
        self.sample_rate = sample_rate
        self.fps = fps
        self.samples_per_frame = sample_rate // fps

        # Video transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Audio processor
        self.audio_processor = Wav2Vec2Processor.from_pretrained("external/facebook/wav2vec2-base-960h")

        # Initialize video capture
        self.cap = None
        self.audio_data = None
        self.current_frame = 0

    def _extract_audio_from_video(self, video_path: str, temp_path: str = 'temp_audio.wav'):
        """Extract audio from video file."""
        import subprocess
        subprocess.call([
            'ffmpeg', '-i', video_path,
            '-acodec', 'pcm_s16le',
            '-ar', str(self.sample_rate),
            '-ac', '1',
            '-y', temp_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        return temp_path

    def load_sources(self, video_path: str, audio_path: str = None):
        """Initialize video and audio sources."""
        # Setup video capture
        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if (fps is not None) and (fps > 0):
            self.fps = fps
        self.current_frame = 0

        # Load audio
        if audio_path is None:
            # Extract audio from video if not provided
            audio_path = self._extract_audio_from_video(video_path)

        audio_array, _ = librosa.load(audio_path, sr=self.sample_rate)
        self.audio_data = self.audio_processor(
            audio_array,
            sampling_rate=self.sample_rate
        ).input_values[0]

    def get_next_window(self):
        """Get next window of frames and corresponding audio."""
        frames = []
        last_valid_frame = None
        frame_count = 0

        for _ in range(self.window_size):
            ret, frame = self.cap.read()
            if not ret:  
                if not frames: 
                    return None, None
                if last_valid_frame is None:
                    last_valid_frame = torch.zeros(3, self.frame_height, self.frame_width)
                frames.append(last_valid_frame.clone())
                continue

            # Process frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frame_tensor = frame.unsqueeze(0)
            frames.append(frame_tensor)
            last_valid_frame = frame_tensor
            frame_count += 1

            self.current_frame += 1

        # Stack frames
        video_window = torch.cat(frames, dim=0)

        # Get corresponding audio window
        start_sample = (self.current_frame - frame_count) * self.samples_per_frame
        end_sample = (self.current_frame - frame_count + self.window_size) * self.samples_per_frame
        audio_window = self.audio_data[start_sample:min(end_sample, len(self.audio_data))]

        # Pad audio if necessary
        expected_audio_length = self.window_size * self.samples_per_frame
        if len(audio_window) < expected_audio_length:
            padding_length = expected_audio_length - len(audio_window)
            audio_window = np.pad(audio_window, (0, padding_length), mode='constant', constant_values=0)

        audio_window = torch.FloatTensor(audio_window)

        return video_window, audio_window

    def process_portrait(self, image_path: str) -> torch.Tensor:
        """Process listener portrait image."""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()


class DyadicInference:
    def __init__(
            self,
            checkpoint_path: str,
            output_dir: str,
            window_size: int = 8,
            momentum: float = 0.0,
            device: str = 'cuda'
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processor
        self.processor = WindowProcessor(window_size=window_size)

        # Initialize model
        self.model = ReactFace(
            img_size=256,
            output_3dmm_dim=58,
            feature_dim=128,
            max_seq_len=800,
            window_size=window_size,
            momentum = momentum,
            device=device
        ).to(self.device)

        # Load pretrained weights from checkpoint
        print(f"Loading pretrained weights from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:  # handle different checkpoint formats
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DataParallel/DistributedDataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded pretrained weights")
        except RuntimeError as e:
            print(f"Error loading weights: {e}")
            print("Attempting to load with strict=False...")
            self.model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded weights with strict=False")

        self.model.eval()

        # Initialize renderer
        self.render = Render(device)

        # Load mean face for 3DMM normalization
        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy')
        ).view(1, 1, -1)

    @torch.no_grad()
    def process_video(
            self,
            speaker_video_path: str,
            speaker_audio_path: str,
            listener_portrait_path: str
    ):
        """Process video in windows to generate reactions."""
        print("Initializing processing...")

        # Load reference portrait
        listener_ref = self.processor.process_portrait(listener_portrait_path)
        listener_ref = listener_ref.to(self.device)

        # Initialize processor with sources
        self.processor.load_sources(speaker_video_path, speaker_audio_path)

        # Initialize model states
        past_reaction_3dmm = torch.zeros(
            1, self.window_size, 58,
            device=self.device
        )
        past_motion_sample = None

        # Storage for generated reactions
        all_reactions = []

        print("Generating reactions...")
        # Process video in windows
        pbar = tqdm()  
        while True:
            # Get next window of data
            video_window, audio_window = self.processor.get_next_window()
            if video_window is None:
                break

            # Prepare inputs
            video_window = video_window.unsqueeze(0).to(self.device)  # Add batch dimension
            audio_window = audio_window.unsqueeze(0).to(self.device)

            # Generate reaction for current window
            current_3dmm, current_motion_sample = self.model.inference_step(
                video_window,
                audio_window,
                past_reaction_3dmm,
                past_motion_sample
            )

            # Store reaction
            all_reactions.append(current_3dmm)

            # Update states
            past_reaction_3dmm = current_3dmm
            past_motion_sample = current_motion_sample

            # Update progress bar with window size
            pbar.update(video_window.size(1))

        pbar.close()

        # Combine all reactions
        listener_3dmm = torch.cat(all_reactions, dim=1)
        listener_3dmm = listener_3dmm + self.mean_face.to(self.device)

        print("Rendering outputs...")
        video_name = Path(speaker_video_path).stem
        render_vectors = listener_3dmm[0]

        # Generate visualizations
        self.render.rendering_2d(
            str(self.output_dir),
            video_name,
            render_vectors,
            listener_ref
        )

        # Save 3DMM coefficients
        np.save(
            self.output_dir / f"{video_name}_3dmm.npy",
            render_vectors.cpu().numpy()
        )

        # Cleanup
        self.processor.cleanup()
        print(f"Processing complete. Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Dyadic Video Inference')
    parser.add_argument('--speaker-video', required=True, help='Path to speaker video')
    parser.add_argument('--speaker-audio', help='Path to speaker audio (optional)')
    parser.add_argument('--listener-portrait', required=True, help='Path to listener portrait')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--window-size', type=int, default=8, help='Window size for inference')
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()

    inferencer = DyadicInference(
        args.checkpoint,
        args.output_dir,
        args.window_size,
        args.momentum,
        args.device
    )

    inferencer.process_video(
        args.speaker_video,
        args.speaker_audio,
        args.listener_portrait
    )


if __name__ == "__main__":
    main()