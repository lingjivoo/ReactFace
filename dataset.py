from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import Wav2Vec2Processor


@dataclass
class DataConfig:
    """Configuration for dataset parameters."""
    dataset_path: str
    img_size: int = 256
    crop_size: int = 224
    clip_length: int = 751
    fps: int = 25
    batch_size: int = 32
    num_workers: int = 4


class VideoTransform:
    """Video frame transformation pipeline."""

    def __init__(self, img_size: int = 256, crop_size: int = 224):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        return self.transform(img)


def load_image(path: str) -> Image.Image:
    """Load an image file as RGB."""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def extract_video_frames(
        video_path: str,
        transform: VideoTransform,
        start_frame: int = 0,
        end_frame: Optional[int] = None
) -> torch.Tensor:
    """Extract and transform frames from a video file.

    Args:
        video_path: Path to video file
        transform: Transform to apply to each frame
        start_frame: Starting frame index
        end_frame: Ending frame index (inclusive)

    Returns:
        Tensor of transformed video frames
    """
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx >= start_frame and (end_frame is None or frame_idx < end_frame):
            ret, frame = cap.retrieve()
            if frame is None:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame).unsqueeze(0))
        elif end_frame is not None and frame_idx >= end_frame:
            break

        frame_idx += 1

    cap.release()
    return torch.cat(frames, dim=0) if frames else torch.empty(0)


class ReactionDataset(Dataset):
    """Dataset for reaction generation task."""

    def __init__(
            self,
            root_path: str,
            split: str,
            img_size: int = 256,
            crop_size: int = 224,
            clip_length: int = 751,
            fps: int = 25,
            load_audio: bool = True,
            load_video_s: bool = True,
            load_video_l: bool = True,
            load_3dmm_s: bool = False,
            load_3dmm_l: bool = False,
            load_ref: bool = True,
            load_neighbour_matrix: bool = False
    ):
        """
        Args:
            root_path: Path to dataset root directory
            split: Dataset split ('train', 'val', or 'test')
            img_size: Size for image resizing
            crop_size: Size for center cropping
            clip_length: Number of frames per clip
            fps: Frames per second
            load_audio: Whether to load audio features
            load_video_s: Whether to load speaker video
            load_video_l: Whether to load listener video
            load_3dmm_s: Whether to load speaker 3DMM parameters
            load_3dmm_l: Whether to load listener 3DMM parameters
            load_ref: Whether to load reference frames
            load_neighbour_matrix: Whether to load neighbour emotion matrix
        """
        super().__init__()

        self.root_path = Path(root_path)
        self.split = split
        self.clip_length = clip_length
        self.fps = fps

        # Loading flags
        self.load_audio = load_audio
        self.load_video_s = load_video_s
        self.load_video_l = load_video_l
        self.load_3dmm_s = load_3dmm_s
        self.load_3dmm_l = load_3dmm_l
        self.load_ref = load_ref
        self.load_neighbour_matrix = load_neighbour_matrix

        # Initialize transforms and processors
        self.transform = VideoTransform(img_size, crop_size)
        self.processor = Wav2Vec2Processor.from_pretrained("external/facebook/wav2vec2-base-960h")

        # Load 3DMM statistics
        self.mean_face = torch.FloatTensor(
            np.load('external/FaceVerse/mean_face.npy')
        ).view(1, 1, -1)
        self.std_face = torch.FloatTensor(
            np.load('external/FaceVerse/std_face.npy')
        ).view(1, 1, -1)

        # Setup paths
        self.audio_path = self.root_path / 'Audio_files'
        self.video_path = self.root_path / 'Video_files'
        self.tdmm_path = self.root_path / '3D_FV_files'

        # Load split data
        self.data_list = self._load_split_data()

        # Load neighbour matrix if needed
        self.neighbour_matrix = None
        if self.load_neighbour_matrix:
            self.neighbour_matrix = np.load(
                self.root_path / f'neighbour_emotion_new_{split}.npy'
            )

    def _load_split_data(self) -> List[Dict[str, str]]:
        """Load and process split data from CSV."""
        df = pd.read_csv(
            self.root_path / f'{self.split}.csv',
            header=None,
            skiprows=1
        )

        speaker_paths = df[1].tolist()
        listener_paths = df[2].tolist()

        # Create speaker-listener pairs
        all_speaker_paths = speaker_paths + listener_paths
        all_listener_paths = listener_paths + speaker_paths

        data_list = []
        for sp, lp in zip(all_speaker_paths, all_listener_paths):
            data_list.append({
                'speaker_video_path': str(self.video_path / f'{sp}.mp4'),
                'speaker_audio_path': str(self.audio_path / f'{sp}.wav'),
                'speaker_3dmm_path': str(self.tdmm_path / f'{sp}.npy'),
                'listener_video_path': str(self.video_path / f'{lp}.mp4'),
                'listener_audio_path': str(self.audio_path / f'{lp}.wav'),
                'listener_3dmm_path': str(self.tdmm_path / f'{lp}.npy')
            })

        return data_list

    def _get_clip_bounds(self, total_length: int) -> Tuple[int, int]:
        """Get start and end frame indices for clip."""
        if self.split == 'train':
            if self.clip_length < total_length:
                start = np.random.randint(0, total_length - self.clip_length)
            else:
                start = 0
        else:
            start = 0
        end = start + self.clip_length
        return start, end

    def _load_video_clip(
            self,
            video_path: str,
            start: int,
            end: int
    ) -> torch.Tensor:
        """Load and transform video clip frames."""
        return extract_video_frames(
            video_path,
            self.transform,
            start_frame=start,
            end_frame=end
        )

    def _load_audio_clip(
            self,
            audio_path: str,
            start: int,
            clip_length: int
    ) -> torch.Tensor:
        """Load and process audio clip with consistent length padding/truncation.

        Args:
            audio_path: Path to audio file
            start: Starting frame index
            clip_length: Desired clip length in frames

        Returns:
            torch.Tensor: Processed audio features of consistent length
        """
        if not self.load_audio:
            return torch.zeros(1)

        # Load audio file
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)

        # Calculate samples per frame
        interval = sampling_rate // self.fps

        # Calculate target length in samples
        target_length = clip_length * interval

        # Process audio features
        audio_features = np.squeeze(
            self.processor(
                speech_array,
                sampling_rate=16000
            ).input_values
        )

        # Extract relevant portion
        start_sample = int(start * interval)
        audio_clip = audio_features[start_sample:start_sample + target_length]

        # Pad or truncate to ensure consistent length
        if len(audio_clip) < target_length:
            # Pad with zeros if too short
            padding = target_length - len(audio_clip)
            audio_clip = np.pad(audio_clip, (0, padding), mode='constant')
        elif len(audio_clip) > target_length:
            # Truncate if too long
            audio_clip = audio_clip[:target_length]

        return torch.FloatTensor(audio_clip)

    def _load_3dmm(
            self,
            path: str,
            start: int,
            clip_length: int
    ) -> torch.Tensor:
        """Load and normalize 3DMM parameters."""
        params = torch.FloatTensor(np.load(path)).squeeze()
        params = params[start:start + clip_length]
        return (params - self.mean_face)[0]

    def _load_reference_frame(
            self,
            video_path: str
    ) -> torch.Tensor:
        """Load reference frame from video."""
        if not self.load_ref:
            return torch.zeros(1)

        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return self.transform(frame)
        return torch.zeros(1)

    def _load_neighbour_3dmm(
            self,
            index: int,
            start: int,
            clip_length: int
    ) -> torch.Tensor:
        """Load neighbouring 3DMM parameters."""
        if not self.load_neighbour_matrix:
            return torch.zeros(1)

        speaker_line = self.neighbour_matrix[index]
        speaker_indices = np.argwhere(speaker_line == 1).reshape(-1)
        max_neighbours = min(len(speaker_indices), 80)

        neighbour_3dmm = []
        for k in range(80):
            if k < max_neighbours:
                neighbour_idx = speaker_indices[k]
                path = self.data_list[neighbour_idx]['listener_3dmm_path']
                params = self._load_3dmm(path, start, clip_length)
            else:
                params = torch.full_like(
                    torch.zeros(clip_length, 58),
                    1e5,
                    dtype=torch.float32
                )
            neighbour_3dmm.append(params.unsqueeze(0))

        return torch.cat(neighbour_3dmm, dim=0)

    def __getitem__(self, index: int) -> Tuple:
        """Get a data sample."""
        data = self.data_list[index]

        # Determine clip bounds
        if self.load_video_s or self.load_video_l:
            cap = cv2.VideoCapture(data['speaker_video_path'])
            total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            params = np.load(data['speaker_3dmm_path'])
            total_length = params.shape[0]

        start, end = self._get_clip_bounds(total_length)

        # Load all required components
        speaker_video = self._load_video_clip(
            data['speaker_video_path'], start, end
        ) if self.load_video_s else torch.zeros(1)

        listener_video = self._load_video_clip(
            data['listener_video_path'], start, end
        ) if self.load_video_l else torch.zeros(1)

        speaker_audio = self._load_audio_clip(
            data['speaker_audio_path'], start, self.clip_length
        )

        speaker_3dmm = self._load_3dmm(
            data['speaker_3dmm_path'], start, self.clip_length
        ) if self.load_3dmm_s else torch.zeros(1)

        listener_3dmm = self._load_3dmm(
            data['listener_3dmm_path'], start, self.clip_length
        ) if self.load_3dmm_l else torch.zeros(1)

        listener_ref = self._load_reference_frame(data['listener_video_path'])

        neighbour_3dmm = self._load_neighbour_3dmm(index, start, self.clip_length)

        return (
            speaker_video,
            speaker_audio,
            speaker_3dmm,
            listener_video,
            listener_3dmm,
            listener_ref,
            neighbour_3dmm,
            data['listener_video_path']
        )

    def __len__(self) -> int:
        return len(self.data_list)


def get_dataloader(
        conf: DataConfig,
        split: str,
        load_audio: bool = False,
        load_video_s: bool = False,
        load_video_l: bool = False,
        load_3dmm_s: bool = False,
        load_3dmm_l: bool = False,
        load_ref: bool = False,
        load_neighbour_matrix: bool = False
) -> DataLoader:
    """Create data loader for specified split."""
    dataset = ReactionDataset(
        root_path=conf.dataset_path,
        split=split,
        img_size=conf.img_size,
        crop_size=conf.crop_size,
        clip_length=conf.clip_length,
        load_audio=load_audio,
        load_video_s=load_video_s,
        load_video_l=load_video_l,
        load_3dmm_s=load_3dmm_s,
        load_3dmm_l=load_3dmm_l,
        load_ref=load_ref,
        load_neighbour_matrix=load_neighbour_matrix
    )

    return DataLoader(
        dataset=dataset,
        batch_size=conf.batch_size,
        shuffle=(split == "train"),
        num_workers=conf.num_workers
    )
