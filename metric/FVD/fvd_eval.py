import torch
import os
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from frechet_video_distance import frechet_video_distance
from typing import List, Tuple


# Configure CPU threads
def set_cpu_threads(num_threads: int = 32):
    """Set the number of CPU threads for various libraries."""
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    torch.set_num_threads(num_threads)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FVD Evaluation between video folders')
    parser.add_argument('--source-dir', required=True, type=str, help="Source video directory")
    parser.add_argument('--target-dir', required=True, type=str, help="Target video directory")
    parser.add_argument('--model-path', default="metric/FVD/pytorch_i3d_model/models/rgb_imagenet.pt",
                        type=str, help="Path to I3D model weights")
    parser.add_argument('--num-videos', type=int, default=100, help="Number of videos to evaluate")
    parser.add_argument('--frame-size', type=int, default=224, help="Frame size for resizing")
    parser.add_argument('--max-frames', type=int, default=750, help="Maximum number of frames to process")
    parser.add_argument('--cpu-threads', type=int, default=32, help="Number of CPU threads to use")
    return parser.parse_args()


def load_video(video_path: str, frame_size: int, max_frames: int = 750) -> np.ndarray:
    """
    Load and preprocess a video file.

    Args:
        video_path: Path to the video file
        frame_size: Size to resize frames to
        max_frames: Maximum number of frames to load

    Returns:
        np.ndarray: Processed video frames of shape (T, H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_size, frame_size), cv2.INTER_AREA)
        frames.append(frame)

    cap.release()

    if len(frames) != max_frames:
        print(f"Warning: Video {video_path} has {len(frames)} frames, expected {max_frames}")
        return None

    return np.stack(frames, axis=0)


def get_matching_videos(source_dir: str, target_dir: str, num_videos: int) -> Tuple[List[str], List[str]]:
    """
    Get matching video files from source and target directories.

    Args:
        source_dir: Directory containing source videos
        target_dir: Directory containing target videos
        num_videos: Number of videos to select

    Returns:
        Tuple of source and target video paths
    """
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.mp4')])
    target_files = sorted([f for f in os.listdir(target_dir) if f.endswith('.mp4')])

    # Find common video names
    common_names = set([f.rsplit('_', 1)[0] for f in source_files]) & \
                   set([f.rsplit('_', 1)[0] for f in target_files])

    # Randomly select videos
    if len(common_names) > num_videos:
        common_names = list(common_names)
        indices = torch.randperm(len(common_names))[:num_videos]
        common_names = [common_names[i] for i in indices]

    source_videos = []
    target_videos = []

    for name in common_names:
        s_file = next(f for f in source_files if f.startswith(name))
        t_file = next(f for f in target_files if f.startswith(name))
        source_videos.append(os.path.join(source_dir, s_file))
        target_videos.append(os.path.join(target_dir, t_file))

    return source_videos, target_videos


def main():
    args = parse_args()
    set_cpu_threads(args.cpu_threads)

    print("Loading videos...")
    source_videos, target_videos = get_matching_videos(
        args.source_dir, args.target_dir, args.num_videos
    )

    # Load and process videos
    source_frames = []
    target_frames = []

    for s_path, t_path in tqdm(zip(source_videos, target_videos), total=len(source_videos)):
        source_video = load_video(s_path, args.frame_size, args.max_frames)
        target_video = load_video(t_path, args.frame_size, args.max_frames)

        if source_video is not None and target_video is not None:
            source_frames.append(source_video)
            target_frames.append(target_video)

    if not source_frames:
        print("Error: No valid video pairs found!")
        return

    # Convert to torch tensors
    source_tensor = torch.from_numpy(np.stack(source_frames, axis=0)).float()
    target_tensor = torch.from_numpy(np.stack(target_frames, axis=0)).float()

    print(f"\nProcessed video shapes:")
    print(f"Source videos: {source_tensor.shape}")
    print(f"Target videos: {target_tensor.shape}")

    # Calculate FVD
    print("\nCalculating FVD...")
    fvd_score = frechet_video_distance(source_tensor, target_tensor, args.model_path)
    print(f"\nFréchet Video Distance: {fvd_score:.4f}")


if __name__ == "__main__":
    main()