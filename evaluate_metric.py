import numpy as np
import torch
import argparse
import os
from metric import (
    compute_FRDvs,
    compute_FRVar,
    compute_FRDiv,
    compute_FRC,
    compute_FRSyn,
    compute_FRC_mp,
    compute_FRD_mp,
    compute_FRSyn_mp
)

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D Face Metrics Computation')
    parser.add_argument('--dataset-path', default="/Path/To/Dataset_root", type=str)
    parser.add_argument('--split', type=str, choices=["val", "test"], required=True)
    parser.add_argument('--gt-speaker-3dmm-path', default="metric/gt/tdmm_speaker.npy", type=str)
    parser.add_argument('--gt-listener-3dmm-path', default="metric/gt/tdmm_listener.npy", type=str)
    parser.add_argument('--gn-listener-3dmm-path',
                        default="PATH/TO/Generated_listener_reactions.npy", type=str)
    return parser.parse_args()


def load_gt_3dmm(path: str, max_frames: int = 750, expected_features: int = 58) -> np.ndarray:
    """Load ground truth 3DMM coefficients (3D format)"""
    data = np.load(path)
    return data[:, :max_frames, :expected_features]


def load_generated_3dmm(path: str, max_frames: int = 750, expected_features: int = 58) -> np.ndarray:
    """Load generated 3DMM coefficients and ensure 4D format"""
    data = np.load(path)

    if len(data.shape) == 3:  # (1612, 750, 58)
        # 添加一个额外的维度使其变为 (1612, 1, 750, 58)
        return np.expand_dims(data, axis=1)[:,:,:max_frames,:]
    elif len(data.shape) == 4:
        return data[:,:,:max_frames,:]
    else:
        raise ValueError(f"Unexpected generated data shape: {data.shape}")


def main():
    args = parse_arguments()

    try:
        # Load ground truth data (3D format)
        gt_speaker_3dmm = load_gt_3dmm(args.gt_speaker_3dmm_path)
        gt_listener_3dmm = load_gt_3dmm(args.gt_listener_3dmm_path)
        print("Ground truth speaker 3DMM:", gt_speaker_3dmm.shape)
        print("Ground truth listener 3DMM:", gt_listener_3dmm.shape)

        # Load generated data (4D format)
        gn_listener_3dmm = load_generated_3dmm(args.gn_listener_3dmm_path)
        print("Generated listener 3DMM:", gn_listener_3dmm.shape)

        # Convert to torch tensors
        gn_tensor = torch.from_numpy(gn_listener_3dmm)
        gt_speaker_tensor = torch.from_numpy(gt_speaker_3dmm)
        gt_listener_tensor = torch.from_numpy(gt_listener_3dmm)

        # Compute metrics
        frdvs = compute_FRDvs(gn_listener_3dmm)
        print('FRDvs:',frdvs)
        frvar = compute_FRVar(gn_listener_3dmm)
        print('FRVar:',frvar)
        frdiv = compute_FRDiv(gn_tensor)
        print('FRDiv:',frdiv)
        frsyn = compute_FRSyn_mp(gn_tensor, gt_speaker_tensor)
        print('FRSyn:',frsyn)
        frc = compute_FRC_mp(args, gn_tensor, gt_listener_tensor)
        print('FRCorr:',frc)

        # Print results
        metrics = {
            'FRDvs': frdvs,
            'FRVar': frvar,
            'FRDiv': frdiv,
            'FRCorr ': frc,
            'FRSyn': frsyn
        }

        for name, value in metrics.items():
            print(f"{name}: {value:.5f}")

        print("\nMetrics Summary:")
        print(" | ".join([f"{name}: {value:.5f}" for name, value in metrics.items()]))

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
