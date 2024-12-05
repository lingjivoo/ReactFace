import os
import sys
import numpy as np
import torch
from torchvision import transforms
from skimage.io import imsave
import skvideo.io
from pathlib import Path
from tqdm import auto
import argparse
import cv2

from utils import torch_img_to_np, _fix_image, torch_img_to_np2
from external.FaceVerse import get_faceverse
from external.PIRender import FaceGenerator


def obtain_seq_index(index, num_frames, semantic_radius = 13):
    seq = list(range(index - semantic_radius, index + semantic_radius + 1))
    seq = [min(max(item, 0), num_frames - 1) for item in seq]
    return seq


def transform_semantic(semantic):
    semantic_list = []
    for i in range(semantic.shape[0]):
        index = obtain_seq_index(i, semantic.shape[0])
        semantic_item = semantic[index, :].unsqueeze(0)
        semantic_list.append(semantic_item)
    semantic = torch.cat(semantic_list, dim = 0)
    return semantic.transpose(1,2)



class Render(object):
    """Computes and stores the average and current value"""

    def __init__(self, device = 'cpu'):
        self.faceverse, _ = get_faceverse(device=device, img_size=224)
        self.faceverse.init_coeff_tensors()
        self.id_tensor = torch.from_numpy(np.load('external/FaceVerse/reference_full.npy')).float().view(1,-1)[:,:150]
        self.pi_render = FaceGenerator().to(device)
        self.pi_render.eval()
        checkpoint = torch.load('external/PIRender/cur_model_fold.pth')
        self.pi_render.load_state_dict(checkpoint['state_dict'])

    def rendering_2d(self, path, ind, listener_vectors, listener_reference):

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 8
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).view(
            listener_vectors.shape[0], C, H, W)

        for i in range(8):
            if i != 7:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)

        out = cv2.VideoWriter(os.path.join(path, ind + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (224, 224))
        for i in range(listener_videos.shape[0]):
            combined_img = np.zeros((224, 224, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = listener_videos[i]
            out.write(combined_img)
        out.release()


    def rendering_3d(self, path, ind, listener_vectors):

        # 3D video
        T = listener_vectors.shape[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:,:52].view(T,-1).to(listener_vectors.get_device())
        self.faceverse.rot_tensor = listener_vectors[:,52:55].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.trans_tensor = listener_vectors[:,55:].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.id_tensor = self.id_tensor.view(1,150).repeat(T,1).view(T,150).to(listener_vectors.get_device())

        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        out = cv2.VideoWriter(os.path.join(path, ind + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (224, 224))
        for i in range(listener_vectors.shape[0]):
            combined_img = np.zeros((224, 224, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = rendered_img_r[i]
            out.write(combined_img)
        out.release()


    def rendering_with_speaker_video(self, path, ind, listener_vectors, speaker_video_clip, listener_reference):

        T = listener_vectors.shape[0]

        self.faceverse.batch_size = T
        self.faceverse.init_coeff_tensors()

        self.faceverse.exp_tensor = listener_vectors[:, :52].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.rot_tensor = listener_vectors[:, 52:55].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.trans_tensor = listener_vectors[:, 55:].view(T, -1).to(listener_vectors.get_device())
        self.faceverse.id_tensor = self.id_tensor.view(1, 150).repeat(T, 1).view(T, 150).to(listener_vectors.get_device())

        pred_dict = self.faceverse(self.faceverse.get_packed_tensors(), render=True, texture=False)
        rendered_img_r = pred_dict['rendered_img']
        rendered_img_r = np.clip(rendered_img_r.cpu().numpy(), 0, 255)
        rendered_img_r = rendered_img_r[:, :, :, :3].astype(np.uint8)

        # 2D video
        semantics = transform_semantic(listener_vectors.detach()).to(listener_vectors.get_device())
        C, H, W = listener_reference.shape
        output_dict_list = []
        duration = listener_vectors.shape[0] // 8
        listener_reference_frames = listener_reference.repeat(listener_vectors.shape[0], 1, 1).view(
            listener_vectors.shape[0], C, H, W)

        for i in range(8):
            if i != 7:
                listener_reference_copy = listener_reference_frames[i * duration:(i + 1) * duration]
                semantics_copy = semantics[i * duration:(i + 1) * duration]
            else:
                listener_reference_copy = listener_reference_frames[i * duration:]
                semantics_copy = semantics[i * duration:]
            output_dict = self.pi_render(listener_reference_copy, semantics_copy)
            fake_videos = output_dict['fake_image']
            fake_videos = torch_img_to_np2(fake_videos)
            output_dict_list.append(fake_videos)

        listener_videos = np.concatenate(output_dict_list, axis=0)
        speaker_video_clip = torch_img_to_np2(speaker_video_clip)

        out = cv2.VideoWriter(os.path.join(path, ind + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (672, 224))
        for i in range(rendered_img_r.shape[0]):
            combined_img = np.zeros((224, 672, 3), dtype=np.uint8)
            combined_img[0:224, 0:224] = speaker_video_clip[i]
            combined_img[0:224, 224:448] = rendered_img_r[i]
            combined_img[0:224, 448:] = listener_videos[i]
            out.write(combined_img)
        out.release()

