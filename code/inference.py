from __future__ import print_function

import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt


from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import hopenet

from skimage import io
import time
from dataloader import FrameLoader
import argparse
import torchvision
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import tqdm
import logging


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--snapshot', dest='snapshot', 
        help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--path_video_csv',
        default='data/video_paths.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--durations_csv',
        default='data/durations.csv',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--batch_size', 
        default=1, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--gpu_number',
        default='0',
        type=str,
        help='GPU Number')
    parser.add_argument(
        '--faces_per_frame',
        default=2,
        type=int,
        help='Max number of faces to keep per frame')
    return parser.parse_args() 


def main():
    args = get_arguments()

    device = th.device(f'cuda:{args.gpu_number}')

    data_mean = th.tensor([0.485, 0.456, 0.406])
    data_std = th.tensor([0.229, 0.224, 0.225])

    idx_tensor = th.FloatTensor([idx for idx in range(66)]).to(device)

    mtcnn = MTCNN(image_size=224,keep_all=True,device=device)

    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = th.load(args.snapshot)
    model.load_state_dict(saved_state_dict)

    model.to(device)
    print('Ready to test network.')
    model.eval()

    dataset = FrameLoader(videos_csv=args.path_video_csv,  durations_csv=args.durations_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)#4)

    max_faces = args.faces_per_frame
    feat_dim = 2048*max_faces

    num_frames = 0
    num_faces = 0
    logging.info(f"Extracting faces feature with maximum {max_faces} faces per frame")
    with th.no_grad():
        for images, out_path in tqdm.tqdm(dataloader):
            if len(images.shape) > 3:
                images = images.squeeze(0)
                n_chunk = images.shape[0]
                num_frames =+ n_chunk
                features = th.cuda.FloatTensor(n_chunk, feat_dim).fill_(0)
                poses = th.cuda.FloatTensor(n_chunk, 3*max_faces).fill_(0)
                n_iter = int(np.ceil(n_chunk / float(args.batch_size)))
                for i in range(n_iter):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    images_batch = images[min_ind:max_ind]
                    embeddings = th.cuda.FloatTensor(feat_dim).fill_(0)
                    predictions = th.cuda.FloatTensor(3*max_faces).fill_(0)
                    try:
                        faces_batch, probs = mtcnn(images_batch, return_prob=True)
                        num_faces += sum([faces.shape[0] for faces in faces_batch if faces is not None])
                        # Normalize batch
                        faces_batch = [faces.sub(data_mean[None, :, None, None]).div(data_std[None, :, None, None]) for faces in faces_batch if faces is not None]
                        
                        for faces in faces_batch:

                            yaw, pitch, roll, feat = model(faces.to(device))

                            feat = feat[0:max_faces].view(-1)
                            embeddings[0:feat.shape[0]] = feat 

                            yaw_predicted = F.softmax(yaw, dim=-1)
                            pitch_predicted = F.softmax(pitch, dim=-1)
                            roll_predicted = F.softmax(roll, dim=-1)
                            
                            # Get continuous predictions in degrees.
                            yaw_predicted = [th.sum(this_yaw * idx_tensor) * 3 - 99 for this_yaw in yaw_predicted]
                            pitch_predicted = [th.sum(this_pitch * idx_tensor) * 3 - 99 for this_pitch in pitch_predicted]
                            roll_predicted = [th.sum(this_roll * idx_tensor) * 3 - 99 for this_roll in roll_predicted]
                            angles = th.tensor([[_yaw, _pitch, _roll] for _yaw,_pitch,_roll in zip(yaw_predicted,pitch_predicted,roll_predicted)])
                            angles = angles[0:max_faces].view(-1)
                            predictions[0:angles.shape[0]] = angles 
                    except:
                        print(f"Error in processing faces of video: {out_path}")
                        continue

                    features[min_ind:max_ind] = embeddings
                    poses[min_ind:max_ind] = predictions
                features = features.cpu().numpy()
                poses = poses.cpu().numpy()
                np.save(out_path[0], features)
                np.save(out_path[0].replace('_feature',''), poses)
            else:
                print(f'Video {out_path} already processed.')
        
        logging.info(f'Done, average number of faces per frame: {num_faces/num_frames}')
        

if __name__ == "__main__":
    main()
