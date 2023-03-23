import os
import torch
from torch.nn.functional import interpolate
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import cv2

from pixloc.utils.data import Paths
from pixloc.localization.feature_extractor import FeatureExtractor
from pixloc.pixlib.utils.experiments import load_experiment
from pixloc.pixlib.datasets.view import read_image
from pixloc.visualization.viz_2d import plot_images, add_text
from pixloc.visualization.animation import VideoWriter, display_video
from pixloc.settings import DATA_PATH, LOC_PATH


def extract_all(args):
    experiment = 'pixloc_megadepth'
    
    device = torch.device('cuda')
    
    pipeline = load_experiment(experiment).to(device).eval()
    net = FeatureExtractor(pipeline.extractor, device, {'resize': None,})
    # net = FeatureExtractor(pipeline.extractor, device)
    
    root = args.root
    images = [image for image in os.listdir(root) if image.endswith(args.image_extention)]
    
    for i in tqdm(range(len(images))):
        image_name = images[i]
        image_path = os.path.join(root, image_name)
        image = read_image(image_path)
        # image = image)

        confs, _, _  = net(image)
        
        shape = confs[0][0].shape
        confs = [interpolate(c[None], size=shape, mode='bilinear', align_corners=False)[0] for c in confs]
        confs = [c.cpu().numpy()[0] for c in confs]
        fine, mid, coarse = confs
    
        fused = (fine*mid*coarse)**(1/3)

        # print(f"Shape: {fused.shape} -- Type: {type(fused)}")
        out_name = image_name.split('.')[0] + '.tiff'
        out_path = os.path.join(args.save_path, out_name)
        cv2.imwrite(out_path, fused)
        


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Extract and save confidence maps for all images in a given folder")
    parser.add_argument("--root", type=str, required=True, help='root path')
    parser.add_argument("--image_extention", type=str, default='.jpg', help='image extention. Default is .jpg')
    parser.add_argument("--save_path", type=str, required=True, help='save path')

    args = parser.parse_args()
    
    extract_all(args)
