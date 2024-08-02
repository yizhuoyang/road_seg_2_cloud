from ultralytics import FastSAM
import cv2
import sys
import torch

# Create a FastSAM model
def sam_infer(model_path,image,ref_points):
    model = FastSAM(model_path)
    results = model(image, points=ref_points, labels=[1,1],imgsz=448)
    return results

