# yolo_predictions.py

import cv2
import numpy as np
import os
import yaml
from yaml.loader import SafeLoader

class YOLO_Pred():
    def __init__(self, onnx_model, data_yaml):
        # load YAML
        with open(data_yaml, mode='r') as f:
            data_yaml = yaml.load(f, Loader=SafeLoader)

        self.labels = data_yaml['names']
        self.nc = data_yaml['nc']

        # load YOLO model
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def predictions(self, image):
        # Modifique esta função para lidar com diferentes formatos de imagem

        if len(image.shape) == 2:
            # Imagem em tons de cinza
            h, w = image.shape
            d = 1
        else:
            # Imagem colorida
            h, w, d = image.shape

        # get the YOLO prediction from the image
        # step-1 convert image into square image (array)
        max_hw = max(h, w)
        input_image = np.zeros((max_hw, max_hw, 3), dtype=np.uint8)
        input_image[0:h, 0:w] = image
        # step-2: get prediction from square array
        INPUT_WH_YOLO = 1280
        blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WH_YOLO, INPUT_WH_YOLO), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        preds = self.yolo.forward()  # detection or prediction from YOLO

        # Restante do seu código permanece inalterado...

        return image

    def generate_colors(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return tuple(colors[ID])
