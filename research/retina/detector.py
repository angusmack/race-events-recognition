import keras
from ..retina.keras_retinanet import models
from ..retina.keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from ..retina.keras_retinanet.utils.visualization import draw_box, draw_caption
from ..retina.keras_retinanet.utils.colors import label_color

import cv2
import numpy as np


class RetinaNet:
    
    def __init__(self, model_path):
        self.detection_model = models.load_model(model_path, backbone_name='resnet50')

        
    def predict_on_batch(self, images):
        return [ self.predict(image) for image in images ]
    
    
    def predict(self, image):
        image, scale = resize_image(preprocess_image(image))
        boxes, scores, _ = self.detection_model.predict(np.expand_dims(image, axis=0))
        boxes = boxes.squeeze()
        scores = scores.squeeze()
        boxes = boxes[scores >= 0.5] / scale
        return boxes


    def draw_boxes(self, image):
        boxes = self.predict(image)
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        for box in boxes:
            b = box.astype(int)
            draw_box(draw, b, color=(255,0,0))
        return draw