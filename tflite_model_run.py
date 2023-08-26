import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import shutil
import gc

from mediapipe_model_maker import image_classifier
from mediapipe_model_maker import object_detector
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


class Video():
    def __init__(self, filename):
        cap = cv2.VideoCapture(filename)
        self.filename = filename
        self.readed = False
        self.length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.time = self.length/self.fps
        self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if not cap.isOpened():
            print("The file doesn't exist, or it's not a video")
            raise FileExistsError
        else:
            self.cap = cap
    
    
    def read_frame(self, n=-1, return_mode='RGB'):
        current_frame = self.current_frame_number()
        if n != -1:
            self.set_frame(n)
        
        cap = self.cap
        self.readed = True

        if cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                return None
            if return_mode == 'RGB':
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if n != -1:
                self.set_frame(current_frame + 1)
            return frame
        return None

    
    def set_frame(self, n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    
    
    def current_frame_number(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    
    def return_list(self, n=-1, return_mode='RGB'):
        current_frame = self.current_frame_number()
        startframe = 0
        
        if n != -1:
            startframe = n
        
        self.set_frame(startframe)
        frames = []
        frame = self.read_frame()

        # while frame[0] is not None:
        while frame is not None:
            if self.current_frame_number() == self.length:
                break
            frames.append(frame)
            frame = self.read_frame(return_mode=return_mode)
        print('Done')
        
        self.set_frame(current_frame)
        
        return frames


def pred_frame_pose(classifier, numpy_image):
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    # Perform image classification on the provided single image.
    classification_result = classifier.classify(mp_image)
    return classification_result.classifications[0].categories[0].category_name


def get_frames(video_path):
    video = Video(video_path)
    return video.return_list()


def pose_mean_algo(model, video_path):
    frames = get_frames(video_path)
    phases = ['P1', 'P2', 'P3', 'P4', 'P5', 'P7', 'P8', 'P10']
    phd = dict([(ph, []) for ph in phases])

    for i in range(0, len(frames), 2):
        # model returns 'P<num>'
        pred = pred_frame_pose(model, frames[i])
        phd[pred] += [i]
    ids = []
    for i, pn in enumerate(list(phd.values())):
        if len(pn) == 0:
            if i == 0:
                ids.append(0)
            else:
                ids.append(ids[i - 1] + 2)
        else:
            if len(ids) == 0:
                ids.append(sum(pn) // len(pn) + 1)
            else:
                if ids[i - 1] < sum(pn) // len(pn) + 1:
                    ids.append(sum(pn) // len(pn) + 1)
                else:
                    ids.append(ids[i - 1] + 2)
    gc.collect()

    return ids


def get_pose_clf(pose_model_path):
    base_options = python.BaseOptions(model_asset_path=pose_model_path)
    options = vision.ImageClassifierOptions(
        base_options=base_options, max_results=4)
    pose_clf = vision.ImageClassifier.create_from_options(options)
    return pose_clf


pose_model_path = '<path_to_tflite_model>'
pose_clf = get_pose_clf(pose_model_path)
y_pred = pose_mean_algo(pose_clf, '<path_to_video>')
print(y_pred)
