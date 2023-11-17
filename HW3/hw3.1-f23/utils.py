import json

import numpy as np
import logging

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

def calculate_iou(gt_bbox, pred_bbox):
    """
    calculate iou 
    args:
    - gt_bbox [array]: 1x4 single gt bbox
    - pred_bbox [array]: 1x4 single pred bbox
    returns:
    - iou [float]: iou between 2 bboxes
    """
    xmin = np.max([gt_bbox[0], pred_bbox[0]])
    ymin = np.max([gt_bbox[1], pred_bbox[1]])
    xmax = np.min([gt_bbox[2], pred_bbox[2]])
    ymax = np.min([gt_bbox[3], pred_bbox[3]])
    
    intersection = max(0, xmax - xmin) * max(0, ymax - ymin)
    gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
    
    union = gt_area + pred_area - intersection
    return intersection / union


def process(image,label):
    """ function to normalize input images """
    image = tf.cast(image/255. ,tf.float32)
    return image,label


def display_metrics(history):
    """ plot loss and accuracy from keras history object """
    f, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(history.history['loss'], linewidth=3)
    ax[0].plot(history.history['val_loss'], linewidth=3)
    ax[0].set_title('Loss', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlabel('Epoch', fontsize=16)
    ax[0].legend(['train loss', 'val loss'], loc='upper right')
    ax[1].plot(history.history['accuracy'], linewidth=3)
    ax[1].plot(history.history['val_accuracy'], linewidth=3)
    ax[1].set_title('Accuracy', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].set_xlabel('Epoch', fontsize=16)
    ax[1].legend(['train acc', 'val acc'], loc='upper left')
    plt.show()

def get_data():
    """ simple wrapper function to get data """
    with open('./data/ground_truth.json') as f:
        ground_truth = json.load(f)
    
    with open('./data/predictions.json') as f:
        predictions = json.load(f)

    return ground_truth, predictions

def get_datasets(imdir):
    """ extract GTSRB dataset from directory 
        directory to be passed as an argument, imdir
    """
    train_dataset = image_dataset_from_directory(imdir, 
                                       image_size=(32, 32),
                                       batch_size=32,
                                       validation_split=0.2,
                                       subset='training',
                                       seed=123,
                                       label_mode='int')

    val_dataset = image_dataset_from_directory(imdir, 
                                        image_size=(32, 32),
                                        batch_size=32,
                                        validation_split=0.2,
                                        subset='validation',
                                        seed=123,
                                        label_mode='int')
    train_dataset = train_dataset.map(process)
    val_dataset = val_dataset.map(process)
    return train_dataset, val_dataset


def get_module_logger(mod_name):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger

