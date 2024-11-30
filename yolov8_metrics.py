import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Path to the saved model weights
weights_path = "/media/robotics-300/8bf075d6-3600-4c70-8f2d-0d11f1ca9e25/robotics300/PycharmProjects/MediaPipe_Yolo_Lstm_222/runs/pose/train5/weights/best.pt"

# Load the trained model
model = YOLO(weights_path)

# Evaluate the model
metrics = model.val()

# Extract metrics
precision = metrics.get('metrics/precision', [])
recall = metrics.get('metrics/recall', [])
map_50 = metrics.get('metrics/mAP_50', [])
map_50_95 = metrics.get('metrics/mAP_50-95', [])
loss_cls = metrics.get('loss/class', [])
loss_box = metrics.get('loss/box', [])
loss_keypoints = metrics.get('loss/keypoints', [])

# Plotting functions
def plot_metric(metric_values, title, ylabel, xlabel="Epochs"):
    """Helper function to plot a metric."""
    plt.figure(figsize=(8, 6))
    plt.plot(metric_values, marker='o', label=title)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.grid()
    plt.legend()
    plt.show()

# Plot metrics
if precision:
    plot_metric(precision, "Precision", "Precision")
if recall:
    plot_metric(recall, "Recall", "Recall")
if map_50:
    plot_metric(map_50, "mAP@0.50", "mAP")
if map_50_95:
    plot_metric(map_50_95, "mAP@0.50:0.95", "mAP")
if loss_cls:
    plot_metric(loss_cls, "Classification Loss", "Loss")
if loss_box:
    plot_metric(loss_box, "Box Loss", "Loss")
if loss_keypoints:
    plot_metric(loss_keypoints, "Keypoints Loss", "Loss")
