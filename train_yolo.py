#Download data from roboflow
import yaml 
import os 
from ultralytics import YOLO
from roboflow import Roboflow


rf = Roboflow(api_key="ADDKEY")
project = rf.workspace("cv-project-gydhs").project("juv_overhead313")
version = project.version(3)
dataset = version.download("yolov11")


#load
model= YOLO("yolo11s.pt")

results = model.train (data="juv_overhead313-1/data.yaml", epochs=10)
results = model.val ()