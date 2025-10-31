import cv2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project")
#from classes.CNN_pytorch import YoloDataset, YoloLoss, YOLOv1, CNN_model, YOLO_visual #import my class CNN
from classes.CNN_chatgpt import YoloDataset, YOLOv1, YoloLoss


#os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project\data\Pascal_VOC\train")


#path = r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project\data\Pascal_VOC\train"

#xml_list = []
#for filename in os.listdir(path):
#    if not filename.endswith('.xml'): continue
#    fullname = os.path.join(path, filename)
#    xml_list.append(fullname)

#my_CNN = CNN_model()
#my_CNN.get_annotation(xml_list)
#my_CNN.encode_pictures()

#os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project")
#my_CNN.df.to_json(r"data\annot_df.json",orient="records")
#torch.save(my_CNN.encoded_picture_annot, r"data\encoded_picture.pt")


df_encoded = torch.load(r"data\encoded_picture.pt",weights_only=False)

df_encoded = df_encoded[:1000]
df_train, df_test = train_test_split(df_encoded,test_size=0.3)
print(df_train)

trainer = YoloLoss()
trainer.train_yolo(df_train,df_test)