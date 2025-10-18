
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class CNN_model():
    def __init__(self):
        self.df = pd.DataFrame(columns=['file_name','ID_class','x_center','y_center','width','height'])
        self.classes = []
        

    def convert_annotation(self,xml_list):
        for xml_file in xml_list:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            img_width = int(root.find("size/width").text)
            img_height = int(root.find("size/height").text)
            
            for obj in root.findall("object"):
                cls = obj.find("name").text
                if cls in self.classes:
                    cls_id = self.classes.index(cls)
                else:
                    self.classes.append(cls)
                    cls_id = self.classes.index(cls)

                xmlbox = obj.find("bndbox")
                xmin = int(xmlbox.find("xmin").text)
                ymin = int(xmlbox.find("ymin").text)
                xmax = int(xmlbox.find("xmax").text)
                ymax = int(xmlbox.find("ymax").text)

                # format our data for yolo
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                self.df.loc[len(self.df)+1] = [xml_file.replace("xml","jpg"),cls_id,x_center,y_center,width,height]
        return
            
    def Yolo1(self,C=None,S=7,B=2):
        if C is None :
            C = len(self.classes)

        model = models.Sequential()
        model.add(layers.Conv2D(64, (7, 7), strides=2, activation='relu', input_shape=(448, 448, 3)))
        model.add(layers.MaxPooling2D((2, 2),strides=2))

        model.add(layers.Conv2D(192, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2),strides=2))

        model.add(layers.Conv2D(128, (1, 1), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (1, 1), activation='relu'))
        model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2),strides=2))
        
        for i in range(4):
            model.add(layers.Conv2D(256, (1, 1), activation='relu'))
            model.add(layers.Conv2D(512, (3, 3), activation='relu'))
        
        model.add(layers.Conv2D(512, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2),strides=2))

        for i in range(2):
            model.add(layers.Conv2D(512, (1, 1), activation='relu'))
            model.add(layers.Conv2D(1024, (3, 3), activation='relu'))

        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), strides=2, activation='relu'))
        model.add(layers.MaxPooling2D((2, 2),strides=2))

        
        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu', input_dim=7*7*1024))
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(S * S * (B * 5 + C), activation='linear'))
        model.add(layers.Reshape((S, S, B * 5 + C)))
        return
    