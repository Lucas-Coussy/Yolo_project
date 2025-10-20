
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import torch
from itertools import product

class CNN_model():
    def __init__(self):
        self.df = pd.DataFrame(columns=['file_name','ID_class','x_center','y_center','width','height'])
        self.encoded_picture_annot = pd.DataFrame(columns=['file_name','encoded_grid'])
        self.classes = []
        self.C = None
        self.B = 2
        self.S = 7
        self.l_coord = 5
        self.l_noobj = 0.5
        self.bbox = 0

    def get_annotation(self,xml_list):
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
                x_center = ((xmin + xmax) / 2) / img_width #normalized coordinates
                y_center = ((ymin + ymax) / 2) / img_height #normalized coordinates
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                self.df.loc[len(self.df)+1] = [xml_file.replace("xml","jpg"),cls_id,x_center,y_center,width,height]
        
        self.C = len(self.classes)
        return

    def encode_annotation(self, img_df):
        y_true = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for _, row in img_df.iterrows():
            cls_id = int(row['ID_class'])
            x, y, w, h = row[['x_center', 'y_center', 'width', 'height']] #normalized coordinates

            # Choose responsible bounding box (first one for simplicity)
            base = self.bbox * 5

            #get the num of the cell the center of the object is
            grid_x = int(x * self.S)
            grid_y = int(y * self.S)
            #make sure the object the num_cell is not out of bound
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            # In yolo we localize the object with it's central cell
            y_true[grid_y, grid_x, base] = 1.0

            # Get the position of the center inside the cell (we know it's in cell (p,k) but we want to know exactly where it is inside)
            x_cell = x * self.S - grid_x
            y_cell = y * self.S - grid_y
            y_true[grid_y, grid_x, base+1:base+3] = torch.tensor([x_cell, y_cell])
            y_true[grid_y, grid_x, base+3:base+5] = torch.tensor([w, h])
            
            # After the 4 first value, we place the class of our object in a one hot encoding fashion
            y_true[grid_y, grid_x, self.B*5 + cls_id] = 1.0

            # exemple of output y_true[grid_x,grid_y] = [x_cell,y_cell,width,height,0,0,0,1,0,0,0,0] if there was only one box other wise there would be zero before and after
            # the position inside the cell (grid_x,grid_y) is x_cell,y_cell, it's width is width, it's height is height and it's class is four

        return y_true
            
    def encode_pictures(self, df=None):
        if df is None:
            df = self.df

        #encoded = []
        for filename in df['file_name'].unique():
            img_df = df[df['file_name'] == filename]
            y_true = self.encode_annotation(img_df)   # encode one image
            self.encoded_picture_annot.loc[len(self.encoded_picture_annot)+1] = [filename, y_true]
            #encoded.append(y_true)
        return 
        
    def Yolo1(self,C=None,S=None,B=None):
        if C is None :
            C = self.C
        else:
            self.C = C
        if S is None:
            S = self.S
        else:
            self.S = S
        if B is None:
            B = self.B
        else:
            self.B = B

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

        #model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        #model.add(layers.Conv2D(1024, (3, 3), strides=2, activation='relu'))
        #model.add(layers.MaxPooling2D((2, 2),strides=2))

        
        #model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu')) #input_dim=7*7*1024
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(S * S * (B * 5 + C), activation='linear'))
        model.add(layers.Reshape((S, S, B * 5 + C)))
        return model
    
    def yolo_loss(self,y_true,y_pred):
        loss_object = 0
        loss_nobject = 0
        loss_proba = 0
        for grid_x, grid_y in product(range(self.S)):
            cell_true = y_true[grid_x,grid_y]
            cell_pred = y_pred[grid_x,grid_y]
            base = self.bbox*5

            loss_object += self.l_coord*self.diff_obj(cell_true[base+1:base+5],cell_pred[base+1:base+5])
            loss_nobject = 0
            for box in range(0,self.B):
                if box != self.bbox:
                    box_base = box*5
                    loss_nobject += self.l_noobj*self.diff_nobj(cell_true[box_base+1:box_base+5],cell_pred[box_base+1:box_base+5])
            if cell_true[base] == 1:
                loss_proba = self.diff_proba(cell_true[self.B*5:self.B*5 + self.C],cell_pred[self.B*5:self.B*5 + self.C])
        return loss_object+loss_nobject+loss_proba
    
    def diff_obj(self,c_true,c_pred):
        xy_loss = (c_true[1]-c_pred[1])**2 + (c_true[2]-c_pred[2])**2
        wh_loss = (c_true[3]**(1/2)-c_pred[3]**(1/2))**2 + (c_true[4]**(1/2)-c_pred[4]**(1/2))**2
        C_loss = (c_true[0]-c_pred[0])**2
        return xy_loss + wh_loss + C_loss
    
    def diff_nobj(self,c_true,c_pred):
        return (c_true[0]-c_pred[0])**2
    
    def diff_proba(self,c_true,c_pred):
        loss_prob = 0
        for i in range(self.C):
            loss_prob += (c_true[i]-c_pred[i])**2

        return loss_prob
    
    def yolo_loss_tf(self,y_true, y_pred, S=7, B=2, C=None, l_coord=5, l_noobj=0.5):
        if C is None:
            C = self.C
        loss_object = 0.0
        loss_nobject = 0.0
        loss_proba = 0.0

        # Flatten grid for easier iteration
        for i in range(S):
            for j in range(S):
                cell_true = y_true[:, i, j, :]  # batch-wise
                cell_pred = y_pred[:, i, j, :]

                # For simplicity, use first box as responsible
                base = self.bbox

                # Object loss
                xy_loss = tf.square(cell_true[:, base+1] - cell_pred[:, base+1]) + tf.square(cell_true[:, base+2] - cell_pred[:, base+2])

                #removed because the model learned negative values to reduce loss because i chipped them
                #cell_pred_w = tf.sqrt(tf.clip_by_value(cell_pred[:, base+3], 1e-6, 1e6)) #avoid negative value for square root
                #cell_pred_h = tf.sqrt(tf.clip_by_value(cell_pred[:, base+4], 1e-6, 1e6)) #avoid negative value for square root

                cell_pred_w = tf.sqrt(tf.abs(cell_pred[:, base+3]) + 1e-6)
                cell_pred_h = tf.sqrt(tf.abs(cell_pred[:, base+4]) + 1e-6)
                wh_loss = tf.square(tf.sqrt(cell_true[:, base+3]) - cell_pred_w) + tf.square(tf.sqrt(cell_true[:, base+4]) - cell_pred_h)
                
                C_loss = tf.square(cell_true[:, base] - cell_pred[:, base])

                loss_object += l_coord * (xy_loss + wh_loss + C_loss)

                # No-object loss
                for b in range(B):
                    if b != base:
                        b_base = b*5
                        loss_nobject += l_noobj * tf.square(cell_true[:, b_base] - cell_pred[:, b_base])

                # Class probability loss (only if object exists)
                mask = tf.expand_dims(tf.cast(cell_true[:, base] == 1.0, tf.float32), axis=-1)
                class_loss = tf.reduce_sum(tf.square(cell_true[:, B*5:B*5+C] - cell_pred[:, B*5:B*5+C]), axis=-1)
                loss_proba += tf.reduce_sum(mask * class_loss)

        total_loss = tf.reduce_mean(loss_object + loss_nobject + loss_proba)
        return total_loss