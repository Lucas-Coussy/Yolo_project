import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Dropout, Flatten, BatchNorm2d
import pandas as pd
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torchvision.models as models

class CNN_model():
    def __init__(self):
        self.df = pd.DataFrame(columns=['file_name','ID_class','x_center','y_center','width','height'])
        self.classes = []
        self.encoded_picture_annot = pd.DataFrame(columns=['file_name','encoded_grid'])
        self.classes_id = []
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
                xmin = float(xmlbox.find("xmin").text)
                ymin = float(xmlbox.find("ymin").text)
                xmax = float(xmlbox.find("xmax").text)
                ymax = float(xmlbox.find("ymax").text)

                xmin = int(round(xmin))
                ymin = int(round(ymin))
                xmax = int(round(xmax))
                ymax = int(round(ymax))

                # format our data for yolo
                x_center = ((xmin + xmax) / 2) / img_width #normalized coordinates
                y_center = ((ymin + ymax) / 2) / img_height #normalized coordinates
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                self.df.loc[len(self.df)+1] = [xml_file.replace("xml","jpg"),cls_id,x_center,y_center,width,height]
        
        self.C = len(self.classes)
        return

    def encode_annotation(self, img_df):
        y_true = torch.zeros((self.S, self.S, self.B*5 + self.C))

        for _, row in img_df.iterrows():
            cls_id = int(row['ID_class'])
            x, y, w, h = row[['x_center', 'y_center', 'width', 'height']]

            grid_x = int(x * self.S)
            grid_y = int(y * self.S)
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            x_cell = x * self.S - grid_x
            y_cell = y * self.S - grid_y

            # ✅ Fill BOTH B predicted box slots
            # over kill to change back
            for b in range(self.B):
                y_true[grid_y, grid_x, b*5:(b+1)*5] = torch.tensor([1.0, x_cell, y_cell, w, h])

            # ✅ Class one-hot
            y_true[grid_y, grid_x, self.B*5 + cls_id] = 1.0

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

class YOLOHead2(nn.Module):
    """The layers added on for detection as described in the paper."""

    def __init__(self, in_channels=1024, S=7, B=2, C=20):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        depth = C + 5 * B   # same as 5B+C in YOLOv1 paper

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 14x14 → 7x7
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            #nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),  # (N, 1024*7*7)

            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            
            Dropout(0.5),
            nn.Linear(4096, S * S * depth)  # final YOLO tensor
        )

    def forward(self, x):
        out = self.net(x)                     # (N, S*S*depth)
        return out.view(-1, self.S, self.S, self.C + 5*self.B)
    
class YOLOHead(nn.Module):
    def __init__(self, in_channels=1280, num_classes=20, B=2):
        super().__init__()
        out_channels = B * 5 + num_classes  # tx,ty,tw,th,obj + classes
        self.conv = nn.Sequential(
            #nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),   
            nn.Conv2d(1024, 1024, kernel_size=1, padding=1),   # (Ch, 14, 14) -> (Ch, 7, 7)
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, out_channels, 1)
        )
        
    def forward(self, x):
        return self.conv(x)  # (N, out, S, S)
    

    
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20, pretrained=True):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.pretrained = pretrained
        
        #pretrained backbone
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1" if self.pretrained else None)
        self.features = mobilenet.features
        self.reduce = nn.Conv2d(1280, 1024, kernel_size=1)
        #self.head = YOLOHead(in_channels=1280, num_classes=C, B=B)
        self.head = YOLOHead2(in_channels=1024,S=7,B=2,C=20)
        #self.detector = nn.Conv2d(1280, B*5 + C, kernel_size=1)

        self.conv20 = Sequential(
            Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            Conv2d(64, 192, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            Conv2d(192, 128, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(128, 256, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(256, 256, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            # 4 repeated blocks
            Conv2d(512, 256, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),

            Conv2d(512, 256, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),

            Conv2d(512, 256, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),

            Conv2d(512, 256, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(256, 512, kernel_size=3, padding=1),
            LeakyReLU(0.1),

            Conv2d(512, 512, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            MaxPool2d(2, 2),

            Conv2d(1024, 512, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 512, kernel_size=1),
            LeakyReLU(0.1),
            Conv2d(512, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
        )

        # final YOLOv1 conv layers
        self.convfinal = Sequential(  
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            BatchNorm2d(1024),
            LeakyReLU(0.1),
        )

        # Fully connected layers for detection
        self.fc = Sequential(
            #Flatten(),
            nn.Flatten(start_dim=1),
            Linear(7 * 7 * 1024, 4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(4096, S * S * (C + B * 5))
        )

    def forward(self, x):
        if self.pretrained == True:
            x = self.features(x)
            x = self.reduce(x)
        else:
            x = self.conv20(x)   # ✅ fixed name

        #x = self.convfinal(x)
        #x = self.fc(x)
        
        #x = x.view(-1, self.S, self.S, self.C + self.B * 5)

        #return x
        #print("features",x.shape)
        #x = nn.AdaptiveAvgPool2d((7,7))(x)
        #print("AdaptiveAvgPool2d",x.shape)
        #out = self.detector(x)
        out = self.head(x)          # (N, B*(5+C), 14, 14)
        #print("head",out.shape)
        #return out.permute(0,2,3,1).contiguous()
        return out

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_obj=5, l_nobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_obj = l_obj
        self.l_nobj = l_nobj

    def iou(self, box1, box2):
        _, x1, y1, w1, h1 = box1
        _, x2, y2, w2, h2 = box2

        box1_x1 = x1 - w1 / 2
        box1_y1 = y1 - h1 / 2
        box1_x2 = x1 + w1 / 2
        box1_y2 = y1 + h1 / 2

        box2_x1 = x2 - w2 / 2
        box2_y1 = y2 - h2 / 2
        box2_x2 = x2 + w2 / 2
        box2_y2 = y2 + h2 / 2

        inter_x1 = torch.max(box1_x1, box2_x1)
        inter_y1 = torch.max(box1_y1, box2_y1)
        inter_x2 = torch.min(box1_x2, box2_x2)
        inter_y2 = torch.min(box1_y2, box2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)

        area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

        return inter_area / (area1 + area2 - inter_area + 1e-6)

    def get_absolute_pos(self, i, j, box, is_pred=True):
        # x,y relative to image
        x_abs = (j + box[1]) / self.S
        y_abs = (i + box[2]) / self.S

        if is_pred:
            # ensure positive width and height
            w_abs = torch.relu(box[3])
            h_abs = torch.relu(box[4])
        else:
            w_abs = box[3]
            h_abs = box[4]

        return torch.stack([box[0], x_abs, y_abs, w_abs, h_abs])

    def yolo_loss(self, pred, target):
        lambda_coord = self.l_obj
        lambda_noobj = self.l_nobj
        batch_size = pred.size(0)
        total_loss = 0.0

        for n in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):
                    pred_boxes = [pred[n, i, j, b*5:(b+1)*5] for b in range(self.B)]
                    pred_classes = pred[n, i, j, self.B*5:]
                    true_box = target[n, i, j, 0:5]
                    true_classes = target[n, i, j, 5:]

                    if true_box[0] == 1:
                        # convert both pred and true boxes to absolute coords
                        pred_abs_boxes = [self.get_absolute_pos(i, j, b, is_pred=True) for b in pred_boxes]
                        true_abs_box = self.get_absolute_pos(i, j, true_box, is_pred=False)

                        ious = torch.stack([self.iou(pb, true_abs_box) for pb in pred_abs_boxes])
                        best_box_idx = torch.argmax(ious)
                        best_box = pred_boxes[best_box_idx]
                        best_abs = pred_abs_boxes[best_box_idx]

                        # --- Coordinate loss in absolute coordinates ---
                        xy_loss = torch.sum((best_abs[1:3] - true_abs_box[1:3])**2)
                        wh_loss = torch.sum((torch.sqrt(best_abs[3:5] + 1e-6) - torch.sqrt(true_abs_box[3:5] + 1e-6))**2)

                        # --- Object confidence loss ---
                        pred_conf = torch.sigmoid(best_box[0])
                        conf_loss_obj = (pred_conf - ious[best_box_idx].detach())**2

                        # --- Class loss ---
                        pred_class_probs = torch.softmax(pred_classes, dim=-1)
                        class_loss = torch.sum((pred_class_probs - true_classes)**2)

                        total_loss += lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + class_loss

                    else:
                        # No-object cells
                        pred_abs_boxes = [self.get_absolute_pos(i, j, b, is_pred=True) for b in pred_boxes]
                        ious = torch.stack([self.iou(pb, self.get_absolute_pos(i,j,true_box,is_pred=False)) for pb in pred_abs_boxes])
                        best_box_idx = torch.argmax(ious)
                        best_box = pred_boxes[best_box_idx]

                        pred_conf = torch.sigmoid(best_box[0])
                        conf_loss_noobj = (pred_conf - 0.0)**2
                        total_loss += lambda_noobj * conf_loss_noobj

        return total_loss / batch_size

    def yolo_loss_stable(self, pred, target):
        lambda_coord = self.l_obj
        lambda_noobj = self.l_nobj

        S = self.S
        B = self.B
        C = self.C
        batch_size = pred.size(0)

        total_loss = 0.0

        for n in range(batch_size):
            for i in range(S):
                for j in range(S):
                    # get my B boxes
                    pred_boxes = [pred[n, i, j, b*5:(b+1)*5] for b in range(B)] # [c1,x1,y1,w1,h1,c2,x2,y2,w2,h2, ... for the B boxes]
                    # get my C classes
                    pred_classes = pred[n, i, j, 5*B:] 
                    # get true box
                    true_box = target[n, i, j, :5] # [c1,x1,y1,w1,h1] the true value
                    # get true classes
                    true_classes = target[n, i, j, 5*B:] #one hot encode of my true class

                    # if object
                    if true_box[0] == 1:

                        # get abs coord
                        true_abs = self.get_absolute_pos(i, j, true_box, is_pred=False)
                        pred_abs = [self.get_absolute_pos(i, j, pb, is_pred=True) for pb in pred_boxes]

                        # Select best predicted box based on IOU
                        ious = torch.stack([self.iou(p, true_abs) for p in pred_abs])
                        best_b = torch.argmax(ious)

                        best_box = pred_boxes[best_b]
                        best_abs = pred_abs[best_b]
                        
                        xy_loss = torch.sum((best_abs[1:3] - true_abs[1:3])**2)
                        
                        wh_loss = torch.sum((torch.sqrt(best_abs[3:5] + 1e-6) - torch.sqrt(true_abs[3:5] + 1e-6))**2)

                        # conf loss
                        pred_conf = torch.sigmoid(best_box[0])
                        conf_loss_obj = (pred_conf - 1.0)**2

                        # classification loss
                        pred_class_probs = torch.softmax(pred_classes, dim=-1)
                        class_loss = torch.sum((pred_class_probs - true_classes)**2)
                        #print("full wh_loss", lambda_coord*wh_loss)
                        #print("full xy_loss", lambda_coord*xy_loss)
                        #print("full conf_loss_obj", conf_loss_obj)
                        #print("full class_loss", class_loss)
                        
                        total_loss += lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + class_loss

                    
                    else:
                        for b in range(B):
                            pred_conf = torch.sigmoid(pred_boxes[b][0])
                            total_loss += lambda_noobj * (pred_conf - 0.0)**2

        return total_loss / batch_size
    
    """
    def get_absolute_pos_vectorized(self, batch_size, tensor, device='cuda'):
        ### get grid with index ij
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.S, device=device, dtype=tensor.dtype),
            torch.arange(self.S, device=device, dtype=tensor.dtype),
            indexing='ij'
        )

        # Check if tensor has B dimension
        if tensor.ndim == 5:  # (N,S,S,B,5)
            # Add grid to each box
            #tensor[..., 1:3] = torch.sigmoid(tensor[..., 1:3])
            tensor[..., 1] += grid_x.unsqueeze(0).unsqueeze(-1)  # j index
            tensor[..., 2] += grid_y.unsqueeze(0).unsqueeze(-1)  # i index
        elif tensor.ndim == 4:  # (N,S,S,5)
            #tensor[..., 1:3] = torch.sigmoid(tensor[..., 1:3])
            tensor[..., 1] += grid_x.unsqueeze(0)  # j index
            tensor[..., 2] += grid_y.unsqueeze(0)  # i index
        else:
            raise ValueError(f"Unexpected tensor shape {tensor.shape}")

        # Normalize x,y to [0,1] relative to image
        tensor[..., 1:3] = tensor[..., 1:3] / self.S
        # Ensure width/height are non-negative
        tensor[..., 3] = tensor[..., 3]**2
        tensor[..., 4] = tensor[..., 4]**2
        #tensor[..., 3:5] = torch.relu(tensor[..., 3:5])

        return tensor
    """

    def get_absolute_pos_vectorized(self, batch_size, tensor, device='cuda'):

        # make a SAFE copy
        t = tensor.clone()

        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.S, device=device, dtype=t.dtype),
            torch.arange(self.S, device=device, dtype=t.dtype),
            indexing='ij'
        )

        if t.ndim == 5:
            t[..., 1] = (t[..., 1] + grid_x.unsqueeze(0).unsqueeze(-1)) / self.S
            t[..., 2] = (t[..., 2] + grid_y.unsqueeze(0).unsqueeze(-1)) / self.S
        else:
            t[..., 1] = (t[..., 1] + grid_x.unsqueeze(0)) / self.S
            t[..., 2] = (t[..., 2] + grid_y.unsqueeze(0)) / self.S

        # YOLOv1: square width & height
        t[..., 3] = t[..., 3].pow(2)
        t[..., 4] = t[..., 4].pow(2)

        return t


    def bbox_iou_vectorized(self, pred_boxes, true_boxes, eps=1e-6):
        """
        pred_boxes: (N, S, S, B, 5)
        true_boxes: (N, S, S, 1, 5)
        returns: IoU → (N, S, S, B)
        """

        # Extract box coords
        pred_x, pred_y = pred_boxes[..., 1], pred_boxes[..., 2]
        pred_w, pred_h = pred_boxes[..., 3], pred_boxes[..., 4]

        true_x, true_y = true_boxes[..., 1], true_boxes[..., 2]
        true_w, true_h = true_boxes[..., 3], true_boxes[..., 4]

        # Convert (x,y,w,h) → (x1,y1,x2,y2)
        # YOLO format is center-based
        pred_x1 = pred_x - pred_w / 2
        pred_x2 = pred_x + pred_w / 2
        pred_y1 = pred_y - pred_h / 2
        pred_y2 = pred_y + pred_h / 2

        true_x1 = true_x - true_w / 2
        true_x2 = true_x + true_w / 2
        true_y1 = true_y - true_h / 2
        true_y2 = true_y + true_h / 2

        # Intersection rectangle
        inter_x1 = torch.max(pred_x1, true_x1)
        inter_x2 = torch.min(pred_x2, true_x2)
        inter_y1 = torch.max(pred_y1, true_y1)
        inter_y2 = torch.min(pred_y2, true_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        # area predictions and truth
        area_pred = pred_w * pred_h
        area_true = true_w * true_h

        # IoU
        iou = inter_area / (area_pred + area_true - inter_area + eps)

        return iou  # (N, S, S, B)

    def yolo_loss_vectorized(self, pred, target, device='cuda'):
        N, _, _, _ = pred.shape
        lambda_coord = self.l_obj
        lambda_noobj = self.l_nobj

        S = self.S
        B = self.B
        C = self.C

        total_loss = 0.0

        pred_boxes = pred[..., :B*5].reshape(N, S, S, B, 5)
        #print("pred_box shape", pred_boxes.shape)
        pred_cls   = pred[..., B*5:]  # (N,S,S,C)
        #print("pred_cls shape",pred_cls.shape)

        true_box = target[..., :5]
        true_cls = target[..., 10:]    # (N,S,S,C)

        obj_mask = (true_box[..., 0] == 1).float()   # (N,S,S)
        noobj_mask = 1 - obj_mask

        pred_boxes_abs = self.get_absolute_pos_vectorized(batch_size=N, tensor=pred_boxes, device=device)
        true_box_abs = self.get_absolute_pos_vectorized(batch_size=N, tensor=true_box, device=device)

        true_box_abs = true_box_abs.unsqueeze(3)
        iou = self.bbox_iou_vectorized(pred_boxes_abs,true_box_abs)

        best_iou, best_idx = torch.max(iou, dim=-1)

        pred_boxes_best = torch.gather(
            pred_boxes,               # (N,S,S,B,5)
            dim=3,                    # gather along B
            index=best_idx[...,None,None].expand(-1,-1,-1,1,5)
        ) # (N,S,S,1,5) select the best box 

        pred_boxes_best = pred_boxes_best.squeeze(3) # (N,S,S,5)

        pred_boxes_obj = pred_boxes_best[obj_mask.bool()]    # shape: (num_obj,5)
        pred_boxes_noobj = pred_boxes_best[noobj_mask.bool()]  # shape: (num_noobj,5)

        true_box_obj = true_box[obj_mask.bool()]
        true_box_noobj = true_box[noobj_mask.bool()] 

        pred_cls_obj = pred_cls[obj_mask.bool()] 
        pred_cls_noobj = pred_cls[noobj_mask.bool()] 

        true_cls_obj = true_cls[obj_mask.bool()] 
        true_cls_noobj = true_cls[noobj_mask.bool()] 
        
        ## obj part
        #print("pred_boxes_obj (x,y,w,h):")
        #print(pred_boxes_obj[..., 1:5])  # x, y, w, h

        #print("true_box_obj (x,y,w,h):")
        #print(true_box_obj[..., 1:5])

        xy_loss = torch.sum((pred_boxes_obj[...,1:3] - true_box_obj[...,1:3])**2)
        print("xy_loss",xy_loss)
        wh_loss = torch.sum((torch.sqrt(torch.relu(pred_boxes_obj[...,3:5]) + 1e-6) - torch.sqrt(torch.relu(true_box_obj[...,3:5] + 1e-6)))**2)
        print("wh_loss",wh_loss)
        #pred_conf_obj = torch.sigmoid(pred_boxes_obj[...,0])
        pred_conf_obj = pred_boxes_obj[...,0]
        #print(pred_conf_obj,torch.ones_like(pred_conf_obj))
        conf_loss_obj = torch.sum((pred_conf_obj - torch.ones_like(pred_conf_obj))**2) #create a one tensor of same dim as pred_conf
        print("conf_loss_obj",conf_loss_obj)
        #pred_class_probs = torch.softmax(pred_cls_obj, dim=-1)
        pred_class_probs = pred_cls_obj
        #print("class ", pred_class_probs,true_cls_obj)
        class_loss = torch.sum((pred_class_probs - true_cls_obj)**2)
        print("class_loss",class_loss)

        ## no obj part
        #pred_conf_noobj = torch.sigmoid(pred_boxes_noobj[...,0])
        pred_conf_noobj = pred_boxes_noobj[...,0]
        #print(pred_conf_noobj,torch.zeros_like(pred_conf_noobj))
        conf_loss_noobj = torch.sum((pred_conf_noobj - torch.zeros_like(pred_conf_noobj))**2)
        print("conf_loss_noobj",conf_loss_noobj)
        total_loss += lambda_coord * (xy_loss + wh_loss) + (conf_loss_obj + class_loss) + lambda_noobj * conf_loss_noobj

        return total_loss / N
    
class YoloDataset(Dataset):
    def __init__(self, X_paths, Y_tensor, img_size=448):
        self.X_paths = X_paths
        self.Y_tensor = Y_tensor
        self.img_size = img_size

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        # Load and preprocess image
        img = cv2.imread(self.X_paths[idx])
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[..., ::-1]  # BGR to RGB
        img = img / 255.0
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C,H,W)

        # Get label
        if isinstance(self.Y_tensor, torch.Tensor):
            label = self.Y_tensor[idx].float()
        else:
            label = torch.tensor(self.Y_tensor[idx], dtype=torch.float32)

        return img, label
  
class YOLO_visual():
    def __init__(self, S=None, B=None, C=None, classes=None, score_threshold=0.3):
        self.S = S
        self.B = B
        self.C = C
        self.classes = classes
        self.score_threshold = score_threshold
    
    def display_yolo_predictions(self,pred_tensor, img_path, score_threshold=0.3):
        """
        Visualize YOLOv1 predictions on a single image.
        
        Args:
            pred_tensor: torch.Tensor of shape (S, S, C + B*5)
            img_path: path to original image
            classes: list of class names (length C)
            S, B, C: YOLO model parameters
            score_threshold: confidence threshold for visualization
        """
        # Ensure CPU NumPy
        if isinstance(pred_tensor, torch.Tensor):
            pred_tensor = pred_tensor.detach().cpu().numpy()

        boxes_list = []
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.resize(img, (448, 448))
        img_h, img_w = img.shape[:2]

        for i in range(self.S):
            for j in range(self.S):
                cell = pred_tensor[i, j, :]
                class_scores = cell[self.B*5:]
                class_probs = torch.softmax(torch.tensor(class_scores), dim=0).numpy()
                print(class_probs)
                # Each bounding box (conf, x, y, w, h)
                for b in range(self.B):
                    start = b * 5
                    conf, x, y, w, h = cell[start:start + 5]
                    conf = 1 / (1 + np.exp(-conf))

                    # Compute image-space coordinates
                    x_img = ((j + x) * img_w) / self.S
                    y_img = ((i + y) * img_h) / self.S
                    w_img = w * img_w
                    h_img = abs(h) * img_h

                    # Determine class with max probability
                    class_id = np.argmax(class_probs)
                    class_prob = class_probs[class_id]
                    final_score = conf * class_prob

                    if final_score > score_threshold:
                        boxes_list.append({
                            'bbox': (x_img, y_img, w_img, h_img),
                            'class_id': class_id,
                            'score': float(final_score)
                        })

        # Draw boxes
        for b in boxes_list:
            x, y, w, h = b['bbox']
            class_id = b['class_id']
            score = b['score']
            color = (0, 255, 0)

            cv2.rectangle(img,
                        (int(x - w/2), int(y - h/2)),
                        (int(x + w/2), int(y + h/2)),
                        color, 2)
            label = f"{self.classes[class_id]}: {score:.2f}"
            cv2.putText(img, label, (int(x - w/2), int(y - h/2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("YOLOv1 Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Yolo_visualize(self,model,img_path,device,score_threshold=0.3):
        model.eval()

        img = cv2.imread(img_path)
        img = cv2.resize(img, (448, 448))
        img_rgb = img[..., ::-1] / 255.0
        img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)[0]  # shape: [S, S, C+B*5]

        # Visualize
        self.display_yolo_predictions(pred, img_path, score_threshold=score_threshold)

    
    def Yolo_visualize_batch(self, model, img_paths, device, score_threshold=0.3):
        model.eval()

        imgs = []
        original_imgs = []

        # Chargement & préprocessing batch
        for img_path in img_paths:
            img = cv2.imread(img_path)
            original_imgs.append(img.copy())
            img = cv2.resize(img, (448, 448))
            img_rgb = img[..., ::-1] / 255.0
            img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
            imgs.append(img_tensor)

        if len(imgs) == 2:
            diff = (imgs[0] - imgs[1]).abs().mean().item()
            maxdiff = (imgs[0] - imgs[1]).abs().max().item()
            print("Mean difference between preprocessed images:", diff)
            print("Max difference:", maxdiff)
        else:
            print("Need exactly 2 images to compare.")

        # Stack en batch : shape -> (batch, 3, 448, 448)
        batch = torch.stack(imgs).to(device)

        with torch.no_grad():
            preds = model(batch)   # shape: (batch, S, S, C+B*5)
            print("Différence max entre prédictions image 0 et image 1 :",(preds[0] - preds[1]).abs().max().item())
        # Boucle image par image
        for i in range(len(img_paths)):
            pred = preds[i]  # prédiction uniquement de l'image i
            if pred.shape[0] == self.C + self.B*5:
                pred = pred.permute(1, 2, 0)
            self.display_yolo_predictions(pred, img_paths[i], score_threshold=score_threshold)

class YOLOVisualizer:
    def __init__(self, S, B, C, classes, score_threshold=0.3):
        self.S = S
        self.B = B
        self.C = C
        self.classes = classes
        self.score_threshold = score_threshold

    def _tensor_to_boxes(self, pred):
        """
        Convert a YOLOv1 prediction tensor (S,S,C+B*5) into a list of boxes:
        Each box = (x_center, y_center, w, h, class_id, score)
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()

        boxes = []

        for i in range(self.S):
            for j in range(self.S):
                cell = pred[i, j]

                class_scores = cell[self.B * 5:]
                probs = np.exp(class_scores - np.max(class_scores))
                probs /= probs.sum()
                class_id = np.argmax(probs)
                class_prob = probs[class_id]

                for b in range(self.B):
                    conf, x, y, w, h = cell[b*5 : b*5+5]
                    conf = 1 / (1 + np.exp(-conf))  # Sigmoid confidence
                    final_score = conf * class_prob

                    if final_score < self.score_threshold:
                        continue

                    # Convert to relative image coordinates
                    x = (j + x) / self.S
                    y = (i + y) / self.S
                    w = w**2
                    h = h**2

                    boxes.append((x, y, w, h, class_id, final_score))

        return boxes

    def _draw_boxes(self, img, boxes):
        H, W = img.shape[:2]

        for (xc, yc, w, h, cls_id, score) in boxes:
            w *= W
            h *= H
            xc *= W
            yc *= H

            x1 = int(xc - w/2)
            y1 = int(yc - h/2)
            x2 = int(xc + w/2)
            y2 = int(yc + h/2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            label = f"{self.classes[cls_id]} {score:.2f}"
            cv2.putText(img, label, (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return img

    def visualize_single(self, model, img_path, device):
        model.eval()

        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (448, 448))
        tensor = torch.tensor(img_resized[..., ::-1]/255, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(tensor)[0]  # (S,S,C+5B)

        boxes = self._tensor_to_boxes(pred)
        output = self._draw_boxes(img_resized, boxes)

        cv2.imshow("YOLO Predictions", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
