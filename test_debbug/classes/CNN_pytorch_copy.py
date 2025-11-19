import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Dropout, Flatten
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
        self.B = 1
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
            #w_abs = box[3]**2
            #h_abs = box[4]**2
        else:
            w_abs = box[3] 
            h_abs = box[4] 

        return torch.stack([box[0], x_abs, y_abs, w_abs, h_abs])

    def yolo_loss(self, pred, target):
        lambda_coord = self.l_obj
        lambda_noobj = self.l_nobj
        print("lambda_coord", lambda_coord)
        print("lambda_noobj", lambda_noobj)
        batch_size = pred.size(0)
        total_loss = 0.0

        for n in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):
                    pred_boxes = [pred[n, i, j, b*5:(b+1)*5] for b in range(self.B)]
                    pred_classes = pred[n, i, j, self.B*5:]
                    true_box = target[n, i, j, 0:5]
                    true_classes = target[n, i, j, self.B*5:]

                    if true_box[0] == 1:
                        print("n,i,j",n,i,j)
                        # convert both pred and true boxes to absolute coords
                        pred_abs_boxes = [self.get_absolute_pos(i, j, b, is_pred=True) for b in pred_boxes]
                        true_abs_box = self.get_absolute_pos(i, j, true_box, is_pred=False)

                        ious = torch.stack([self.iou(pb, true_abs_box) for pb in pred_abs_boxes])
                        print("iou",ious)
                        best_box_idx = torch.argmax(ious)
                        best_box = pred_boxes[best_box_idx]
                        best_abs = pred_abs_boxes[best_box_idx]

                        # --- Coordinate loss in absolute coordinates ---
                        print("heh")
                        print("pred xy",best_abs[1:3])
                        print("true xy",true_abs_box[1:3])
                        xy_loss = torch.sum((best_abs[1:3] - true_abs_box[1:3])**2)
                        print("xy_loss",xy_loss)
                        print("pred wh",best_abs[3:5])
                        print("true wh",true_abs_box[3:5])
                        wh_loss = torch.sum((torch.sqrt(best_abs[3:5] + 1e-5) - torch.sqrt(true_abs_box[3:5] + 1e-5))**2)
                        print("wh_loss",wh_loss)
                        # --- Object confidence loss ---
                        pred_conf = torch.sigmoid(best_box[0])
                        #pred_conf = best_box[0]
                        print("pred_conf",pred_conf)
                        #conf_loss_obj = (pred_conf - ious[best_box_idx].detach())**2
                        #conf_loss_obj = (pred_conf - 1)**2
                        #conf_target = torch.clamp(self.iou(best_abs, true_abs_box).detach(), 0, 1)
                        #conf_target = 1.0 + self.iou(best_abs, true_abs_box).detach() * 0  # =1

                        conf_target = 1
                        print("conf_target",conf_target)
                        conf_loss_obj = (pred_conf - conf_target)**2
                        print("conf_loss_obj",conf_loss_obj)
                        # --- Class loss ---
                        pred_class_probs = torch.softmax(pred_classes, dim=-1)
                        print("true_classes",true_classes)
                        print("pred_class_probs",pred_class_probs)
                        class_loss = torch.sum((pred_class_probs - true_classes)**2)

                        #total_loss += lambda_coord * (xy_loss + wh_loss) + 10*conf_loss_obj + 1000*class_loss
                        #total_loss += lambda_coord * (xy_loss + 115*wh_loss) + 130*conf_loss_obj + 1300*class_loss
                        total_loss += lambda_coord * (xy_loss + wh_loss) + conf_loss_obj + class_loss
                    else:
                        # No-object cells: apply to all boxes
                        for b in range(self.B):
                            pred_conf = torch.sigmoid(pred_boxes[b][0])
                            #pred_conf = pred_boxes[b][0]
                            print("pred_conf no obj",pred_conf)
                            total_loss += lambda_noobj * (pred_conf - 0.0)**2


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
  
class ConvBlock(nn.Module):
    """A block of Conv2D -> BatchNorm -> ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class YOLOBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(YOLOBackbone, self).__init__()
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        # Use the feature extractor part only
        self.features = mobilenet.features

    def forward(self, x):
        x = self.features(x)  # Output shape ~ (batch, 1280, H/32, W/32)
        return x

class YOLOHead(nn.Module):
    def __init__(self, grid_size, num_classes, num_anchors):
        super(YOLOHead, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.detector = nn.Conv2d(1280, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        return self.detector(x).permute(0, 2, 3, 1).contiguous()
    
class YOLO(nn.Module):
    def __init__(self, grid_size=7, num_classes=20, num_anchors=3):
        super(YOLO, self).__init__()
        self.backbone = YOLOBackbone()
        self.head = YOLOHead(grid_size, num_classes, num_anchors)

    def forward(self, x):
        features = self.backbone(x)
        features = nn.AdaptiveAvgPool2d((7,7))(features)
        predictions = self.head(features)
        return predictions


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
