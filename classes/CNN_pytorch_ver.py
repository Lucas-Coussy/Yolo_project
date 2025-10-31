import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Linear, Dropout, Flatten


# ---------------------------
#  YOLOv1 Model (simplified)
# ---------------------------
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()

        self.feature_extractor = Sequential(
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

            # final YOLOv1 conv layers
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
            Conv2d(1024, 1024, kernel_size=3, padding=1),
            LeakyReLU(0.1),
        )

        # Fully connected layers for detection
        self.fc = Sequential(
            Flatten(),
            Linear(7 * 7 * 1024, 4096),
            LeakyReLU(0.1),
            Dropout(0.5),
            Linear(4096, S * S * (C + B * 5))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x


# ---------------------------
#  YOLOv1 Loss Function
# ---------------------------
class YoloLoss(nn.Module):
        def __init__(self, S=7, B=2, C=20, l_coord=5, l_noobj=0.5):
            super(YoloLoss, self).__init__()
            self.S = S
            self.B = B
            self.C = C
            self.l_coord = l_coord
            self.l_noobj = l_noobj

        def compute_iou(self, box1, box2):
            """
            box shape: (..., 4) => (x, y, w, h)
            assumes coords are relative to cell, normalized 0-1
            """
            # convert (x, y, w, h) to (x1, y1, x2, y2)
            b1_x1 = box1[..., 0] - box1[..., 2] / 2
            b1_y1 = box1[..., 1] - box1[..., 3] / 2
            b1_x2 = box1[..., 0] + box1[..., 2] / 2
            b1_y2 = box1[..., 1] + box1[..., 3] / 2

            b2_x1 = box2[..., 0] - box2[..., 2] / 2
            b2_y1 = box2[..., 1] - box2[..., 3] / 2
            b2_x2 = box2[..., 0] + box2[..., 2] / 2
            b2_y2 = box2[..., 1] + box2[..., 3] / 2

            # intersection
            inter_x1 = torch.max(b1_x1, b2_x1)
            inter_y1 = torch.max(b1_y1, b2_y1)
            inter_x2 = torch.min(b1_x2, b2_x2)
            inter_y2 = torch.min(b1_y2, b2_y2)

            inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
            b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
            union = b1_area + b2_area - inter_area

            iou = inter_area / (union + 1e-6)
            return iou

        def forward(self, pred, target):
            """
            pred, target: [batch, S, S, B*5 + C]
            """
            pred_boxes = pred[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
            pred_classes = pred[..., self.B*5:]
            true_boxes = target[..., :self.B*5].view(-1, self.S, self.S, self.B, 5)
            true_classes = target[..., self.B*5:]

            # IoU between each predicted box and true box
            ious = self.compute_iou(pred_boxes[..., 1:5], true_boxes[..., 1:5])

            # choose best box per cell
            best_box = ious.argmax(dim=-1).unsqueeze(-1)  # shape [B, S, S, 1]
            best_iou = ious.max(dim=-1, keepdim=True).values

            obj_mask = (true_boxes[..., 0:1] > 0).float()

            # confidence targets = IoU for responsible box, 0 otherwise
            conf_target = best_iou * obj_mask

            # predicted confidences
            pred_conf = torch.sigmoid(pred_boxes[..., 0:1])

            # coordinate and size losses (only for object cells)
            xy_loss = torch.sum(obj_mask * (pred_boxes[..., 1:3] - true_boxes[..., 1:3]) ** 2)
            wh_loss = torch.sum(obj_mask * (torch.sqrt(torch.abs(pred_boxes[..., 3:5] + 1e-6)) -
                                            torch.sqrt(torch.abs(true_boxes[..., 3:5] + 1e-6))) ** 2)

            # confidence loss
            conf_loss_obj = torch.sum(obj_mask * (pred_conf - conf_target) ** 2)
            conf_loss_noobj = torch.sum((1 - obj_mask) * (pred_conf - conf_target) ** 2)

            # class loss
            class_loss = torch.sum(obj_mask * (pred_classes - true_classes) ** 2)

            total = (
                self.l_coord * (xy_loss + wh_loss)
                + conf_loss_obj
                + self.l_noobj * conf_loss_noobj
                + class_loss
            )

            return total / pred.size(0)
        

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
    
    class IoU():
        def __init__(self):
            pass

        def iou(self,bbox_predicted,bbox_actual):

            """
            One thing to note here about the coordinates which even I got a bit confused a bout at the beginning is top left means (0,0)
            So as we move towards the right the x value increases(which is actually pretty normal) but the thing which i didnt realize initially
            is that as we move downwards value of y increases(which is the opposite of how we normally percieve y values)

            """
            x_min_predicted,y_min_predicted,x_max_predicted,y_max_predicted=self.bbox_coordinate_format(bbox_predicted)
            x_min_actual,y_min_actual,x_max_actual,y_max_actual=self.bbox_coordinate_format(bbox_actual)

            x_intersection_right=min(x_max_actual,x_max_predicted)
            x_intersection_left=max(x_min_actual,x_min_predicted)
            y_intersection_lower=min(y_max_actual,y_max_predicted)
            y_intersection_upper=max(y_min_actual,y_min_predicted)

            w_inter=max(0,x_intersection_right-x_intersection_left)
            h_inter=max(0,y_intersection_lower-y_intersection_upper)

            w_box1=max(0,x_max_predicted-x_min_predicted)
            h_box1=max(0,y_max_predicted-y_min_predicted)

            w_box2=max(0,x_max_actual-x_min_actual)
            h_box2=max(0,y_max_actual-y_min_actual)

            area_box1=w_box1*h_box1

            area_box2=w_box2*h_box2
            
            area_inter=w_inter*h_inter

            area_union=area_box1+area_box2-area_inter

            if area_union==0:
                return 0
            else:
                iou=area_inter/area_union
                return iou

        def bbox_coordinate_format(self,bbox):
            """
            So basically the bbox coordinate predicted by the model are bw 0 and 1

            eg x = 0.5 means  box is centered horizontally at the middle of the image 
            y = 0.5 means box is centered vertically at the middle
            w = 0.2 means the box takes up 20% of the image width
            h = 0.3 means the box takes up 30% of image height

            This fn maps these values to the actual bbox coordinates in the image
            """
            
            img_width, img_height = 448, 448

            _,x,y,w,h=bbox

            x_center = x * img_width
            y_center = y * img_height
            box_width = w * img_width
            box_height = h * img_height

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            return x1.item(),y1.item(),x2.item(),y2.item()
