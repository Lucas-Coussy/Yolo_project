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
        y_true = torch.zeros((self.S, self.S, 5 + self.C))

        for _, row in img_df.iterrows():
            cls_id = int(row['ID_class'])
            x, y, w, h = row[['x_center', 'y_center', 'width', 'height']] #normalized coordinates

            #get the num of the cell the center of the object is
            grid_x = int(x * self.S)
            grid_y = int(y * self.S)
            #make sure the object the num_cell is not out of bound
            grid_x = min(grid_x, self.S - 1)
            grid_y = min(grid_y, self.S - 1)

            # In yolo we localize the object with it's central cell
            y_true[grid_y, grid_x, 0] = 1.0

            # Get the position of the center inside the cell (we know it's in cell (p,k) but we want to know exactly where it is inside)
            x_cell = x * self.S - grid_x
            y_cell = y * self.S - grid_y
            y_true[grid_y, grid_x, 1:3] = torch.tensor([x_cell, y_cell])
            y_true[grid_y, grid_x, 3:5] = torch.tensor([w, h])
            
            # After the 4 first value, we place the class of our object in a one hot encoding fashion
            y_true[grid_y, grid_x, 5 + cls_id] = 1.0

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
    
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

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
        x = self.feature_extractor(x)   # ✅ fixed name
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + self.B * 5)
        return x

"""
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_obj=5, l_nobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_obj = l_obj
        self.l_nobj = l_nobj

    def bbox_coordinate_format(self,bbox):
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


    def iou(self,bbox_predicted,bbox_actual):
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
        
    def yolo_loss(self, pred, target):
        tot_loss = 0
        for img_ind in range(len(pred)):
            for i in range(self.S):
                for j in range(self.S):
                    boxes = []
                    for bbox in range(self.B):
                        boxes.append(pred[img_ind,i,j,bbox*5:(bbox+1)*5])

                    true_box = target[img_ind,i,j,0:5]
                    best_box_ind = np.argmax([self.iou(box,true_box) for box in boxes])
                    best_box = boxes[best_box_ind]

                    if true_box[0]==1:
                        pred_conf = torch.sigmoid(pred[img_ind,i,j,self.B*5:])
                        xy_loss = torch.sum((best_box[1:3] - true_box[1:3]) ** 2)
                        wh_loss = torch.sum((torch.sqrt(torch.abs(best_box[3:5] + 1e-6)) - torch.sqrt(torch.abs(true_box[3:5] + 1e-6))) ** 2)
                        conf_loss_obj = torch.sum((pred_conf - target[img_ind,i,j,5:]) ** 2)
                        class_loss = torch.sum((best_box[0] - true_box[0]) ** 2)
                        tot_loss += self.l_obj * (xy_loss+wh_loss) + class_loss + conf_loss_obj
                    else: 
                        class_loss = torch.sum((best_box[0] - true_box[0]) ** 2)
                        tot_loss += self.l_nobj * class_loss 

        return tot_loss
"""

"""
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_obj=5, l_nobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_obj = l_obj
        self.l_nobj = l_nobj

    def bbox_coordinate_format(self,bbox):
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


    def iou(self, bbox_predicted, bbox_actual):
        # bbox_predicted, bbox_actual: tensors of shape [5] -> (C, x, y, w, h)
        # C: confidence, x,y,w,h normalized between 0-1
        
        _, x1, y1, w1, h1 = bbox_predicted
        _, x2, y2, w2, h2 = bbox_actual
        
        # Convert to corner coordinates
        x1_min = x1 - w1/2
        y1_min = y1 - h1/2
        x1_max = x1 + w1/2
        y1_max = y1 + h1/2
        
        x2_min = x2 - w2/2
        y2_min = y2 - h2/2
        x2_max = x2 + w2/2
        y2_max = y2 + h2/2

        # Intersection
        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)
        
        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter_area + 1e-6
        
        return inter_area / union

        
    def yolo_loss(self, pred, target):
        tot_loss = 0
        for img_ind in range(len(pred)):
            for i in range(self.S):
                for j in range(self.S):
                    boxes = []
                    for bbox in range(self.B):
                        boxes.append(pred[img_ind,i,j,bbox*5:(bbox+1)*5])

                    true_box = target[img_ind,i,j,0:5]
                    ious = torch.stack([self.iou(box, true_box) for box in boxes])  # tensor on GPU
                    best_box_ind = torch.argmax(ious).item()  # convert to Python int for indexing
                    best_box = boxes[best_box_ind]

                    if true_box[0]==1:
                        pred_conf = torch.sigmoid(pred[img_ind,i,j,self.B*5:])
                        xy_loss = torch.sum((best_box[1:3] - true_box[1:3]) ** 2)
                        wh_loss = torch.sum((torch.sqrt(torch.abs(best_box[3:5] + 1e-6)) - torch.sqrt(torch.abs(true_box[3:5] + 1e-6))) ** 2)
                        conf_loss_obj = torch.sum((pred_conf - target[img_ind,i,j,5:]) ** 2)
                        class_loss = torch.sum((best_box[0] - true_box[0]) ** 2)
                        tot_loss += self.l_obj * (xy_loss+wh_loss) + class_loss + conf_loss_obj
                    else: 
                        class_loss = torch.sum((best_box[0] - true_box[0]) ** 2)
                        tot_loss += self.l_nobj * class_loss 

        return tot_loss
"""

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_obj=5, l_nobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_obj = l_obj
        self.l_nobj = l_nobj

    def bbox_coordinate_format(self,bbox):
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


    def iou(self, bbox_predicted, bbox_actual):
        """
        bbox_predicted, bbox_actual: tensors of shape [5] -> (conf, x, y, w, h)
        All coordinates are expected in normalized [0,1] image space.
        """

        # Ensure numeric stability
        eps = 1e-6

        # Apply sigmoid to x, y to constrain between 0–1
        _, x1, y1, w1, h1 = bbox_predicted
        _, x2, y2, w2, h2 = bbox_actual

        x1 = torch.sigmoid(x1)
        y1 = torch.sigmoid(y1)
        w1 = torch.abs(w1)
        h1 = torch.abs(h1)

        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)
        w2 = torch.clamp(w2, 0, 1)
        h2 = torch.clamp(h2, 0, 1)

        # Convert to corners
        x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
        y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
        x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
        y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2

        # Intersection
        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - inter_area + eps

        return inter_area / union

        
    def yolo_loss(self, pred, target):
        """
        pred: (N, S, S, B*5 + C)
        target: (N, S, S, 5 + C)
        """
        lambda_coord = self.l_obj
        lambda_noobj = self.l_nobj
        batch_size = pred.size(0)

        total_loss = 0.0

        for n in range(batch_size):
            for i in range(self.S):
                for j in range(self.S):

                    # === Split predictions ===
                    # Two predicted boxes per cell
                    pred_boxes = [pred[n, i, j, b*5:(b+1)*5] for b in range(self.B)]
                    pred_classes = pred[n, i, j, self.B*5:]  # class predictions

                    # === Ground truth ===
                    true_box = target[n, i, j, 0:5]
                    true_classes = target[n, i, j, 5:]

                    # if there’s an object in this cell
                    if true_box[0] == 1:

                        # --- Find the best matching predicted box ---
                        ious = torch.stack([self.iou(pred_box, true_box) for pred_box in pred_boxes])
                        best_box_idx = torch.argmax(ious)
                        best_box = pred_boxes[best_box_idx]

                        # --- Coordinate loss ---
                        # Apply sigmoid to x, y to keep them in [0,1]
                        pred_xy = torch.sigmoid(best_box[1:3])
                        true_xy = true_box[1:3]
                        xy_loss = torch.sum((pred_xy - true_xy) ** 2)

                        # Width & height: predict square roots (YOLO trick)
                        pred_wh = best_box[3:5]
                        true_wh = true_box[3:5]
                        wh_loss = torch.sum((torch.sqrt(torch.abs(pred_wh + 1e-6)) -
                                            torch.sqrt(torch.abs(true_wh + 1e-6))) ** 2)

                        # --- Object confidence loss ---
                        pred_conf = torch.sigmoid(best_box[0])
                        iou_score = ious[best_box_idx].detach()  # used as target for confidence
                        conf_loss_obj = (pred_conf - iou_score) ** 2

                        # --- Class loss ---
                        pred_class_probs = torch.softmax(pred_classes, dim=-1)
                        class_loss = torch.sum((pred_class_probs - true_classes) ** 2)

                        # --- Combine ---
                        total_loss += (
                            lambda_coord * (xy_loss + wh_loss) +
                            conf_loss_obj +
                            class_loss
                        )

                    else:
                        # --- No-object cells ---
                        for b in range(self.B):
                            pred_conf = torch.sigmoid(pred[n, i, j, b*5])
                            conf_loss_noobj = (pred_conf - 0.0) ** 2
                            total_loss += lambda_noobj * conf_loss_noobj

        total_loss = total_loss / batch_size
        return total_loss

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
                    #x = 1 / (1 + np.exp(-x))  # sigmoid
                    #y = 1 / (1 + np.exp(-y))
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