import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import Sequential, Conv2d, LeakyReLU, MaxPool2d, Flatten, Linear, Dropout
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------
# 1️⃣ Dataset Class
# --------------------------
class YoloDataset(Dataset):
    def __init__(self, df, img_size=448):
        """
        df: pandas DataFrame with columns ['file_name','encoded_grid']
        encoded_grid: tensor of shape [S, S, 5+C]
        """
        self.df = df
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['file_name']
        y_true = row['encoded_grid']

        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img[..., ::-1] / 255.0  # BGR -> RGB, normalize
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2,0,1)  # (C,H,W)

        return img_tensor, y_true.float()


# --------------------------
# 2️⃣ YOLOv1 Model
# --------------------------
class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.feature_extractor = Sequential(
            Conv2d(3, 64, 7, 2, 3), LeakyReLU(0.1), MaxPool2d(2,2),
            Conv2d(64,192,3,1,1), LeakyReLU(0.1), MaxPool2d(2,2),
            Conv2d(192,128,1), LeakyReLU(0.1),
            Conv2d(128,256,3,1,1), LeakyReLU(0.1),
            Conv2d(256,256,1), LeakyReLU(0.1),
            Conv2d(256,512,3,1,1), LeakyReLU(0.1), MaxPool2d(2,2),
            # 4 repeated blocks
            Conv2d(512,256,1), LeakyReLU(0.1),
            Conv2d(256,512,3,1,1), LeakyReLU(0.1),
            Conv2d(512,256,1), LeakyReLU(0.1),
            Conv2d(256,512,3,1,1), LeakyReLU(0.1),
            Conv2d(512,256,1), LeakyReLU(0.1),
            Conv2d(256,512,3,1,1), LeakyReLU(0.1),
            Conv2d(512,256,1), LeakyReLU(0.1),
            Conv2d(256,512,3,1,1), LeakyReLU(0.1),
            Conv2d(512,512,1), LeakyReLU(0.1),
            Conv2d(512,1024,3,1,1), LeakyReLU(0.1),
            MaxPool2d(2,2),
            Conv2d(1024,512,1), LeakyReLU(0.1),
            Conv2d(512,1024,3,1,1), LeakyReLU(0.1),
            Conv2d(1024,512,1), LeakyReLU(0.1),
            Conv2d(512,1024,3,1,1), LeakyReLU(0.1),
            Conv2d(1024,1024,3,1,1), LeakyReLU(0.1),
            Conv2d(1024,1024,3,2,1), LeakyReLU(0.1),
            Conv2d(1024,1024,3,1,1), LeakyReLU(0.1),
            Conv2d(1024,1024,3,1,1), LeakyReLU(0.1),
        )

        self.fc = Sequential(
            Flatten(),
            Linear(7*7*1024, 4096), LeakyReLU(0.1), Dropout(0.5),
            Linear(4096, S*S*(C + B*5))
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.C + self.B*5)
        return x

# --------------------------
# 3️⃣ Vectorized YOLO Loss
# --------------------------
class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, l_obj=5, l_noobj=0.5):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.l_obj = l_obj
        self.l_noobj = l_noobj

    def iou_tensor(self, boxes1, boxes2):
        """boxes1, boxes2: (..., 4) -> x,y,w,h normalized"""
        x1, y1, w1, h1 = boxes1.unbind(-1)
        x2, y2, w2, h2 = boxes2.unbind(-1)

        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        inter_xmin = torch.max(x1_min, x2_min)
        inter_ymin = torch.max(y1_min, y2_min)
        inter_xmax = torch.min(x1_max, x2_max)
        inter_ymax = torch.min(y1_max, y2_max)

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        union_area = w1 * h1 + w2 * h2 - inter_area + 1e-6
        return inter_area / union_area

    def forward(self, pred, target):
        N = pred.shape[0]
        B, S, C = self.B, self.S, self.C

        # Separate predicted boxes
        pred_boxes = pred[..., :B*5].view(N, S, S, B, 5)
        pred_conf = torch.sigmoid(pred_boxes[..., 0])
        pred_xy = pred_boxes[..., 1:3]
        pred_wh = pred_boxes[..., 3:5]
        pred_classes = pred[..., B*5:]

        # Target
        target_box = target[..., :5]
        target_conf = target_box[..., 0]
        target_xy = target_box[..., 1:3]
        target_wh = target_box[..., 3:5]
        target_classes = target[..., 5:]

        # IOU per predicted box
        ious = torch.stack([self.iou_tensor(pred_xy[..., b, :], target_xy) for b in range(B)], dim=-1)
        best_box = torch.argmax(ious, dim=-1)  # (N,S,S)

        # Mask for selecting best box
        best_mask = F.one_hot(best_box, num_classes=B).bool()
        best_mask = best_mask.unsqueeze(-1)

        # Select best boxes
        best_pred_xy = torch.sum(pred_xy * best_mask, dim=3)
        best_pred_wh = torch.sum(pred_wh * best_mask, dim=3)
        best_pred_conf = torch.sum(pred_conf * best_mask.squeeze(-1), dim=3)

        # Object mask
        obj_mask = target_conf.unsqueeze(-1)
        noobj_mask = 1 - obj_mask

        # Losses
        xy_loss = torch.sum(obj_mask * (best_pred_xy - target_xy)**2)
        wh_loss = torch.sum(obj_mask * (torch.sqrt(best_pred_wh + 1e-6) - torch.sqrt(target_wh + 1e-6))**2)
        conf_loss_obj = torch.sum(obj_mask.squeeze(-1) * (best_pred_conf - ious.max(-1).values)**2)
        conf_loss_noobj = torch.sum(noobj_mask.squeeze(-1) * (pred_conf**2))
        class_loss = torch.sum(obj_mask * (pred_classes - target_classes)**2)

        total_loss = self.l_obj * (xy_loss + wh_loss) + conf_loss_obj + self.l_noobj*conf_loss_noobj + class_loss
        return total_loss

# --------------------------
# 4️⃣ Training Script
# --------------------------
def train_yolo(df_train, df_test, S=7, B=2, C=20, epochs=50, batch_size=8, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = YoloDataset(df_train)
    test_dataset = YoloDataset(df_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = YOLOv1(S=S, B=B, C=C).to(device)
    criterion = YoloLoss(S=S, B=B, C=C)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for imgs, labels in progress_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "yolov1_trained.pth")
    print("💾 Model saved as yolov1_trained.pth")
    return model, train_losses

os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project")
df_encoded = torch.load(r"data\encoded_picture.pt",weights_only=False)

df_encoded = df_encoded[:1000]
df_train, df_test = train_test_split(df_encoded,test_size=0.3)
print(df_train)

train_yolo(df_train,df_test)