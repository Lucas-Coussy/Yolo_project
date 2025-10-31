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

os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project\classes")
from classes.CNN_pytorch import YoloDataset, YoloLoss, YOLOv1, CNN_model, YOLO_visual #import my class CNN

os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project\data\Pascal_VOC\train")

path = r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project\data\Pascal_VOC\train"

xml_list = []
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    xml_list.append(fullname)

my_CNN = CNN_model()
my_CNN.get_annotation(xml_list)
my_CNN.encode_pictures()

os.chdir(r"C:\Users\Lucas\Desktop\vacantion classes\Yolo project")

my_CNN.df.to_json(r"data\annot_df.json",orient="records")
torch.save(my_CNN.encoded_picture_annot, r"data\encoded_picture.pt")

df_encoded = torch.load(r"data\encoded_picture.pt",weights_only=False)

X_train, X_test, y_train, y_test = train_test_split(df_encoded['file_name'],df_encoded['encoded_grid'],test_size=0.4)

X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

X_train = X_train[:4000]
y_train = y_train[:4000]

train_dataset = YoloDataset(X_train, y_train)
test_dataset = YoloDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# -------------------------------
# 2️⃣ Initialize Model, Loss, Optimizer
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
S, B, C = 7, 2, 20

model = YOLOv1(S=S, B=B, C=C).to(device)
criterion = YoloLoss(S=S, B=B, C=C)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# -------------------------------
# 3️⃣ Training Loop with Progress Bar
# -------------------------------
num_epochs = 60
train_losses = []

print(f"Starting training on {device} for {num_epochs} epochs...\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

    for imgs, labels in progress_bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion.yolo_loss(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "yolov1_trained.pth")
print("\n💾 Model saved as yolov1_trained.pth")

# -------------------------------
# 4️⃣ Plot Training Loss Curve
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title("YOLOv1 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()