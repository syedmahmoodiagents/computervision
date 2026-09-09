import cv2
import torch
import torch.nn as nn

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.maxm = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(256, 128)
    self.linear2 = nn.Linear(128, 3)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.maxm(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.maxm(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    
    return x

model = MyModel()
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))

image = cv2.imread("zero.jpeg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (8, 8))
imag_f = resized.reshape(1, 8, 8)
req_image = 255 - imag_f
row_image = req_image.reshape(1, 1, 8, 8)  # Add batch and channel dimensions

tor_image = torch.FloatTensor(row_image)

model.eval()
with torch.no_grad():
    Yp = model(tor_image)
    print(Yp.shape)
    sx = torch.softmax(Yp, dim=1)
    predicted_img = torch.argmax(sx, dim=1)
    print("Predicted class:", predicted_img.item())

