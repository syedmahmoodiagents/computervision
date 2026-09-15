import torch
from torchvision import models
import streamlit as st
import cv2
import numpy as np
from mtcnn import MTCNN

m = models.vgg16(weights=None)
m.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2, bias=True)

m.load_state_dict(torch.load('vgg_weights.pth', weights_only=True))
m.eval()

for param in m.parameters():
    param.requires_grad = False

for param in m.classifier[6].parameters():
    param.requires_grad = True


#################################

def crop_faces(image_path, output_prefix="face"):

    detector = MTCNN()
    image = cv2.imread(image_path)

    if image is not None:

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image_rgb)

        x, y, width, height = detections[0]['box']
        x, y = max(0, x), max(0, y)
        cropped_face = image[y:y+height, x:x+width]
        return cropped_face


imgfile = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if imgfile is not None:
    file_bytes = imgfile.read()
    npimg = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    # image_rgb = cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(npimg)

    x, y, width, height = detections[0]['box']
    x, y = max(0, x), max(0, y)
    cropped_face = npimg[y:y+height, x:x+width]

    # st.image(cropped_face, channels="RGB")
    
    resized = cv2.resize(cropped_face, (224, 224))
    imag_f = resized.reshape(1, 3, 224, 224)
    tor_image = torch.FloatTensor(imag_f)

    m.eval()
    with torch.no_grad():
        Yp = m(tor_image)
        sx = torch.softmax(Yp, dim=1)
        predicted_img = torch.argmax(sx, dim=1)
        
        if predicted_img.item() == 0:
            st.title("Predicted class: Tom")
        else:
            st.title("Predicted class: Leo")
        