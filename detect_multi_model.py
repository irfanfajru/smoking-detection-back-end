import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

def detect_multi_model(inputImage, threshold=0.25, saveDir="storage/result", weights=["weights/yolov7.pt","weights/best_cigarette.pt"], imageSize=640):
    # Load YOLOv7 models
    model1_weights = weights[0]
    model2_weights = weights[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = attempt_load(model1_weights, map_location=device)
    model1.eval()

    model2 = attempt_load(model2_weights, map_location=device)
    model2.eval()

    # Load and preprocess the image
    image_path = f"storage/upload/{inputImage}"
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = 640

    # Resize the image to the model's input size
    img_resized = cv2.resize(img, (img_size, img_size))
    img_tensor = torch.from_numpy(img_resized).to(device).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Perform object detection with the first model
    with torch.no_grad():
        pred1 = model1(img_tensor)[0]
    detections1 = non_max_suppression(pred1, 0.4, 0.5)

    # Process detections from the first model
    for det in detections1[0]:
        det = scale_coords(img_tensor.shape[2:], det[None, :], img.shape[:2]).round()
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = map(int, xyxy)
            
            # Crop the bounding box region from the original image
            cropped_img = imcrop(img,map(int,xyxy))
            
            # Preprocess the cropped image for the second model
            cropped_img_resized = cv2.resize(cropped_img, (img_size, img_size))
            cropped_img_tensor = torch.from_numpy(cropped_img_resized).to(device).permute(2, 0, 1).float() / 255.0
            cropped_img_tensor = cropped_img_tensor.unsqueeze(0)
            
            # Perform object detection with the second model
            with torch.no_grad():
                pred2 = model2(cropped_img_tensor)[0]
            detections2 = non_max_suppression(pred2, 0.4, 0.5)
            
            # Draw bounding boxes on the cropped image
            for det2 in detections2[0]:
                det2 = scale_coords(cropped_img_tensor.shape[2:], det2[None, :], cropped_img.shape[:2]).round()
                for *xyxy2, conf2, cls2 in reversed(det2):
                    x1_2, y1_2, x2_2, y2_2 = map(int, xyxy2)
                    x1_abs = x1 + x1_2
                    y1_abs = y1 + y1_2
                    x2_abs = x1 + x2_2
                    y2_abs = y1 + y2_2
                    label = f'{model2.names[int(cls2)]} {conf2:.2f}'
                    plot_one_box([x1_abs, y1_abs, x2_abs, y2_abs], img, label=label, color=[0,0,244])

    # Display or save the final image with bounding boxes
    cv2.imwrite(f"{saveDir}/{inputImage}", img)


def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    cropedImg = img[y1:y2, x1:x2, :]
    return cropedImg

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2