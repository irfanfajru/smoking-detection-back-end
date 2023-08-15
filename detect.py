import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import uuid
# second model
from detect_cigar import detect_cigar

def detect(inputImage, classes=None, threshold=0.25, saveDir="storage/result", weights=["weights/yolov7.pt"], imageSize=640, save_img=True):
    source, weights, view_img, save_txt, imgsz, trace = inputImage, weights, False, False, imageSize, False
    # Directories
    save_dir = Path(saveDir)
    # Initialize
    set_logging()
    device = select_device("")
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights[0], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
            next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=False)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, threshold, threshold, classes=classes, agnostic=False)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                detailDetection = []
                for rider in find_riders(reversed(det)):
                    # detect cigar from second model
                    cropImagePath,cropImage = imcrop(im0,map(int, rider["personBox"]))
                    result = detect_cigar(cropImagePath,weights=[weights[1]],imageSize=640)
                    tempDetail = []
                    if result["merokok"]:
                        # write bbox for person and detect cigar
                        for resBox in result['boxes']:
                            tempDetail.append({
                                "cls":resBox['cls'],
                                'conf':f"{resBox['conf']:.2f}"
                            })
                            plot_one_box(resBox['box'], cropImage, label=f'{resBox["cls"]} {resBox["conf"]:.2f}',
                                            color=[0,0,220], line_thickness=1)
                        
                        # combine bbox from croped img to im0
                        im0 = combine_img(im0,cropImage,map(int,rider["personBox"]))
                        # create bbox for person who smoking
                        label = f'Merokok'
                        plot_one_box(rider["combinedBox"], im0, label=label,
                                    color=[0,0,220], line_thickness=2)
                    else:
                        label = "Tidak Merokok"
                        plot_one_box(rider["combinedBox"], im0, label=label,
                                    color=[0,100,0], line_thickness=2)

                    # add detail detection
                    detailDetection.append({
                        "label":label,
                        "conf":f"{result['score']:.2f}",
                        "detail":tempDetail,
                    })

                    # delete temp file
                    os.remove(cropImagePath)

            # Print time (inference + NMS)
            print(
                f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
                print(
                    f" The image with the result is saved in: {save_path}")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    # print(detailDetection)
    return detailDetection

def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    cropedImg = img[y1:y2, x1:x2, :]
    # temp crop image
    path = f"storage/temp/{uuid.uuid4()}.jpg"
    cv2.imwrite(path,cropedImg)
    return path,cropedImg

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

def combine_img(img,cropedImg,bbox):
    x1,y1,x2,y2 = bbox
    img[y1:y2, x1:x2, :] = cropedImg
    return img

# iou func
def bbox_iou(box1, box2, x1y1x2y2=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is 4

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    return iou.item(),merge_boxes(box1,box2)

def merge_boxes(box1, box2):
    return [min(box1[0], box2[0]), 
        min(box1[1], box2[1]), 
        max(box1[2], box2[2]),
        max(box1[3], box2[3])]

def check_overlap(person_bbox, motorcycle_bbox):
    x_min_person, y_min_person, x_max_person, y_max_person = person_bbox
    x_min_motorcycle, y_min_motorcycle, x_max_motorcycle, y_max_motorcycle = motorcycle_bbox
    
    y_bottom_person = y_max_person
    
    if y_min_motorcycle <= y_bottom_person <= y_max_motorcycle:
        return True
    else:
        return False

def find_riders(det):
    riders = []
    combined = set()
    det1 = 0
    for *xyxy, _, cls in reversed(det):
        det2 = 0
        for *xyxy2, _, cls2 in reversed(det):
            if (int(cls) == 0 and int(cls2) == 3) and (det1 not in combined and det2 not in combined):
                iou,combinedBox = bbox_iou(xyxy,xyxy2)
                if (iou >= 0.5 or iou <= 0.7) and check_overlap(xyxy,xyxy2):
                    riders.append({
                        "combinedBox":combinedBox,
                        "personBox":xyxy,
                        "motorcycleBox":xyxy2,
                    })
                    combined.add(det1)
                    combined.add(det2)
            det2+=1
        det1+=1
    print(combined)
    return riders



