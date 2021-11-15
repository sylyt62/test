import myconf
import sys
import random
sys.path.insert(0, 'yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import letterbox
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import numpy as np
import shutil
import time
from math import pi, atan
import cv2
from torch import Tensor, zeros, no_grad, from_numpy
import torch.backends.cudnn as cudnn


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def genColors(n):
    '''
    generate n colors by HEX CODE. Such as:
        Red  # FF0000         rgb(255, 0, 0)
        Maroon  # 800000         rgb(128, 0, 0)
        Yellow  # FFFF00         rgb(255, 255, 0)
        Olive  # 808000         rgb(128, 128, 0)
    '''
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(n)]
    return colors


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def draw_stats(img, stats):
    for i, key in enumerate(stats.keys()):
        if '2' in key:
            term = key.replace('2', ' to ')
        else:
            term = key
        stat = '{}: {}'.format(term, stats[key])
        cv2.putText(img, stat, (50, 20 * (i + 1)), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)
    return img


def analyse_detections(deepsort, step_back=3, pixel_thred=2):
    stats = {
        'still': [],
        'top2down': [], 'down2top': [],
        'right2left': [], 'left2right': [],
        'topRight2downLeft': [], 'downLeft2topRight': [],
        'topLeft2downRight': [], 'downRight2topLeft': []
    }

    for track in deepsort.tracker.tracks:
        if track.is_confirmed():
            route = track.history
            if len(route) >= step_back:
                dx, dy = route[-1][0] - route[-step_back][0], route[-1][1] - route[-step_back][1]
                if (dx ** 2 + dy ** 2) ** 0.5 < pixel_thred:
                    stats['still'].append(track.track_id)
                else:
                    angle = atan(abs(dy) / abs(dx))
                    if dx >= 0 and dy >= 0:
                        angle += pi / 2
                    elif dx > 0 and dy < 0:
                        angle = pi / 2 - angle
                    elif dx <= 0 and dy <= 0:
                        angle += 3 / 2 * pi
                    elif dx < 0 and dy > 0:
                        angle = 3 / 2 * pi - angle

                    if 0 <= angle <= pi / 8 or 15 / 8 * pi <= angle <= 2 * pi:
                        stats['down2top'].append(track.track_id)
                    elif 1 / 8 * pi < angle <= 3 / 8 * pi:
                        stats['downLeft2topRight'].append(track.track_id)
                    elif 3 / 8 * pi < angle <= 5 / 8 * pi:
                        stats['left2right'].append(track.track_id)
                    elif 5 / 8 * pi < angle <= 7 / 8 * pi:
                        stats['topLeft2downRight'].append(track.track_id)
                    elif 7 / 8 * pi < angle <= 9 / 8 * pi:
                        stats['top2down'].append(track.track_id)
                    elif 9 / 8 * pi < angle <= 11 / 8 * pi:
                        stats['topRight2downLeft'].append(track.track_id)
                    elif 11 / 8 * pi < angle <= 13 / 8 * pi:
                        stats['right2left'].append(track.track_id)
                    elif 13 / 8 * pi < angle <= 15 / 8 * pi:
                        stats['downRight2topLeft'].append(track.track_id)
    return stats


def infer(origin_img_q, result_img_q):
    out, source = myconf.OUTPUT, myconf.SOURCE
    yolo_weights, deep_sort_weights = myconf.YOLO_WEIGHTS, myconf.DEEPSORT_WEIGHTS
    show_vid, save_vid, save_txt, evaluate = True, True, False, False
    augment, agnostic_nms = False, False
    imgsz = myconf.IMG_SIZE
    classes = myconf.CLASSES
    # Initialize
    device = select_device('0') if myconf.USE_GPU else select_device('cpu')

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(myconf.DEEPSORT_CONFIG)
    # attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    classes = [i for i in range(len(names)) if names[i] in classes]

    # Run inference
    if device.type != 'cpu':
        model(zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    is_opened = True
    end_cnt = 0

    while is_opened:
        if origin_img_q.qsize() == 0:
            # print('origin_img_q qsize is 0. [{}]')
            end_cnt += 1
            continue

        # get frame from original stream queue (cv2.cap.read(): BGR)
        im0s = origin_img_q.get()

        # yolov5 preprocess method
        img = letterbox(im0s, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xnxn
        img = np.ascontiguousarray(img)

        img = from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, myconf.CONF_THRED, myconf.NMS_THRED, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per frame
            s, im0 = '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, names[int(c)])  # add to string

                xywh_bboxs = []
                confs = []
                cats = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    # to deep sort format
                    x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
                    xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                    xywh_bboxs.append(xywh_obj)
                    confs.append([conf.item()])
                    cats.append([cls.item()])

                xywhs = Tensor(xywh_bboxs)
                confss = Tensor(confs)
                cats = Tensor(cats)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, cats)

                if len(outputs) > 0:
                    # count traffic volume by direction
                    stats = analyse_detections(deepsort)

                    infer_time_str = 'Infer Time: {}'.format(round(t2 - t1, 4))
                    result_img_q.put((im0s, cats.transpose(1, 0).tolist()[0], confss.transpose(1, 0).tolist()[0],
                                      outputs[:, :4], infer_time_str, stats))

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5m6.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with no_grad():
        infer(args)

