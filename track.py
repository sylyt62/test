import sys, subprocess
import myconf

sys.path.insert(0, 'yolov5')

from yolov5.utils.google_utils import attempt_download
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from math import atan, pi

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


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


def draw_boxes(img, bbox, identities=None, names=None, offset=(0, 0)):
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
        if key == 't2d':
            term = 'Top to Down'
        elif key == 'd2t':
            term = 'Down to Top'
        elif key == 'l2r':
            term = 'Left to Right'
        elif key == 'r2l':
            term = 'Right to Left'
        elif key == 'tl2dr':
            term = 'TopLeft to DownRight'
        elif key == 'dr2tl':
            term = 'DownRight to TopLeft'
        elif key == 'tr2dl':
            term = 'TopRight to DownLeft'
        elif key == 'dl2tr':
            term = 'DownLeft to TopRight'
        else:
            term = 'Still'
        stat = '{}: {}'.format(term, stats[key])
        cv2.putText(img, stat, (50, 20 * (i + 1)), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)
    return img


def analyse_detections(deepsort, step_back=3, pixel_thred=2):
    stats = {
        'still': [],
        't2d': [], 'd2t': [],
        'l2r': [], 'r2l': [],
        'tl2dr': [], 'dr2tl': [],
        'tr2dl': [], 'dl2tr': []
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
                        stats['d2t'].append(track.track_id)
                    elif 1 / 8 * pi < angle <= 3 / 8 * pi:
                        stats['dl2tr'].append(track.track_id)
                    elif 3 / 8 * pi < angle <= 5 / 8 * pi:
                        stats['l2r'].append(track.track_id)
                    elif 5 / 8 * pi < angle <= 7 / 8 * pi:
                        stats['tl2dr'].append(track.track_id)
                    elif 7 / 8 * pi < angle <= 9 / 8 * pi:
                        stats['t2d'].append(track.track_id)
                    elif 9 / 8 * pi < angle <= 11 / 8 * pi:
                        stats['tr2dl'].append(track.track_id)
                    elif 11 / 8 * pi < angle <= 13 / 8 * pi:
                        stats['r2l'].append(track.track_id)
                    elif 13 / 8 * pi < angle <= 15 / 8 * pi:
                        stats['dr2tl'].append(track.track_id)
    return stats


def detect(opt):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate = \
        opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.evaluate
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    attempt_download(deep_sort_weights, repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #    if os.path.exists(out):
    #        pass
    #        shutil.rmtree(out)  # delete output folder
    #    os.makedirs(out)  # make new output folder
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model = model.half()  # to FP16

    # Get category names
    names = model.module.names if hasattr(model, 'module') else model.names
    classes = [i for i in range(len(names)) if names[i] in opt.classes]

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = os.path.basename(source).split('.')[0]
    txt_path = os.path.join(str(Path(out)), txt_file_name + '.txt')

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

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

                xywhs = torch.Tensor(xywh_bboxs)
                confss = torch.Tensor(confs)
                cats = torch.Tensor(cats)

                # pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0, cats)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)
                    # to MOT format
                    tlwh_bboxs = xyxy_to_tlwh(bbox_xyxy)

                    # Write MOT compliant results to file
                    if save_txt:
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            bbox_top = tlwh_bbox[0]
                            bbox_left = tlwh_bbox[1]
                            bbox_w = tlwh_bbox[2]
                            bbox_h = tlwh_bbox[3]
                            identity = output[-1]
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_top,
                                                            bbox_left, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

                # draw stats
                if myconf.DRAW_STATS:
                    stats = analyse_detections(deepsort)
                    draw_stats(im0, stats)

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

            # Stream results as RTSP
            if opt.stream_rtsp:
                print(frame_idx)
                if frame_idx == 0:
                    rtsp_server = 'rtsp://localhost:8554/test'
                    vide_wh_str = '{}x{}'.format(im0.shape[1], im0.shape[0])
                    fps = 30
                    command = ['ffmpeg',
                               '-re',
                               '-s', vide_wh_str,
                               '-r', str(fps),  # rtsp fps (from input server)
                               '-i', '-',

                               # You can change ffmpeg parameter after this item.
                               '-pix_fmt', 'yuv420p',
                               '-r', '30',  # output fps
                               '-g', '50',
                               '-c:v', 'libx264',
                               '-b:v', '2M',
                               '-bufsize', '64M',
                               '-maxrate', "4M",
                               '-preset', 'veryfast',
                               '-rtsp_transport', 'tcp',
                               '-segment_times', '5',
                               '-f', 'rtsp',
                               rtsp_server]
                    process = subprocess.Popen(command, stdin=subprocess.PIPE)
                    print(' '.join(command))
                else:
                    _, frame = cv2.imencode('.png', im0)
                    process.stdin.write(frame.tobytes())

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, default='yolov5/weights/yolov5m6.pt', help='model.pt path')
    parser.add_argument('--deep_sort_weights', type=str, default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    parser.add_argument('--stream-rtsp', action='store_true', help='push rtsp stream')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)


