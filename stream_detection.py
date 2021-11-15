# import cv2, json, sys, time, requests
from cv2 import VideoCapture, namedWindow, imshow, waitKey, WINDOW_FREERATIO
from json import loads
from sys import argv, exit
from time import localtime, sleep, strftime
from requests import post
from multiprocessing import Queue, Process, set_start_method
from numpy import random, copy
from infer_stream import infer, draw_boxes, draw_stats
import myconf
from redis import StrictRedis


def receive_from_redis(host, port, password=None, db=0, queue_name=myconf.REDIS_SERVER_QUEUE):
    try:
        client = StrictRedis(host=host, port=port, password=password, db=db, charset="utf-8", decode_responses=True)
        msg = client.lpop(queue_name)
        return msg
    except Exception as re:
        print("Redis connection exception : {}".format(re))
        return None


def xyxy2xywh(xyxy):
    xywh = []
    for box in xyxy:
        x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
        w, h = abs(box[2] - box[0]), abs(box[3] - box[1])
        xywh.append([x_center, y_center, w, h])
    return xywh


def rand_str(l):
    pool = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    s = ''
    for _ in range(l):
        s += pool[random.randint(len(pool))]
    return s


def queue_img_put(q_origin, rtsp_url):
    cap = VideoCapture(rtsp_url)
    end_cnt = 0

    while True:
        if end_cnt > myconf.MAX_END_COUNT:
            exit('Reached max end count [{}]'.format(end_cnt))
        is_opened, frame = cap.read()
        if is_opened:
            end_cnt = 0
            q_origin.put(frame)
            # print('Put new frame.')
            if q_origin.qsize() > 1:
                q_origin.get()
                # print('Clean old frame.')
            else:
                print('No frame been cached.')
                sleep(0.01)
        else:
            print('Camera closed. End count: {}'.format(end_cnt))
            end_cnt += 1
            sleep(1)
            cap.release()


def queue_img_get(q_results, rtsp_url, window_name=None):
    if window_name:
        namedWindow(window_name, flags=WINDOW_FREERATIO) if window_name else None

    url = myconf.JAVA_BACKEND_URL
    i = 0
    while True:
        if q_results.qsize() < 1:
            print('Result img queue is empty. Nothing to get.')
        (origin_img, label_ids, scores, boxes, infer_time_str, stats) = q_results.get()
        print('Get new frame. {}'.format(origin_img.shape))
        img = copy(origin_img)
        box_info = (boxes, scores, label_ids)

        '''report and draw'''
        current_time = strftime("%Y-%m-%d %X", localtime())
        if myconf.IF_PUSH_TO_BACKEND:
            body = myconf.BODY
            body['url'] = rtsp_url
            body['time'] = current_time
            for k in body['traffic'].keys():
                body['traffic'][k] = stats[k]
            body['videoHeight'], body['videoWidth'] = origin_img.shape[:2]
            body['bbox'] = xyxy2xywh(boxes)
            print('Pushing traffic volume to JAVA backend ...\n{}'.format(body))
            body = {'body': str(body)}
            post(url, str(body), verify=False)
        if window_name:
            img = draw_boxes(img, boxes, label_ids)
            img = draw_stats(img, stats)
            imshow(window_name, img)
            waitKey(1)


def run_cmd(url_received):
    set_start_method(method='spawn')

    origin_img_q = Queue(maxsize=2)
    result_img_q = Queue(maxsize=4)

    processes = [
        Process(target=queue_img_put, args=(origin_img_q, url_received,)),
        Process(target=infer, args=(origin_img_q, result_img_q, )),
        Process(target=queue_img_get, args=(result_img_q, url_received, myconf.WIN_NAME,))
    ]

    [setattr(process, "daemon", True) for process in processes]
    [process.start() for process in processes]
    [process.join() for process in processes]


def run_redis():
    set_start_method(method='spawn')
    detection_process_dict = {}
    i = 0
    while True:
        msg_from_redis = receive_from_redis(myconf.REDIS_HOST, myconf.REDIS_PORT, myconf.REDIS_PASSWORD,
                                            myconf.REDIS_SERVER_QUEUE)

        if msg_from_redis is None:
            print('[Main Info] Main process waiting count {} ...'.format(i))
            sleep(1)
            i += 1
        else:
            i = 0
            print('[Main Info] Received message from redis:\n{}'.format(msg_from_redis))
            msg_from_redis = loads(msg_from_redis)
            url, operation = msg_from_redis['url'], msg_from_redis['detection']
            if operation == 1:
                if url not in detection_process_dict.keys():
                    print('[Child Info] Starting new detction process for URL: {}'.format(url))
                    origin_img_q = Queue(maxsize=2)
                    result_img_q = Queue(maxsize=4)
                    processes = [
                        Process(target=queue_img_put, args=(origin_img_q, url,)),
                        Process(target=infer, args=(origin_img_q, result_img_q,)),
                        Process(target=queue_img_get, args=(result_img_q, url, myconf.WIN_NAME, myconf.PUSH_RTSP,))
                    ]
                    [setattr(process, "daemon", True) for process in processes]
                    [process.start() for process in processes]
                    # [process.join() for process in processes]
                    detection_process_dict.update({url: processes})
                    print('{}'.format(list(detection_process_dict.keys())))
                else:
                    print('[Child Warn] URL: {} is already been detecting.'.format(url))
            elif operation == 0:
                if url not in detection_process_dict.keys():
                    print('[Child Warn] URL: {} not in runnig list. Cannot be closed.'.format(url))
                else:
                    print('[Child Info] Closing URL: {}'.format(url))
                    for p in detection_process_dict[url]:
                        p.terminate()
                        p.join()
                    del detection_process_dict[url]


if __name__ == '__main__':
    try:
        url_received = argv[1]
        if url_received.startswith('rtsp://'):
            print('An RTSP URL is received. - {}'.format(url_received))
        else:
            print('Received an URL but not RTSP URL - {}'.format(url_received))
        method = 'cmd'
    except:
        print('No URL passed. Using redis mode')
        method = 'redis'

    if method == 'cmd':
        run_cmd(url_received)
    elif method == 'redis':
        run_redis()
    else:
        print('Mode not supported.')
        exit(0)
