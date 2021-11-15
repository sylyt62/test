# SQL and Redis
MYSQL_HOST = '172.30.2.135'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'golocus'
MYSQL_DB = 'SDAT'
REDIS_HOST = 'localhost'
REDIS_PORT = '6379'  # aliyun
REDIS_SERVER_QUEUE = 'serverMsg'
REDIS_PASSWORD = None

PUSH_RTSP = True
RTSP_URL = 'rtsp://127.0.0.1:8554/'


MAX_END_COUNT = 10
JAVA_BACKEND_URL = ''
IF_PUSH_TO_BACKEND = True
WIN_NAME = None

USE_GPU = False

SOURCE = "E:\Workspace\yolov5_deepsort\Yolov5_DeepSort_Pytorch\\videos\\traffic_low_vision.mp4"
OUTPUT = "E:\Workspace\yolov5_deepsort\Yolov5_DeepSort_Pytorch\inference\output"

YOLO_WEIGHTS = "yolov5/weights/yolov5m6.pt"
DEEPSORT_WEIGHTS = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
DEEPSORT_CONFIG = "deep_sort_pytorch/configs/deep_sort.yaml"

NMS_THRED = 0.4
CONF_THRED = 0.5
IMG_SIZE = 640


CLASSES = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'train', 'ship', 'boat']

DRAW_STATS = False
BODY = {
    'url': '',
    'time': '',
    'traffic': {
        'still': 0,                 # 静止的
        'top2down': 0,              # 从上到下
        'topRight2downLeft': 0,     # 右上到左下
        'right2left': 0,            # 从右到左
        'downRight2topLeft': 0,     # 右下到左上
        'down2top': 0,              # 从下到上
        'downLeft2topRight': 0,     # 左下到右上
        'left2right': 0,            # 从左到右
        'topLeft2downRight': 0      # 左上到右下
    },
    'videoWidth': 0,
    'videoHeight': 0,
    'bbox': []     # [包围框中心x, 包围框中心y, 包围框宽度, 包围框高度]
}

