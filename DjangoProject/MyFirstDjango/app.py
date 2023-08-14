from fastapi import FastAPI, UploadFile, File
from typing import List
import torch
import cv2
import numpy as np
import random

# 导入你的自定义模块
from MyFirstDjango.models.experimental import attempt_load
from MyFirstDjango.utils.datasets import MyLoadImages
from MyFirstDjango.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, set_logging
from MyFirstDjango.utils.plots import plot_one_box
from MyFirstDjango.utils.torch_utils import select_device, load_classifier

app = FastAPI()

class simulation_opt:
    def __init__(self, weights='static/07211657_best.pt',
                 img_size=640, conf_thres=0.25,
                 iou_thres=0.45, device='', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok

class detectapi:
    def __init__(self, weights, img_size=640):
        self.opt = simulation_opt(weights=weights, img_size=img_size)
        weights, imgsz = self.opt.weights, self.opt.img_size

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDAd

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(
                self.device).eval()

        # read names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.names = ['Vial' if name == 'P1' else 'Ampoule' if name == 'P2' else name for name in self.names]
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, source):  # 使用时，调用这个函数
        if type(source) != list:
            raise TypeError('source must be a list which contain  pictures read by cv2')
        dataset = MyLoadImages(source, img_size=self.imgsz, stride=self.stride)

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                next(self.model.parameters())))  # run once
        # t0 = time.time()
        result = []
        '''
        for path, img, im0s, vid_cap in dataset:'''
        for img, im0s in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = self.model(img, augment=self.opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)
            # t2 = time_synchronized()

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
                # Print time (inference + NMS)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Process detections
            det = pred[0]  # 原来的情况是要保持图片，因此多了很多关于保持路径上的处理。另外，pred
            # 其实是个列表。元素个数为batch_size。由于对于我这个api，每次只处理一个图片，
            # 所以pred中只有一个元素，直接取出来就行，不用for循环。
            im0 = im0s.copy()  # 这是原图片，与被传进来的图片是同地址的，需要copy一个副本，否则，原来的图片会受到影响
            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            result_txt = []
            # 对于一张图片，可能有多个可被检测的目标。所以结果标签也可能有多个。
            # 每被检测出一个物体，result_txt的长度就加一。result_txt中的每个元素是个列表，记录着
            # 被检测物的类别引索，在图片上的位置，以及置信度
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results

                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (int(cls.item()), [int(_.item()) for _ in xyxy], conf.item())  # label format
                    result_txt.append(line)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
            result.append((im0, result_txt))  # 对于每张图片，返回画完框的图片，以及该图片的标签列表。
        return result, len(result_txt)
    
@app.post("/detect/")
async def detect_objects(files: List[UploadFile]):
    results = []

    for file in files:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        detector = detectapi(weights='static/07211657_best.pt', img_size=640)
        detection_results, _ = detector.detect([image])
        
        vial_count = 0
        ampoule_count = 0
        for _, result_txt in detection_results:
            for cls, _, _ in result_txt:
                if cls == 0:  # Vial
                    vial_count += 1
                elif cls == 1:  # Ampoule
                    ampoule_count += 1
        
        result = {
            "vial_count": vial_count,
            "ampoule_count": ampoule_count,
            "total_count": vial_count + ampoule_count
        }
        results.append(result)
    
    return results

# 原来的命令行参数解析改为 FastAPI 的方式
# 这里你可能需要根据需要调整参数的默认值或者将一些参数设置为可选
class FastAPIOptions:
    def __init__(self, weights='static/07211657_best.pt',
                 img_size=640, conf_thres=0.25,
                 iou_thres=0.45, device='', view_img=False,
                 classes=None, agnostic_nms=False,
                 augment=False, update=False, exist_ok=False):
        self.weights = weights
        self.source = None
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.view_img = view_img
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.update = update
        self.exist_ok = exist_ok

opt = FastAPIOptions()  # 创建 FastAPIOptions 对象，根据需要设置参数的值

if __name__ == '__main__':
    print("Starting FastAPI server...")

# uvicorn app:app --reload
# uvicorn main:app --reload








