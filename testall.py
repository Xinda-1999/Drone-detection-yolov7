import argparse
import json
import os
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_synchronized, TracedModel

from glob import glob


def test(data,
         weights=None,
         batch_size=50,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         augment=False,
         model=None,
         dataloader=None,
         save_hybrid=False,  # for hybrid auto-labelling
         compute_loss=None,
         half_precision=True,
         trace=False,
         v5_metric=False,
         output_path='',
         testset_path=''):
    # Initialize/load model and set device
    training = model is not None

    device = select_device(opt.device, batch_size=batch_size)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check img_size
    
    if trace:
        model = TracedModel(model, device, imgsz)

    # Half
    half = device.type != 'cpu' and half_precision  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    check_dataset(data)  # check

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    whatsInTestset = glob(testset_path+'/*')
    video_dirs = []
    for d in whatsInTestset:
        if not 'cache' in d:
            # 有时会生成cache文件，应避免干扰
            video_dirs.append(d)
    n_video = len(video_dirs)
    for video_i, video_dir in enumerate(video_dirs):
        print(video_i+1, 'in', n_video, ': 正在处理', video_dir.split('/')[-1])

        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        """ dataloader = create_dataloader(data[task], imgsz, batch_size, gs, opt, pad=0.5, rect=True, workers = 0,
                                        prefix=colorstr(f'{task}: '))[0] """
        
        dataloader = create_dataloader(video_dir, imgsz, batch_size, gs, opt, pad=0.5, rect=True, workers = 0)[0]

        if v5_metric:
            print("Testing with YOLOv5 AP metric...")
        
        seen = 0
        s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        loss = torch.zeros(3, device=device)

        text = []
        for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader)):
            img = img.to(device, non_blocking=True)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            nb, _, height, width = img.shape  # batch size, channels, height, width

            with torch.no_grad():
                # Run model
                out, train_out = model(img, augment=augment)  # inference and training outputs

                # Compute loss
                if compute_loss:
                    loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

                # Run NMS
                targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
                lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
                out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)

            # Statistics per image
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                path = Path(paths[si])
                seen += 1
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem

                if len(pred) == 0:
                    # 没有发现无人机
                    text.append((image_id, []))
                    continue

                # Predictions
                
                predn = pred.clone()
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred


                # 更新text
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                p,b = pred.tolist()[0], box.int().tolist()[0]   # 只检查confidence最高的一个候选，因为无人机只有一个
                
                if p[4]>0.6:
                    # 发现无人机
                    text.append((image_id, b))
                else:
                    # 没有发现无人机，或候选框confidence不足
                    text.append((image_id, []))  


        assert len(text) == seen   # 确保列表长度等于帧数

        # 保存text
        text = sorted(text, key=lambda x:x[0])     
        text = [x[1] for x in text]
        video_name = video_dir.split('\\')[-1]
        txt_path = os.path.join(output_path, video_name + '.txt')
        with open(txt_path, 'w') as f:
            txt = '{"res":' + str(text) + "}"
            f.write(txt)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp7/weights/best.pt', help='model.pt path(s)')#
    parser.add_argument('--data', type=str, default='data/VOC.yaml', help='*.data path')#
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')#
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')#
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')#
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')#
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')#
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')#
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')#
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    # JM定义的
    parser.add_argument('--output-path', type=str, default='./output_txt', help='存放所有输出text的目录')
    parser.add_argument('--testset-path', type=str, default='./Origin_dataset/test', help='存放所有输出text的目录')
    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    glob(opt.testset_path+'/*')

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.augment,
             trace=not opt.no_trace,
             v5_metric=opt.v5_metric,
             output_path = opt.output_path,
             testset_path = opt.testset_path
             )
