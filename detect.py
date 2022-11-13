import argparse
import os
import platform
from posixpath import basename
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import sys

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_labels, rbox_to_poly,
    xyxy2xywh, plot_one_rotated_box, strip_optimizer, set_logging, rotate_non_max_suppression)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.evaluation_utils import rbox2txt
import pprint


def poly_to_obj_info(poly, plot_robx, score):
    if isinstance(plot_robx, torch.Tensor):
        plot_robx = plot_robx.cpu().float().numpy()
    obj_info = {}
    x0 = poly[0][0]
    y0 = poly[0][1]
    x1 = poly[0][0]
    y1 = poly[0][1]
    for i in range(0,len(poly)):
        poly[i][0] = round(poly[i][0], 3)
        poly[i][1] = round(poly[i][1], 3)
        if poly[i][0] >= x1:
            x1 = poly[i][0]
        elif poly[i][0] <= x0:
            x0 = poly[i][0]
        if poly[i][1] >= y1:
            y1 = poly[i][1]
        elif poly[i][1] <= y0:
            y0 = poly[i][1]
    obj_info["aabb"] = [[x0, y0], [x1, y1]]
    obj_info["angle"] = round(float(plot_robx[4]), 3)
    obj_info["center"] = [float(plot_robx[0]), float(plot_robx[1])]
    obj_info["width"] = float(plot_robx[2])
    obj_info["length"] = float(plot_robx[3])
    obj_info["polygon"] = poly.tolist()
    obj_info["score"] = score
    obj_info["label"] = 1.0
    return obj_info

def detect(opt, model=None, progress_callback=None, save_img=False):
    '''
    input: save_img_flag
    output(result):
    '''
    # 获取输出文件夹，输入路径，权重，参数等参数
    out, source, weights, view_img, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    if not os.path.exists(weights):
        print("no weight:", weights)
    is_video = False
    if not os.path.isfile(source):
        opt.save_json = False
    if opt.save_json:
        dir, video_name = os.path.split(source)
        file_name, file_type = os.path.splitext(video_name)
        video_types = ['.mp4', '.avi', '.mpeg', '.flv', '.rmvb', '.mov', '.3gp', '.mpg', '.wmv']
        if  file_type.lower() in video_types:
            opt.save_json = True
            if opt.obj_path == "":
                opt.obj_path = os.path.join(dir, file_name + '/' + file_name + '_objs')
            if not os.path.exists(opt.obj_path):
                os.mkdir(opt.obj_path)
            is_video = True
        elif os.path.isdir(source):
            if not os.path.exists(opt.obj_path):
                os.mkdir(opt.obj_path)

    # 移除之前的输出文件夹,并新建输出文件夹
    # if os.path.exists(out):
    #     shutil.rmtree(out)  # delete output folder
    # os.makedirs(out)  # make new output folder
    if not os.path.exists(out):
        os.mkdir(out)
    # Initialize
    set_logging()
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:
        device = select_device(opt.device)
        # 加载Float32模型，确保用户设定的输入图片分辨率能整除最大步长s=32(如不能则调整为能整除并返回)
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    # 如果设备为gpu，使用Float16
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 设置Float16
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    # 通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类别名字    names = ['person', 'bicycle', 'car',...,'toothbrush']
    names = model.module.names if hasattr(model, 'module') else model.names
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    # 进行一次前向推理,测试程序是否正常  向量维度（1，3，imgsz，imgsz）
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    """
        path 图片/视频路径  'E:\...\bus.jpg'
        img 进行resize+pad之后的图片   1*3*re_size1*resize2的张量 (3,img_height,img_weight)
        img0 原size图片   (img_height,img_weight,3)          
        cap 当读取图片时为None，读取视频时为视频源   
    """
    frame_id = 0
    obj_id = 0
    obj_detect_results = {}
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        # 图片也设置为Float16
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if img.ndimension() == 3:
            # (in_channels,size1,size2) to (1,in_channels,img_height,img_weight)
            img = img.unsqueeze(0)  # 在[0]维增加一个维度

        # Inference
        t1 = time_synchronized()
        """
        model:
        input: in_tensor (batch_size, 3, img_height, img_weight)
        output: 推理时返回 [z,x]
        z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), nc)
        x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes]) 
        '''
               
        前向传播 返回pred[0]的shape是(1, num_boxes, nc)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于w
        num_boxes = 3 * h/32 * w/32 + 3 * h/16 * w/16 + 3 * h/8 * w/8
        pred[0][..., 0:4] 预测框坐标为xywh(中心点+宽长)格式
        pred[0][..., 4]为objectness置信度
        pred[0][..., 5:5+nc]为分类结果
        pred[0][..., 5+nc:]为Θ分类结果
        """
        # pred : (batch_size, num_boxes, no)  batch_size=1
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        # 进行NMS
        # pred : list[tensor(batch_size, num_conf_nms, [xylsθ,conf,classid])] θ∈[0,179]
        #pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        pred = rotate_non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, without_iouthres=False)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        #print('Done. (%.3fs)' % (t2 - t1), pred.shape)
        #continue

        # Process detections
        for i, det in enumerate(pred):  # i:image index  det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            file_name, file_type =  os.path.splitext(str((Path(p).name)))

            save_path = str(Path(out) / Path(p).name)  # 图片保存路径+图片名字
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            #print(txt_path)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            objs_info = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :5] = scale_labels(img.shape[2:], det[:, :5], im0.shape).round()

                # Print results    det:(num_nms_boxes, [xylsθ,conf,classid]) θ∈[0,179]
                for c in det[:, -1].unique():  # unique函数去除其中重复的元素，并按元素（类别）由大到小返回一个新的无元素重复的元组或者列表
                    n = (det[:, -1] == c).sum()  # detections per class  每个类别检测出来的素含量
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string 输出‘数量 类别,’

                # Write results  det:(num_nms_boxes, [xywhθ,conf,classid]) θ∈[0,179]
                for *rbox, conf, cls in reversed(det):  # 翻转list的排列结果,改为类别由小到大的排列
                    # rbox=[tensor(x),tensor(y),tensor(w),tensor(h),tsneor(θ)] θ∈[0,179]
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    # if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        class_name = '%s' % names[int(cls)]
                        conf_str = '%.3f' % conf
                        # rbox2txt(rbox, classname, conf_str, Path(p).stem, str(out + '/result_txt/result_before_merge'))
                        #plot_one_box(rbox, im0, label=label, color=colors[int(cls)], line_thickness=2)
                        if conf < 0.5:
                            cc = (0, 255, 255)
                        elif conf < 0.7:
                            cc = (0, 255, 0)
                        else:
                            cc = (0, 0, 255)
                        if view_img:
                            plot_robx = rbox
                            plot_one_rotated_box(plot_robx, im0, label=label, color=cc, line_thickness=2, pi_format=False)
                        plot_robx = rbox
                        poly = rbox_to_poly(plot_robx, im0, pi_format=False)
                        objs_info.append(poly_to_obj_info(poly, plot_robx, float(conf)))
                        if opt.save_to_txt:
                            coorindate  = ""
                            for p0 in poly:
                                for p1 in p0:
                                    coorindate += (str(float(p1))+" ")
                            if class_name not in obj_detect_results:
                                obj_detect_results[class_name] = []

                            obj_detect_results[class_name].append(file_name + " " + str(float(conf)) + " " + coorindate)
            out_json_path = os.path.join(opt.obj_path, "%05d.json" % (frame_id))
            frame_id += 1
            if opt.save_json:
                if is_video:
                    with open(out_json_path, 'w') as f:
                        f.write(pprint.pformat(objs_info, width=500, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))
                else:
                    vehicle_markers = []
                    for obj_info in objs_info:
                        vehicle_marker = {}
                        if obj_info["width"] > obj_info["length"]:
                            vehicle_marker["heading_angle"] = -obj_info["angle"]
                            vehicle_marker["width"] = obj_info["length"]
                            vehicle_marker["length"] = obj_info["width"]
                        else:
                            vehicle_marker["heading_angle"] = 90-obj_info["angle"]
                            vehicle_marker["width"] = obj_info["width"]
                            vehicle_marker["length"] = obj_info["length"]
                        vehicle_marker["frame_id"] = 0
                        vehicle_marker["height"] = 30
                        vehicle_marker["id"] = obj_id
                        obj_id += 1
                        vehicle_marker["score"] = obj_info["score"]
                        vehicle_marker["x"] = obj_info["center"][0]
                        vehicle_marker["y"] = obj_info["center"][1]
                        vehicle_markers.append([vehicle_marker])
                    dir, file_name = os.path.split(p)
                    image_name, image_type = os.path.splitext(file_name)
                    out_json_path = os.path.join(opt.obj_path, "%s.vehicle_markers.json" % (image_name))
                    with open(out_json_path, 'w') as f:
                        f.write(pprint.pformat(vehicle_markers, width=500, indent=1).replace("'", "\"").replace("True", "true").replace("False", "false"))

            if progress_callback:
                progress_callback(frame_id/dataset.nframes)
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results 播放结果
            if opt.view_img:
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("result", im0)
                k = cv2.waitKey(1)  # 1 millisecond
                if k == 27:
                    exit(0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                    pass
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    cv2.destroyAllWindows()
    if opt.save_to_txt:
        for key in obj_detect_results:
            txt_path = os.path.join(out, "Task1_" + key + ".txt")
            with open(txt_path, "a",encoding="utf-8") as f:
                for obj in obj_detect_results[key]:
                    f.write(obj+'\n')
    print('   All Done. (%.3fs)' % (time.time() - t0))

"""
    weights:训练的权重
    source:测试数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    output:网络预测之后的图片/视频的保存路径
    img-size:网络输入图片大小
    conf-thres:置信度阈值
    iou-thres:做nms的iou阈值
    device:设置设备
    view-img:是否展示预测之后的图片/视频，默认False
    save-txt:是否将预测的框坐标以txt文件形式保存，默认False
    classes:设置只保留某一部分类别，形如0或者0 2 3
    agnostic-nms:进行nms是否将所有类别框一视同仁，默认False
    augment:推理的时候进行多尺度，翻转等操作(TTA)推理
    update:如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.join(os.path.abspath(os.path.dirname(__file__)),"last.pt"), help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='DOTA_demo_view/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--obj-path', type=str, default='', help='output folder')  # output folder
    parser.add_argument('--output', type=str, default='./output', help='json output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=3840, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', type=bool, default=False, help='display results')
    parser.add_argument('--save-json', type=bool, default=False, help='save results to *.json')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--save-img', action='store_true', help='save images')
    parser.add_argument('--save-to-txt', type=bool, default=False, help='save results to txt')
    opt = parser.parse_args()
    return opt

def main(progress_callback=None, model=None):
    opt  = parse_args()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt, progress_callback, model=model)
                # 去除pt文件中的优化器等信息
                strip_optimizer(opt.weights)
        else:
            detect(opt, model=model, progress_callback=progress_callback)

def run(video_path, img_size, save_json, obj_path, device, progress_callback):
    if type(img_size) != str or type(device) != str :
        print("img-size or device or save_json is not str")
        return
    sys.argv = [sys.argv[0], '--source', video_path]
    sys.argv += ['--img-size', img_size]
    sys.argv += ['--save-json', save_json]
    sys.argv += ['--obj-path', obj_path]
    sys.argv += ['--device', device]
    main(progress_callback)

def run_to_val(model, val_path, img_size, output=None, save_to_txt=False, view_img=True):
    sys.argv = [sys.argv[0], '--source', val_path]
    sys.argv += ['--img-size', str(img_size)]
    sys.argv += ['--save-to-txt', str(save_to_txt)]
    sys.argv += ['--view-img', str(view_img)]
    if output is not None:
        sys.argv += ['--output', output]
    main(model=model)

def test():
    run('/home/me/Documents/eagle/data/011/011.mp4', '3840', False, '/home/me/Documents/eagle/data/011/011_objs', '1', None)

def test_val():
    val=[
    '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0011_private170/images/',
    '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0014_longyaolu_c1000e/val/images',
    '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0013_longyaolu_dji/val/images',
    '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/part2_dataset']
    output_path = "output"
    import shutil
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.mkdir(output_path)
    for path in val:
        run_to_val(model=None, val_path=path, img_size=768, save_to_txt=True, output=output_path)

if __name__ == '__main__':
    test_val()
    # main()
    # test()
