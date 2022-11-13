# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Hongjie NI, based on code from Bharath Hariharan
# --------------------------------------------------------
"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
#import cPickle
import numpy as np
import cv2
from utils.general import rbox_to_poly
from shapely.geometry import Polygon

class Validation(object):
    def __init__(self) -> None:
        super().__init__()
        self.img_type_lists = {'.jpg', '.bmp', '.png', '.jpeg', '.rgb', '.tif'}
        self.class_recs = {}
        self.img_names = {}
        self.file_lists = []

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def obb_rotated_box(self, point):
        if len(point) % 2 != 0:
            return None
        poly_points = []
        i = 0
        while i < len(point):
            poly_points.append((point[i], point[i + 1]))
            i += 2
        return Polygon(poly_points)

    def voc_eval(
            self,
            detpath,
            class_name,
            # cachedir,
            ovthresh=0.25,
            use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])
        Top level function that does the PASCAL VOC evaluation.
        detpath: Path to detections
            detpath.format(classname) should produce the detection results file.
        annopath: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        imagesetfile: Text file containing the list of images, one image per line.
        classname: Category name (duh)
        cachedir: Directory for caching the annotations
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)
        # assumes imagesetfile is a text file with each line an image name
        # cachedir caches the annotations in a pickle file

        class_recs = {}
        npos = 0
        for img_name in self.class_recs:
            R = [obj for obj in self.class_recs[img_name] if obj['name'] == class_name]
            bbox = np.array([x['obb'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[img_name] = {
                'obb': bbox,
                'difficult': difficult,
                'det': det
            }

        # read dets
        detfile = detpath.format(class_name)
        if not os.path.exists(detfile):
            return 0, 0, 0
        with open(detfile, 'r') as f:
            lines = f.readlines()

        split_lines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in split_lines]
        confidence = np.array([float(x[1]) for x in split_lines])

        #print('check confidence: ', confidence)

        BB = np.array([[float(z) for z in x[2:]] for x in split_lines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)

        #print('check sorted_scores: ', sorted_scores)
        #print('check sorted_ind: ', sorted_ind)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        #print('check imge_ids: ', image_ids)
        #print('imge_ids len:', len(image_ids))
        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        if class_recs == {}:
            return 0, 0, 0
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['obb'].astype(float)

            bbox = bb.tolist()
            poly1 = self.obb_rotated_box(bb)
            origin_bb_area = poly1.area
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                inters = []
                each_box_area = []
                for obb in BBGT:
                    poly2 = Polygon(obb)
                    each_box_area.append(poly2.area)
                    if not poly1.intersects(poly2):
                        inters.append(0)
                    else:
                        inters.append(poly1.intersection(poly2).area)  # 相交面积

                each_box_area = np.array(each_box_area, dtype=np.float32)
                inters = np.array(inters, dtype=np.float32)
                # union
                uni = (each_box_area + origin_bb_area - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                ## if there exist 2
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                    # print('filename:', image_ids[d])
            else:
                fp[d] = 1.

        # compute precision recall

        # print('check fp:', fp)
        # print('check tp', tp)

        print('class_name:',class_name,' npos num:', npos)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)
        return rec, prec, ap


    # val image directory
    # detection result directory
    # annotation directory
    # class names set
    # if the dataset format is yolo obb, the datasets should be convert to dota format
    def run(self,
            val_image_path,
            detection_path,
            class_names,
            log_dir="log_dir"):

        detection_path += '/Task1_{:s}.txt'
        self.yolo_to_dota(val_image_path, class_names)
        obb_mAP_results = []
        for i in range(0,10):
            print('class_names:', class_names)
            obb_mAP_results.append(self.get_obb_mAP(detection_path, class_names, round(0.5+0.05*i,2)))

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        map_log_path = os.path.join(log_dir,"log_mAP.log")
        with open(map_log_path,"a",encoding="utf-8") as f:
            from datetime import datetime
            f.write(datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+"\n")
            f.write(str(class_names)+"\n")
            for i in range(0,len(obb_mAP_results)):
                f.write(str(round(0.5+0.05*i,2))+"@mAP\n")
                f.write(str(obb_mAP_results[i][0])+"\n")
                f.write(str(obb_mAP_results[i][1])+"\n")

    def get_obb_mAP(self, detection_path, class_names, ovthresh):
        classaps = []
        map = 0
        for class_name in class_names:
            rec, prec, ap = self.voc_eval(detection_path,
                                          class_name,
                                          ovthresh=ovthresh,
                                          use_07_metric=True)
            map = map + ap
            # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
            # print(ap)
            classaps.append(ap)

        map = round(map / len(class_names), 3)
        print('mAP@'+str(ovthresh)+":", map)
        classaps = 100 * np.array(classaps)
        print('classaps: ', classaps)
        return map, classaps

    def get_file_lists(self, parent_path, file_types=[]):
        file_lists = os.listdir(parent_path)
        for file_name in file_lists:
            file_path = os.path.join(parent_path, file_name)
            if os.path.isdir(file_path):
                self.get_file_lists(file_path, file_types)
            else:
                _, file_type = os.path.splitext(file_name)
                if file_type in file_types:
                    self.file_lists.append(file_path)
                elif file_types == []:
                    self.file_lists.append(file_path)

    def yolo_to_dota(self, yolo_result_path, class_names):
        self.class_recs = {}
        self.img_names = []
        datasets_path = []
        if type(yolo_result_path) is not list:
            datasets_path.append(yolo_result_path)
        else:
            datasets_path = yolo_result_path
        for path in datasets_path:
            self.file_lists = []
            self.get_file_lists(path,[".txt"])
            for txt_file_path in self.file_lists:
                file_name,_ = os.path.splitext(os.path.basename(txt_file_path))
                with open(txt_file_path, "r", encoding="utf-8") as f:
                    if file_name not in self.class_recs:
                        self.class_recs[file_name] = []
                        self.img_names.append(file_name)
                    for line in f.readlines():
                        data = line.split()
                        #origin yolo bbox coorinadte value [0,1]
                        # image = cv2.imread(image_path)
                        # plot_robx = [
                        #     float(data[1]) * image.shape[1],
                        #     float(data[2]) * image.shape[0],
                        #     float(data[3]) * image.shape[1],
                        #     float(data[4]) * image.shape[0],
                        #     float(data[5])
                        # ]
                        plot_robx = [
                            float(data[1]),
                            float(data[2]),
                            float(data[3]),
                            float(data[4]),
                            float(data[5])
                        ]
                        poly = rbox_to_poly(plot_robx, None, pi_format=False)
                        self.class_recs[file_name].append({"name":class_names[int(data[0])], "obb":poly.tolist(), "difficult": 0})

def main():
    # detpath = r'E:\documentation\OneDrive\documentation\DotaEvaluation\evluation_task2\evluation_task2\faster-rcnn-nms_0.3_task2\nms_0.3_task\Task2_{:s}.txt'
    # annopath = r'I:\dota\testset\ReclabelTxt-utf-8\{:s}.txt'
    # imagesetfile = r'I:\dota\testset\va.txt'

    detpath = r'output'
    annopath = [
  '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0011_private170',
  '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0014_longyaolu_c1000e/val',
  '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/0013_longyaolu_dji/val',
  '/media/me/65F33762C14D581B/NI/datasets/drone_yolov5_obb/part2_dataset'
]  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    classnames = ['car', 'bus', 'min_bus', 'truck', 'van', 'motor', 'bike', 'ped', 'big_truck']
    validation = Validation()
    validation.run(annopath, detpath, classnames)


if __name__ == '__main__':
    main()