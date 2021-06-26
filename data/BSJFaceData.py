# -*- coding=utf-8 -*-

import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import json
import random

def _resize_subtract_mean(image, image_size):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (image_size[2], image_size[1]), interpolation=interp_method)
    image = image.astype(np.float32)
    image = (image - 127.5) * 0.0078125
    return image.transpose(2, 0, 1)

class BSJFaceData(data.Dataset):
    def __init__(self, root, image_size=[3, 640, 640], n_points=40, preproc=None, transform=None):
        self.preproc = preproc
        self.root = root
        self.image_size = image_size
        self.n_points = n_points

        self.transform = transform

        self.imgs_path = []

        self.target_bbox     = 0
        self.target_landmark = 0
        self.target_group_id = 0
        for parent, _, fnames in os.walk(self.root):
            for fname in fnames:
                _, ext = os.path.splitext(fname)
                if ext != ".png" and ext != ".jpg":
                    continue
                fname = os.path.join(parent, fname)
                tname = fname.replace(".jpg", ".json").replace(".png", ".json")

                if os.path.isfile(fname) is False or os.path.isfile(tname) is False:
                    continue
                
                json_dict = {}
                with open(tname, "r") as f:
                    json_dict = json.load(f)
 
                if "shapes" not in json_dict:
                    continue
                
                shapes = json_dict['shapes']
                
                group_id = []
                for shape in shapes:
                    if "shape_type" not in shape:
                        continue
                    
                    if shape["shape_type"] == "rectangle":
                        self.target_bbox += 1
                    elif shape["shape_type"] == "polygon":
                        self.target_landmark += 1

                #     if "group_id" in shape and "group_id" is not None:
                #         if shape['group_id'] not in group_id:
                #             group_id.append("group_id")

                # target_group_id += len(group_id)
                img_path = {"name": fname, "tname": tname}
                self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)
    
    def __str__(self):
        return "train image {} lanmark, dataset imgae size {}.all image {}. target bbox {}. target landmark {}".format(
            self.n_points, self.image_size, len(self.imgs_path), self.target_bbox, self.target_landmark)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]["name"]
        tname = self.imgs_path[index]["tname"]
        json_dict = {}
        with open(tname, "r") as f:
            json_dict = json.load(f)
        
        shapes = json_dict['shapes']
        face_group = {}
        for shape in shapes:
            if "group_id" not in shape and "group_id" is None:
                continue
            if "points" not in shape and "shape_type" not in shape:
                continue
            if shape["group_id"] not in face_group:
                face_group[shape["group_id"]] = {"face":[], "landmark": []}

            if shape["shape_type"] == "rectangle":
                face_group[shape["group_id"]]["face"]       = shape["points"]
            elif shape["shape_type"] == "polygon":
                face_group[shape["group_id"]]["landmark"]   = shape["points"]
        image = cv2.imread(img_path)
        # print("img_path: ", img_path)
        height, width, _ = image.shape

        h_rate = float(self.image_size[1]) / float(height)
        w_rate = float(self.image_size[2]) / float(width)
        # boxes_t *= [w_rate, h_rate, w_rate, h_rate]

        # boxes_t[:, 0::2] /= self.image_size[2]
        # boxes_t[:, 1::2] /= self.image_size[1]

        annotations = np.zeros((0, self.n_points + 5), dtype=np.float)
        for idx, key in enumerate(face_group):
            face_info = face_group[key]
            annotation = np.zeros((1, self.n_points + 5), dtype=np.float)
            # bbox
            # annotation[0, 0] = float(face_info["face"][0][0]) * w_rate / self.image_size[2]     # x1
            # annotation[0, 1] = float(face_info["face"][0][1]) * h_rate / self.image_size[1]     # y1
            # annotation[0, 2] = float(face_info["face"][1][0]) * w_rate / self.image_size[2]     # x2
            # annotation[0, 3] = float(face_info["face"][1][1]) * h_rate / self.image_size[1]     # y2
            annotation[0, 0] = float(face_info["face"][0][0])     # x1
            annotation[0, 1] = float(face_info["face"][0][1])     # y1
            annotation[0, 2] = float(face_info["face"][1][0])     # x2
            annotation[0, 3] = float(face_info["face"][1][1])     # y2

            # landmarks
            # for i in range(int(self.n_points/2)):
            #     points = face_info["landmark"][i]
            #     annotation[0, i * 2 + 4] = float(points[0])    # x
            #     annotation[0, i * 2 + 5] = float(points[1])    # y
            #     # annotation[0, i * 2 + 4] = float(points[0]) * w_rate / self.image_size[2]   # x
            #     # annotation[0, i * 2 + 5] = float(points[1]) * h_rate / self.image_size[1]   # y

            if self.n_points == 40:
                for i in range(20):
                    points = face_info["landmark"][i]
                    annotation[0, i * 2 + 4] = float(points[0])    # x
                    annotation[0, i * 2 + 5] = float(points[1])    # y
                    # annotation[0, i * 2 + 4] = float(points[0]) * w_rate / self.image_size[2]   # x
                    # annotation[0, i * 2 + 5] = float(points[1]) * h_rate / self.image_size[1]   # y
            elif self.n_points == 10:
                pointsArr = face_info["landmark"]
                # 左眼坐标
                annotation[0, 4] = float(pointsArr[6][0]+pointsArr[8][0]) / 2
                annotation[0, 5] = float(pointsArr[7][1]+pointsArr[9][1]) / 2

                # 右眼坐标
                annotation[0, 6] = float(pointsArr[10][0]+pointsArr[12][0]) / 2
                annotation[0, 7] = float(pointsArr[11][1]+pointsArr[13][1]) / 2

                # 鼻尖坐标
                annotation[0, 8] = float(pointsArr[14][0])
                annotation[0, 9] = float(pointsArr[14][1])

                # 左嘴角坐标
                annotation[0, 10] = float(pointsArr[15][0])
                annotation[0, 11] = float(pointsArr[15][1])

                # 右嘴角坐标
                annotation[0, 12] = float(pointsArr[17][0])
                annotation[0, 13] = float(pointsArr[17][1])


            # 是否关键点坐标
            if np.sum(annotation[0, 4:]) == 0:
                annotation[0, -1] = -1
            else:
                annotation[0, -1] = 1

            annotations = np.append(annotations, annotation, axis=0)


        # image = _resize_subtract_mean(image, self.image_size) #QLHua注意 在在线数据扩增中 注释这行代码，先不resize, 扩增完再resize
        target = np.array(annotations)

        #数据扩增
        if self.preproc is not None:
            image, target = self.preproc(image, target)

        if self.transform is not None:
            image, target = self.transform(image, target)
    
        return torch.from_numpy(image), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    #batch format --> [( tensor(), array() ), ..., ( tensor(), array() )]
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

if __name__ == '__main__':
    root = "/data_ssd/holyland/face_20_landmark"
    bsj = BSJFaceData(root)
    image, annotations = bsj.__getitem__(0)
    print(image)
    print("image.shape: ", image.shape)
    print(annotations)
    # for annotation in annotations:
    #     x1, y1, x2, y2 = annotation[:4]
    #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))

    # cv2.imshow("image", image)
    # cv2.waitKey()
    