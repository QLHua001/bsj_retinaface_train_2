import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, landmark_num, preproc=None):
        self.preproc = preproc
        self.landmark_num = landmark_num
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)


    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        print("getitem_index:")
        print(index)
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape
        print(self.imgs_path[index])

        labels = self.words[index]
        if landmark_num == 5:
            annotations = np.zeros((0, 15)) #QLHUA TEST landmark 15 --> 4+1+40=45
        elif landmark_num == 20:
            annotations = np.zeros((0, 45)) #QLHUA TEST landmark 15 --> 4+1+40=45
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            if landmark_num == 5:
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x   #QLHUA TEST landmark
                annotation[0, 5] = label[5]    # l0_y   #QLHUA TEST landmark
                annotation[0, 6] = label[7]    # l1_x   #QLHUA TEST landmark
                annotation[0, 7] = label[8]    # l1_y   #QLHUA TEST landmark
                annotation[0, 8] = label[10]   # l2_x   #QLHUA TEST landmark
                annotation[0, 9] = label[11]   # l2_y   #QLHUA TEST landmark
                annotation[0, 10] = label[13]  # l3_x   #QLHUA TEST landmark
                annotation[0, 11] = label[14]  # l3_y   #QLHUA TEST landmark
                annotation[0, 12] = label[16]  # l4_x   #QLHUA TEST landmark
                annotation[0, 13] = label[17]  # l4_y   #QLHUA TEST landmark
                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1      #QLHUA TEST landmark
                else:
                    annotation[0, 14] = 1       #QLHUA TEST landmark
            elif landmark_num == 20:
                annotation = np.zeros((1, 45))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x   #QLHUA TEST landmark
                annotation[0, 5] = label[5]    # l0_y   #QLHUA TEST landmark
                annotation[0, 6] = label[7]    # l1_x   #QLHUA TEST landmark
                annotation[0, 7] = label[8]    # l1_y   #QLHUA TEST landmark
                annotation[0, 8] = label[10]   # l2_x   #QLHUA TEST landmark
                annotation[0, 9] = label[11]   # l2_y   #QLHUA TEST landmark
                annotation[0, 10] = label[13]  # l3_x   #QLHUA TEST landmark
                annotation[0, 11] = label[14]  # l3_y   #QLHUA TEST landmark
                annotation[0, 12] = label[16]  # l4_x   #QLHUA TEST landmark
                annotation[0, 13] = label[17]  # l4_y   #QLHUA TEST landmark
                annotation[0, 14] = label[19]    # l5_x   #QLHUA TEST landmark
                annotation[0, 15] = label[20]    # l5_y   #QLHUA TEST landmark
                annotation[0, 16] = label[22]    # l6_x   #QLHUA TEST landmark
                annotation[0, 17] = label[23]    # l6_y   #QLHUA TEST landmark
                annotation[0, 18] = label[25]   # l7_x   #QLHUA TEST landmark
                annotation[0, 19] = label[26]   # l7_y   #QLHUA TEST landmark
                annotation[0, 20] = label[28]  # l8_x   #QLHUA TEST landmark
                annotation[0, 21] = label[29]  # l8_y   #QLHUA TEST landmark
                annotation[0, 22] = label[31]  # l9_x   #QLHUA TEST landmark
                annotation[0, 23] = label[32]  # l9_y   #QLHUA TEST landmark
                annotation[0, 24] = label[34]    # l10_x   #QLHUA TEST landmark
                annotation[0, 25] = label[35]    # l10_y   #QLHUA TEST landmark
                annotation[0, 26] = label[37]    # l11_x   #QLHUA TEST landmark
                annotation[0, 27] = label[38]    # l11_y   #QLHUA TEST landmark
                annotation[0, 28] = label[40]   # l12_x   #QLHUA TEST landmark
                annotation[0, 29] = label[41]   # l12_y   #QLHUA TEST landmark
                annotation[0, 30] = label[43]  # l13_x   #QLHUA TEST landmark
                annotation[0, 31] = label[44]  # l13_y   #QLHUA TEST landmark
                annotation[0, 32] = label[46]  # l14_x   #QLHUA TEST landmark
                annotation[0, 33] = label[47]  # l14_y   #QLHUA TEST landmark
                annotation[0, 34] = label[49]    # l15_x   #QLHUA TEST landmark
                annotation[0, 35] = label[50]    # l15_y   #QLHUA TEST landmark
                annotation[0, 36] = label[52]    # l16_x   #QLHUA TEST landmark
                annotation[0, 37] = label[53]    # l16_y   #QLHUA TEST landmark
                annotation[0, 38] = label[55]   # l17_x   #QLHUA TEST landmark
                annotation[0, 39] = label[56]   # l17_y   #QLHUA TEST landmark
                annotation[0, 40] = label[58]    # l18_x   #QLHUA TEST landmark
                annotation[0, 41] = label[59]    # l18_y   #QLHUA TEST landmark
                annotation[0, 42] = label[61]    # l19_x   #QLHUA TEST landmark
                annotation[0, 43] = label[62]    # l19_y   #QLHUA TEST landmark
                if (annotation[0, 4]<0):
                    annotation[0, 44] = -1      #QLHUA TEST landmark
                else:
                    annotation[0, 44] = 1       #QLHUA TEST landmark

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        print("target: ")
        print(target.shape)
        print(target)
        if self.preproc is not None:
            img, target = self.preproc(img, target)
        print("after preproc: ")
        print(img.shape)
        print(target.shape)
        print(target)
        return torch.from_numpy(img), target

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
