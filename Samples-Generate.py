from pycocotools.coco import COCO
import numpy as np
import json
import skimage.io as io
import cv2
from skimage.transform import resize
import os
from copy import deepcopy

import pycocotools
from pycocotools import mask as mask_util
import pycocotools._mask as _mask
import multiprocessing as mp

from color_jitter import ColorJitter
import pycocotools.mask as mask_utils

'''
coco.dataset: dict
    images: list[dict]
    annotations: list[dict]
'''

max_orig_annid = 900100581904
transform_dict = {'brightness':0.1026, 'contrast':0.0935, 'sharpness':0.8386, 'color':0.1592}
colorJitter = ColorJitter(transform_dict)

def getMask(ann):
    poly_specs = ann['dp_masks']
    segm = np.zeros((256,) * 2, dtype=np.float32)
    for i in range(14):
        poly_i = poly_specs[i]
        if poly_i:
            mask_i = mask_utils.decode(poly_i)
            segm[mask_i > 0] =1
    mask = coco.annToMask(ann)
    x1, y1, w, h = [int(xywh) for xywh in ann['bbox']]
    x2 = x1+w
    y2 = y1+h
    _segm = cv2.resize(segm, (w, h))
    mask_segm = np.zeros(mask.shape, dtype=np.float32)
    mask_segm[y1:y2, x1:x2] = mask_segm[y1:y2, x1:x2] + _segm
    mask_segm += mask
    mask_segm[mask_segm>=0.3] = 1
    mask_segm[mask_segm<0.3] = 0

    return mask_segm

def get_seg(ann):
    seg_poly = ann['segmentation']
    x1, y1, w, h = [int(xywh) for xywh in ann['bbox']]
    x2 = x1 + w
    y2 = y1 + h
    segm = np.zeros((h, w), dtype=np.float32)
    if len(seg_poly) == 0:
        segm = np.zeros((h, w), dtype=np.float32)
    else:
        rle = _mask.frPyObjects(
            seg_poly, h, w)
        if type(rle) == list:
            mask_numpy = _mask.decode(rle)
        else:
            mask_numpy = _mask.decode([rle])[:, :, 0]
        mask_numpy = np.array(mask_numpy, dtype=np.float32)
        if mask_numpy.ndim < 3:
            mask_numpy = mask_numpy[:, :, None]
        mask_numpy = np.sum(mask_numpy, axis=2)  # connect the divided part
        segm[mask_numpy > 0] = 1

    mask = coco.annToMask(ann)
    mask_segm = np.zeros(mask.shape, dtype=np.float32)
    mask_segm[y1:y2, x1:x2] = mask_segm[y1:y2, x1:x2] + segm
    # mask_segm[x1:x2, y1:y2] = mask_segm[x1:x2, y1:y2] + segm
    mask_segm += mask
    mask_segm[mask_segm >= 0.3] = 1
    mask_segm[mask_segm < 0.3] = 0

    return mask_segm

def deepcopy_dict(x):
    y = {}
    for key, value in x.items():
        y[deepcopy(key)] = deepcopy(value)
    return y

def deepcopy_list(x):
    y = []
    for a in x:
        y.append(deepcopy_dict(a))
    return y

def computeIoU(moddified_bbox, orig_bbox):
    cx1 = moddified_bbox[0]
    cy1 = moddified_bbox[1]
    cx2 = cx1 + moddified_bbox[2]
    cy2 = cy1 + moddified_bbox[3]

    gx1 = orig_bbox[0]
    gy1 = orig_bbox[1]
    gx2 = gx1 + orig_bbox[2]
    gy2 = gy1 + orig_bbox[3]

    carea = (cx2 - cx1) * (cy2 - cy1)
    garea = (gx2 - gx1) * (gy2 - gy1)

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h

    return max(area / carea, area / garea)

def computeMaskIoU(mask_add, mask_orig):
    return np.sum((mask_add+mask_orig)>1.5) / np.sum((mask_add+mask_orig)>=0.9)

def IoUsLowerThanThres(moddified_bbox, orig_bboxes, thres=0.3):
    for orig_bbox in orig_bboxes:
        if computeIoU(moddified_bbox, orig_bbox) > thres:
            return False
    return True

def modify_ann(ann, x_shift, y_shift, scale_w, scale_h):
    x_orig, y_orig, w_orig, h_orig = ann['bbox']
    ann['bbox'] = [x_shift, y_shift, int(w_orig*scale_w), int(h_orig*scale_h)]
    for i in range(len(ann['segmentation'])):
        segm = np.array(ann['segmentation'][i])
        segm[::2] = (segm[::2] - x_orig) * scale_w + x_shift
        segm[1::2] = (segm[1::2] - y_orig) * scale_h + y_shift
        ann['segmentation'][i] = segm.tolist()
    kpts = np.array(ann['keypoints'])
    kpts[::3] = (kpts[::3] - x_orig) * scale_w + x_shift
    kpts[1::3] = (kpts[1::3] - y_orig) * scale_h + y_shift
    ann['keypoints'] = kpts.tolist()

if __name__ == '__main__':
    coco_synthesize = {}
    dataDir = ""
    annTrainFile = ""
    syn_imgDir = "%s" % dataDir
    syn_json_file = dataDir + ""
    if not os.path.exists(syn_imgDir):
        print(syn_imgDir)
        os.makedirs(syn_imgDir)

    coco = COCO(annTrainFile)
    print("--"*20)
    print("Initlizing coco_synthesize")
    coco_synthesize['images'] = []
    coco_synthesize['categories'] = [coco.dataset['categories'][0]]
    coco_synthesize['annotations'] = []


    print("--"*20)
    print("loading coco_instances for background images")
    InstanceAnnFile = ""
    coco_instances = COCO(InstanceAnnFile)
    person_catIds = coco_instances.getCatIds(catNms=['person'])
    person_imgIds = coco_instances.getImgIds(catIds=person_catIds)
    all_imgIds = coco_instances.getImgIds()
    bg_imgIds = list(set(all_imgIds) - set(person_imgIds))
    bg_imgs = coco_instances.loadImgs(bg_imgIds)  
    print("**"*20)
    print("{} images have no person in instances dataset".format(len(bg_imgs)))  

    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)  
    all_annIds = coco.getAnnIds(imgIds, catIds)
    all_anns = coco.loadAnns(all_annIds)
    id_used = {deepcopy(ann['id']): 0 for ann in all_anns}
    print("**"*20)
    print("{} images have person in instances dataset".format(len(imgs)))  
    print("{} keypoints annotations in train2017 dataset".format(len(all_anns)))  

    buffers = []
    print("--"*20)
    print("save all keypoints annotations into buffer")
    for img in imgs:
        try:
            I = io.imread("%s%s" %('', img['file_name']))
        except FileNotFoundError:
            I = io.imread("%s%s" %('', img['file_name']))
        if len(I.shape) == 0:
            continue
        if len(I.shape) == 2:
            I = np.expand_dims(I, -1).repeat(3, -1)
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        try:
            anns = coco.loadAnns(annIds)
        except KeyError:
            continue
        anns_keyopints = [ann for ann in anns if 'keypoints' in ann.keys()]
        useful_anns_keyopints = [ann for ann in anns_keyopints if ann['num_keypoints'] > 6 and len(ann['segmentation']) == 1]
        for ann_keypoints in useful_anns_keyopints:
            buffers.append([I, deepcopy(ann_keypoints), img])
    print("**" * 20)
    print("{} keypoints annotations can be used".format(len(buffers)))  


    print("--"*20)
    print("Add more keypoints annotations to more background images")
    for i, bg_img in enumerate(bg_imgs):
        if len(buffers) < 10:
            break

        I_bg = io.imread("%s" %bg_img['file_name'])
        if len(I_bg.shape) == 0:
            continue
        if len(I_bg.shape) == 2:
            I_bg = np.expand_dims(I_bg, -1).repeat(3, -1)

        coco_synthesize['images'].append(bg_img)

        num_add = np.random.randint(1, 16)
        bboxes_keypoints = []
        real_num_add = 0
        for j in range(num_add):
            index_buf = np.random.randint(len(buffers))
            img_add, ann_add, img_info = deepcopy(buffers[index_buf])
            h_img, w_img, _ = I_bg.shape
            x0, y0, w_bbox_add, h_bbox_add = [int(xywh) for xywh in ann_add['bbox']]
            x1 = x0 + w_bbox_add
            y1 = y0 + h_bbox_add

            # For w,h of bbox rescale
            if max(w_bbox_add, h_bbox_add) < 100:  # small
                scale_w = np.random.uniform(1.5, min(2, (w_img - 100) / w_bbox_add - 0.1))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif max(w_bbox_add, h_bbox_add) < 200:  # medium
                scale_w = np.random.uniform(1.2, min(1.8, (w_img - 50) / w_bbox_add - 0.05))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif max(w_bbox_add, h_bbox_add) < 300:
                scale_w = np.random.uniform(0.9, min(1.5, (w_img - 10) / w_bbox_add - 0.05))
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif w_bbox_add > w_img - 5:
                scale_w = np.random.uniform(0.7, (w_img - 10) / w_bbox_add - 0.05)
                scale_h = np.random.uniform(0.98 * scale_w, 1.02 * scale_w)
            elif h_bbox_add > h_img - 5:
                scale_h = np.random.uniform(0.7, (h_img - 10) / h_bbox_add - 0.05)
                scale_w = np.random.uniform(0.98 * scale_h, 1.02 * scale_h)
            else:
                scale_h = np.random.uniform(0.8, (h_img - 10) / h_bbox_add - 0.1)
                scale_w = np.random.uniform(0.98 * scale_h, 1.02 * scale_h)

            # For (x0, y0) shift
            try:
                x_shifts = [np.random.randint(5, w_img - w_bbox_add * scale_w) for _ in range(10000)]
                y_shifts = [np.random.randint(5, h_img - h_bbox_add * scale_h) for _ in range(10000)]
            except ValueError:
                continue

            add_flag = False
            for x_shift, y_shift in zip(x_shifts, y_shifts):
                modified_bbox = [x_shift, y_shift, w_bbox_add * scale_w, h_bbox_add * scale_h]
                if IoUsLowerThanThres(modified_bbox,bboxes_keypoints, 0.6):
                    add_flag = True
                    real_num_add += 1
                    break

            # if this ann_add not satisfies IoU with original bbox < 0.5, just continue
            if not add_flag:
                continue

            mask_add = coco.annToMask(ann_add)
            id_used[ann_add['id']] += 1
            ann_add['id'] = ann_add['id'] + max_orig_annid * (id_used[ann_add['id']])

            ann_add['image_id'] = bg_imgs[i]['id']
            ann_add['area'] = int(ann_add['area'] * scale_w * scale_h)
            modify_ann(ann_add, x_shift, y_shift, scale_w, scale_h)
            coco_synthesize['annotations'].append(ann_add)
            bboxes_keypoints.append(ann_add['bbox'])
            
            h_img_bbox = int(h_bbox_add * scale_h)
            w_img_bbox = int(w_bbox_add * scale_w)
            img_add = colorJitter(img_add)
            img_bbox_add = cv2.resize(img_add[y0: y1, x0: x1], (w_img_bbox, h_img_bbox))
            mask_bbox_add = cv2.resize(mask_add[y0: y1, x0: x1], (w_img_bbox, h_img_bbox))
            mask_bbox_add = np.expand_dims(mask_bbox_add, -1)
            I_bg[y_shift: y_shift + h_img_bbox, x_shift: x_shift + w_img_bbox] = \
                I_bg[y_shift: y_shift + h_img_bbox, x_shift: x_shift + w_img_bbox] * (1 - mask_bbox_add) + \
                img_bbox_add * mask_bbox_add
        print("[{}/{}] add {} annotations on image {}".format(i, len(bg_imgs), real_num_add, bg_img['file_name']))
        io.imsave(syn_imgDir+bg_img['file_name'], I_bg)

    print("Save annotations into {}".format(syn_json_file))
    with open(syn_json_file, 'w') as fp:
        json.dump(coco_synthesize, fp)
    print("num of annotations: ", len(coco_synthesize["annotations"]))
    print("num of images: ", len(coco_synthesize["images"]))