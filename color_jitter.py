import numpy as np 
from PIL import ImageEnhance
from PIL import Image

transform_type_dict = dict(
    brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
    sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
)

class ColorJitter(object):
    def __init__(self, transform_dict):
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
    
    def __call__(self, img):
        out = Image.fromarray(img)
        rand_num = np.random.uniform(0, 1, len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            out = transformer(out).enhance(r)
        
        return np.array(out)

if __name__ == "__main__":
    import skimage.io as io
    from pycocotools.coco import COCO
    transform_dict = {'brightness':0.1026, 'contrast':0.0935, 'sharpness':0.8386, 'color':0.1592}
    colorJitter = ColorJitter(transform_dict)
    annFile = "/data2/wangxuanhan/datasets/coco2014/annotations/person_keypoints_train2017.json"
    coco = COCO(annFile)
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    imgs = coco.loadImgs(imgIds)  # list[dict]

    for img in imgs[:30]:
        I = io.imread("%strain2017/%s" %('/data2/wangxuanhan/datasets/coco2014/', img['file_name']))
        io.imsave('./color_test/%s' %img['file_name'], I)
        io.imsave('./color_test/%s' %img['file_name'] + '_color.jpg', colorJitter(I))

