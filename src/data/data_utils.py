import random
import numpy as np
import torch
import cv2

import albumentations as aug
import albumentations.augmentations.functional as augf
from albumentations.core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform

from surface_loss_utils import numpy_make_one_hot, one_hot2dist

MASK_THR = 525

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle, input_shape):
    width, height = input_shape[:2]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T

def build_masks(rles, input_shape, output_shape=None):
    assert len(rles) == 4

    depth = len(rles)
    if output_shape:
        masks = np.zeros((depth, *output_shape))
    else:
        masks = np.zeros((depth, *input_shape))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            mask = rle2mask(rle, input_shape)
            if output_shape:
                if input_shape != output_shape:
                    mask = cv2.resize(mask, output_shape[::-1], interpolation=cv2.INTER_NEAREST)
            masks[i, :, :] = mask * (i+1)
    
    return masks.sum(axis=0)


def build_rles(masks, n_cls=4, thr=MASK_THR):
    result = []
    for i in range(1, n_cls+1):
        m = (masks==i).astype(np.uint8)
        if m.sum() < thr:
            result.append('')
        else:
            result.append(mask2rle(m))
    return result


def build_labels(rles):
    depth = len(rles)
    labels = np.zeros(depth, dtype=np.float32)

    for i, rle in enumerate(rles):
        labels[i] = 1 if type(rle) == str else 0

    return labels


class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.
    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: 0.0625.
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Default: 0.1.
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: 45.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (list of ints [r, g, b]): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (scalar or list of ints): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        p (float): probability of applying the transform. Default: 0.5.
    Targets:
        image, mask, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_REFLECT_101, value=None, mask_value=None, always_apply=False, p=0.5):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit = to_tuple(shift_limit)
        self.scale_limit = to_tuple(scale_limit, bias=1.0)
        self.rotate_limit = to_tuple(rotate_limit)
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR, **params):
        return augf.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0, **params):
        return augf.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)

    def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0, interpolation=cv2.INTER_LINEAR,
                          **params):
        return augf.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)

    def get_params(self):
        return {'angle': random.uniform(self.rotate_limit[0], self.rotate_limit[1]),
                'scale': random.uniform(self.scale_limit[0], self.scale_limit[1]),
                'dx': random.uniform(self.shift_limit[0], self.shift_limit[1]),
                'dy': 0} # Modified based on the dataset

    def apply_to_bbox(self, bbox, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, **params):
        return augf.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, **params)

    def get_transform_init_args(self):
        return {
            'shift_limit': self.shift_limit,
            'scale_limit': to_tuple(self.scale_limit, bias=-1.0),
            'rotate_limit': self.rotate_limit,
            'interpolation': self.interpolation,
            'border_mode': self.border_mode,
            'value': self.value,
            'mask_value': self.mask_value
        }


def do_random_salt_pepper_noise(image, noise = 0.0005):
    height,width = image.shape[:2]
    num_salt = int(noise*width*height)

    # Salt mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [255,255,255]

    # Pepper mode
    yx = [np.random.randint(0, d - 1, num_salt) for d in image.shape[:2]]
    image[tuple(yx)] = [0,0,0]

    return image


class SaltPepperNoise(ImageOnlyTransform):

    def __init__(self, level_limit=0.0005, always_apply=False, p=0.5):
        super(SaltPepperNoise, self).__init__(always_apply, p)
        self.level_limit = level_limit

    def apply(self, img, level=0.0001, **params):
        return do_random_salt_pepper_noise(img, noise=level)

    def get_params(self):
        return {"level": random.uniform(0, self.level_limit)}

    def get_transform_init_args_names(self):
        return ("level_limit",)


def do_width_random_crop(image, mask, w):
    height, width = image.shape[:2]
    x = 0
    if width > w:
        x = np.random.choice(width-w)
    image_c = image[:, x:x+w, :]
    mask_c  = mask[:, x:x+w]
    
    if (image_c < 10).mean() > 0.85 or image_c.mean() < 15:
        image_c = image[:, :w, :]
        mask_c  = mask[:, :w]
        
    if (image_c < 10).mean() > 0.85:
        image_c = image[:, -w:, :]
        mask_c  = mask[:, -w:]
        
    return image_c, mask_c


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x,y=0,0
    if width>w:
        x = np.random.choice(width-w)
    if height>h:
        y = np.random.choice(height-h)
    image = image[y:y+h,x:x+w,:]
    mask  = mask [y:y+h,x:x+w]

    #---
    if (w,h)!=(width,height):
        image = cv2.resize( image, dsize=(width,height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask,  dsize=(width,height), interpolation=cv2.INTER_NEAREST)

    return image, mask


def aug_train(p=1, shift_limit=0.0625, scale_limit=0.1):
    aug_list = [
        aug.HorizontalFlip(p=0.5),
        aug.VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=8, p=0.7),
        aug.OneOf([
            aug.RandomBrightness(limit=0.1, p=1),
            aug.RandomContrast(limit=0.1, p=1),
            aug.RandomGamma(gamma_limit=(90, 110), p=1)
        ], p=0.5),
        aug.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5)
    ]

    return aug.Compose(aug_list, p=p)


def aug_heavy(prob=0.9):
    return aug.Compose([
    	aug.Flip(),
        aug.OneOf([
            aug.CLAHE(clip_limit=2, p=.5),
            aug.IAASharpen(p=.25),
            aug.IAAEmboss(p=.25),
        ], p=.35),
        aug.OneOf([
            aug.IAAAdditiveGaussianNoise(p=.3),
            aug.GaussNoise(p=.7),
            SaltPepperNoise(level_limit=0.0002, p=.7),
            aug.ISONoise(p=.3),
            ], p=.5),
        aug.OneOf([
            aug.MotionBlur(p=.2),
            aug.MedianBlur(blur_limit=3, p=.3),
            aug.Blur(blur_limit=3, p=.5),
        ], p=.4),
        aug.OneOf([
            aug.RandomContrast(p=.5),
            aug.RandomBrightness(p=.5),
            aug.RandomGamma(p=.5),
        ], p=.4),
        aug.ShiftScaleRotate(shift_limit=.0625, scale_limit=0.1, rotate_limit=12, p=.7),
        aug.OneOf([
            aug.GridDistortion(p=.2),
            aug.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=.2),
            aug.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=.2),
        ], p=.6),
        aug.HueSaturationValue(p=.5),
    ], p=prob)


def aug_medium(prob=1):
	return aug.Compose([
	    aug.Flip(),
	    aug.OneOf([
            aug.CLAHE(clip_limit=2, p=.5),
            aug.IAASharpen(p=.25),
            ], p=0.35),
	    aug.OneOf([
	        aug.RandomContrast(),
	        aug.RandomGamma(),
	        aug.RandomBrightness(),
	        ], p=0.3),
	    aug.OneOf([
	        aug.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
	        aug.GridDistortion(),
	        aug.OpticalDistortion(distort_limit=2, shift_limit=0.5),
	        ], p=0.3),
	    aug.ShiftScaleRotate(rotate_limit=12),
	    aug.OneOf([
            aug.GaussNoise(p=.35),
            SaltPepperNoise(level_limit=0.0002, p=.7),
            aug.ISONoise(p=.7),
            ], p=.5),
	    aug.Cutout(num_holes=3, p=.25),
	], p=prob)


def numpy_mask2dist(mask):
    one_hot_mask = numpy_make_one_hot(mask, 5)
    dist_map = one_hot2dist(one_hot_mask)
    return dist_map