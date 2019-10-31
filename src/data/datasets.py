import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2

from data_utils import *
import policies
import augmentation_transforms


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
standardize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class BalanceCropSurfaceTrainSet(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1, resize=None):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_train()
        self.sample_type = sample_type
        self.resize = resize
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = cv2.imread(os.path.join(self.data_dir, ids[i]), 1)
        if self.resize:
            image = cv2.resize(image, (int(self.resize[0]*6.25), self.resize[0]), interpolation=cv2.INTER_LINEAR)
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600), (self.resize[0], int(self.resize[0]*6.25)))
            image, mask = do_width_random_crop(image, mask, self.resize[1])
        else:
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600))
            image, mask = do_width_random_crop(image, mask, 512)
        data_dict = {"image": image, "mask": mask}

        augmented = self.augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        dist_map = torch.tensor(numpy_mask2dist(mask.astype(int)), dtype=torch.float32).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        
        return image, mask, dist_map


class BalanceCropTrainSet(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1, resize=None):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_medium()
        self.sample_type = sample_type
        self.resize = resize
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = cv2.imread(os.path.join(self.data_dir, ids[i]), 1)
        if self.resize:
            image = cv2.resize(image, (int(self.resize[0]*6.25), self.resize[0]), interpolation=cv2.INTER_LINEAR)
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600), (self.resize[0], int(self.resize[0]*6.25)))
            image, mask = do_width_random_crop(image, mask, self.resize[1])
        else:
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600))
            image, mask = do_width_random_crop(image, mask, 512)
        data_dict = {"image": image, "mask": mask}

        augmented = self.augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        
        return image, mask


class BalanceTrainSet(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_train(p=1, shift_limit=0., scale_limit=0.)
        self.sample_type = sample_type
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = cv2.imread(os.path.join(self.data_dir, ids[i]), 1)
        mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600))
        image, mask = do_random_crop_rescale(image, mask, 1568, 224)
        data_dict = {"image": image, "mask": mask}

        augmented = self.augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        
        return image, mask
    

class TrainSet(Dataset):
    """Training dataset"""

    def __init__(self, train_ids, train_df, data_dir):
        self.train_ids = train_ids
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_train()

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):       
        image = cv2.imread(os.path.join(self.data_dir, self.train_ids[idx]), 1)
        mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == self.train_ids[idx], 'EncodedPixels'], (256, 1600))
        data_dict = {"image": image, "mask": mask}

        augmented = self.augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        return image, mask


    
class ValSet(Dataset):
    """Training dataset"""

    def __init__(self, val_ids, valid_df, data_dir, resize=None):
        self.val_ids = val_ids
        self.valid_df = valid_df
        self.data_dir = data_dir
        self.resize = resize

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.val_ids[idx]), 1)
        mask = build_masks(self.valid_df.loc[self.valid_df['ImageId'] == self.val_ids[idx], 'EncodedPixels'], (256, 1600))

        if self.resize:
            image = cv2.resize(image, self.resize[::-1])

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        return image, mask


# Classification Datasets
class BalanceTrainSetCLS(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_train()
        self.sample_type = sample_type
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = cv2.imread(os.path.join(self.data_dir, ids[i]), 1)
        label = build_labels(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'])
        data_dict = {"image": image}

        augmented = self.augmentor(**data_dict)
        image = augmented["image"]

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        label = torch.from_numpy(label).to(DEVICE)
        
        return image, label


class ValSetCLS(Dataset):
    """Training dataset"""

    def __init__(self, val_ids, valid_df, data_dir):
        self.val_ids = val_ids
        self.valid_df = valid_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.val_ids[idx]), 1)
        label = build_labels(self.valid_df.loc[self.valid_df['ImageId'] == self.val_ids[idx], 'EncodedPixels'])

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        label = torch.from_numpy(label).to(DEVICE)
        return image, label



class BalanceTrainSetRandAugCLS(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.aug_policies = policies.randaug_policies()
        self.sample_type = sample_type
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):

        chosen_policy = self.aug_policies[np.random.choice(len(self.aug_policies))]
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = Image.open(os.path.join(self.data_dir, ids[i]))
        label = build_labels(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'])

        image = augmentation_transforms.apply_policy(chosen_policy, image)
        image = np.array(image)

        if random.uniform(0, 1) > 0.5:
            if random.uniform(0, 1) > 0.25:
                image = do_random_salt_pepper_noise(image, noise=random.uniform(0, 0.0002))
            else:
                image = augmentation_transforms.cutout_numpy(image, size=10)
        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        label = torch.from_numpy(label).to(DEVICE)
        
        return image, label


class ValSetPILCLS(Dataset):
    """Training dataset"""

    def __init__(self, val_ids, valid_df, data_dir):
        self.val_ids = val_ids
        self.valid_df = valid_df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, idx):
        # image = cv2.imread(os.path.join(self.data_dir, self.val_ids[idx]), 1)
        image = Image.open(os.path.join(self.data_dir, self.val_ids[idx]))
        label = build_labels(self.valid_df.loc[self.valid_df['ImageId'] == self.val_ids[idx], 'EncodedPixels'])

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        label = torch.from_numpy(label).to(DEVICE)
        return image, label


class BalanceTrainSetRandAugCLSUnsup(Dataset):
    """Training dataset"""

    def __init__(self, test_ids, data_dir, length):
        self.test_ids = test_ids
        self.data_dir = data_dir
        self.aug_policies = policies.randaug_policies_unsup()
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.test_ids)-1)

        chosen_policy = self.aug_policies[np.random.choice(len(self.aug_policies))]
        
        image_easy = Image.open(os.path.join(self.data_dir, self.test_ids[idx]))
        if random.uniform(0, 1) > 0.5:
            image_easy = image_easy.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > 0.5:
            image_easy = image_easy.transpose(Image.FLIP_TOP_BOTTOM)

        image_hard = image_easy.copy()
        image_hard = augmentation_transforms.apply_policy(chosen_policy, image_hard)
        image_hard = np.array(image_hard)

        if random.uniform(0, 1) > 0.5:
            if random.uniform(0, 1) > 0.25:
                image_hard = do_random_salt_pepper_noise(image_hard, noise=random.uniform(0, 0.0002))
            else:
                image_hard = augmentation_transforms.cutout_numpy(image_hard, size=10)

        image_easy = standardize(transforms.functional.to_tensor(image_easy)).to(DEVICE)
        image_hard = standardize(transforms.functional.to_tensor(image_hard)).to(DEVICE)
        
        return image_easy, image_hard


class BalanceCropTrainSetCS(Dataset):
    """Training dataset"""

    def __init__(self, ids_list, train_df, data_dir, sample_type=1, resize=None):
        self.ids_list = ids_list
        self.train_df = train_df
        self.data_dir = data_dir
        self.augmentor = aug_medium()
        self.sample_type = sample_type
        self.resize = resize
        self.iter_list = np.zeros(5, dtype=np.int32)

    def __len__(self):
        return len(self.ids_list[0])

    def __getitem__(self, idx):
        
        if self.sample_type == 0:
            oe = idx % 8
            cur_cls = oe//2+1 if oe%2==0 else 0
        elif self.sample_type == 1:
            cur_cls = idx % 5
        else:
            raise ValueError("Invalid sample_type: please choose from [0, 1]")

        ids = self.ids_list[cur_cls]
        i = self.iter_list[cur_cls] % len(ids)
        self.iter_list[cur_cls] += 1
        
        image = cv2.imread(os.path.join(self.data_dir, ids[i]), 1)
        if self.resize:
            image = cv2.resize(image, (int(self.resize[0]*6.25), self.resize[0]), interpolation=cv2.INTER_LINEAR)
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600), (self.resize[0], int(self.resize[0]*6.25)))
            image, mask = do_width_random_crop(image, mask, self.resize[1])
        else:
            mask = build_masks(self.train_df.loc[self.train_df['ImageId'] == ids[i], 'EncodedPixels'], (256, 1600))
            image, mask = do_width_random_crop(image, mask, 512)
        data_dict = {"image": image, "mask": mask}

        augmented = self.augmentor(**data_dict)
        image, mask = augmented["image"], augmented["mask"]

        label = torch.zeros(4, dtype=torch.float)
        for c in range(1, 5):
            if (mask == c).sum() > 0:
                label[c-1] = 1

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        label = label.to(DEVICE)
        
        return image, mask, label


class ValSetCS(Dataset):
    """Training dataset"""

    def __init__(self, val_ids, valid_df, data_dir, resize=None):
        self.val_ids = val_ids
        self.valid_df = valid_df
        self.data_dir = data_dir
        self.resize = resize

    def __len__(self):
        return len(self.val_ids)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.val_ids[idx]), 1)
        mask = build_masks(self.valid_df.loc[self.valid_df['ImageId'] == self.val_ids[idx], 'EncodedPixels'], (256, 1600))

        if self.resize:
            image = cv2.resize(image, self.resize[::-1])

        label = torch.zeros(4, dtype=torch.float)
        for c in range(1, 5):
            if (mask == c).sum() > 0:
                label[c-1] = 1

        image = standardize(transforms.functional.to_tensor(image)).to(DEVICE)
        mask = torch.from_numpy(mask).long().to(DEVICE)
        label = label.to(DEVICE)
        
        return image, mask, label