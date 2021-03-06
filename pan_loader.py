import numpy as np
import random
import torch
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
from PIL import Image
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import os
import json
import os.path
import colorsys


inf = float('inf')
nan = float('nan')


class CocoDataset(data.Dataset):

    def __init__(self, img_dir, seg_dir, ann, config):
        self.img_dir = img_dir
        self.seg_dir = seg_dir
        self.coco_data = self.index_annotations(ann)
        self.config = config
        self.catMap = self.build_cat_map()

    def index_annotations(self, ann):
        # create map with coco image index as key
        d = {}
        for i in ann['annotations']:
            coco_index = i['image_id']
            d[coco_index] = {
                'segments_info': i['segments_info'],
                'segments_file': i['file_name'],
                'image_id': i['image_id']
            }
        for i in ann['images']:
            coco_index = i['id']
            image_file = i['file_name']
            d[coco_index]['image_file'] = image_file

        return list(d.values())

    # coco category ids remapped to contiguous range(133+1)
    def build_cat_map(self):
        config = self.config
        coco_cat_ids = config.CAT_IDS
        catMap = {}
        for i in range(config.NUM_CATS):
            catMap[coco_cat_ids[i]] = i
        return catMap

    def __getitem__(self, index):
        try:
            # 0. read coco data as is; if no instances of required criteria then
            # return None and filter in collate
            data = self.load_data(index)

            # 1. remove unwanted data
            # 2. fixed resolution.
            # 3. Data Augmentation: skipped for now
            data = self.standardize_data(*data)

            # 4. Target generation:
            return self.generate_targets(*data)

        except:
            print("problem loading image index: %d %d" % (index, self.coco_data[index]['image_id']))
            return self[random.choice(range(len(self)))]

    def load_data(self, index):
        coco_data = self.coco_data
        config = self.config

        ignore_cat_ids = config.IGNORE_CAT_IDS

        ann = coco_data[index]
        image_id = ann['image_id']
        segments_info = ann['segments_info']
        segments_file = ann['segments_file']
        image_file = ann['image_file']
        img_file = Image.open(os.path.join(self.img_dir, image_file))
        img = np.array(img_file.convert('RGB'))
        img_file.close()

        instance_masks = []
        cat_ids = []

        coco_seg_file = Image.open(os.path.join(self.seg_dir,
                                                segments_file))
        coco_seg = np.array(coco_seg_file.convert('RGB'), dtype=np.uint8)
        coco_seg_file.close()

        seg_id = self.rgb2id(coco_seg)
        ignore_cat_ids = np.array(config.IGNORE_CAT_IDS)
        for s in segments_info:
            mask = np.where(seg_id == s['id'], 1, 0)
            iscrowd = s['iscrowd']
            cat_id = self.catMap[s['category_id']]
            if (s['iscrowd'] != 1) and (cat_id not in ignore_cat_ids) and np.sum(mask) > 144:
                instance_masks.append(mask)
                cat_ids.append(self.catMap[s['category_id']])

        cat_ids = np.array(cat_ids)
        instance_masks = np.array(instance_masks)

        return img, instance_masks, cat_ids

    def standardize_data(self, img, instance_masks, cat_ids):
        img, instance_masks = self.resize_data(img, instance_masks)

        # img, instance_masks, cat_ids = self.data_augment(img, instance_masks, cat_ids)
        return img, instance_masks, cat_ids

    def generate_targets(self, img, instance_masks, cat_ids):
        from scipy.ndimage import convolve

        impulses = []
        for mask in instance_masks:
            # single size lp filter, not multi scale targetting
            lp_filter = np.ones((13, 13))
            # convolve and check locations where the conv is maximum
            smooth_mask = convolve(mask, lp_filter, mode='constant', cval=0.0)
            idx = np.where(smooth_mask == np.max(smooth_mask))
            p, q = random.choice(list(zip(idx[0], idx[1])))

            iw, ih = self.config.IMPULSE_SIZE

            impulse = np.zeros_like(mask)
            impulse[p - iw:p + iw, q - ih:q + ih] = 1
            impulses.append(impulse)

        impulses = np.array(impulses)

        # kind of out of place. put in standardize_data??
        img = img / 255
        img -= self.config.MEAN_PIXEL
        img /= self.config.STD_PIXEL
        img = np.moveaxis(img, 2, 0)

        imgs = np.stack([img for i in range(impulses.shape[0])], 0)

        n = impulses.shape[0]
        # areas = np.sum(impulses, (1,2))
        # idx = np.argsort(areas)[-min(n,8):]
        l = np.arange(n)
        np.random.shuffle(l)
        idx = l[:min(n, 8)]

        return imgs[idx], impulses[idx], instance_masks[idx], cat_ids[idx]

    def rgb2id(self, color):
        return color[:, :,
                     0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]

    def resize_data(self, img, instance_masks):
        config = self.config

        w, h = config.WIDTH, config.HEIGHT
        img = self.resize_image(img, (w, h), "RGB")
        instance_masks = np.array(
            [self.resize_image(m, (w, h), "L") for m in instance_masks])

        return img, instance_masks

    def resize_image(self, img, size, mode):
        interpolation = {"RGB": Image.BICUBIC, "L": Image.NEAREST}[mode]
        img_obj = Image.fromarray(img.astype(np.uint8), mode)
        img_obj.thumbnail(size, interpolation)

        (w, h) = img_obj.size
        padded_img = Image.new(mode, size, "black")
        padded_img.paste(img_obj, ((size[0] - w) // 2, (size[1] - h) // 2))

        return np.array(padded_img)

    def __len__(self):
        return len(self.coco_data)


# custom collate function
# instead of using torch.cat, we return as list.
# this is because impulses, instance_masks are of variable size and
# cannot be concatenated. also if we dont do this, we have to send dummy masks
# or replicate image to fit dimensions of impulses,

# images, impulses, instance_masks, cat_ids

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    z = []
    n = len(batch[0])
    for i in range(n):
        z.append(torch.cat([torch.from_numpy(it[i]) for it in batch], 0))
    return z


def get_loader(img_dir, seg_dir, ann, config):
    coco_dataset = CocoDataset(img_dir, seg_dir, ann, config)
    data_loader = data.DataLoader(dataset=coco_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  collate_fn=collate_fn,
                                  shuffle=True,
                                  pin_memory=config.PIN_MEMORY,
                                  num_workers=config.NUM_WORKERS
                                  )
    return data_loader

if __name__ == '__main__':
    import base_config
    import visualize

    config = base_config.Config()

    data_dir = "/home/aravind/dataset/"
    ann_dir = data_dir + "annotations/panoptic/"

    # val_img_dir = data_dir + "train2017/"
    # val_seg_dir = ann_dir + "panoptic_train2017/"
    # val_ann_json = ann_dir + "panoptic_train2017.json"

    val_img_dir = data_dir + "val2017/"
    val_seg_dir = ann_dir + "panoptic_val2017/"
    val_ann_json = ann_dir + "panoptic_val2017.json"

    with open(val_ann_json, "r") as f:
        val_ann = json.load(f)

    val_dataset = CocoDataset(val_img_dir, val_seg_dir, val_ann, config)
    l = len(val_dataset)
    index = random.choice(list(range(len(val_dataset))))
    # index = 19726
    data = [torch.from_numpy(it) for it in val_dataset[index]]
    # imgs, impulses, instance_masks, cat_ids = val_dataset[index]
    visualize.visualize_targets(data, config, "new")
    print(index)

    # val_loader = get_loader(val_img_dir, val_seg_dir, val_ann, config)
    # for i, data in enumerate(val_loader, 0):
    #     print("Batch:%d" % i)
    #     visualize.visualize_targets(data, config, "hehe")
    #     input()
