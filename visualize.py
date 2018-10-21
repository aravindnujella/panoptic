# TODO: Clean up
# 

import numpy as np
import random
from PIL import Image
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

import colorsys


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
            image[:, :, c])
    return image


def extract_bbox(mask):
    m = np.where(mask != 0)
    # y1,x1,y2,x2. bottom right just outside of blah
    return np.min(m[0]), np.min(m[1]), np.max(m[0]) + 1, np.max(m[1]) + 1


def create_labelled_image(img, mask, class_name):
    img = Image.fromarray(img.astype(np.uint8))
    draw = ImageDraw.Draw(img)
    # y1, x1, y2, x2 = extract_bbox(masks)
    draw.rectangle(((0, 0), (40, 20)), fill="black")
    font = ImageFont.truetype("./data/Aaargh.ttf", 14)
    draw.text((5, 5), class_name, font=font, fill=(255, 255, 255))
    return img


# def visualize_targets(img, masks, class_response, base_impulse, config):
#     g = config.GRID_SHAPE

#     img = np.moveaxis(img, 0, 2)
#     img *= config.STD_PIXEL
#     img += config.MEAN_PIXEL
#     img *= 255
#     class_ids = np.argmax(class_response, 0).reshape(-1)
#     N = class_ids.shape[0]
#     response_colors = random_colors(N)
#     impulse_colors = random_colors(N)
#     for i in range(N):
#         masked_img = img.copy()
#         masked_img = apply_mask(masked_img, masks[i], response_colors[i])
#         masked_img = apply_mask(masked_img, base_impulse[i], impulse_colors[i])
#         masked_img = create_labelled_image(masked_img, masks[i], config.CLASS_NAMES[class_ids[i]])
#         masked_img.save("./results/" + str(i) + ".png", "PNG")


def visualize_targets(img, masks, class_ids, base_impulse, config):
    img = np.moveaxis(img, 0, 2)
    img *= config.STD_PIXEL
    img += config.MEAN_PIXEL
    img *= 255

    N = class_ids.shape[0]
    response_colors = random_colors(N)
    impulse_colors = random_colors(N)
    for i in range(N):
        masked_img = img.copy()
        masked_img = apply_mask(masked_img, masks[i], response_colors[i])
        masked_img = apply_mask(masked_img, base_impulse[i], impulse_colors[i])
        masked_img = create_labelled_image(masked_img, masks[i], config.CAT_NAMES[class_ids[i]])
        masked_img.save("./results/" + str(i) + ".png", "PNG")


def visualize_coco_data(img, masks, cat_ids, config):

    img = np.moveaxis(img, 0, 2)
    img *= config.STD_PIXEL
    img += config.MEAN_PIXEL
    img *= 255
    class_ids = np.argmax(class_response, 0).reshape(-1)
    N = class_ids.shape[0]
    response_colors = random_colors(N)
    impulse_colors = random_colors(N)
    for i in range(N):
        masked_img = img.copy()
        masked_img = apply_mask(masked_img, masks[i], response_colors[i])
        masked_img = apply_mask(masked_img, base_impulse[i], impulse_colors[i])
        masked_img = create_labelled_image(masked_img, masks[i], config.CAT_NAMES[class_ids[i]])
        masked_img.save("./results/" + str(i) + ".png", "PNG")



# Master function
# img = image as returned by the __getitem__ in [cwh] format
# masks = list of masks returned by __getitem__ in [wh] format;
# (in [0,1] continous interval) 
# cat_ids = cat_ids, len(cat_ids) == len(masks)
# impulses of same shape as masks
# config with which the __getitem__ is used
# name = name you want to give
# images saved will be 0_$name, 1_$name etc 

def visualize_data(img, impulses, masks, cat_ids, config, name):
    img = np.moveaxis(img, 0, 2)
    img *= config.STD_PIXEL
    img += config.MEAN_PIXEL
    img *= 255

    N = cat_ids.shape[0]
    response_colors = random_colors(N)
    impulse_colors = random_colors(N)
    for i in range(N):
        masked_img = img.copy()
        masked_img = apply_mask(masked_img, masks[i], response_colors[i])
        masked_img = apply_mask(masked_img, impulses[i], impulse_colors[i])
        masked_img = create_labelled_image(masked_img, masks[i], config.CAT_NAMES[int(cat_ids[i])])
        masked_img.save(("./results/%d_%s" ".png")%(i,name), "PNG")

