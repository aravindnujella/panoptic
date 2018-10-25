import numpy as np

class Config():
    NAME = "InSegm"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    NUM_WORKERS = 16
    PIN_MEMORY = True
    VALIDATION_STEPS = 20
    MAX_INSTANCES = 8

    CAT_NAMES = ['BG'] + [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    CAT_IDS = [0] + [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
        85, 86, 87, 88, 89, 90, 92, 93, 95, 100, 107, 109, 112, 118, 119, 122,
        125, 128, 130, 133, 138, 141, 144, 145, 147, 148, 149, 151, 154, 155,
        156, 159, 161, 166, 168, 171, 175, 176, 177, 178, 180, 181, 184, 185,
        186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
        200
    ]
    IGNORE_CAT_NAMES = ['BG'] 
    MEAN_PIXEL = np.array(
        [0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, -1)
    STD_PIXEL = np.array(
        [0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, -1)

    IMPULSE_SIZE = (16, 16)
    MIN_STUFF_AREA = 10 * 10

    def __init__(self):
        self.WIDTH = 32 * 7
        self.HEIGHT = 32 * 7
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.IMAGE_SHAPE = (self.WIDTH, self.HEIGHT, 3)
        # 133 + 1 in panoptic
        self.NUM_CATS = len(self.CAT_NAMES)
        self.IGNORE_CAT_IDS = [
            self.CAT_NAMES.index(c) for c in self.IGNORE_CAT_NAMES
        ]

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

if __name__ == '__main__':
    config = Config()
    config.display()