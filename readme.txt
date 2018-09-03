TODO: A basic proof that your modelling (simultaneous semantic and instance segmentation) is possible
> split stuff islands
> implement convolutional targetting


[2018-09-02 18:25]
blue print for target generation:
coco data -> image, class_ids, segment image
segment image -> remove crowds
segment image -> segments
segments -> filter some classes
resize(image, segments) 