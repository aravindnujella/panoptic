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


[2018-10-07 18:50]
> its proven that simultaneous instance and semantic can be done but current modelling is not the best the way to do it
> current set of targets: 1) implement loss.py 2) implement train.py 3) verify model correctness in model.py


