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



[2018-10-14 10:05]
memory required is too high. may be use wing conv? i.e, add a linear layer. 

4*f -> f


[2018-10-14 14:35]


[2018-10-17 01:12]
(2 iresnets with wings)
can train 12 instances @ 224*224 resolution
unable to train even 1 instance @ 448*448 resolution


[2018-10-18 23:38]
turns out there was issue with del(outs) in train loop. placing a delete to clear memory makes it possible to ...

[2018-10-18 23:43]
with clever deletes, i was able to run upto batch size of 4 on 448*448 resolution on 2 * resnet50 with full params. could clever freeing up of data save me?



[2018-10-21 22:47]
TODO: implement dataparallel :)

[2018-10-22 00:47]
"network not learning anything; possible bug locns: iresnet, mb, cb, lossfn, not letting copy_filters train, many inputs from same image,;"


[2018-10-22 00:53]
iresnet working fine 