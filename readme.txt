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


[2018-10-22 14:26]
some problem with mask branch. 
> is it using too much memory?
> mask_layer seems to have some issue with memory

[2018-10-22 14:28]
> problem is with mask layer using bias=None instead of bias=False


[2018-10-23 00:17]
TODO:
> move unpack images from model.py. it is not the memory usage bottleneck
> change collate_fn in pan_loader.py to do it the pytorch way. this opens up possibility of multi gpu
> change the cudify_list etc.
> the model is still not learning anything.



[2018-10-23 16:04]
> seems like issue is isufficient number of parameters?
> it could also be that im training on big instances which are easy for trivial soln.
  (just predict everything as +ve)


[2018-10-23 16:08]
> yup. degenerate network


[2018-10-24 03:00]
Description of the model tried:

why can't I learn anything with the resnet model?
potential locations for mistakes:
1) iresnet could have some errors -> I've manually verified that there are no mistakes.
2) under representational power of the chosen model
3) under training
4) bad loss function: mask_loss, class_loss
5) bad design of class branch and loss branch: not taking translation equivariance into account
6) bad ordering of training data: several images from same image => batchnorm could behave in ...? way.
7) 


[2018-10-26 07:02]
> renamed classes and variable names to be separate in iresnet.py
> turns out that running_var, running mean as parameters/ buffers doesn't make difference; However we register them as buffers that don't require grad
> It requires about 6k images to see the results of training
> memory usage (4.5GB): 1 image, max_instances = 16, 224*224, resnet50 is partially frozen(pretrained weights are all frozen), only mask_branch is trained



[2018-11-02 06:30]
TODO:
> (L) cleanup: remove limit on min size of inputs (panloader), 
> (H) learn for non trivial inputs
> (L) clean up squeeze and unsqueeze of instance_masks, instances. 
    => rework how visualize  deals with them 
> (L) rework so that entire workflow uses hyperparameters as defined in config instead of local ones.
> (H) add residual in residual_block
> (H) unable to learn even trivial case if wingi is enabled



[2018-11-03 05:29]
> git diff HEAD^ -- ":!*.ipynb" to ignore ipynb from git diff


[2018-11-03 08:31]
jupyter lab


[2018-11-03 11:49]
> first non trivial model. still suffering from similar deficiencies as vgg model