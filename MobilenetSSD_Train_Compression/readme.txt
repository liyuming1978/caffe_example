1.reference:
get from https://github.com/chuanqi305/MobileNet-SSD and opitmization(caffe) from https://github.com/yonghenglh6/DepthwiseConvolution/tree/master/caffe

2.how:
A.please reference https://github.com/chuanqi305/MobileNet-SSD to prepare data
B.use transferTypeToDepthwiseConvolution.py to change to DepthwiseConvolution
C.use train.sh to train, note: I just get 72.3456 while chuanqi305 is 72.7 (maybe DepthwiseConvolution?)

3.how to compress (need more research)
please use model_pruning.py to compress, now is 71.0498 and compression is little
I think, mabye it should compress mobilenet firstly and then train this mobilenet with imagenet and your data
and then, train the ssd with compressed mobilenet model.