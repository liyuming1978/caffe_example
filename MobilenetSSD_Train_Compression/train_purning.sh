#!/bin/sh
if ! test -f example_purning/MobileNetSSD_train_pruning_ok.prototxt ;then
	echo "error: example_new/MobileNetSSD_train_pruning_ok.prototxt does not exist."
	echo "please use the gen_model.sh to generate your own model."
        exit 1
fi
mkdir -p snapshot_pruning
../../build/tools/caffe train -solver="solver_train_pruning.prototxt" \
-weights="example_purning/mobilenet_iter_120000_pruning.caffemodel" \
-gpu 0 
