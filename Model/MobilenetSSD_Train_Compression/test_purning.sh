#!/bin/sh
latest=example_purning/mobilenet_iter_120000_pruning.caffemodel
#latest=$(ls -t snapshot/*.caffemodel | head -n 1)
if test -z $latest; then
	exit 1
fi
../../build/tools/caffe train -solver="solver_test_pruning.prototxt" \
--weights=$latest \
-gpu 0
