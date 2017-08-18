#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
ROOT=$DIR/..
BatchSizes=( 1 2 4 8 16 32 64 128 )
TimeStamp=$(date +"%Y%m%d%H%M")
ModelDir=$ROOT/models/Baidu
BenchDir=$ROOT/models/Baidu/${TimeStamp}

Protos=( AlexNet-merged VGG16-merged ResNet101-merged GoogleNetV2-merged GoogleNetV1-merged )
#Protos=( AlexNet-merged VGG16-merged ResNet101-merged GoogleNetV2-merged GoogleNetV1-merged )

echo "Create benchmark dir ${BenchDir}"
mkdir -p ${BenchDir}
echo "Remove spatial kernel cache"
rm .spatialkernels/* -f

for proto in ${Protos[@]}; do
  for BS in ${BatchSizes[@]} ; do
    echo "Generate model file for ${proto} with batch size ${BS}"
    cat ${ModelDir}/${proto}.prototxt |sed  "s/input_\(.*\) 1/input_\1 ${BS}/g" > ${BenchDir}/${proto}-${BS}.prototxt
    echo "Warming up and tuning"
    $ROOT/build/tools/caffe time --gpu 0 --iterations 10 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}.log 2>&1
    echo "Benchmarking"
    $ROOT/build/tools/caffe time --gpu 0 --iterations 20 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt >> ${BenchDir}/${proto}-${BS}.log 2>&1

    echo "Generate model file for ${proto} with batch size ${BS} and do HALF_ALL mode"
    #cat ${BenchDir}/${proto}-${BS}.prototxt |sed  's/#half_precision_mode/half_precision_mode/g' > ${BenchDir}/${proto}-${BS}-fp16.prototxt
    echo "Warming up and tuning HALF_ALL"
    $ROOT/build/tools/caffe-fp16.bin time --gpu 0 --iterations 10 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}-fp16.log 2>&1
    echo "Benchmarking HALF_ALL"
    $ROOT/build/tools/caffe-fp16.bin time --gpu 0 --iterations 20 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt >> ${BenchDir}/${proto}-${BS}-fp16.log 2>&1

  done;
done;


#    $ROOT/build/tools/caffe time --gpu 0 --iterations 10 --phase TEST --model $proto 2>&1 | python $ROOT/scripts/process_bench_log.py $proto


#export PYTHONPATH=$ROOT/python
#export PATH=.:$PATH
#for proto in ${alexnet_files[@]} ${googlenet_files[@]} ${vgg_files[@]} ; do
#    $ROOT/build/tools/caffe time --gpu 0 --iterations 10 --phase TEST --model $proto 2>&1 | python $ROOT/scripts/process_bench_log.py $proto
#done

