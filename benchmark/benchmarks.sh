#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}"  )" && pwd  )"
ROOT=$DIR/
BatchSizes=( 1 2 4 8 16 32 64 128 )
TimeStamp=$(date +"%Y%m%d%H%M")
ModelDir=$ROOT
BenchDir=$ROOT/${TimeStamp}

Protos=( AlexNet-merged VGG16-merged ResNet101-merged GoogleNetV2-merged GoogleNetV1-merged)

echo "Create benchmark dir ${BenchDir}"
mkdir -p ${BenchDir}
echo "Remove spatial kernel cache"
rm .spatialkernels/* -f

printf "\n\nStart Test in Time:%s\n" ${TimeStamp}  >> result.txt
printf "%-30s" "---------------------------- "  >> result.txt
for BS in ${BatchSizes[@]} ; do
	printf "%-10s" ${BS} >> result.txt
done;
echo '' >> result.txt

printf "\n\nStart Test in Time:%s\n" ${TimeStamp}  >> result-fp16.txt
printf "%-30s" "---------------------------- "  >> result-fp16.txt
for BS in ${BatchSizes[@]} ; do
	printf "%-10s" ${BS} >> result-fp16.txt
done;
echo '' >> result-fp16.txt
	
for proto in ${Protos[@]}; do
	printf "%-30s" ${proto}  >> result.txt
	printf "%-30s" ${proto}  >> result-fp16.txt
	
  for BS in ${BatchSizes[@]} ; do
    echo "Generate model file for ${proto} with batch size ${BS}"
    cat ${ModelDir}/${proto}.prototxt |sed  "s/input_\(.*\) 1/input_\1 ${BS}/g" > ${BenchDir}/${proto}-${BS}.prototxt
    echo "Warming up and tuning"
    $HOME/code/caffe/build/tools/caffe time --gpu 0 --iterations 10 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}.log 2>&1
    echo "Benchmarking"
    $HOME/code/caffe/build/tools/caffe time --gpu 0 --iterations 20 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}.log 2>&1
		result=`grep -o -E 'Average Forward pass: [0-9.]*' ${BenchDir}/${proto}-${BS}.log`
		i2=`echo $result| awk -F ": " '{print $2}'`
		c=`awk -v x=1000.0 -v y=$i2 -v b=${BS} 'BEGIN{printf "%.1f\n",x*b/y}'`
		printf "%-10.1f" ${c}>> result.txt
		
    echo "Generate model file for ${proto} with batch size ${BS} and do HALF_ALL mode"
    #cat ${BenchDir}/${proto}-${BS}.prototxt |sed  's/#half_precision_mode/half_precision_mode/g' > ${BenchDir}/${proto}-${BS}-fp16.prototxt
    echo "Warming up and tuning HALF_ALL"
    $HOME/code/caffe/build/tools/caffe-fp16.bin time --gpu 0 --iterations 10 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}-fp16.log 2>&1
    echo "Benchmarking HALF_ALL"
    $HOME/code/caffe/build/tools/caffe-fp16.bin time --gpu 0 --iterations 20 --phase TEST --model ${BenchDir}/${proto}-${BS}.prototxt > ${BenchDir}/${proto}-${BS}-fp16.log 2>&1
		result=`grep -o -E 'Average Forward pass: [0-9.]*' ${BenchDir}/${proto}-${BS}-fp16.log`
		i2=`echo $result| awk -F ": " '{print $2}'`
		c=`awk -v x=1000.0 -v y=$i2 -v b=${BS} 'BEGIN{printf "%.1f\n",x*b/y}'`
		printf "%-10.1f" ${c}>> result-fp16.txt
		
  done;
	echo '' >> result.txt
	echo '' >> result-fp16.txt
done;

rm -rf ${BenchDir}

