cmake_minimum_required(VERSION 2.8.8)
set(Caffe_DIR /home/lym/work/code/caffe/build/)
set(ViennaCL_INCLUDE_DIR /home/lym/local/include/) 
find_package(Caffe)
include_directories(
	${Caffe_INCLUDE_DIRS}
    ${Caffe_DIR}/install/include/
	${Caffe_DIR}/../include/
    ${ViennaCL_INCLUDE_DIR}
)
add_definitions(${Caffe_DEFINITIONS})
add_executable(classification_fp16 classification_fp16.cpp)
target_link_libraries(classification_fp16 ${Caffe_LIBRARIES})