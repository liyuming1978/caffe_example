cmake_minimum_required(VERSION 2.8.8)

include(cmake/FindFFmpeg.cmake)
include(cmake/FindLibYuv.cmake)
#include(cmake/FindSDL.cmake)
include(cmake/FindXCB.cmake)

set(Caffe_DIR /home/lym/work/code/caffe/build/)
set(ViennaCL_INCLUDE_DIR /home/lym/local/include/) 
find_package(Caffe)
include_directories(
	${Caffe_INCLUDE_DIRS}
	${Caffe_DIR}/install/include/
	${Caffe_DIR}/../include/
	${ViennaCL_INCLUDE_DIR}
	${FFMPEG_INCLUDE_DIRS}
	${LIBYUV_INCLUDE_DIR}
	${SDL_INCLUDE_DIR}
	${XCB_INCLUDE_DIRS}
)
add_definitions(${Caffe_DEFINITIONS})
add_compile_options(-std=c++11)
add_executable(mobilenet_ssd mobilenet_ssd.cpp kernels.cpp detector.cpp videodec.cpp XCBShow.cpp)
target_link_libraries(mobilenet_ssd ${Caffe_LIBRARIES} ${FFMPEG_LIBRARIES} -Wl,-Bsymbolic ${LIBYUV_LIBRARY} ${SDL_LIBRARY} ${XCB_LIBRARIES})
