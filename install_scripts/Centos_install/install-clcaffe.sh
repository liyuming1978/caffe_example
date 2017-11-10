#!/bin/bash
###yum install########################
##if proxy, please set proxy at /etc/yum.conf  with proxy=http://XXX:XXX
sudo yum install -y epel-release
sudo yum makecache
sudo yum install -y protobuf-devel leveldb-devel snappy-devel opencv-devel hdf5-devel protobuf-compiler boost-devel
sudo yum install -y gflags-devel glog-devel lmdb-devel  libstdc++ git vim
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y ocl-icd-devel opencl-headers

#liyuming mark: numpy will update atlas to 3.10, it will make cmake error for caffe,
#so if you want python, you must install cent6 package for numpy ... by yourself
#sudo yum install -y python-pip python-devel
#sudo pip install --upgrade pip -i http://mirrors.aliyun.com/pypi/simple
#sudo pip install --upgrade virtualenv -i http://mirrors.aliyun.com/pypi/simple
#sudo pip install numpy -i http://mirrors.aliyun.com/pypi/simple
#sudo pip install scikit-image -i http://mirrors.aliyun.com/pypi/simple
#sudo pip install protobuf -i http://mirrors.aliyun.com/pypi/simple

##atlas must use 3.8.4##
sudo rpm -ivh atlas-3.8.4-2.el6.x86_64.rpm
sudo rpm -ivh atlas-devel-3.8.4-2.el6.x86_64.rpm

##upgrade cmake
mkdir -p $HOME/code
cd $HOME/code
mkdir -p cmake
wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz    
tar xvf cmake-3.6.2.tar.gz && cd cmake-3.6.2/
./bootstrap
gmake
sudo gmake install

#copy opencv cmake
sudo cp /usr/lib64/cmake/OpenCV/OpenCVConfig.cmake /usr/lib64/cmake/OpenCV/OpenCVModules.cmake

#clean and build
rm -rf $HOME/code/caffe
rm -rf $HOME/code/viennacl-dev
rm -rf $HOME/code/isaac
###intall viennacl-dev########################
mkdir -p $HOME/code
cd $HOME/code
git clone https://github.com/viennacl/viennacl-dev.git
cd viennacl-dev
mkdir build && cd build
cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$HOME/local - DOPENCL_LIBRARY=/opt/intel/opencl/libOpenCL.so ..
make -j4
make install
cd $HOME/code
git clone https://github.com/intel/isaac
cd isaac
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/local .. && make -j4
make install

###intall caffe########################
mkdir -p $HOME/code
cd $HOME/code
git clone https://github.com/01org/caffe.git
cd caffe
git checkout inference-optimize
mkdir build && cd build
export ISAAC_HOME=$HOME/local
export Atlas_ROOT_DIR=/usr/lib64/atlas

cmake .. -DUSE_GREENTEA=ON -DUSE_CUDA=OFF -DUSE_INTEL_SPATIAL=ON -DBUILD_docs=0 -DUSE_ISAAC=ON -DViennaCL_INCLUDE_DIR=$HOME/local/include -DOPENCL_LIBRARIES=/opt/intel/opencl/libOpenCL.so -DOPENCL_INCLUDE_DIRS=/opt/intel/opencl/include

make all -j4
make install

####add env############################
grep -q "export VIENNACL_CACHE_PATH"  $HOME/.bashrc
if  [ $?  -ne  0 ] 
then # if not exist
	echo "" >> $HOME/.bashrc
	echo "export VIENNACL_CACHE_PATH=${HOME}/.cache/clcaffe_cache/viennacl/" >> $HOME/.bashrc
fi

####install clpeak############################
cd $HOME/code
git clone https://github.com/krrishnarraj/clpeak.git
cd clpeak
mkdir build && cd build
cmake ..
make -j4
./build/clpeak
