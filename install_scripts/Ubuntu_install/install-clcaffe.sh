#!/bin/bash
###apt-get install########################
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler python-opencv
sudo apt-get install -y --no-install-recommends libboost-all-dev
sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y libstdc++6  libatlas-base-dev git cmake vim
sudo apt-get install -y python-pip python-dev build-essential 
sudo apt-get install -y ocl-icd-opencl-dev opencl-headers ocl-icd-libopencl1
sudo pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip install --upgrade virtualenv -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple

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
git checkout 2a73199d862b63b41497098fa926f0e35ab41fc2
mkdir build && cd build
export ISAAC_HOME=$HOME/local

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
rm -rf ${HOME}/.cache/clcaffe_cache/

####install clpeak############################
cd $HOME/code
git clone https://github.com/krrishnarraj/clpeak.git
cd clpeak
mkdir build && cd build
cmake ..
make -j4
./clpeak
