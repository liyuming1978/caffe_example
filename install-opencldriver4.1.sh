#!/bin/bash
echo "please make sure kernel version is over 4.7"
sudo apt-get update
sudo apt-get upgrade
mkdir -p $HOME/opencl
cd $HOME/opencl
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/11396/SRB4.1_linux64.zip
unzip SRB4.1_linux64.zip
mkdir -p ./intel-opencl
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-cpu-r4.1-61547.x86_64.tar.xz
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-devel-r4.1-61547.x86_64.tar.xz
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-r4.1-61547.x86_64.tar.xz
sudo cp -R intel-opencl/* /
sudo ldconfig  
sudo apt-get install clinfo
clinfo
rm -rf $HOME/opencl