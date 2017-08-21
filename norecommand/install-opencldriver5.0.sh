#!/bin/bash
echo "please make sure kernel version is over 4.7"
sudo apt-get update
sudo apt-get upgrade
mkdir -p $HOME/opencl
cd $HOME/opencl
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/11396/SRB5.0_linux64.zip
unzip SRB5.0_linux64.zip
mkdir -p ./intel-opencl
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-cpu-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-devel-r5.0-63503.x86_64.tar.xz
tar -C intel-opencl -Jxf $HOME/opencl/intel-opencl-r5.0-63503.x86_64.tar.xz
sudo cp -R intel-opencl/* /
sudo ldconfig  
sudo apt-get install clinfo
clinfo
rm -rf $HOME/opencl