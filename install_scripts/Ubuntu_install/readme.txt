please use >ubuntu16.04 (kernel>=3.7.10)
1. install-opencldriver4.1.sh
2. install-clcaffe.sh
done

if /usr/bin/ld: /usr/local/lib/libgflags.a(gflags.cc.o): relocation R_X86_64_32S against `.rodata' can not be used when making a shared object; recompile with -fPIC

please rebuild glag with  
cmake .. -DBUILD_SHARED_LIBS=ON
make -j2 && sudo make -j2 install
