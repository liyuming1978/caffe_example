@echo off
@setlocal EnableDelayedExpansion

echo must install openclsdk,python27,git,cmake,vs 2015 for desktop (not for win10)
echo if download tar from https://github.com/willyd/caffe-builder/releases/ (see WindowsDownloadPrebuiltDependencies.cmake)fail, please copy it to build and delete libraries dir

for /f "delims=" %%t in ('python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"') do set py_path_str=%%t

cd %~sdp0
git clone https://github.com/dlfcn-win32/dlfcn-win32
cd dlfcn-win32
cmake -G "Visual Studio 14 2015 Win64" .
cmake --build . --config Release
cd %~sdp0
::git clone https://github.com/ptillet/isaac.git (isaac build wrong, please use intel isaac)
git clone https://github.com/intel/isaac.git   
cd isaac
mkdir build
cd build
cmake -G "Visual Studio 14 2015 Win64" ..
cmake --build . --config Release

cd %~sdp0
git clone https://github.com/01org/caffe.git
cd caffe
git checkout inference-optimize
git pull
git clone https://github.com/viennacl/viennacl-dev.git

set BUILD_PYTHON=1
set BUILD_PYTHON_LAYER=1
set USE_INTEL_SPATIAL=1
set USE_GREENTEA=1
set USE_ISAAC=1
set RUN_TESTS=0
set RUN_INSTALL=1
::set PYTHON_VERSION=3
call scripts\build_win.cmd

cd %~sdp0
for /f  "tokens=1,2 delims==" %%b in (%~sdp0\caffe\build\CMakeCache.txt) do (
	if "%%b"=="OPENCL_LIBRARIES:FILEPATH" set OPENCL_LIBRARIES=%%c
	if "%%b"=="PYTHON_LIBRARY:FILEPATH" set PYTHON_LIBRARY=%%c
)
set OPENCL_LIBRARIES=%OPENCL_LIBRARIES:/=\%
set PYTHON_LIBRARY=%PYTHON_LIBRARY:/=\%

if not exist "%~sdp0\caffe\build\install\" (
	echo do not find caffe build
)else (
	echo "build finish"
)

