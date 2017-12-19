@echo off
@setlocal EnableDelayedExpansion

echo must install openclsdk,python27,git,cmake,vs 2015 for desktop (not for win10)
echo if download tar from https://github.com/willyd/caffe-builder/releases/ (see WindowsDownloadPrebuiltDependencies.cmake)fail, please copy it to build and delete libraries dir

for /f "delims=" %%t in ('python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"') do set py_path_str=%%t

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
	echo "start copying"

	:: copy lib and include
	copy /y %~sdp0\dlfcn-win32\Release\dl.dll %~sdp0\caffe\build\install\bin
	copy /y %~sdp0\isaac\build\lib\Release\isaac.dll %~sdp0\caffe\build\install\bin
	copy /y %~sdp0\dlfcn-win32\Release\dl.dll %~sdp0\caffe\build\install\python\caffe
	copy /y %~sdp0\isaac\build\lib\Release\isaac.dll %~sdp0\caffe\build\install\python\caffe
	copy /y %~sdp0\dlfcn-win32\Release\dl.dll %~sdp0\caffe\build\tools\Release
	copy /y %~sdp0\isaac\build\lib\Release\isaac.dll %~sdp0\caffe\build\tools\Release

	copy /y %~sdp0\caffe\build\libraries\lib\boost_python-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_system-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_thread-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_filesystem-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_regex-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_chrono-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_date_time-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\boost_atomic-vc140-mt-1_61.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\glog.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\gflags.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\libprotobuf.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\caffehdf5_hl.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\caffehdf5.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\caffezlib.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\lmdb.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\leveldb.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\snappy_static.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\lib\libopenblas.dll.a %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\x64\vc14\lib\opencv_highgui310.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\x64\vc14\lib\opencv_videoio310.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\x64\vc14\lib\opencv_imgcodecs310.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\x64\vc14\lib\opencv_imgproc310.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\caffe\build\libraries\x64\vc14\lib\opencv_core310.lib %~sdp0\caffe\build\install\lib
	copy /y %~sdp0\isaac\build\lib\Release\isaac.lib %~sdp0\caffe\build\install\lib

	copy /y "%OPENCL_LIBRARIES%" %~sdp0\caffe\build\install\lib
	copy /y "%PYTHON_LIBRARY%" %~sdp0\caffe\build\install\lib

	xcopy %~sdp0\caffe\build\libraries\include %~sdp0\caffe\build\install\include /s /h /c /y 
	move /y %~sdp0\caffe\build\install\include\boost-1_61\boost %~sdp0\caffe\build\install\include\boost 
	rmdir /s/q %~sdp0\caffe\build\install\include\boost-1_61
	mkdir %~sdp0\caffe\build\install\include\viennacl
	xcopy %~sdp0\caffe\viennacl-dev\viennacl %~sdp0\caffe\build\install\include\viennacl /s /h /c /y 
	mkdir  %~sdp0\caffe\build\install\include\CL
	xcopy %~sdp0\caffe\viennacl-dev\CL %~sdp0\caffe\build\install\include\CL /s /h /c /y 
	mkdir  %~sdp0\caffe\build\install\include\3rdparty
	xcopy %~sdp0\caffe\include\caffe\3rdparty %~sdp0\caffe\build\install\include\3rdparty /s /h /c /y
	

	:: install python
	cd %py_path_str%\..\..\Scripts
	pip install protobuf -i https://pypi.tuna.tsinghua.edu.cn/simple
	echo "###############################################"
	echo "copy caffe\build\install\python\caffe to  %py_path_str%\caffe"
	::cd %py_path_str%
	::mkdir caffe
	::xcopy %~sdp0\caffe\build\install\python\caffe .\caffe /s /h /c /y 
	echo "copy done"
)
pause

