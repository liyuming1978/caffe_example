@echo off
@setlocal EnableDelayedExpansion

:: download https://dist.nuget.org/win-x86-commandline/latest/nuget.exe

cd %~sdp0
mkdir include
mkdir lib
cd lib
mkdir Release
cd %~sdp0
xcopy %~sdp0\..\caffe\build\install\include %~sdp0\include /s /h /c /y 
xcopy %~sdp0\..\caffe\build\install\lib %~sdp0\lib\Release /s /h /c /y 

nuget.exe pack clCaffe_Package.nuspec