### FindFFmpeg.cmake ---
## ffmpeg-yami can only link static ffmpeg, build share ffmpeg fails...


include(FindPackageHandleStandardArgs)
include(GetPrerequisites)

if(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIRS)
  set(FFMPEG_FOUND TRUE)
else(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIRS)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC libavcodec QUIET)
    pkg_check_modules(_FFMPEG_AVDEVICE libavdevice QUIET)
    pkg_check_modules(_FFMPEG_AVFILTER libavfilter QUIET)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat QUIET)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil QUIET)
    pkg_check_modules(_FFMPEG_SWRESAMPLE libswresample QUIET)
    pkg_check_modules(_FFMPEG_SWSCALE libswscale QUIET)
		pkg_check_modules(_FFMPEG_YAMI yami QUIET)
		pkg_check_modules(_FFMPEG_VAAPI va QUIET)
  endif(PKG_CONFIG_FOUND)

  find_path(FFMPEG_AVCODEC_INCLUDE_DIR
    NAMES avcodec.h
    PATHS /opt/ffmpeg/include ${_FFMPEG_AVCODEC_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libavcodec)

  find_path(FFMPEG_AVDEVICE_INCLUDE_DIR
    NAMES avdevice.h
    PATHS  /opt/ffmpeg/include ${_FFMPEG_AVDEVICE_INCLUDE_DIRS}/usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libavdevice)

  find_path(FFMPEG_AVFILTER_INCLUDE_DIR
    NAMES avfilter.h
    PATHS  /opt/ffmpeg/include ${_FFMPEG_AVFILTER_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libavfilter)

  find_path(FFMPEG_AVFORMAT_INCLUDE_DIR
    NAMES avformat.h
    PATHS /opt/ffmpeg/include ${_FFMPEG_AVFORMAT_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libavformat)

  find_path(FFMPEG_AVUTIL_INCLUDE_DIR
    NAMES avutil.h
    PATHS /opt/ffmpeg/include ${_FFMPEG_AVUTIL_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libavutil)

  find_path(FFMPEG_SWRESAMPLE_INCLUDE_DIR
    NAMES swresample.h
    PATHS /opt/ffmpeg/include ${_FFMPEG_SWRESAMPLE_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libswresample)

  find_path(FFMPEG_SWSCALE_INCLUDE_DIR
    NAMES swscale.h
    PATHS /opt/ffmpeg/include ${_FFMPEG_SWRESAMPLE_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include
    PATH_SUFFIXES ffmpeg libswscale)

  find_library(FFMPEG_AVCODEC_LIBRARY
    NAMES avcodec
    PATHS /opt/ffmpeg/lib ${_FFMPEG_AVCODEC_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_AVDEVICE_LIBRARY
    NAMES avdevice
    PATHS /opt/ffmpeg/lib ${_FFMPEG_AVDEVICE_LIBRARY_DIRS}  /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_AVFILTER_LIBRARY
    NAMES avfilter
    PATHS /opt/ffmpeg/lib ${_FFMPEG_AVDEVICE_LIBRARY_DIRS}  /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_AVFORMAT_LIBRARY
    NAMES avformat
    PATHS /opt/ffmpeg/lib ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}  /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_AVUTIL_LIBRARY
    NAMES avutil
    PATHS /opt/ffmpeg/lib ${_FFMPEG_AVUTIL_LIBRARY_DIRS}  /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_SWRESAMPLE_LIBRARY
    NAMES swresample
    PATHS /opt/ffmpeg/lib ${_FFMPEG_SWRESAMPLE_LIBRARY_DIRS}  /usr/lib /usr/local/lib /opt/local/lib)

  find_library(FFMPEG_SWSCALE_LIBRARY
    NAMES swscale
    PATHS  /opt/ffmpeg/lib ${_FFMPEG_SWSCALE_LIBRARY_DIRS} /usr/lib /usr/local/lib /opt/local/lib)
		
  find_library(FFMPEG_YAMI_LIBRARY
    NAMES yami
    PATHS /opt/yami/libyami/lib ${_FFMPEG_YAMI_LIBRARY_DIRS}  /opt/ffmpeg/lib /opt/yami/libyami/lib)	
		
  find_library(FFMPEG_VAAPI_LIBRARY
    NAMES va
    PATHS /opt/yami/vaapi/lib$ ${_FFMPEG_VAAPI_LIBRARY_DIRS} /opt/yami/vaapi/lib)
		
  find_library(FFMPEG_VAAPIDRM_LIBRARY
    NAMES va-drm
    PATHS /opt/yami/vaapi/lib$ ${_FFMPEG_VAAPI_LIBRARY_DIRS} /opt/yami/vaapi/lib)
		
  if(FFMPEG_AVCODEC_LIBRARY)
    set(FFMPEG_FOUND TRUE)
  endif(FFMPEG_AVCODEC_LIBRARY)

  if (FFMPEG_FOUND)
    get_filename_component(FFMPEG_INCLUDE_DIR ${FFMPEG_AVCODEC_INCLUDE_DIR} PATH)
    set(FFMPEG_INCLUDE_DIRS
      ${FFMPEG_INCLUDE_DIR}
      ${FFMPEG_AVCODEC_INCLUDE_DIR}
      ${FFMPEG_AVDEVICE_INCLUDE_DIR}
      ${FFMPEG_AVFILTER_INCLUDE_DIR}
      ${FFMPEG_AVFORMAT_INCLUDE_DIR}
      ${FFMPEG_SWRESAMPLE_INCLUDE_DIR}
      ${FFMPEG_SWSCALE_INCLUDE_DIR})
    set(FFMPEG_LIBRARIES 
      ${FFMPEG_AVFORMAT_LIBRARY}
      ${FFMPEG_AVCODEC_LIBRARY}
      ${FFMPEG_SWRESAMPLE_LIBRARY}
      ${FFMPEG_SWSCALE_LIBRARY}
      ${FFMPEG_AVDEVICE_LIBRARY}
      ${FFMPEG_AVFILTER_LIBRARY}
      ${FFMPEG_AVUTIL_LIBRARY}
			${FFMPEG_YAMI_LIBRARY}
			${FFMPEG_VAAPIDRM_LIBRARY}
			${FFMPEG_VAAPI_LIBRARY})
  endif (FFMPEG_FOUND)

  mark_as_advanced(FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARIES)
  mark_as_advanced(
    FFMPEG_AVCODEC_INCLUDE_DIR
    FFMPEG_AVDEVICE_INCLUDE_DIR
    FFMPEG_AVFILTER_INCLUDE_DIR
    FFMPEG_AVFORMAT_INCLUDE_DIR
    FFMPEG_AVUTIL_INCLUDE_DIR
    FFMPEG_SWSCALE_INCLUDE_DIR 
    FFMPEG_SWRESAMPLE_INCLUDE_DIR)
  mark_as_advanced(
    FFMPEG_AVCODEC_LIBRARY
    FFMPEG_AVDEVICE_LIBRARY
    FFMPEG_AVFILTER_LIBRARY
    FFMPEG_AVFORMAT_LIBRARY
    FFMPEG_AVUTIL_LIBRARY
    FFMPEG_SWSCALE_LIBRARY 
    FFMPEG_SWRESAMPLE_LIBRARY)
endif (FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIRS)
