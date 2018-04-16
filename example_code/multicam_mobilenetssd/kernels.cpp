// AUTOMATICALLY GENERATED FILE, DO NOT EDIT
#include "caffe/common.hpp"
#ifdef USE_GREENTEA
#include "kernels.hpp"
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#ifdef DISABLE_DOUBLE_SUPPORT
  #define DOUBLE_SUPPORT "#define DISABLE_DOUBLE_SUPPORT\n"
#else
  #define DOUBLE_SUPPORT "#define ENABLE_DOUBLE_SUPPORT\n"
#endif  // DISABLE_DOUBLE_SUPPORT
namespace caffe {
#ifdef USE_INDEX_64
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n#define TYPE_HALF 3\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_fp16)\n#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define HALF_SUPPORT_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if  defined(cl_intel_subgroups) && !defined(HAS_INTEL_SUBGROUPS)\n#undef cl_intel_subgroups\n#endif\n\n#if defined(HAS_INTEL_SUBGROUPS) && !defined(cl_intel_subgroups)\n#define cl_intel_subgroups\n#endif\n\n#ifndef NULL\n#define NULL (void*)0\n#endif";  // NOLINT
static std::string definitions_64 = DOUBLE_SUPPORT "// Types used for parameters, offset computations and so on\n#define int_tp long\n#define uint_tp unsigned long\n\n// Definitions used to cast the types above as needed\n#define int_tpc long\n#define uint_tpc unsigned long";  // NOLINT
#else
static std::string header = DOUBLE_SUPPORT "#ifndef __OPENCL_VERSION__\n#define __kernel\n#define __global\n#define __constant\n#define __local\n#define get_global_id(x) 0\n#define get_global_size(x) 0\n#define get_local_id(x) 0\n#define get_local_size(x) 0\n#define FLT_MAX 0\n#define FLT_MIN 0\n#define cl_khr_fp64\n#define cl_amd_fp64\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#define CLK_LOCAL_MEM_FENCE\n#define CLK_GLOBAL_MEM_FENCE\n#define Dtype float\n#define barrier(x)\n#define atomic_cmpxchg(x, y, z) x\n#define signbit(x) x\n#define int_tp long\n#define uint_tp unsigned long\n#define int_tpc long\n#define uint_tpc unsigned long\n#endif\n\n#define CONCAT(A,B) A##_##B\n#define TEMPLATE(name,type) CONCAT(name,type)\n\n#define TYPE_FLOAT 1\n#define TYPE_DOUBLE 2\n#define TYPE_HALF 3\n\n#if defined(cl_khr_fp64)\n#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#elif defined(cl_amd_fp64)\n#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n#ifndef DISABLE_DOUBLE_SUPPORT\n#define DOUBLE_SUPPORT_AVAILABLE\n#endif //DISABLE_DOUBLE_SUPPORT\n#endif\n\n#if defined(cl_khr_fp16)\n#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n#define HALF_SUPPORT_AVAILABLE\n#endif\n\n#if defined(cl_khr_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if defined(cl_khr_global_int32_base_atomics)\n#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n#define ATOMICS_32_AVAILABLE\n#endif\n\n#if  defined(cl_intel_subgroups) && !defined(HAS_INTEL_SUBGROUPS)\n#undef cl_intel_subgroups\n#endif\n\n#if defined(HAS_INTEL_SUBGROUPS) && !defined(cl_intel_subgroups)\n#define cl_intel_subgroups\n#endif\n\n#ifndef NULL\n#define NULL (void*)0\n#endif";  // NOLINT
static std::string definitions_32 = DOUBLE_SUPPORT "// Types used for parameters, offset computations and so on\n#define int_tp int\n#define uint_tp unsigned int\n\n// Definitions used to cast the types above as needed\n#define int_tpc int\n#define uint_tpc unsigned int";  // NOLINT
#endif
static std::vector<std::vector<std::string>> cl_kernels{
    {"#ifndef __OPENCL_VERSION__",    // NOLINT
"#include \"header.cl\"",    // NOLINT
"#endif",    // NOLINT
"",    // NOLINT
"__kernel void TEMPLATE(preprocess, Dtype)(const int_tp n,float scale,float mean,__global const float* indata,",    // NOLINT
"__global Dtype* outdata) {",    // NOLINT
"for (int_tp index = get_global_id(0); index < n; index += get_global_size(0))",    // NOLINT
"{",    // NOLINT
"outdata[index] = (indata[index]+mean)*scale;",    // NOLINT
"}",    // NOLINT
"}",    // NOLINT
"",    // NOLINT
""}   // NOLINT
};
static std::string cl_kernel_names[] = {
    "preprocess"   // NOLINT
};
viennacl::ocl::program & RegisterMyCommonKernels(viennacl::ocl::context *ctx) {
  std::stringstream ss;
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] == std::string("common")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  std::string kernel_string = ss.str();
  const char* kernel_program = kernel_string.c_str();
  string options;
  ctx->build_options(options);
  viennacl::ocl::program &program = ctx->add_program(kernel_program,
      "kernel_program");
  return program;
}
template <typename Dtype>
viennacl::ocl::program & RegisterMyKernels(viennacl::ocl::context *ctx) {
  std::stringstream ss;
  std::stringstream int64_base_atomics;
  int64_base_atomics << "\n\n";  // NOLINT
  int64_base_atomics << "#if defined(cl_khr_int64_base_atomics)" << "\n\n";  // NOLINT
  int64_base_atomics << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable" << "\n\n";  // NOLINT
  int64_base_atomics << "#define ATOMICS_64_AVAILABLE" << "\n\n";  // NOLINT
  int64_base_atomics << "#endif" << "\n\n";  // NOLINT
  if(ctx->devices()[0].extensions().find("cl_khr_int64_base_atomics")!= std::string::npos) {
    header += int64_base_atomics.str();
  }
#ifdef USE_INDEX_64
  ss << header << "\n\n";  // NOLINT
  ss << definitions_64 << "\n\n";  // NOLINT
#else
  ss << header << "\n\n";  // NOLINT
  ss << definitions_32 << "\n\n";  // NOLINT
#endif
  if (std::is_same<Dtype, float>::value) { // NOLINT
  ss << "#define Dtype float" << "\n\n";  // NOLINT
  ss << "#define Dtype2 float2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 float4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 float8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 float16" << "\n\n";  // NOLINT
  ss << "#define as_Dtype as_float" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_float2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_float4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_float8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_float16" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_FLOAT" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    for (int j = 0; j < cl_kernels[i].size(); ++j) {
      ss << cl_kernels[i][j] << "\n\n";
    }
  }
  }
  if (std::is_same<Dtype, double>::value) { // NOLINT
  ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  ss << "#define Dtype double" << "\n\n";  // NOLINT
  ss << "#define Dtype2 double2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 double4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 double8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 double16" << "\n\n";  // NOLINT
  ss << "#define as_Dtype as_double" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_double2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_double4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_double8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_double16" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE double" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] != std::string("fft")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ss << "#endif  // DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  }
  if (std::is_same<Dtype, half_float::half>::value) { // NOLINT
  ss << "#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)" << "\n\n";  // NOLINT
  if (ctx->devices()[0].extensions().find("cl_intel_subgroups")!= std::string::npos   // NOLINT
      && ctx->devices()[0].extensions().find("cl_intel_subgroups_short")== std::string::npos) { // NOLINT
    std::cerr << "Fatal error: Intel iGPU device found but doesn\'t support cl_intel_subgroups_short." << std::endl; // NOLINT
    std::cerr << "Please upgrade the GPU driver and OpenCL SDK." << std::endl; // NOLINT
    std::cerr << "For iGPU platforms before Gen9, fp16 is not supported." << std::endl; // NOLINT
    exit(-1);
  }
  ss << "#define Dtype half" << "\n\n";  // NOLINT
  ss << "#define Dtype2 half2" << "\n\n";  // NOLINT
  ss << "#define Dtype4 half4" << "\n\n";  // NOLINT
  ss << "#define Dtype8 half8" << "\n\n";  // NOLINT
  ss << "#define Dtype16 half16" << "\n\n";  // NOLINT
  ss << "#define as_Dtype as_half" << "\n\n";  // NOLINT
  ss << "#define as_Dtype2 as_half2" << "\n\n";  // NOLINT
  ss << "#define as_Dtype4 as_half4" << "\n\n";  // NOLINT
  ss << "#define as_Dtype8 as_half8" << "\n\n";  // NOLINT
  ss << "#define as_Dtype16 as_half16" << "\n\n";  // NOLINT
  ss << "#define TYPE TYPE_HALF" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MAX HALF_MAX" << "\n\n";  // NOLINT
  ss << "#define DTYPE_MIN HALF_MIN" << "\n\n";  // NOLINT
  ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
  for (int i = 0; i < cl_kernels.size(); ++i) {
    if (cl_kernel_names[i] != std::string("fft")) {
      for (int j = 0; j < cl_kernels[i].size(); ++j) {
        ss << cl_kernels[i][j] << "\n\n";
      }
    }
  }
  ss << "#endif  // HALF_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
  }
  std::string kernel_string = ss.str();
  const char* kernel_program = kernel_string.c_str();
  string options;
#ifdef USE_FFT
  options = " -DFFT ";
#endif
#ifdef HAS_HALF_SUPPORT
  options += " -DHAS_HALF_SUPPORT ";
#endif
  if(ctx->devices()[0].extensions().find("cl_intel_subgroups")!= std::string::npos) {
    options += " -DHAS_INTEL_SUBGROUPS ";
  }
  bool is_beignet = ctx->devices()[0].opencl_c_version().find("beignet")
                    != std::string::npos;
  if (!is_beignet)
    options += (" -cl-no-subgroup-ifp ");
  ctx->build_options(options);
  viennacl::ocl::program &program = ctx->add_program(kernel_program,
      "kernel_program");
  return program;
}
#ifdef HAS_HALF_SUPPORT
template
viennacl::ocl::program & RegisterMyKernels<half>(viennacl::ocl::context *ctx);
#endif
template
viennacl::ocl::program & RegisterMyKernels<float>(viennacl::ocl::context *ctx);
template
viennacl::ocl::program & RegisterMyKernels<double>(viennacl::ocl::context *ctx);
int getMyKernelBundleCount() {
  return cl_kernels.size();
}
template<typename Dtype>
std::string getMyKernelBundleSource(int index) {
  std::stringstream ss;
#ifdef USE_INDEX_64
  ss << header << "\n\n";  // NOLINT
  ss << definitions_64 << "\n\n";  // NOLINT
#else
  ss << header << "\n\n";  // NOLINT
  ss << definitions_32 << "\n\n";  // NOLINT
#endif
  if (std::is_same<Dtype, float>::value) {
    ss << "#define Dtype float" << "\n\n";  // NOLINT
    ss << "#define Dtype2 float2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 float4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 float8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 float16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_FLOAT" << "\n\n";  // NOLINT
    ss << "#define as_Dtype as_float" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_float2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_float4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_float8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_float16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  } else if (std::is_same<Dtype, double>::value) {
    ss << "#ifdef DOUBLE_SUPPORT_AVAILABLE" << "\n\n";  // NOLINT
    ss << "#define Dtype double" << "\n\n";  // NOLINT
    ss << "#define Dtype2 double2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 double4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 double8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 double16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_DOUBLE" << "\n\n";  // NOLINT
    ss << "#define as_Dtype as_double" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_double2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_double4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_double8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_double16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE double" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX FLT_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN FLT_MIN" << "\n\n";  // NOLINT
  } else {
    ss << "#if defined(HALF_SUPPORT_AVAILABLE) && defined(HAS_HALF_SUPPORT)" << "\n\n";  // NOLINT
    ss << "#define Dtype half" << "\n\n";  // NOLINT
    ss << "#define Dtype2 half2" << "\n\n";  // NOLINT
    ss << "#define Dtype4 half4" << "\n\n";  // NOLINT
    ss << "#define Dtype8 half8" << "\n\n";  // NOLINT
    ss << "#define Dtype16 half16" << "\n\n";  // NOLINT
    ss << "#define TYPE TYPE_HALF" << "\n\n";  // NOLINT
    ss << "#define as_Dtype as_half" << "\n\n";  // NOLINT
    ss << "#define as_Dtype2 as_half2" << "\n\n";  // NOLINT
    ss << "#define as_Dtype4 as_half4" << "\n\n";  // NOLINT
    ss << "#define as_Dtype8 as_half8" << "\n\n";  // NOLINT
    ss << "#define as_Dtype16 as_half16" << "\n\n";  // NOLINT
    ss << "#define KERNEL_ARG_DTYPE float" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MAX HALF_MAX" << "\n\n";  // NOLINT
    ss << "#define DTYPE_MIN HALF_MIN" << "\n\n";  // NOLINT
  }
  for (int j = 0; j < cl_kernels[index].size(); ++j) {
    ss << cl_kernels[index][j] << "\n\n";
  }
  if (std::is_same<Dtype, float>::value) {
  } else {
    ss << "#endif" << "\n\n";  // NOLINT
  }
  return ss.str();
}
template std::string getMyKernelBundleSource<half>(int index);
template std::string getMyKernelBundleSource<float>(int index);
template std::string getMyKernelBundleSource<double>(int index);
std::string getMyKernelBundleName(int index) {
  return cl_kernel_names[index];
}
}  // namespace caffe
#endif
