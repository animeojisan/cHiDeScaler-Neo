cmake_minimum_required(VERSION 3.24)
project(neo_dlssnr_feature18_backend LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
  message(FATAL_ERROR "The DLSSNR Feature 18 backend requires Windows x64 + MSVC.")
endif()

if(NOT DEFINED DLSS_SDK_ROOT AND DEFINED ENV{DLSS_SDK_ROOT})
  set(DLSS_SDK_ROOT "$ENV{DLSS_SDK_ROOT}")
endif()
if(NOT DLSS_SDK_ROOT)
  get_filename_component(_neo_root "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
  set(_bundled_hint "${_neo_root}/external/DLSS")
  if(EXISTS "${_bundled_hint}/include/nvsdk_ngx.h")
    set(DLSS_SDK_ROOT "${_bundled_hint}")
  endif()
endif()
if(NOT DLSS_SDK_ROOT OR NOT EXISTS "${DLSS_SDK_ROOT}/include/nvsdk_ngx.h")
  message(FATAL_ERROR "NVIDIA DLSS/NGX SDK not found. Set DLSS_SDK_ROOT or clone NVIDIA/DLSS to external/DLSS.")
endif()

set(_ngx_lib_candidates
  "${DLSS_SDK_ROOT}/lib/Windows_x86_64/x86_64/nvsdk_ngx_d.lib"
  "${DLSS_SDK_ROOT}/lib/Windows_x86_64/x86_64/nvsdk_ngx_s.lib"
  "${DLSS_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib"
  "${DLSS_SDK_ROOT}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib")
set(NGX_LIB "")
foreach(candidate IN LISTS _ngx_lib_candidates)
  if(EXISTS "${candidate}")
    set(NGX_LIB "${candidate}")
    break()
  endif()
endforeach()
if(NOT NGX_LIB)
  message(FATAL_ERROR "Could not find nvsdk_ngx_d.lib/nvsdk_ngx_s.lib below ${DLSS_SDK_ROOT}/lib")
endif()

add_library(neo_dlssnr_backend SHARED bridge.cpp)
target_compile_features(neo_dlssnr_backend PRIVATE cxx_std_20)
target_compile_definitions(neo_dlssnr_backend PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_include_directories(neo_dlssnr_backend PRIVATE
  "${DLSS_SDK_ROOT}/include"
  "${CMAKE_CURRENT_LIST_DIR}/../include")
target_link_libraries(neo_dlssnr_backend PRIVATE d3d12 dxgi version "${NGX_LIB}")
set_target_properties(neo_dlssnr_backend PROPERTIES
  OUTPUT_NAME "neo_dlssnr_backend"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(MSVC)
  target_compile_options(neo_dlssnr_backend PRIVATE /W4 /permissive- /EHsc)
endif()
