cmake_minimum_required(VERSION 3.10)
add_compile_options(-std=c++11 -static)

project(3DSIFT LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_30,code=compute_30)
#list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_32,code=compute_32)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_35,code=compute_35)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_37,code=compute_37)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=compute_52)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_53,code=compute_53)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_60,code=compute_60)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_61,code=compute_61)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_62,code=compute_62)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_70,code=compute_70)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_72,code=compute_72)
list( APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=compute_75)



#-----------------------------------------------------------------------------
# Build options
#-----------------------------------------------------------------------------

option(BUILD_FEATEXTRACT "Build featExtract." ON)
option(BUILD_FEATMATCHMULTIPLE "Build featMatchMultiple." ON)

#-----------------------------------------------------------------------------
# Git protocole option
#-----------------------------------------------------------------------------
option(USE_GIT_PROTOCOL "If behind a firewall turn this off to use http instead." OFF)

set(git_protocol "git")
if(NOT USE_GIT_PROTOCOL)
  set(git_protocol "https")
endif()

#-----------------------------------------------------------------------------
# Enable and setup External project global properties
#-----------------------------------------------------------------------------
include(ExternalProject)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
endif()

#-----------------------------------------------------------------------------
# OpenMP
#-----------------------------------------------------------------------------

find_package(OpenMP)
if (OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#-----------------------------------------------------------------------------
# Zlib
#-----------------------------------------------------------------------------

message(STATUS "Zlib library will be installed.")

ExternalProject_Add(Zlib
  SOURCE_DIR "${PROJECT_BINARY_DIR}/deps/zlib"
  BINARY_DIR "${PROJECT_BINARY_DIR}/deps/zlib-build"
  INSTALL_DIR "${PROJECT_BINARY_DIR}/deps/zlib-install"
  GIT_REPOSITORY "${git_protocol}://github.com/madler/zlib.git"
  GIT_TAG "50893291621658f355bc5b4d450a8d06a563053d"
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  -DCMAKE_BIN_DIR:PATH=<INSTALL_DIR>/bin
  -DINSTALL_INC_DIR:PATH=<INSTALL_DIR>/include
  -DINSTALL_LIB_DIR:PATH=<INSTALL_DIR>/lib
  -DINSTALL_MAN_DIR:PATH=<INSTALL_DIR>/share/man
  -DINSTALL_PKGCONFIG_DIR:PATH=<INSTALL_DIR>/share/pkgconfig)

if(WIN32)
  set(ZLIB_LIB_BASE_NAME "zlibstatic")
  set(ZLIB_LIB_NAME_RELEASE "${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(ZLIB_LIB_NAME_DEBUG "${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_BASE_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX}")
elseif(UNIX)
  set(ZLIB_LIB_BASE_NAME "z")
  set(ZLIB_LIB_NAME_RELEASE "${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(ZLIB_LIB_NAME_DEBUG "${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
else()
  # MacOSX
endif()

ExternalProject_Get_Property(Zlib install_dir)
set(ZLIB_LIBRARY_DIR ${install_dir}/lib)
set(ZLIB_INCLUDE_DIR ${install_dir}/include)
set(ZLIB_BINARY_DIR ${install_dir}/bin)

add_library(zlib STATIC IMPORTED)
set_target_properties(zlib PROPERTIES IMPORTED_LOCATION_DEBUG "${ZLIB_LIBRARY_DIR}/${ZLIB_LIB_NAME_DEBUG}")
set_target_properties(zlib PROPERTIES IMPORTED_LOCATION_RELEASE "${ZLIB_LIBRARY_DIR}/${ZLIB_LIB_NAME_RELEASE}")

if(${CMAKE_BUILD_TYPE} MATCHES Debug)
  set(${ZLIB_LIB_NAME} ${ZLIB_LIB_NAME_DEBUG})
else()
  set(${ZLIB_LIB_NAME} ${ZLIB_LIB_NAME_RELEASE})
endif()

#-----------------------------------------------------------------------------
# Niftilib
#-----------------------------------------------------------------------------

message(STATUS "Nifti library will be installed.")

ExternalProject_Add(Nifticlib
  SOURCE_DIR "${PROJECT_BINARY_DIR}/deps/nifticlib"
  BINARY_DIR "${PROJECT_BINARY_DIR}/deps/nifticlib-build"
  INSTALL_DIR "${PROJECT_BINARY_DIR}/deps/nifticlib-install"
  GIT_REPOSITORY "${git_protocol}://gitlab.com/slckr/nifticlib.git"
  GIT_TAG "e26a94e947c210104223f9f49737392c742c1c5b"
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  -DZLIB_INCLUDE_DIR:PATH=${ZLIB_INCLUDE_DIR}
  -DZLIB_LIBRARY_DEBUG:PATH=${ZLIB_LIBRARY_DIR}/${ZLIB_LIB_NAME_DEBUG}
  -DZLIB_LIBRARY_RELEASE:PATH=${ZLIB_LIBRARY_DIR}/${ZLIB_LIB_NAME_RELEASE}
  DEPENDS Zlib)

set(NIFTIIO_LIB_BASE_NAME "niftiio")
set(NIFTIIO_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${NIFTIIO_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")

set(NIFTICDF_LIB_BASE_NAME "nifticdf")
set(NIFTICDF_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${NIFTICDF_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")

set(ZNZ_LIB_BASE_NAME "znz")
set(ZNZ_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${ZNZ_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Get_Property(Nifticlib install_dir)
set(NIFTI_LIBRARY_DIR ${install_dir}/lib)
set(NIFTI_INCLUDE_DIR ${install_dir}/include/nifti)
set(NIFTI_BINARY_DIR ${install_dir}/bin)

add_library(niftiio STATIC IMPORTED)
set_target_properties(niftiio PROPERTIES IMPORTED_LOCATION "${NIFTI_LIBRARY_DIR}/${NIFTIIO_LIB_NAME}")

add_library(nifticdf STATIC IMPORTED)
set_target_properties(nifticdf PROPERTIES IMPORTED_LOCATION "${NIFTI_LIBRARY_DIR}/${NIFTICDF_LIB_NAME}")

add_library(znz STATIC IMPORTED)
set_target_properties(znz PROPERTIES IMPORTED_LOCATION "${NIFTI_LIBRARY_DIR}/${ZNZ_LIB_NAME}")

#-----------------------------------------------------------------------------
# FLANN
#-----------------------------------------------------------------------------

message(STATUS "FLANN library will be installed.")

ExternalProject_Add(FLANN
  SOURCE_DIR "${PROJECT_BINARY_DIR}/deps/FLANN"
  BINARY_DIR "${PROJECT_BINARY_DIR}/deps/FLANN-build"
  INSTALL_DIR "${PROJECT_BINARY_DIR}/deps/FLANN-install"
  GIT_REPOSITORY "${git_protocol}://gitlab.com/slckr/flann"
  GIT_TAG "94d066a07d288044ba8f9534b365dbd2171f106a"
  CMAKE_ARGS
  -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  -DBUILD_MATLAB_BINDINGS:BOOL=OFF
  -DBUILD_PYTHON_BINDINGS:BOOL=OFF
  -DBUILD_EXAMPLES:BOOL=OFF
  )

set(FLANN_LIB_BASE_NAME "flann_s")
set(FLANN_CPP_LIB_BASE_NAME "flann_cpp_s")

set(FLANN_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${FLANN_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(FLANN_CPP_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${FLANN_CPP_LIB_BASE_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Get_Property(FLANN install_dir)
set(FLANN_LIBRARY_DIR ${install_dir}/lib)
set(FLANN_INCLUDE_DIR ${install_dir}/include)
set(FLANN_BINARY_DIR ${install_dir}/bin)

add_library(flann STATIC IMPORTED)
set_target_properties(flann PROPERTIES IMPORTED_LOCATION "${FLANN_LIBRARY_DIR}/${FLANN_LIB_NAME}")

add_library(flann_cpp STATIC IMPORTED)
set_target_properties(flann_cpp PROPERTIES IMPORTED_LOCATION "${FLANN_LIBRARY_DIR}/${FLANN_CPP_LIB_NAME}")

#-----------------------------------------------------------------------------
# 3DSIFT
#-----------------------------------------------------------------------------

add_subdirectory(cuda_common)
add_subdirectory(src_common)
add_subdirectory(feat_common)

if(BUILD_FEATEXTRACT)
	add_subdirectory(featExtract)
endif()

if(BUILD_FEATMATCHMULTIPLE)
	add_subdirectory(featMatchMultiple)
endif()
