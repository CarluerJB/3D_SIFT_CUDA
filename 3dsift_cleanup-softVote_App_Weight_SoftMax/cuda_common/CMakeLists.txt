cmake_minimum_required(VERSION 3.0)

project(cuda_common CXX CUDA)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${NIFTI_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR})
link_directories(${NIFTI_LIBRARY_DIR} ${ZLIB_LIBRARY_DIR})
file(GLOB SOURCE_FILES *.cu)
cuda_add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} OPTIONS -arch sm_61)
target_link_libraries(${PROJECT_NAME} niftiio znz zlib src_common feat_common)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

