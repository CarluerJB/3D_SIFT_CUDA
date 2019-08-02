cmake_minimum_required(VERSION 3.0)

project(src_common CXX CUDA)

file(GLOB SOURCE_FILES *.cpp)
include_directories(${CUDA_INCLUDE_DIRS})
set(cuda_common_files "/home/carluer/Documents/ETS/3dsift_cleanup-softVote_App_Weight_SoftMax/cuda_common")
include_directories(${cuda_common_files})
include_directories(${NIFTI_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR})
link_directories(${NIFTI_LIBRARY_DIR} ${ZLIB_LIBRARY_DIR})
add_library(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} cuda_common niftiio znz zlib)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
