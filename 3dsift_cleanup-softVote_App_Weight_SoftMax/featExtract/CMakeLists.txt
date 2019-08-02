cmake_minimum_required(VERSION 3.0)

project(featExtract)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${NIFTI_INCLUDE_DIR} ${ZLIB_INCLUDE_DIR})
link_directories(${NIFTI_LIBRARY_DIR} ${ZLIB_LIBRARY_DIR})
set(cuda_common_files "/home/carluer/Documents/ETS/3dsift_cleanup-softVote_App_Weight_SoftMax/cuda_common")
include_directories(${cuda_common_files})
add_executable(${PROJECT_NAME} featExtract.cpp)
target_link_libraries(${PROJECT_NAME} niftiio znz zlib src_common feat_common cuda_common)

install(TARGETS featExtract RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
