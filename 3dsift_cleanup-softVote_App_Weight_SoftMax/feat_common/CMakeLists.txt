cmake_minimum_required(VERSION 3.0)

project(feat_common)

include_directories(${NIFTI_INCLUDE_DIR} ${FLANN_INCLUDE_DIR})

file(GLOB SOURCE_FILES *.cpp)
add_library(${PROJECT_NAME} ${SOURCE_FILES})
add_dependencies(${PROJECT_NAME} Nifticlib FLANN)
target_link_libraries(${PROJECT_NAME} src_common)
target_compile_definitions(${PROJECT_NAME} PUBLIC FLANN_STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
