cmake_minimum_required(VERSION 3.0)

project(featMatchMultiple)




include_directories(${NIFTI_INCLUDE_DIR} ${FLANN_INCLUDE_DIR})
link_directories(${NIFTI_LIBRARY_DIR} ${FLANN_LIBRARY_DIR} ${ZLIB_LIBRARY_DIR})

add_executable(${PROJECT_NAME} featMatchMultiple.cpp)
target_link_libraries(${PROJECT_NAME} feat_common niftiio znz zlib flann)
install(TARGETS featMatchMultiple RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})
