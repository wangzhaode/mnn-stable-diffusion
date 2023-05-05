cmake_minimum_required(VERSION 3.0)
project(stable-diffusion-mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

option(BUILD_FOR_ANDROID "Build for android." OFF)

# include dir
include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

# libs dir
link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

# source files
FILE(GLOB SRCS ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

if (BUILD_FOR_ANDROID)
    add_library(MNN SHARED IMPORTED)
    add_library(MNN_Express SHARED IMPORTED)
    add_library(MNNOpenCV SHARED IMPORTED)
    set_target_properties(
        MNN
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libs/libMNN.so
        )
    set_target_properties(
        MNN_Express
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libs/libMNN_Express.so
        )
    set_target_properties(
        MNNOpenCV
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/libs/libMNNOpenCV.so
        )
    add_executable(main ${SRCS})
    target_link_libraries(main MNN MNN_Express MNNOpenCV log)
else()
    # target
    add_executable(main ${SRCS})
    if (MSVC)
        target_link_libraries(main MNN)
    else()
        target_link_libraries(main MNN MNN_Express MNNOpenCV)
    endif()
endif()