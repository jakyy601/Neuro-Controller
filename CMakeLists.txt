cmake_minimum_required(VERSION 3.19)

project(Neural_Network)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/inc/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.c"
)

add_executable(main ${all_SRCS})
#link_directories(${CMAKE_SOURCE_DIR}/inc)
#add_executable(main src/main.c)