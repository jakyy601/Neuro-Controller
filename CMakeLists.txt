cmake_minimum_required(VERSION 3.19)

project(Neural_Network)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

file(GLOB main_srcs
    "${PROJECT_SOURCE_DIR}/inc/neural_controller.h"
    "${PROJECT_SOURCE_DIR}/src/main.c"
    "${PROJECT_SOURCE_DIR}/src/neural_controller.c"
)

file(GLOB test_srcs
    "${PROJECT_SOURCE_DIR}/src/test.c"
)

add_library(
    pbPlots
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/pbPlots.c"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/pbPlots.h"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/supportLib.c"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/supportLib.c"
)

add_library(
    cJSON
    "${PROJECT_SOURCE_DIR}/lib/cJSON/cJSON.c"
    "${PROJECT_SOURCE_DIR}/lib/cJSON/cJSON.h"
)

target_include_directories(
    pbPlots PUBLIC 
    "${PROJECT_SOURCE_DIR}/lib/pbPlots"
    cJSON
    "${PROJECT_SOURCE_DIR}/lib/cJSON")

add_executable(main ${main_srcs})
add_executable(test ${test_srcs})

target_link_libraries(main PRIVATE pbPlots cJSON)
target_link_libraries(test PRIVATE pbPlots)
