cmake_minimum_required(VERSION 3.19)

project(Neural_Network)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

file(
    GLOB main_srcs
    "${PROJECT_SOURCE_DIR}/src/neuralController.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralController.h"
    "${PROJECT_SOURCE_DIR}/src/ncHelper.c"
    "${PROJECT_SOURCE_DIR}/inc/ncHelper.h"
    "${PROJECT_SOURCE_DIR}/src/main.c"

)

file(
    GLOB test_srcs
    "${PROJECT_SOURCE_DIR}/src/test.c"
)

file(
    GLOB test2_srcs
    "${PROJECT_SOURCE_DIR}/src/test2.c"
    "${PROJECT_SOURCE_DIR}/src/ncHelper.c"
    "${PROJECT_SOURCE_DIR}/inc/ncHelper.h"
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

add_library(
    minIni
    "${PROJECT_SOURCE_DIR}/lib/minIni/minIni.c"
    "${PROJECT_SOURCE_DIR}/lib/minIni/minIni.h"
)

target_include_directories(
    pbPlots PUBLIC 
    "${PROJECT_SOURCE_DIR}/lib/pbPlots"
    cJSON PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/cJSON"
    minIni PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/minIni")

add_executable(main ${main_srcs})
add_executable(test ${test_srcs})
add_executable(test2 ${test2_srcs})

target_link_libraries(main PRIVATE pbPlots cJSON minIni)
target_link_libraries(test PRIVATE pbPlots)
target_link_libraries(test2 PRIVATE pbPlots)
