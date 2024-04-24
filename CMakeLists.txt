cmake_minimum_required(VERSION 3.18.4)

project(Neural_Network)

include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

file(
    GLOB main_srcs
    "${PROJECT_SOURCE_DIR}/src/neuralController.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralController.h"
    "${PROJECT_SOURCE_DIR}/src/main.c"
)

file(
    GLOB test3_srcs
    "${PROJECT_SOURCE_DIR}/src/neuralController.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralController.h"
    "${PROJECT_SOURCE_DIR}/src/ncHelper.h"
    "${PROJECT_SOURCE_DIR}/src/ncHelper.c"
    "${PROJECT_SOURCE_DIR}/misc/test3.c"
)

add_library(
    pbPlots
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/pbPlots.c"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/pbPlots.h"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/supportLib.c"
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/supportLib.c"
)

add_library(
    minIni
    "${PROJECT_SOURCE_DIR}/lib/minIni/minIni.c"
    "${PROJECT_SOURCE_DIR}/lib/minIni/minIni.h"
)

add_library(
    pthread
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/_ptw32.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/pthread.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/sched.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/semaphore.h"
)

target_include_directories(
    pbPlots PUBLIC 
    "${PROJECT_SOURCE_DIR}/lib/pbPlots"
    cJSON PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/cJSON"
    minIni PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/minIni")

target_link_libraries(pthread -lpthread -LC:/src/vcpkg/vcpkg/installed/x64-windows/lib)
set_target_properties(pthread PROPERTIES LINKER_LANGUAGE C)


add_executable(main ${main_srcs})
add_executable(test3 ${test3_srcs})

target_link_libraries(main PRIVATE pbPlots minIni pthread)
target_link_libraries(test3 PRIVATE pbPlots minIni pthread)

