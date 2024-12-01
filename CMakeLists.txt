cmake_minimum_required(VERSION 3.18.4)

project(Neural_Network)

# project directories
include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

# put source files into a variable
file(
    GLOB main_srcs
    "${PROJECT_SOURCE_DIR}/src/neuralController.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralController.h"
    "${PROJECT_SOURCE_DIR}/src/main.c"
)

add_library(
    pthread
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/_ptw32.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/pthread.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/sched.h"
    "${PROJECT_SOURCE_DIR}/lib/pthreads/include/semaphore.h"
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
# project internal libraries
target_include_directories(
    pbPlots PUBLIC 
    "${PROJECT_SOURCE_DIR}/lib/pbPlots"
    minIni PUBLIC
    "${PROJECT_SOURCE_DIR}/lib/minIni")

# link project external libraries
target_link_libraries(pthread -lpthread -LC:/src/vcpkg/vcpkg/installed/x64-windows/lib)
set_target_properties(pthread PROPERTIES LINKER_LANGUAGE C)

# create target
add_executable(main ${main_srcs})

# link project libraries
target_link_libraries(main PRIVATE pbPlots minIni pthread)

