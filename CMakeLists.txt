cmake_minimum_required(VERSION 3.18.4)

project(Neural_Network)

# project directories
include_directories(
    ${CMAKE_SOURCE_DIR}/inc
    ${CMAKE_SOURCE_DIR}/src
)

# put source files into a variable
file(
    GLOB neuralController
    "${PROJECT_SOURCE_DIR}/src/neuralController.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralController.h"
)

file(
    GLOB neuralControllerInterfaceSrc
    "${PROJECT_SOURCE_DIR}/src/neuralControllerInterface.c"
    "${PROJECT_SOURCE_DIR}/inc/neuralControllerInterface.h"
)

file(
    GLOB main
    "${PROJECT_SOURCE_DIR}/src/main.c"
    "${PROJECT_SOURCE_DIR}/inc/main.h"
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
    "${PROJECT_SOURCE_DIR}/lib/pbPlots/supportLib.h"
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
    "${PROJECT_SOURCE_DIR}/lib/minIni"
)

# link project external libraries
target_link_libraries(pthread -lpthread m)
set_target_properties(pthread PROPERTIES LINKER_LANGUAGE C)

# find python installation for the neuralControllerInterface
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# create target library for python
Python3_add_library(neuralControllerInterface MODULE ${neuralController} ${neuralControllerInterfaceSrc})
set_target_properties(neuralControllerInterface PROPERTIES
    PREFIX ""
    SUFFIX ".${Python3_SOABI}.so"
)
target_include_directories(neuralControllerInterface PRIVATE ${Python3_INCLUDE_DIRS})

# create target main
add_executable(main ${main} ${neuralController})

# link project libraries
target_link_libraries(main PRIVATE pbPlots minIni pthread)
target_link_libraries(neuralControllerInterface PRIVATE Python3::Module)

#add_library(libNeuralController SHARED ${neuralController})
#set_target_properties(libNeuralController PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
#set_target_properties(libNeuralController PROPERTIES OUTPUT_NAME "NeuralController")