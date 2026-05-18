cmake_minimum_required(VERSION 3.22)

project(Neural_Network)

# generated library goes into the scripts folder
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/scripts")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

# set sources
set(NeuralController_Sources
    ${PROJECT_SOURCE_DIR}/src/neuralController.c
)

# set include directories
set(Includes
    ${PROJECT_SOURCE_DIR}/inc
)

add_library(neuralController STATIC
    ${NeuralController_Sources}
)

target_include_directories(neuralController PRIVATE
    ${Includes}
)

set(main_Sources
    ${PROJECT_SOURCE_DIR}/src/main.c
)

add_library(minIni STATIC
    ${PROJECT_SOURCE_DIR}/lib/minIni/minIni.c
)

target_include_directories(minIni PUBLIC
    ${PROJECT_SOURCE_DIR}/lib/minIni
)

# add target main
add_executable(main ${NeuralController_Sources} src/main.c)

# add includes
target_include_directories(main PRIVATE
    ${Includes}
)

# link project libraries
target_link_libraries(main PRIVATE minIni neuralController m)

# create target library for python
add_library(neuralControllerInterface SHARED ${NeuralController_Sources})
target_include_directories(neuralControllerInterface PRIVATE ${Includes})
target_link_libraries(neuralControllerInterface PRIVATE neuralController m)
set_target_properties(neuralControllerInterface PROPERTIES PREFIX "")
