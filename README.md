# Neuro Controller
Implementation of a feed forward neural network with online learning through backpropagation and online learning that controls a simulated PTn plant. 

## Folder structure
```text
├── build                       CMake's build folder, configured through CMakePreset.json
├── doc                         Several .tex and .puml files that were used in the corresponding Master Thesis        
├── inc                         Include directory
├── scripts                     Script folder for the Python testing
├── src                         Source folder
│   ├── activation.c            Implementation of the activation functions
│   ├── main.c                  Very rough self-test routine for the neural controller
│   ├── neuralController.c      The main algorithm, also documented with public API
│   ├── pid.c                   A implementation for a PID controller
│   ├── pt2.c                   A C-simulation for a PT2 plant
├── CMakeLists.txt              Declaration of targets
├── CMakePreset.json            Declaration of CMake presets
├── README.md                   This file
```

## Neural Network
The network code is found in src/neuralController.c and the corresponding header file neuralController.h
Using CMake either a full executable using the src/main.c can be built, a shared library that is then consumed by the tests scripts in scripts/ or a STATIC CMake
target that can be consumed by another exectuable.
Consume with 
```
target_link_libraries(myTarget
    PUBLIC
        neuralController
)
```

## Building
There are 2 presets that can be used to build the target, so either use the normal preset for the Python testing that creates a shared library, which on Linux
also is directly put into scripts/ for consumption by the Python scripts:
```
cmake --preset shared
cmake --build build/shared
```
or use the self-test preset which compiles the main.c for a quick test to see if everything works or execute together with valgrind for memory leak verification
```
cmake --preset test
cmake --build build/test
```