# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "C:/Users/Jakob/esp/esp-idf/components/bootloader/subproject"
  "F:/work/esp32-gateway/build/bootloader"
  "F:/work/esp32-gateway/build/bootloader-prefix"
  "F:/work/esp32-gateway/build/bootloader-prefix/tmp"
  "F:/work/esp32-gateway/build/bootloader-prefix/src/bootloader-stamp"
  "F:/work/esp32-gateway/build/bootloader-prefix/src"
  "F:/work/esp32-gateway/build/bootloader-prefix/src/bootloader-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "F:/work/esp32-gateway/build/bootloader-prefix/src/bootloader-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "F:/work/esp32-gateway/build/bootloader-prefix/src/bootloader-stamp${cfgdir}") # cfgdir has leading slash
endif()
