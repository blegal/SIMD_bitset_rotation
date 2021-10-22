cmake_minimum_required(VERSION 3.7)

project(bit_compressor)

SET(CMAKE_BUILD_TYPE Release)

#SET (CMAKE_C_COMPILER   /usr/local/Cellar/llvm/13.0.0_1/bin/clang)
#SET (CMAKE_CXX_COMPILER /usr/local/Cellar/llvm/13.0.0_1/bin/clang++)

SET (CMAKE_C_COMPILER   /usr/local/Cellar/gcc/11.2.0/bin/gcc-11)
SET (CMAKE_CXX_COMPILER /usr/local/Cellar/gcc/11.2.0/bin/g++-11)


SET (CMAKE_CXX_STANDARD 17)

SET (CMAKE_CXX_FLAGS "")
SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -g0 -std=c++17 -march=native -mtune=native -funroll-loops")

#SET (CMAKE_EXE_LINKER_FLAGS "-lm")

# Generate the source files list
file (GLOB_RECURSE source_files src/*)

# Add includes
include_directories (src)

# Object
add_library (dec-obj OBJECT ${source_files})

add_executable(bit_compressor $<TARGET_OBJECTS:dec-obj>)
