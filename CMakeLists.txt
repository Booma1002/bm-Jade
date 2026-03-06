cmake_minimum_required(VERSION 3.20)
project(my_engine)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# --- PYTHON SETUP ---
set(PYTHON_EXECUTABLE "C:/Users/Booma/anaconda3/envs/BinderProject/python.exe")
execute_process(
        COMMAND "${PYTHON_EXECUTABLE}" -m pybind11 --includes
        OUTPUT_VARIABLE PYBIND_INCLUDES_RAW
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "-I" "" PYBIND_INCLUDES_CLEAN "${PYBIND_INCLUDES_RAW}")
string(REPLACE " " ";" PYBIND_INCLUDE_LIST "${PYBIND_INCLUDES_CLEAN}")
include_directories(${PYBIND_INCLUDE_LIST})

# --- TARGET 1: THE LIBRARY (Python Module) ---
# Compiles bindings.cpp AND the core engine files so Python can access them
add_library(my_engine SHARED
        bindings/bindings.cpp
        src/Tile.cpp
        src/Dispatcher.cpp
        src/TileKernelsCpu.cpp
        src/TileOperator.cpp
        src/TileKernelsGpu.cpp
)
set_target_properties(my_engine PROPERTIES SUFFIX ".pyd")
# Let the library see the .hpp files
target_include_directories(my_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- TARGET 2: THE PLAYGROUND (C++ Executable) ---
# Compiles the standalone C++ environment for local testing
add_executable(cpp_runner
        examples/playground.cpp
        examples/main.cpp
        src/Tile.cpp
        src/Dispatcher.cpp
        src/TileKernelsCpu.cpp
        src/TileOperator.cpp
        src/TileKernelsGpu.cpp
        src/Registry.cpp
        src/Storage.cpp
        src/Allocator.cpp
        include/header/Logger.hpp
        src/Logger.cpp
        src/Logger.cpp
        src/TileFactories.cpp
        src/TileTransformations.cpp
        src/TileMath.cpp
        src/TileReductions.cpp
        include/temp/TileFactories.tpp
        include/temp/TileTransformations.tpp
        include/header/Engine.hpp
        include/header/Heatmap.hpp
        src/Heatmap.cpp
)
# Let the executable see the .hpp files
target_include_directories(cpp_runner PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(cpp_runner PUBLIC OpenMP::OpenMP_CXX)
    target_link_options(cpp_runner PUBLIC "-fopenmp")
endif()