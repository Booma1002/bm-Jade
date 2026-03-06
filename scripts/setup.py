from setuptools import setup, Extension
import pybind11

# 1. Define the Extension
cpp_args = ['/std:c++17', '/O2'] # MSVC flags for C++17 and Optimization

ext_modules = [
    Extension(
        'my_engine',
        ['bindings.cpp'],  # Only compile the bindings file (which includes tile.hpp)
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

# 2. Setup (Force disable auto-discovery)
setup(
    name='my_engine',
    version='0.0.1',
    ext_modules=ext_modules,
    # This block prevents the error by telling setuptools "I have no Python packages"
    packages=[],
    py_modules=[],
)