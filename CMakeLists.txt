# ------------------------------------------------------------------------------
#        A Modular Optimization framework for Localization and mApping
#                               (MOLA)
#
# Copyright (C) 2018-2021, Jose Luis Blanco-Claraco, contributors (AUTHORS.md)
# All rights reserved.
# Released under GNU GPL v3. See LICENSE file
# ------------------------------------------------------------------------------

# Minimum CMake vesion: limited by CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
cmake_minimum_required(VERSION 3.4)

# Tell CMake we'll use C++ for use in its tests/flags
project(mola-input-kaist-dataset LANGUAGES CXX)

# MOLA CMake scripts: "mola_xxx()"
find_package(mola-common REQUIRED)

# find dependencies:
find_mola_package(mola-kernel)

find_package(mrpt-obs REQUIRED)
find_package(mrpt-maps REQUIRED)  # for CObservationPointCloud
find_package(mrpt-math REQUIRED)  # for MRPT Eigen utilities

# -----------------------
# define lib:
file(GLOB_RECURSE LIB_SRCS src/*.cpp src/*.h)
file(GLOB_RECURSE LIB_PUBLIC_HDRS include/*.h)

mola_add_library(
	TARGET ${PROJECT_NAME}
	SOURCES ${LIB_SRCS} ${LIB_PUBLIC_HDRS}
	PRIVATE_LINK_LIBRARIES
		mola-kernel
		mrpt::obs
		mrpt::math
		mrpt::maps
	CMAKE_DEPENDENCIES
		mola-kernel
		mrpt-maps
		mrpt-obs
)
