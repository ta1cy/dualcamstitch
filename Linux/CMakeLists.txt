cmake_minimum_required(VERSION 3.10)
project(dualcamstitch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(OpenCV REQUIRED)

add_executable(dualcamstitch_base dualcamstitch_base.cpp)
target_include_directories(dualcamstitch_base PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(dualcamstitch_base ${OpenCV_LIBS})