cmake_minimum_required(VERSION 3.16)
project(hungarian_eigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(hungarian_eigen_lib
    src/hungarian_eigen.cpp
)

target_include_directories(hungarian_eigen_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hungarian_eigen_lib
    PUBLIC
        Eigen3::Eigen
)

add_executable(hungarian_eigen
    src/main.cpp
)

target_link_libraries(hungarian_eigen
    PRIVATE
        hungarian_eigen_lib
)

enable_testing()
add_subdirectory(tests)
