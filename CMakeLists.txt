cmake_minimum_required(VERSION 2.8)
project(xx)

add_definitions(-D CUDA_USE_STATIC_CUDA_RUNTIME=OFF)
find_package(OpenCV 2.4.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread system)

add_definitions(-DCOMPILEDWITHC11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   message(STATUS "Using flag -std=c++11.")
add_definitions(-DCOMPILEDWITHC0X)
if (OpenCV_FOUND)
	message(fasfdfafdadfda)
endif()
include_directories(
${EIGEN3_INCLUDE_DIR}
${Boost_INCLUDE_DIRS}
${Pangolin_INCLUDE_DIRS}
)

add_executable(msckf test.cpp)
target_link_libraries(msckf
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${EIGEN3_LIBS}
${Boost_LIBRARIES})

add_executable(try  src/msckf.cc)
target_link_libraries(try
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${EIGEN3_LIBS}
${Boost_LIBRARIES})


