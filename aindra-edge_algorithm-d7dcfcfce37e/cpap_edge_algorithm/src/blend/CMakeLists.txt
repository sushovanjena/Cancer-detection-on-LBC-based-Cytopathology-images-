cmake_minimum_required(VERSION 2.8)

project(algoblend)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_capi "test_capi.cpp" )
target_link_libraries(test_capi ${PROJECT_NAME})

add_library(${PROJECT_NAME} SHARED "blend.cpp" "capi.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)
