project(HNSWLIB_JNI)

cmake_minimum_required(VERSION 3.6.0)

find_package(Java REQUIRED)
include(UseJava)

set(HNSWLIB_JNI_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -march=x86-64 -Wall -pedantic -mavx -msse4 -mf16c -fvisibility=hidden")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

find_package(JNI)
if (DEFINED JNI_INCLUDE_DIRS)
    message (STATUS "JNI_INCLUDE_DIRS=${JNI_INCLUDE_DIRS}")
    message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")
endif()

include_directories(${JNI_INCLUDE_DIRS} src/main/includes src/main/cpp)

set(SOURCE_FILES src/main/cpp/hnswLibJni.cpp src/main/cpp/mathlib_jni.cpp)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $ENV{CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $ENV{CMAKE_RUNTIME_OUTPUT_DIRECTORY})
add_library(HNSWLIB_OBJ OBJECT ${SOURCE_FILES})
add_library(HNSWLIB_JNI SHARED ${SOURCE_FILES})

target_link_libraries(HNSWLIB_JNI)

include(CTest)
enable_testing()

# Prepare doctest for other targets to use
add_library(doctest INTERFACE)
target_include_directories(doctest INTERFACE src/test/cpp/doctest.h)

# Make test executable
add_executable(tests_knn src/test/cpp/main.cpp src/test/cpp/knn_distance_test.cpp src/test/cpp/knn_index_test.cpp src/test/cpp/knn_search_test.cpp $<TARGET_OBJECTS:HNSWLIB_OBJ> src/test/cpp/float8_test.cpp)
target_link_libraries(tests_knn doctest)
add_test(NAME tests COMMAND $<TARGET_FILE:tests_knn>)