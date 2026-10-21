cmake_minimum_required(VERSION 3.5)
project(MyOpenCVProject)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

# 查找 OpenCV 库
find_package(OpenCV 4.5.4 REQUIRED)

# 包含 OpenCV 头文件目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成可执行文件（我们只生成一个，叫 my_app，源码用 main.cpp）
add_executable(my_app main.cpp)

# 将 OpenCV 库链接到我们的程序上
target_link_libraries(my_app ${OpenCV_LIBS})