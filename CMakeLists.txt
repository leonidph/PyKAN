cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(kanTorch)


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

message(STATUS "===========================================================")
message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "===========================================================")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} ")
message(STATUS " version: ${Torch_VERSION}")




include_directories( /usr/include/torch/csrc/api/include
                     /usr/local/include/opencv4    )

add_executable(kanTorch
                main.cpp
                DataSrc.h DataSrc.cc
                task.h task.cpp
                kan.h kan.cpp
                rnn.h rnn.cpp
                simpleNet.h simpleNet.cpp
                simpleFunc.h simpleFunc.cpp
                chebkan.cpp chebkan.hpp)
target_link_libraries(${PROJECT_NAME}  "${TORCH_LIBRARIES}"  ${OpenCV_LIBS} )



message(STATUS "*******************************************************************")
message(STATUS " TARGET_LINK_LIBRARIES: ${TARGET_LINK_LIBRARIES}")
message(STATUS "*******************************************************************")

set_property(TARGET kanTorch PROPERTY CXX_STANDARD 17)
