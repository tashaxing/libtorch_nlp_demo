cmake_minimum_required(VERSION 3.0)

project(nlp_predicator)

if (UNIX)
	add_definitions(-std=c++11)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)
find_package(Torch REQUIRED)

set (SRC
	src/nlp_predicator.cpp
)

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME}
	${TORCH_LIBRARIES}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)