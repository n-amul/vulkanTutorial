cmake_minimum_required(VERSION 3.17)

set(PROJECT_NAME vulkan_tutorial)
set(CMAKE_CXX_STANDARD 17)


set(WINDOW_NAME "Vulkan")
set(WINDOW_WIDTH 800)
set(WINDOW_HEIGHT 600)

project(${PROJECT_NAME})
add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/easylogging++.cc src/easylogging++.h
    )

include(Dependency.cmake)

# 우리 프로젝트에 include / lib 관련 옵션 추가
target_include_directories(${PROJECT_NAME} PUBLIC ${DEP_INCLUDE_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC ${DEP_LIB_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${DEP_LIBS})

#vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)

target_compile_definitions(${PROJECT_NAME} PUBLIC
WINDOW_NAME="${WINDOW_NAME}"
WINDOW_WIDTH=${WINDOW_WIDTH}
WINDOW_HEIGHT=${WINDOW_HEIGHT}
)

add_dependencies(${PROJECT_NAME} ${DEP_LIST})

if (MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC /wd4819)
endif()