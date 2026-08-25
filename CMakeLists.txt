cmake_minimum_required(VERSION 3.20)

project(gemma_chatbot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)

add_executable(gemma_chatbot
    src/main.cpp
)

target_link_libraries(gemma_chatbot PRIVATE llama)