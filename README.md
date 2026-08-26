# Local Gemma AI Chatbot

A locally running AI chatbot powered by Google's Gemma 4 E4B model using
`llama.cpp`, `llama-cpp-python`, Vulkan, and GPU acceleration.

## Overview

This project runs a quantized Gemma language model locally on a Windows PC
without requiring a cloud-based LLM API or an OpenAI API key.

The model is executed using the `llama.cpp` inference engine and accelerated
using Vulkan on an NVIDIA RTX 3050 Laptop GPU.

The project currently provides an interactive terminal-based chatbot through
Python, while also containing a C++/CMake implementation for llama.cpp.

## Architecture

```text
User
  ↓
Python chatbot
  ↓
llama-cpp-python
  ↓
llama.cpp
  ↓
Vulkan
  ↓
NVIDIA RTX 3050
  ↓
Gemma 4 E4B Q4_K_M