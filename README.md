# Local Gemma AI Chatbot

A locally running AI chatbot powered by Google's Gemma 4 E4B model using
llama.cpp, llama-cpp-python, Vulkan, and GPU acceleration.

## Overview

This project runs a quantized Gemma language model locally on a Windows PC
without requiring a cloud-based LLM API.

The model is executed using the llama.cpp inference engine and accelerated
using Vulkan on an NVIDIA RTX 3050 Laptop GPU.

## Architecture

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

## Features

- Local LLM inference
- Gemma 4 E4B model
- GGUF quantized model
- Vulkan GPU acceleration
- NVIDIA RTX 3050 support
- Python interface
- Interactive terminal chatbot

## Hardware

- CPU: AMD Ryzen 5 8645HS
- RAM: 16 GB DDR5
- GPU: NVIDIA GeForce RTX 3050 6 GB
- OS: Windows

## Current Status

Working prototype.

The model can be loaded locally and used through an interactive
terminal chatbot.

## Future Improvements

- Conversation memory
- Streaming responses
- Chat commands
- Improved system prompts
- Graphical user interface
- Tool integration
- Image generation integration
