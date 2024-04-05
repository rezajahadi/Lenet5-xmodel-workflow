# Vitis AI PyTorch Design Process

This repository provides an implementation guide for deploying PyTorch models using the Vitis AI framework on Xilinx® evaluation boards. The design process outlined here guides users through the steps necessary to convert a Python description of a neural network model to a compiled model that can be executed on Xilinx® hardware.

**Baseline Implementation:**

The baseline implementation presented here is derived from the AMD Vitis AI tutorial. While the core concepts remain similar, the implementation has been tailored to suit the Xilinx® platform.

## Overview

The process involves several key steps:

1. **Model Design**: Begin by designing your neural network model in PyTorch. This includes defining the architecture, specifying layers, and configuring parameters.

2. **Quantization**: Perform quantization to convert the model's parameters and activations to lower precision. This step is crucial for optimizing the model to run efficiently on FPGA hardware.

3. **Compilation**: Utilize the Vitis AI compiler to compile the quantized model. This step transforms the model into a format compatible with Xilinx® FPGAs.

4. **Deployment**: Finally, deploy the compiled model onto the Xilinx® evaluation board. This step involves loading the model onto the board and executing inference tasks.

## Getting Started

To begin, ensure you have the following prerequisites installed:

- Python (with PyTorch)
- Vitis AI
- Xilinx® evaluation board setup

Follow these steps to run the compiled model on your Xilinx® evaluation board:

1. **Clone Repository**: Clone this repository to your local machine.

2. **Model Design**: Design your neural network model using PyTorch. Ensure that the model architecture is compatible with Vitis AI requirements.

3. **Quantization**: Perform quantization on the model using the provided scripts or customize them to suit your specific requirements.

4. **Compilation**: Use the Vitis AI compiler to compile the quantized model. Refer to the documentation for detailed instructions on how to compile models for your target hardware.

5. **Deployment**: Deploy the compiled model onto your Xilinx® evaluation board. Follow the provided instructions to load the model onto the board and execute inference tasks.

## Contributing

Contributions to this repository are welcome. If you have suggestions for improvements or encounter any issues, please feel free to open an issue or submit a pull request.
