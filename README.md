# transfomer.cpp — Bit-Serial Inference Engine for CPU
---
An experimental CPU inference project that accelerates **linear layers** using **bit-slicing (bit-planes)** and **bit-serial computation**.  
It replaces conventional FP GEMM-style multiplication with **bitwise primitives** such as **AND** and **POPCOUNT** (plus lightweight accumulation).

## Description
---
This project converts an **INT4-quantized linear layer** into **four bit-planes** (one per bit position) and computes the layer output via **bit-serial accumulation**.

## Switch Boards: Bit-Plane Storage
---
Instead of storing weights as packed INT4 nibbles, this project stores weights as **four physically separated bit-planes**:

- **Board 3 (MSB)**: dominates large-magnitude contribution  
- **Board 0 (LSB)**: refines fine-grained precision

This representation makes the linear layer explicitly **bit-addressable** at runtime.

## Mathematical Formulation
---
Let an INT4 weight matrix \(W\) be decomposed into four binary matrices \(\mathbf{B}^{(i)}\):

\[
W = \sum_{i=0}^{3} 2^i \cdot \mathbf{B}^{(i)}
\]

Then the linear operation becomes:

\[
Y = WX = \sum_{i=0}^{3} 2^i \cdot (\mathbf{B}^{(i)}X)
\]

Each term \((\mathbf{B}^{(i)}X)\) is computed using bitwise operations and popcount-based accumulation.

## Current Status
---
- [x] INT4 conversion of FP32 weights
- [x] Bit-plane slicing into 4 switch boards (per linear layer)
- [x] Project structure focused on CPU execution using bitwise primitives (AND / POPCOUNT)