//
// Created by mrjak on 23-04-2020.
//

/* Inspired by https://pytorch.org/tutorials/advanced/cpp_extension.html */
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <stdio.h>

// ---------------------------------------------------------------------------------- //
void add_cuda(at::Tensor A, at::Tensor B, at::Tensor C);

// ---------------------------------------------------------------------------------- //
void add(at::Tensor A, at::Tensor B, at::Tensor C) { add_cuda(A, B, C); }

// ---------------------------------------------------------------------------------- //
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m
) {
m.def("add",                 &add,              "");
}