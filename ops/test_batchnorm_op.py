#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest

import paddle
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

# @OpTestTool.skip_if(
#     not is_compiled_with_device(), "x86 test will be skipped due to timeout."
# )
# class TestBatchNormTrainOp(OpTest):
#     def setUp(self):
#         self.attrs = {
#             "x_shape": [2, 16, 8, 8],
#             "x_dtype": "float32",
#         }
#         self.prepare_inputs()

#     def prepare_inputs(self):
#         self.x_np = self.random(
#             shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
#         )

#     def build_paddle_program(self, target):
#         x = paddle.to_tensor(self.x_np)
#         batch_norm = paddle.nn.BatchNorm(
#             self.attrs["x_shape"][1], act=None, is_test=False
#         )
#         out = batch_norm(x)

#         self.paddle_outputs = [out]

#     # Note: If the forward and backward operators are run in the same program,
#     # the forward result will be incorrect.
#     def build_cinn_program(self, target):
#         builder = NetBuilder("batch_norm")
#         x = builder.create_input(
#             self.nptype2cinntype(self.attrs["x_dtype"]),
#             self.attrs["x_shape"],
#             "x",
#         )
#         scale = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 1.0, 'scale', 'float32'
#         )
#         bias = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 0.0, 'bias', 'float32'
#         )
#         mean = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 0.0, 'mean', 'float32'
#         )
#         variance = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 1.0, 'variance', 'float32'
#         )

#         out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)

#         prog = builder.build()
#         forward_res = self.get_cinn_output(
#             prog, target, [x], [self.x_np], out, passes=[]
#         )
#         self.cinn_outputs = [forward_res[0]]

#     def test_check_results(self):
#         self.check_outputs_and_grads()

# @OpTestTool.skip_if(
#     not is_compiled_with_device(), "x86 test will be skipped due to timeout."
# )
# class TestBatchNormBackwardOp(OpTest):
#     def setUp(self):
#         self.attrs = {
#             "x_shape": [2, 16, 8, 8],
#             "x_dtype": "float32",
#         }
#         self.prepare_inputs()

#     def prepare_inputs(self):
#         self.x_np = self.random(
#             shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
#         )
#         self.y_np = self.random(
#             shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
#         )

#     def build_paddle_program(self, target):
#         x = paddle.to_tensor(self.x_np, stop_gradient=False)
#         batch_norm = paddle.nn.BatchNorm(
#             self.attrs["x_shape"][1], act=None, is_test=False
#         )
#         out = batch_norm(x)

#         self.paddle_outputs = [out]
#         self.paddle_grads = self.get_paddle_grads([out], [x], [self.y_np])

#     # Note: If the forward and backward operators are run in the same program,
#     # the forward result will be incorrect.
#     def build_cinn_program(self, target):
#         builder = NetBuilder("batch_norm")
#         x = builder.create_input(
#             self.nptype2cinntype(self.attrs["x_dtype"]),
#             self.attrs["x_shape"],
#             "x",
#         )
#         scale = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 1.0, 'scale', 'float32'
#         )
#         bias = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 0.0, 'bias', 'float32'
#         )
#         mean = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 0.0, 'mean', 'float32'
#         )
#         variance = builder.fill_constant(
#             [self.attrs["x_shape"][1]], 1.0, 'variance', 'float32'
#         )

#         out = builder.batchnorm(x, scale, bias, mean, variance, is_test=False)

#         prog = builder.build()
#         forward_res = self.get_cinn_output(
#             prog, target, [x], [self.x_np], out, passes=[]
#         )
#         self.cinn_outputs = [forward_res[0]]

#         builder_grad = NetBuilder("batch_norm_grad")
#         dout = builder_grad.create_input(
#             self.nptype2cinntype(self.attrs["x_dtype"]),
#             self.attrs["x_shape"],
#             "dout",
#         )
#         x_g = builder_grad.create_input(
#             self.nptype2cinntype(self.attrs["x_dtype"]),
#             self.attrs["x_shape"],
#             "x_g",
#         )
#         scale_g = builder_grad.fill_constant(
#             scale.shape(), 1.0, 'scale_g', 'float32'
#         )
#         save_mean = builder_grad.create_input(
#             self.nptype2cinntype('float32'), out[1].shape(), "save_mean"
#         )
#         save_variance = builder_grad.create_input(
#             self.nptype2cinntype('float32'), out[2].shape(), "save_variance"
#         )

#         out_grad = builder_grad.batch_norm_grad(
#             dout, x_g, scale_g, save_mean, save_variance
#         )
#         prog = builder_grad.build()
#         backward_res = self.get_cinn_output(
#             prog,
#             target,
#             [dout, x_g, save_mean, save_variance],
#             [self.y_np, self.x_np, forward_res[1], forward_res[2]],
#             out_grad,
#             passes=[],
#         )
#         self.cinn_grads = [backward_res[0]]

#     def test_check_results(self):
#         self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestBatchNormInferOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [2, 16, 8, 8],
            "x_dtype": "float32",
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np)
        batch_norm = paddle.nn.BatchNorm(
            self.attrs["x_shape"][1], act=None, is_test=True
        )
        out = batch_norm(x)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("batch_norm")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["x_dtype"]),
            self.attrs["x_shape"],
            "x",
        )
        scale = builder.fill_constant(
            [self.attrs["x_shape"][1]], 1.0, 'scale', 'float32'
        )
        bias = builder.fill_constant(
            [self.attrs["x_shape"][1]], 0.0, 'bias', 'float32'
        )
        mean = builder.fill_constant(
            [self.attrs["x_shape"][1]], 0.0, 'mean', 'float32'
        )
        variance = builder.fill_constant(
            [self.attrs["x_shape"][1]], 1.0, 'variance', 'float32'
        )

        out = builder.batchnorm(x, scale, bias, mean, variance, is_test=True)

        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.x_np], out, passes=[]
        )
        self.cinn_outputs = [forward_res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()

if __name__ == "__main__":
    unittest.main()
