#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from paddle import _C_ops
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestPool2dOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [1, 3, 32, 32],
            "data_format": "NCHW",
            "dtype": "float32",
            "pooling_type": "max",
            "kernel_size": [2, 2],
            "stride": [1, 1],
            "padding": [0, 0],
            "padding_algorithm": "VALID",
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "adaptive": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["shape"], dtype=self.attrs["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        out = _C_ops.pool2d(
            x,
            self.attrs["kernel_size"],
            self.attrs["stride"],
            self.attrs["padding"],
            self.attrs["ceil_mode"],
            self.attrs["exclusive"],
            self.attrs["data_format"],
            self.attrs["pooling_type"],
            self.attrs["global_pooling"],
            self.attrs["adaptive"],
            self.attrs["padding_algorithm"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["shape"], "x"
        )
        out = builder.pool2d(
            x,
            pooling_type=self.attrs["pooling_type"],
            kernel_size=self.attrs["kernel_size"],
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            ceil_mode=self.attrs["ceil_mode"],
            exclusive=self.attrs["exclusive"],
            data_format=self.attrs["data_format"],
            global_pooling=self.attrs["global_pooling"],
            adaptive=self.attrs["adaptive"],
            padding_algorithm=self.attrs["padding_algorithm"],
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.x_np], [out], passes=[]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

class TestPool2dOpPadding(TestPool2dOp):
    def setUp(self):
        self.attrs = {
            "shape": [1, 3, 32, 32],
            "data_format": "NCHW",
            "dtype": "float32",
            "pooling_type": "max",
            "kernel_size": [2, 2],
            "stride": [1, 1],
            "padding": [1, 1],
            "padding_algorithm": "SAME",
            "global_pooling": False,
            "ceil_mode": True,
            "exclusive": True,
            "adaptive": False,
        }
        self.prepare_inputs()

class TestPool2dOpAvg(TestPool2dOp):
    def setUp(self):
        self.attrs = {
            "shape": [1, 3, 32, 32],
            "data_format": "NCHW",
            "dtype": "float32",
            "pooling_type": "avg",
            "kernel_size": [3, 3],
            "stride": [2, 2],
            "padding": [0, 0],
            "padding_algorithm": "VALID",
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "adaptive": False,
        }
        self.prepare_inputs()

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestPool2dOpNHWC(TestPool2dOp):
    def setUp(self):
        self.attrs = {
            "shape": [1, 32, 32, 3],
            "data_format": "NHWC",
            "dtype": "float32",
            "pooling_type": "max",
            "kernel_size": [2, 2],
            "stride": [1, 1],
            "padding": [0, 0],
            "padding_algorithm": "VALID",
            "global_pooling": False,
            "ceil_mode": False,
            "exclusive": True,
            "adaptive": False,
        }
        self.prepare_inputs()

# class TestPool2dOpAdaptive(TestPool2dOp):
#     def setUp(self):
#         self.attrs = {
#             "shape": [32, 3, 64, 64],
#             "data_format": "NCHW",
#             "dtype": "float32",
#             "pooling_type": "avg",
#             "kernel_size": [5, 5],
#             "stride": [3, 3],
#             "padding": [1, 1],
#             "padding_algorithm": "EXPLICIT",
#             "global_pooling": False,
#             "ceil_mode": False,
#             "exclusive": True,
#             "adaptive": True,
#         }
#         self.prepare_inputs()

#     def test_check_results(self):
#         self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestPool2dBackwardOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [1, 32, 32, 3],
            "data_format": "NHWC",
            "dtype": "float32",
            "pooling_type": "max",
            "kernel_size": [2, 2],
            "stride": [1, 1],
            "padding": [0, 0],
            "padding_algorithm": "SAME",
            "global_pooling": False,
            "ceil_mode": True,
            "exclusive": False,
            "adaptive": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["shape"], dtype=self.attrs["dtype"]
        )
        self.dy_np = self.random(
            shape=self.attrs["shape"], dtype=self.attrs["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        forward_out = _C_ops.pool2d(
            x,
            self.attrs["kernel_size"],
            self.attrs["stride"],
            self.attrs["padding"],
            self.attrs["ceil_mode"],
            self.attrs["exclusive"],
            self.attrs["data_format"],
            self.attrs["pooling_type"],
            self.attrs["global_pooling"],
            self.attrs["adaptive"],
            self.attrs["padding_algorithm"],
            True,  # Need in paddlepaddle-2.4.2, will be removed in paddlepaddle-2.5
        )
        self.paddle_outputs = [forward_out]
        self.paddle_grads = self.get_paddle_grads(
            [forward_out], [x], [self.dy_np]
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("pool2d")
        # forward
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["shape"], "x"
        )
        y = builder.pool2d(
            x,
            kernel_size=self.attrs["kernel_size"],
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            ceil_mode=self.attrs["ceil_mode"],
            exclusive=self.attrs["exclusive"],
            data_format=self.attrs["data_format"],
            pooling_type=self.attrs["pooling_type"],
            global_pooling=self.attrs["global_pooling"],
            adaptive=self.attrs["adaptive"],
            padding_algorithm=self.attrs["padding_algorithm"],
        )
        # backward
        dy = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["shape"], "dy"
        )
        dx = builder.pool2d_grad(
            x,
            y,
            dy,
            kernel_size=self.attrs["kernel_size"],
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            ceil_mode=self.attrs["ceil_mode"],
            exclusive=self.attrs["exclusive"],
            data_format=self.attrs["data_format"],
            pooling_type=self.attrs["pooling_type"],
            global_pooling=self.attrs["global_pooling"],
            adaptive=self.attrs["adaptive"],
            padding_algorithm=self.attrs["padding_algorithm"],
        )
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, dy], [self.x_np, self.dy_np], [y, dx], passes=[]
        )
        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1]]

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestPool2dBackwardOpPadding(TestPool2dBackwardOp):
    def setUp(self):
        self.attrs = {
            "shape": [1, 3, 32, 32],
            "data_format": "NHWC",
            "dtype": "float32",
            "pooling_type": "max",
            "kernel_size": [3, 3],
            "stride": [2, 2],
            "padding": [1, 1],
            "padding_algorithm": "VALID",
            "global_pooling": False,
            "ceil_mode": True,
            "exclusive": False,
            "adaptive": False,
        }
        self.prepare_inputs()

# class TestPool2dBackwardOpAvg(TestPool2dBackwardOp):
#     def setUp(self):
#         self.attrs = {
#             "shape": [1, 3, 32, 32],
#             "data_format": "NHWC",
#             "dtype": "float32",
#             "pooling_type": "avg",
#             "kernel_size": [2, 2],
#             "stride": [2, 2],
#             "padding": [0, 0],
#             "padding_algorithm": "SAME",
#             "global_pooling": True,
#             "ceil_mode": False,
#             "exclusive": True,
#             "adaptive": False,
#         }
#         self.prepare_inputs()

# class TestPool2dBackwardOpAdaptive(TestPool2dBackwardOp):
#     def setUp(self):
#         self.attrs = {
#             "shape": [32, 3, 64, 64],
#             "data_format": "NHWC",
#             "dtype": "float32",
#             "pooling_type": "avg",
#             "kernel_size": [3, 3],
#             "stride": [2, 2],
#             "padding": [1, 1],
#             "padding_algorithm": "EXPLICIT",
#             "global_pooling": False,
#             "ceil_mode": False,
#             "exclusive": True,
#             "adaptive": True,
#         }
#         self.prepare_inputs()

if __name__ == "__main__":
    unittest.main()
