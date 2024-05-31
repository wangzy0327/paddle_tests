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

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestMatmulOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [128, 64],
            "y_shape": [64, 32],
            "transx": False,
            "transy": False,
            "dtype": "float32",
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.attrs["x_shape"], self.attrs["dtype"])
        self.y_np = self.random(self.attrs["y_shape"], self.attrs["dtype"])

    def paddle_func(self, x, y):
        return paddle.matmul(
            x,
            y,
            transpose_x=self.attrs["transx"],
            transpose_y=self.attrs["transy"],
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        y = paddle.to_tensor(self.y_np, stop_gradient=True)
        out = self.paddle_func(x, y)
        self.paddle_outputs = [out]

    def cinn_func(self, builder, x, y):
        return builder.matmul(
            x,
            y,
            transpose_x=self.attrs["transx"],
            transpose_y=self.attrs["transy"],
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("matmul")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["x_shape"], "x"
        )
        y = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["y_shape"], "y"
        )
        out = self.cinn_func(builder, x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.x_np, self.y_np], [out], passes=[]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestMatmulOpBatch(TestMatmulOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [5, 4, 16],
            "y_shape": [5, 16, 32],
            "transx": False,
            "transy": False,
            "dtype": "float32",
        }
        self.prepare_inputs()

class TestMatmulOpRowVector(TestMatmulOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [4, 16],
            "y_shape": [16],
            "transx": False,
            "transy": False,
            "dtype": "float32",
        }
        self.prepare_inputs()

class TestMatmulOpColVector(TestMatmulOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [4, 16],
            "y_shape": [16, 1],
            "transx": False,
            "transy": False,
            "dtype": "float32",
        }
        self.prepare_inputs()

class TestMatmulOpTrans(TestMatmulOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [16, 4],
            "y_shape": [32, 16],
            "transx": True,
            "transy": True,
            "dtype": "float32",
        }
        self.prepare_inputs()

class TestMatmulOpBatchTrans(TestMatmulOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [10, 12, 128, 64],
            "y_shape": [10, 12, 128, 64],
            "transx": False,
            "transy": True,
            "dtype": "float32",
        }
        self.prepare_inputs()

if __name__ == "__main__":
    unittest.main()
