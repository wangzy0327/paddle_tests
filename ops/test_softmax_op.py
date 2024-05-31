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
import paddle.nn.functional as F
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSoftmaxOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [1, 64, 32],
            "dtype": "float32",
            "axis": 1,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.attrs["shape"], self.attrs["dtype"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = F.softmax(x, axis=self.attrs["axis"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("softmax")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["shape"], "x"
        )
        out = builder.softmax(x, axes=[self.attrs["axis"]])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

if __name__ == "__main__":
    unittest.main()
