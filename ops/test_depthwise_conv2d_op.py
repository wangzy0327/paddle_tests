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
import paddle.nn as nn
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestDepthwiseConv2dOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [3, 32, 32, 16],
            "w_shape": [16, 1, 3, 3],
            "dtype": "float32",
            "data_format": "NHWC",
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "groups": 16,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["dtype"]
        )
        self.w_np = self.random(
            shape=self.attrs["w_shape"], dtype=self.attrs["dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = nn.initializer.Assign(self.w_np)
        if self.attrs["data_format"] == "NCHW":
            c_axis = 1
        elif self.attrs["data_format"] == "NHWC":
            c_axis = 3
        else:
            raise ValueError("Unknown data_format")
        conv = nn.Conv2D(
            in_channels=self.attrs["x_shape"][c_axis],
            out_channels=self.attrs["x_shape"][c_axis],
            kernel_size=self.attrs["kernel_size"],
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            dilation=self.attrs["dilation"],
            groups=self.attrs["groups"],
            weight_attr=weight,
            bias_attr=False,
            data_format=self.attrs["data_format"],
        )
        y = conv(x)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("depthwise_conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["x_shape"], "x"
        )
        weight = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]),
            self.attrs["w_shape"],
            "weight",
        )

        if self.attrs["data_format"] == "NCHW":
            y = builder.depthwise_conv2d(
                x,
                weight,
                strides=self.attrs["stride"],
                paddings=self.attrs["padding"],
                dilations=self.attrs["dilation"],
                groups=self.attrs["groups"],
                data_format=self.attrs["data_format"],
            )
        elif self.attrs["data_format"] == "NHWC":
            weight_t = builder.transpose(weight, [0, 2, 3, 1])
            y = builder.depthwise_conv2d(
                x,
                weight_t,
                strides=self.attrs["stride"],
                paddings=self.attrs["padding"],
                dilations=self.attrs["dilation"],
                groups=self.attrs["groups"],
                data_format=self.attrs["data_format"],
            )

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, weight], [self.x_np, self.w_np], [y], passes=[]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestDepthwiseConv2dOpPadding(TestDepthwiseConv2dOp):
    def setUp(self):
        self.attrs = {
            "x_shape": [3, 32, 32, 16],
            "w_shape": [16, 1, 3, 3],
            "dtype": "float32",
            "data_format": "NHWC",
            "kernel_size": [3, 3],
            "stride": [1, 1],
            "padding": [1, 1],
            "dilation": [1, 1],
            "groups": 16,
        }
        self.prepare_inputs()

if __name__ == "__main__":
    unittest.main()
