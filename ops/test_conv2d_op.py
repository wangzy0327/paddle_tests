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
import paddle.nn.functional as F
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestConv2dOp(OpTest):
    def setUp(self):
        print("TestConv2dOp setup")
        self.attrs = {
            "x_shape": [3, 32, 32, 16],
            "w_shape": [16, 16, 3, 3],
            "dtype": "float32",
            "data_format": "NHWC",
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "groups": 1,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        print("TestConv2dOp prepare inputs")
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["dtype"]
        )
        self.w_np = self.random(
            shape=self.attrs["w_shape"], dtype=self.attrs["dtype"]
        )

    def build_paddle_program(self, target):
        print("TestConv2dOp build paddle program")        
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        weight = paddle.to_tensor(self.w_np, stop_gradient=True)
        y = F.conv2d(
            x,
            weight,
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            dilation=self.attrs["dilation"],
            groups=self.attrs["groups"],
            data_format=self.attrs["data_format"],
        )
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        print("TestConv2dOp build cinn program") 
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["x_shape"], "x"
        )
        weight = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]),
            self.attrs["w_shape"],
            "weight",
        )

        if self.attrs["data_format"] == "NCHW":
            y = builder.conv2d(
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
            y = builder.conv2d(
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
            prog,
            target,
            [x, weight],
            [self.x_np, self.w_np],
            [y],
            passes=[],
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestConv2dOpPadding(TestConv2dOp):
    def setUp(self):
        print("TestConv2dOp Padding setup") 
        self.attrs = {
            "x_shape": [3, 32, 32, 16],
            "w_shape": [16, 16, 3, 3],
            "dtype": "float32",
            "data_format": "NHWC",
            "stride": [2, 2],
            "padding": [1, 1],
            "dilation": [1, 1],
            "groups": 1,
        }
        self.prepare_inputs()

def infer_output_shape(x_shape, w_shape, stride, padding, dilation):
    print("infer_output_shape") 
    batch_size, in_height, in_width, _ = x_shape
    out_channels, _, kernel_height, kernel_width = w_shape
    stride_height, stride_width = stride
    pad_height, pad_width = padding
    dilation_height, dilation_width = dilation

    out_height = (in_height + 2 * pad_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * pad_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
    return [batch_size, out_height, out_width, out_channels]

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestConv2dBackwardOp(OpTest):
    def setUp(self):
        print("TestConv2dBackwardOp setUp") 
        self.attrs = {
            "x_shape": [3, 32, 32, 16],
            "w_shape": [16, 16, 3, 3],
            "dtype": "float32",
            "data_format": "NHWC",
            "stride": [1, 1],
            "padding": [0, 0],
            "dilation": [1, 1],
            "groups": 1,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        print("TestConv2dBackwardOp prepare inputs")
        self.attrs["y_shape"] = infer_output_shape(self.attrs["x_shape"], self.attrs["w_shape"], self.attrs["stride"], self.attrs["padding"], self.attrs["dilation"])
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["dtype"]
        )
        self.w_np = self.random(
            shape=self.attrs["w_shape"], dtype=self.attrs["dtype"]
        )
        self.dy_np = self.random(
            shape=self.attrs["y_shape"], dtype=self.attrs["dtype"]
        )

    def build_paddle_program(self, target):
        print("TestConv2dBackwardOp build_paddle_program")        
        x = paddle.to_tensor(self.x_np, stop_gradient=False)
        weight = paddle.to_tensor(self.w_np, stop_gradient=False)
        y = F.conv2d(
            x,
            weight,
            stride=self.attrs["stride"],
            padding=self.attrs["padding"],
            dilation=self.attrs["dilation"],
            groups=self.attrs["groups"],
            data_format=self.attrs["data_format"],
        )
        self.paddle_outputs = [y]
        self.paddle_grads = self.get_paddle_grads(
            [y], [x, weight], [self.dy_np]
        )

    def build_cinn_program(self, target):
        print("TestConv2dBackwardOp build_cinn_program") 
        builder = NetBuilder("conv2d")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]), self.attrs["x_shape"], "x"
        )
        weight = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]),
            self.attrs["w_shape"],
            "weight",
        )
        dy = builder.create_input(
            self.nptype2cinntype(self.attrs["dtype"]),
            self.attrs["y_shape"],
            "dy",
        )

        if self.attrs["data_format"] == "NCHW":
            y = builder.conv2d(
                x,
                weight,
                strides=self.attrs["stride"],
                paddings=self.attrs["padding"],
                dilations=self.attrs["dilation"],
                groups=self.attrs["groups"],
                data_format=self.attrs["data_format"],
            )
            x_grad = builder.conv(
                weight,
                dy,
                data_format=self.attrs["data_format"],
                conv_type="backward_data",
                output_shape=x.shape(),
            )
            weight_grad = builder.conv(
                x,
                dy,
                data_format=self.attrs["data_format"],
                conv_type="backward_filter",
                output_shape=weight.shape(),
            )
        elif self.attrs["data_format"] == "NHWC":
            weight_t = builder.transpose(weight, [0, 2, 3, 1])
            y = builder.conv2d(
                x,
                weight_t,
                strides=self.attrs["stride"],
                paddings=self.attrs["padding"],
                dilations=self.attrs["dilation"],
                groups=self.attrs["groups"],
                data_format=self.attrs["data_format"],
            )
            x_grad = builder.conv(
                weight_t,
                dy,
                data_format=self.attrs["data_format"],
                conv_type="backward_data",
                output_shape=x.shape(),
            )
            w_grad = builder.conv(
                x,
                dy,
                data_format=self.attrs["data_format"],
                conv_type="backward_filter",
                output_shape=weight_t.shape(),
            )
            weight_grad = builder.transpose(w_grad, [0, 3, 1, 2])

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, weight, dy],
            [self.x_np, self.w_np, self.dy_np],
            [y, x_grad, weight_grad],
            passes=[],
        )
        self.cinn_outputs = [res[0]]
        self.cinn_grads = [res[1], res[2]]

    def test_check_results(self):
        print("TestConv2dBackwardOp test_check_results") 
        self.check_outputs_and_grads()

if __name__ == "__main__":
    unittest.main()
