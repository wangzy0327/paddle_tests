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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestArangeOp(OpTest):
    def setUp(self):
        self.attrs = {
            "start": 0,
            "end": 100,
            "step": 1,
            "dtype": "int32",
        }

    def build_paddle_program(self, target):
        out = paddle.arange(
            self.attrs["start"],
            self.attrs["end"],
            self.attrs["step"],
            self.attrs["dtype"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("arange")
        out = builder.arange(
            self.attrs["start"],
            self.attrs["end"],
            self.attrs["step"],
            self.attrs["dtype"],
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestArgMaxOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [64, 16],
            "dtype": "float32",
            "axis": 1,
            "keepdim": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            self.attrs["shape"], self.attrs["dtype"], low=0, high=10
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.argmax(x, self.attrs["axis"], self.attrs["keepdim"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argmax")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        out = builder.argmax(x, self.attrs["axis"], self.attrs["keepdim"])
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.x_np], [out]
        )
        self.cinn_outputs = np.array(forward_res).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestArgMinOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [64, 16],
            "dtype": "float32",
            "axis": 1,
            "keepdim": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            self.attrs["shape"], self.attrs["dtype"], low=0, high=10
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.argmin(x, self.attrs["axis"], self.attrs["keepdim"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argmax")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        out = builder.argmin(x, self.attrs["axis"], self.attrs["keepdim"])
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.x_np], [out]
        )
        self.cinn_outputs = np.array(forward_res).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestArgSortOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [64, 16],
            "dtype": "float32",
            "axis": 1,
            "descending": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(self.attrs["shape"], self.attrs["dtype"])

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.argsort(x1, self.attrs["axis"], self.attrs["descending"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argsort")
        x = builder.create_input(
            self.nptype2cinntype(self.x_np.dtype), self.x_np.shape, "x"
        )
        out = builder.argsort(x, self.attrs["axis"], not self.attrs["descending"])
        prog = builder.build()
        forward_res = self.get_cinn_output(prog, target, [x], [self.x_np], out)
        self.cinn_outputs = np.array([forward_res[0]]).astype("int64")

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestBitcastConvertOp(OpTest):
    def setUp(self):
        self.init_case()

    # input[(3, 1), int32] --> output[(3, 1, 4), uint8]
    def init_case(self):
        from struct import pack, unpack
        data = np.random.random([3, 1]).astype(np.int32)
        packed = pack(data.size * 'i', *data.flatten())
        self.inputs = {"x": data}
        self.outputs = {
            "y": np.array(unpack('12B', packed), dtype='uint8').reshape(
                (3, 1, 4)
            ),
            "output_type": "uint8",
        }

    def build_paddle_program(self, target):
        y = paddle.to_tensor(self.outputs["y"], stop_gradient=False)
        self.paddle_outputs = [y]

    def build_cinn_program(self, target):
        builder = NetBuilder("bitcast_convert")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.bitcast_convert(x, self.outputs["output_type"])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()

class TestBroadcastToOp(OpTest):
    def setUp(self):
        self.init_case()

    def init_case(self):
        self.inputs = {"x": np.random.random([6]).astype("float32")}
        self.out_shape = [4, 5, 6]
        self.broadcast_axes = [2]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.broadcast_to(x, shape=self.out_shape)

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("BroadcastTo")
        x = builder.create_input(self.nptype2cinntype(self.inputs["x"].dtype), self.inputs["x"].shape, "x")
        out = builder.broadcast_to(
            x, out_shape=self.out_shape, broadcast_axes=self.broadcast_axes
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestConcatOp(OpTest):
    def setUp(self):
        self.attrs = {
            "axis": 0,
            "dtype": "float32",
            "shapes": [[10, 3, 5], [4, 3, 5]],
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {}
        for i, shape in enumerate(self.attrs["shapes"]):
            name = "x" + str(i)
            self.inputs[name] = self.random(shape, self.attrs["dtype"])

    def paddle_inputs(self, inputs):
        return [
            paddle.to_tensor(data, stop_gradient=True)
            for _, data in inputs.items()
        ]

    def cinn_inputs(self, builder, inputs):
        return [
            builder.create_input(
                self.nptype2cinntype(data.dtype), data.shape, name
            )
            for name, data in inputs.items()
        ]

    def build_paddle_program(self, target):
        out = paddle.concat(x=self.paddle_inputs(self.inputs), axis=self.attrs["axis"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("concat")
        input_list = self.cinn_inputs(builder, self.inputs)
        out = builder.concat(input_list, axis=self.attrs["axis"])

        prog = builder.build()

        input_datas = [data for _, data in self.inputs.items()]

        res = self.get_cinn_output(prog, target, input_list, input_datas, [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestConstantOp(OpTest):
    def setUp(self):
        self.attrs = {
            "dtype": "float32",
            "shape": [10, 3, 5],
            "constant_value": 10,
        }
        if "float" in self.attrs["dtype"]:
            self.value = float(self.attrs["constant_value"])

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.value, dtype=self.attrs["dtype"])
        self.paddle_outputs = [x]

    def build_cinn_program(self, target):
        builder = NetBuilder("constant")
        x = builder.constant(self.value, "x", self.attrs["dtype"])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [x])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestDropoutInferOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [128, 64, 32],
            "x_dtype": "float32",
            "p": 0.5,
            "mode": "downscale_in_infer",
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
        )
        if self.attrs["mode"] == 'upscale_in_train':
            self.attrs["cinn_mode"] = 'upscale_in_train'
        elif self.attrs["mode"] == 'downscale_in_infer':
            self.attrs["cinn_mode"] = 'downgrade_in_infer'
        else:
            raise f"Unknown mode for dropout_infer: {self.attrs['mode']}"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = F.dropout(
            x, p=self.attrs["p"], mode=self.attrs["mode"], training=False
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("dropout_infer")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["x_dtype"]),
            self.attrs["x_shape"],
            "x",
        )
        out = builder.dropout_infer(x, self.attrs["p"], self.attrs["cinn_mode"])
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestExpandDimsOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [32, 64],
            "axes_shape": [0, 2],
            "x_dtype": "float32",
        }
        self.init_case()

    def init_case(self):
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
        )

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = paddle.unsqueeze(x, self.attrs["axes_shape"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("expand_dims")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["x_dtype"]),
            self.attrs["x_shape"],
            "x",
        )
        out = builder.expand_dims(x, self.attrs["axes_shape"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestGatherOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [16, 32],
            "index": [32],
            "axis": 1,
            "x_dtype": "float32",
        }

    def build_paddle_program(self, target):
        dtype = self.attrs["x_dtype"]
        axis = self.attrs["axis"]
        x_shape = self.attrs["x_shape"]
        index_shape = self.attrs["index"]
        # Paddle does not support negative axis values.
        axis = axis if axis >= 0 else len(x_shape) + axis
        x = np.random.randn(*x_shape).astype(dtype)
        index = np.random.randint(0, x_shape[axis], index_shape).astype("int32")
        self.data = [x, index]
        x = paddle.to_tensor(x, stop_gradient=False)
        index = paddle.to_tensor(index, stop_gradient=False)
        out = paddle.gather(x, index, axis)
        self.paddle_outputs.append(out)

    def build_cinn_program(self, target):
        dtype = self.attrs["x_dtype"]
        axis = self.attrs["axis"]
        builder = NetBuilder("gather")
        x = builder.create_input(self.nptype2cinntype(dtype), self.attrs["x_shape"], "x")
        index = builder.create_input(self.nptype2cinntype("int32"), self.attrs["index"], "index")
        out = builder.gather(x, index, axis=axis)
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x, index], self.data, [out])
        self.cinn_outputs.extend(res)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestOneHotOp(OpTest):
    def setUp(self):
        self.attrs = {
            "x_shape": [32, 64],
            "x_dtype": "int32",
            "depth": 10,
            "axis": -1,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.x_np = self.random(
            shape=self.attrs["x_shape"], dtype=self.attrs["x_dtype"]
        )
        self.dtype = "float32"

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.x_np, stop_gradient=True)
        out = F.one_hot(x, num_classes=self.attrs["depth"])

        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("one_hot")
        x = builder.create_input(
            self.nptype2cinntype(self.attrs["x_dtype"]),
            self.attrs["x_shape"],
            "x",
        )
        on_value = builder.fill_constant(
            [1], 1, 'on_value', dtype=self.attrs["x_dtype"]
        )
        off_value = builder.fill_constant(
            [1], 0, 'off_value', dtype=self.attrs["x_dtype"]
        )
        out = builder.one_hot(
            x,
            on_value,
            off_value,
            depth=self.attrs["depth"],
            axis=self.attrs["axis"],
            dtype=self.dtype,
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.x_np], [out])

        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestRepeatOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [10, 3, 5],
            "dtype": "float32",
            "repeats": 2,
            "axis": 1,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        dims = len(self.attrs["shape"])
        axis = self.attrs["axis"]
        axis = min(axis, dims - 1)
        axis = max(axis, -dims)
        self.inputs = {
            "x": self.random(self.attrs["shape"], self.attrs["dtype"], -1.0, 1.0),
            "repeats": self.attrs["repeats"],
            "axis": axis,
        }

    def build_paddle_program(self, target):
        x = np.repeat(
            self.inputs["x"], self.inputs["repeats"], self.inputs["axis"]
        )
        out = paddle.to_tensor(x, stop_gradient=True)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("repeat")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.repeat(x, self.inputs["repeats"], self.inputs["axis"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestReshapeOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [1, 1024, 4],
            "dtype": "float32",
            "target_shape": [1, 2048, 2],
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.attrs["shape"], self.attrs["dtype"]),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.reshape(x, self.attrs["target_shape"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reshape_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.reshape(x, self.attrs["target_shape"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestReverseOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [10, 3, 5],
            "dtype": "float32",
            "axes": [0, 1],
            "net_builder_api": "reverse",
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        dims = len(self.attrs["shape"])
        axes = self.attrs["axes"].copy()
        for i in range(len(axes)):
            axes[i] = min(axes[i], dims - 1)
            axes[i] = max(axes[i], -dims)
        self.inputs = {
            "x": self.random(self.attrs["shape"], self.attrs["dtype"]),
            "axes": axes,
        }
        self.net_builder_api = self.attrs["net_builder_api"]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        if self.net_builder_api == "reverse":
            out = paddle.reverse(x, self.inputs["axes"])
        elif self.net_builder_api == "flip":
            out = paddle.flip(x, self.inputs["axes"])
        else:
            raise NotImplementedError
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reverse")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        if self.net_builder_api == "reverse":
            out = builder.reverse(x, self.inputs["axes"])
        elif self.net_builder_api == "flip":
            out = builder.flip(x, self.inputs["axes"])
        else:
            raise NotImplementedError

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSelectOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [32, 64],
            "dtype": "float32",
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "Condition": self.random(self.attrs["shape"], "bool"),
            "X": self.random(self.attrs["shape"], self.attrs["dtype"]),
            "Y": self.random(self.attrs["shape"], self.attrs["dtype"]),
        }

    def build_paddle_program(self, target):
        c = paddle.to_tensor(self.inputs["Condition"], stop_gradient=True)
        x = paddle.to_tensor(self.inputs["X"], stop_gradient=True)
        y = paddle.to_tensor(self.inputs["Y"], stop_gradient=True)

        out = paddle.where(c, x, y)
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("select")
        c = builder.create_input(
            self.nptype2cinntype(self.inputs["Condition"].dtype),
            self.inputs["Condition"].shape,
            "Condition",
        )
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["X"].dtype),
            self.inputs["X"].shape,
            "X",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["Y"].dtype),
            self.inputs["Y"].shape,
            "Y",
        )

        out = builder.select(c, x, y)
        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [c, x, y],
            [self.inputs["Condition"], self.inputs["X"], self.inputs["Y"]],
            [out],
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSliceOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [10, 12],
            "dtype": "float32",
            "axes": [0, 1],
            "starts": [2, 2],
            "ends": [5, 3],
            "strides": [1, 1],
            "decrease_axis": [1],
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "inputs": self.random(self.attrs["shape"], self.attrs["dtype"])
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["inputs"], stop_gradient=True)
        res = paddle.strided_slice(
            x, self.attrs["axes"], self.attrs["starts"], self.attrs["ends"], self.attrs["strides"]
        )
        out_shape = []
        for i in range(len(res.shape)):
            if i in self.attrs["decrease_axis"]:
                self.assertEqual(res.shape[i], 1)
            else:
                out_shape.append(res.shape[i])

        if len(out_shape) == 0:
            out_shape = [1]
        res = paddle.reshape(res, out_shape)
        self.paddle_outputs = [res]

    def build_cinn_program(self, target):
        builder = NetBuilder("slice")
        inputs = builder.create_input(
            self.nptype2cinntype(self.inputs["inputs"].dtype),
            self.inputs["inputs"].shape,
            "inputs",
        )
        out = builder.slice(
            inputs,
            axes=self.attrs["axes"],
            starts=self.attrs["starts"],
            ends=self.attrs["ends"],
            strides=self.attrs["strides"],
            decrease_axis=self.attrs["decrease_axis"],
        )

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [inputs], [self.inputs["inputs"]], [out]
        )
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSortOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [64, 16],
            "dtype": "float32",
            "axis": 0,
            "descending": False,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.attrs["shape"], self.attrs["dtype"])}

    def build_paddle_program(self, target):
        x1 = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.sort(x1, self.attrs["axis"], self.attrs["descending"])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sort")
        x1 = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.sort(x1, self.attrs["axis"], not self.attrs["descending"])
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x1], [self.inputs["x"]], [out]
        )

        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSplitOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [9, 9, 5],
            "dtype": "float32",
            "num_or_sections": [2, 3, 4],
            "axis": 0,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {
            "x": self.random(self.attrs["shape"], self.attrs["dtype"], -1.0, 1.0)
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        if len(self.attrs["num_or_sections"]) == 1:
            num = self.attrs["num_or_sections"][0]
        else:
            num = self.attrs["num_or_sections"]
        out = paddle.split(x, num_or_sections=num, axis=self.attrs["axis"])
        self.paddle_outputs = out

    def build_cinn_program(self, target):
        builder = NetBuilder("split")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.split(
            x, num_or_sections=self.attrs["num_or_sections"], axis=self.attrs["axis"]
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], out)
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestSqueezeOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [64, 32, 1],
            "dtype": "float32",
            "axes": [],
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.attrs["shape"], self.attrs["dtype"])}

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.squeeze(x, self.attrs["axes"])
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("squeeze")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.squeeze(x, self.attrs["axes"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestTopKOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [4, 32, 8], 
            "dtype": "float32",
            "k": 4,
            "axis": 1,
            "largest": True,
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.attrs["shape"], self.attrs["dtype"])}

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        if x.shape[self.attrs["axis"]] < self.attrs["k"]:
            self.attrs["k"] = x.shape[self.attrs["axis"]]
        out = paddle.topk(x, self.attrs["k"], self.attrs["axis"])
        self.paddle_outputs = [out[0], out[1]]

    def build_cinn_program(self, target):
        builder = NetBuilder("topk")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.top_k(x, self.attrs["k"], self.attrs["axis"], self.attrs["largest"])
        prog = builder.build()
        forward_res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out[0], out[1]]
        )
        self.cinn_outputs = forward_res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestTransposeOp(OpTest):
    def setUp(self):
        self.attrs = {
            "shape": [16, 8, 4, 2], 
            "dtype": "float32",
            "axes": [0, 2, 1, 3],
        }
        self.prepare_inputs()

    def prepare_inputs(self):
        self.inputs = {"x": self.random(self.attrs["shape"], self.attrs["dtype"])}

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=True)
        out = paddle.transpose(
            x,
            [
                axis + len(self.inputs["x"].shape) if axis < 0 else axis
                for axis in self.attrs["axes"]
            ],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("transpose_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = builder.transpose(x, self.attrs["axes"])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])
        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

if __name__ == "__main__":
    unittest.main()
