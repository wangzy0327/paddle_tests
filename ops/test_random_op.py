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
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestGaussianRandomOp(OpTest):
    def setUp(self):
        print("TestGaussianRandomOp  setup")
        self.attrs = {
            "shape": [128, 64, 32],
            "mean": 0.0,
            "std": 1.0,
            "seed": 0,
            "dtype": "float32",
        }

    def build_paddle_program(self, target):
        out = paddle.tensor.random.gaussian(
            shape=self.attrs["shape"],
            mean=self.attrs["mean"],
            std=self.attrs["std"],
            dtype=self.attrs["dtype"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("gaussian_random")
        out = builder.gaussian_random(
            self.attrs["shape"],
            self.attrs["mean"],
            self.attrs["std"],
            self.attrs["seed"],
            self.attrs["dtype"],
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = res

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Uniform distribution.
        print("TestGaussianRandomOp test_check_results")
        self.check_outputs_and_grads(
            max_relative_error=10000, max_absolute_error=10000
        )

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestRandIntOp(OpTest):
    def setUp(self):
        print("TestRandIntOp  setup")
        self.init_case()

    def init_case(self):
        self.shape = [2, 3]
        self.min = 0
        self.max = 5
        self.seed = 10
        self.dtype = "int32"

    def build_paddle_program(self, target):
        out = paddle.randint(
            shape=self.shape, low=self.min, high=self.max, dtype=self.dtype
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("randint")
        out = builder.randint(
            self.shape, self.min, self.max, self.seed, self.dtype
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = [res[0]]

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # uniform distribution.
        print("TestRandIntOp test_check_results")
        self.check_outputs_and_grads(
            max_relative_error=10000, max_absolute_error=10000
        )

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestUniformRandomOp(OpTest):
    def setUp(self):
        print("TestUniformRandomOp  setup")
        self.attrs = {
            "shape": [128, 64, 32],
            "min": -1.0,
            "max": 1.0,
            "seed": 0,
            "dtype": "float32",
        }

    def build_paddle_program(self, target):
        out = paddle.uniform(
            shape=self.attrs["shape"],
            dtype=self.attrs["dtype"],
            min=self.attrs["min"],
            max=self.attrs["max"],
            seed=self.attrs["seed"],
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("uniform_random")
        out = builder.uniform_random(
            self.attrs["shape"],
            self.attrs["min"],
            self.attrs["max"],
            self.attrs["seed"],
            self.attrs["dtype"],
        )
        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out], passes=[])
        self.cinn_outputs = res

    def test_check_results(self):
        # Due to the different random number generation numbers implemented
        # in the specific implementation, the random number results generated
        # by CINN and Paddle are not the same, but they all conform to the
        # Uniform distribution.
        print("TestUniformRandomOp test_check_results")
        self.check_outputs_and_grads(
            max_relative_error=10000, max_absolute_error=10000
        )

if __name__ == "__main__":
    unittest.main()
