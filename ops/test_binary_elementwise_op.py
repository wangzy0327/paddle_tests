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

import numpy as np
import paddle

from op_test_helper import TestBinaryOp

class TestAddOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.add(x, y, axis)

class TestBitwiseAndOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_and(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_and(x, y, axis)

class TestBitwiseOrOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_or(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_or(x, y, axis)

class TestBitwiseXorOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def paddle_func(self, x, y):
        return paddle.bitwise_xor(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.bitwise_xor(x, y, axis)
    
class TestCompareEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.equal(x, y, axis)
    
class TestCompareNotEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.not_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.not_equal(x, y, axis)
    
class TestCompareGreaterThanOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.greater_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.greater_than(x, y, axis)

class TestCompareLessThanOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.less_than(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.less_than(x, y, axis)
    
class TestCompareGreaterEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.greater_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.greater_equal(x, y, axis)

class TestCompareLessEqualOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.less_equal(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.less_equal(x, y, axis)

class TestDivideOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.divide(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.divide(x, y, axis)

class TestIsCloseOp(TestBinaryOp):
    def setUp(self):
        self.attrs = {
            "rtol": 1e-05,
            "atol": 1e-06,
            "equal_nan": False,
        }
        self.init_case()

    def paddle_func(self, x, y):
        return paddle.isclose(
            x, y, self.attrs["rtol"], self.attrs["atol"], self.attrs["equal_nan"]
        )

    def cinn_func(self, builder, x, y, axis):
        return builder.isclose(
            x, y, self.attrs["rtol"], self.attrs["atol"], self.attrs["equal_nan"]
        )

class TestLeftShiftOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10)

    def paddle_func(self, x, y):
        return paddle.bitwise_left_shift(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.left_shift(x, y, axis)

class TestMaxOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.maximum(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.max(x, y, axis)

class TestMinOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.minimum(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.min(x, y, axis)

class TestModOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.mod(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.mod(x, y, axis)

class TestMultiplyOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.multiply(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.multiply(x, y, axis)

class TestPowOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.pow(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.pow(x, y, axis)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestRightShiftOp(TestBinaryOp):
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

    def get_y_data(self):
        return self.random([32, 64], 'int32', 1, 10)

    def paddle_func(self, x, y):
        return paddle.bitwise_right_shift(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.right_shift(x, y, axis)

class TestSubtractOp(TestBinaryOp):
    def paddle_func(self, x, y):
        return paddle.subtract(x, y)

    def cinn_func(self, builder, x, y, axis):
        return builder.subtract(x, y, axis)

del(TestBinaryOp)

if __name__ == "__main__":
    unittest.main()
