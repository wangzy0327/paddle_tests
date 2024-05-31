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

from op_test_helper import TestUnaryOp

class TestAbsOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.abs(x)

    def cinn_func(self, builder, x):
        return builder.abs(x)
    
class TestAcosOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.acos(x)

    def cinn_func(self, builder, x):
        return builder.acos(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestAcoshOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.acosh(x)

    def cinn_func(self, builder, x):
        return builder.acosh(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestAsinOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.asin(x)

    def cinn_func(self, builder, x):
        return builder.asin(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestAsinhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.asinh(x)

    def cinn_func(self, builder, x):
        return builder.asinh(x)
    
class TestAtanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.atan(x)

    def cinn_func(self, builder, x):
        return builder.atan(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(max_relative_error=1e-3, max_absolute_error=1e-3)

class TestAtanhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.atanh(x)

    def cinn_func(self, builder, x):
        return builder.atanh(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)
    
class TestBitwiseNotOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.bitwise_not(x)

    def cinn_func(self, builder, x):
        return builder.bitwise_not(x)
    
    def get_x_data(self):
        return self.random([32, 64], 'int32', 1, 10000)

class TestCastOp(TestUnaryOp):
    def setUp(self):
        self.attrs = {
            "d_dtype": "int32",
        }
        self.init_case()

    def paddle_func(self, x):
        return paddle.cast(x, self.attrs["d_dtype"])
    
    def cinn_func(self, builder, x):
        return builder.cast(x, self.attrs["d_dtype"])

class TestCbrtOp(TestUnaryOp):
    def build_paddle_program(self, target):
        numpy_out = np.cbrt(self.inputs["x"])
        out = paddle.to_tensor(numpy_out, stop_gradient=False)
        self.paddle_outputs = [out]

    def cinn_func(self, builder, x):
        return builder.cbrt(x)
    
class TestCeilOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.ceil(x)

    def cinn_func(self, builder, x):
        return builder.ceil(x)
    
class TestCosOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.cos(x)

    def cinn_func(self, builder, x):
        return builder.cos(x)
    
class TestCoshOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.cosh(x)

    def cinn_func(self, builder, x):
        return builder.cosh(x)
    
class TestErfOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.erf(x)

    def cinn_func(self, builder, x):
        return builder.erf(x)

class TestExpOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.exp(x)

    def cinn_func(self, builder, x):
        return builder.exp(x)
    
class TestFloorOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.floor(x)

    def cinn_func(self, builder, x):
        return builder.floor(x)
    
class TestGeluOp(TestUnaryOp):
    def paddle_func(self, x):
        return F.gelu(x)

    def cinn_func(self, builder, x):
        return builder.gelu(x)

class TestIsFiniteOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isfinite(x)

    def cinn_func(self, builder, x):
        return builder.is_finite(x)
    
    def get_x_data(self):
        x = self.random([32, 64], 'float32', -10.0, 10.0)
        num = x.size // 2
        indices = np.random.choice(x.size, num, replace=False)
        np.put(x, indices[:num//2], np.nan)
        np.put(x, indices[num//2:num*3//4], np.inf)
        np.put(x, indices[num*3//4:], -np.inf)
        return x

class TestIsInfOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isinf(x)

    def cinn_func(self, builder, x):
        return builder.is_inf(x)
    
    def get_x_data(self):
        x = self.random([32, 64], 'float32', -10.0, 10.0)
        num = x.size // 2
        indices = np.random.choice(x.size, num, replace=False)
        np.put(x, indices[:num//2], np.nan)
        np.put(x, indices[num//2:num*3//4], np.inf)
        np.put(x, indices[num*3//4:], -np.inf)
        return x

class TestIsNanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.isnan(x)

    def cinn_func(self, builder, x):
        return builder.is_nan(x)
    
    def get_x_data(self):
        x = self.random([32, 64], 'float32', -10.0, 10.0)
        num = x.size // 2
        indices = np.random.choice(x.size, num, replace=False)
        np.put(x, indices[:num//2], np.nan)
        np.put(x, indices[num//2:num*3//4], np.inf)
        np.put(x, indices[num*3//4:], -np.inf)
        return x
    
class TestLogOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.log(x)

    def cinn_func(self, builder, x):
        return builder.log(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestNegativeOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.neg(x)

    def cinn_func(self, builder, x):
        return builder.negative(x)

class TestReluOp(TestUnaryOp):
    def paddle_func(self, x):
        return F.relu(x)

    def cinn_func(self, builder, x):
        return builder.relu(x)

class TestRoundOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.round(x)

    def cinn_func(self, builder, x):
        return builder.round(x)

class TestSigmoidOp(TestUnaryOp):
    def paddle_func(self, x):
        return F.sigmoid(x)

    def cinn_func(self, builder, x):
        return builder.sigmoid(x)
    
class TestSignOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sign(x)

    def cinn_func(self, builder, x):
        return builder.sign(x)
    
class TestSinOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sin(x)

    def cinn_func(self, builder, x):
        return builder.sin(x)

class TestSinhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sinh(x)

    def cinn_func(self, builder, x):
        return builder.sinh(x)

class TestSqrtOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.sqrt(x)

    def cinn_func(self, builder, x):
        return builder.sqrt(x)
    
    def test_check_results(self):
        self.check_outputs_and_grads(equal_nan=True)

class TestTanOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.tan(x)

    def cinn_func(self, builder, x):
        return builder.tan(x)

class TestTanhOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.tanh(x)

    def cinn_func(self, builder, x):
        return builder.tanh(x)

class TestTruncOp(TestUnaryOp):
    def paddle_func(self, x):
        return paddle.trunc(x)

    def cinn_func(self, builder, x):
        return builder.trunc(x)

del(TestUnaryOp)

if __name__ == "__main__":
    unittest.main()
