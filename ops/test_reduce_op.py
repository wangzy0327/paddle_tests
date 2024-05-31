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

from op_test_helper import TestReduceOp

class TestReduceSumOp(TestReduceOp):
    def paddle_func(self, x):
        return paddle.sum(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_sum(x, self.dim, self.keep_dim)
    
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [1]
        self.keep_dim = False

class TestReduceProdOp(TestReduceOp):
    def paddle_func(self, x):
        return paddle.prod(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_prod(x, self.dim, self.keep_dim)
    
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False

class TestReduceMaxOp(TestReduceOp):
    def paddle_func(self, x):
        return paddle.max(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_max(x, self.dim, self.keep_dim)
    
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

class TestReduceMinOp(TestReduceOp):
    def paddle_func(self, x):
        return paddle.min(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_min(x, self.dim, self.keep_dim)
    
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = [0, 1]
        self.keep_dim = False

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

class TestAllOp(TestReduceOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        return paddle.all(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_all(x, self.dim, self.keep_dim)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

class TestAnyOp(TestReduceOp):
    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "bool")}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        return paddle.any(x, axis=self.dim, keepdim=self.keep_dim)

    def cinn_func(self, builder, x):
        return builder.reduce_any(x, self.dim, self.keep_dim)

    def test_check_results(self):
        self.check_outputs_and_grads(all_equal=True)

del(TestReduceOp)

if __name__ == "__main__":
    unittest.main()
