import numpy as np
import paddle
from paddle.cinn.frontend import NetBuilder

from op_test import OpTest, OpTestTool, is_compiled_with_device

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestUnaryOp(OpTest):
    def setUp(self):
        # print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.init_case()

    def get_x_data(self):
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_axis_value(self):
        return -1

    def init_case(self):
        self.inputs = {"x": self.get_x_data()}

    def paddle_func(self, x):
        raise NotImplementedError

    def cinn_func(self, builder, x):
        raise NotImplementedError

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)

        out = self.paddle_func(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("unary_op_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out]
        )

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestBinaryOp(OpTest):
    def setUp(self):
        # print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.init_case()

    def get_x_data(self):
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_y_data(self):
        return self.random([32, 64], 'float32', -10.0, 10.0)

    def get_axis_value(self):
        return -1

    def init_case(self):
        self.inputs = {"x": self.get_x_data(), "y": self.get_y_data()}
        self.axis = self.get_axis_value()

    def paddle_func(self, x, y):
        raise NotImplementedError

    def cinn_func(self, builder, x, y, axis):
        raise NotImplementedError

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)

        def get_unsqueeze_axis(x_rank, y_rank, axis):
            self.assertTrue(
                x_rank >= y_rank,
                "The rank of x should be greater or equal to that of y.",
            )
            axis = axis if axis >= 0 else x_rank - y_rank
            unsqueeze_axis = (
                np.arange(0, axis).tolist()
                + np.arange(axis + y_rank, x_rank).tolist()
            )

            return unsqueeze_axis

        unsqueeze_axis = get_unsqueeze_axis(
            len(self.inputs["x"].shape), len(self.inputs["y"].shape), self.axis
        )
        y_t = (
            paddle.unsqueeze(y, axis=unsqueeze_axis)
            if len(unsqueeze_axis) > 0
            else y
        )
        out = self.paddle_func(x, y_t)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("binary_op_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        y = builder.create_input(
            self.nptype2cinntype(self.inputs["y"].dtype),
            self.inputs["y"].shape,
            "y",
        )
        out = self.cinn_func(builder, x, y, axis=self.axis)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()

@OpTestTool.skip_if(
    not is_compiled_with_device(), "x86 test will be skipped due to timeout."
)
class TestReduceOp(OpTest):
    def setUp(self):
        # print(f"\nRunning {self.__class__.__name__}: {self.case}")
        self.init_case()

    def init_case(self):
        self.inputs = {"x": self.random([10, 10, 10], "float32", -1.0, 1.0)}
        self.dim = []
        self.keep_dim = False

    def paddle_func(self, x):
        raise NotImplementedError

    def cinn_func(self, builder, x):
        raise NotImplementedError

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = self.paddle_func(x)
        self.paddle_outputs = [out]

    # Note: If the forward and backward operators are run in the same program,
    # the forward result will be incorrect.
    def build_cinn_program(self, target):
        builder = NetBuilder("reduce_op_test")
        x = builder.create_input(
            self.nptype2cinntype(self.inputs["x"].dtype),
            self.inputs["x"].shape,
            "x",
        )
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res

    def test_check_results(self):
        self.check_outputs_and_grads()