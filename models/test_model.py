
import os
import time
from numpy.testing import assert_allclose
os.environ['FLAGS_prim_all'] = 'true'
# os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
# os.environ['FLAGS_group_schedule_tiling_first'] = '1'
# os.environ['FLAGS_enable_pir_api'] = '1'
# os.environ['FLAGS_use_cinn'] = '1'
# os.environ['FLAGS_cinn_bucket_compile'] = '1'
import paddle
from paddle.vision.models import resnet18, resnet50, squeezenet1_1, mobilenet_v2
from paddleclas import EfficientNetB0 as efficientnet_b0
from facenet import FaceNet

paddle.device.set_device('mlu:0')
# model
# model = resnet18(pretrained=True)
# model = efficientnet_b0(pretrained=True)
model = FaceNet()
# x = paddle.rand([1, 3, 224, 224])
x = paddle.rand([1, 3, 240, 320])
# paddle inference
print("Running paddle inference...")
model.eval()
paddle_output = model(x)

# paddle.device.set_device('cpu')
# model_cpu = resnet18(pretrained=True)
# x_cpu = paddle.to_tensor(x)
# print("Running cpu inference...")
# model_cpu.eval()
# cpu_output = model_cpu(x_cpu)

paddle.set_flags({
    # "FLAGS_allow_cinn_ops": ";".join(cinn_ops_allowed),
})
print("Running paddle with cinn inference...")
build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True
static_model = paddle.jit.to_static(
    model,
    build_strategy=build_strategy,
    full_graph=True,
)
static_model.eval()
cinn_output = static_model(x)
paddle.device.synchronize()

assert_allclose(paddle_output.numpy(), cinn_output.numpy(), rtol=1e-5, atol=1e-5)
print("Test passed.")
