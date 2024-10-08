import paddle
import os
import time
import numpy as np
import paddleclas
import facenet


def set_flags(use_cinn):
    if use_cinn:
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_prim_all'] = 'true'
        os.environ['FLAGS_prim_enable_dynamic'] = 'true'
        # os.environ['FLAGS_print_ir'] = '1'
        os.environ['FLAGS_enable_pir_api'] = '1'
        os.environ['FLAGS_use_cinn'] = '1'
        os.environ['FLAGS_cinn_bucket_compile'] = '1'
        os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'
    else:
        os.environ['FLAGS_use_cinn'] = '0'

def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )

def benchmark(net, input, repeat=5, warmup=3):
    # warm up
    for _ in range(warmup):
        net(input)
        paddle.device.synchronize()
    # time
    t = []
    for _ in range(repeat):
        t1 = time.time()
        net(input)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("--[benchmark] Run for %d times, the average latency is: %f ms" % (repeat, np.mean(t)))

class TestBase:
    def __init__(self):
        device_info = paddle.get_device()
        print("Current Paddle device : %s"%(device_info))
        self.net = None
        self.input = None
        self.cinn_net = None

    def to_eval(self, use_cinn):
        set_flags(use_cinn)
        if use_cinn:
            if not self.cinn_net:
                self.cinn_net = to_cinn_net(self.net)
            net = self.cinn_net
        else:
            net = self.net
        net.eval()
        return net

    def eval(self, use_cinn):
        net = self.to_eval(use_cinn)
        out = net(self.input)
        return out

    def check_cinn_output(self):
        pd_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), pd_out.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, self.input)

class TestResNet18(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddle.vision.models.resnet18(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class TestResNet50(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddle.vision.models.resnet50(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class TestSqueezeNet(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddle.vision.models.squeezenet1_1(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class TestMobileNetV2(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddle.vision.models.mobilenet_v2(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class EfficientNetB0(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddleclas.EfficientNetB0(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class FaceNet(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = facenet.FaceNet()
        self.input = paddle.randn([batch_size, 3, 240, 320])                

if __name__ == "__main__":
    print("Test ResNet18 ........")
    model = TestResNet18()       
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test ResNet50 ........")
    model = TestResNet50()       
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test SqueezeNet ........")
    model = TestSqueezeNet()
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test MobileNet ........")
    model = TestMobileNetV2()
    # model.check_cinn_output()    
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test EfficientNetB0 ........")
    # model.check_cinn_output()    
    model = EfficientNetB0()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test FaceNet ........")
    model = FaceNet()
    # model.check_cinn_output()    
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)          
