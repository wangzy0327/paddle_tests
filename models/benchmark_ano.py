import paddle
import os
import time
import numpy as np
# import paddleclas
import facenet
import paddle.vision
import ppdet
from ppdet.core.workspace import load_config, create
from ppdet.engine import Trainer

import paddleseg

import paddleocr

from paddle import nn  
from paddle.io import DataLoader, TensorDataset

class LSTMModel(nn.Layer):  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  
        super(LSTMModel, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  
        self.fc = nn.Linear(hidden_size, num_classes)  
  
    def forward(self, x):  
        # 初始化隐藏状态和单元状态  
        h0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])  
        c0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])  
  
        # 前向传播LSTM  
        out, _ = self.lstm(x, (h0, c0))  
  
        # 取最后一个时间步的输出  
        out = self.fc(out[:, -1, :])  
        return out

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
    
def benchmark_ano(net, input, repeat=5, warmup=3):
    # warm up
    for _ in range(warmup):
        net.ocr(input)
        paddle.device.synchronize()
    # time
    t = []
    for _ in range(repeat):
        t1 = time.time()
        net.ocr(input)
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
        
class TestBaseAno:
    def __init__(self, model_path=None, use_cinn=False):
        device_info = paddle.get_device()
        print("Current Paddle device : %s" % device_info)
        self.model_path = model_path
        self.use_cinn = use_cinn
        self.net = None
        self.input = None
        self.cinn_net = None

    def init_model(self):
        self.net = paddleocr.PaddleOCR(det_model_dir=self.model_path, use_gpu=True, use_cinn=self.use_cinn)
        if self.use_cinn and self.net._use_cinn and not self.cinn_net:
            self.cinn_net = to_cinn_net(self.net)

    def to_eval(self, use_cinn):
        set_flags(use_cinn)
        if use_cinn:
            if not self.cinn_net:
                self.cinn_net = to_cinn_net(self.net)
            net = self.cinn_net
        else:
            net = self.net
        return net

    def eval(self, input, use_cinn):
        net = self.to_eval(use_cinn)
        out = net.ocr(input, cls=False)
        return out

    def check_cinn_output(self, input):
        pd_out = self.eval(input, use_cinn=False)
        cinn_out = self.eval(input, use_cinn=True)
        # Since PaddleOCR output is not a simple tensor, we compare the first element of the first box's coordinates for simplicity
        np.testing.assert_allclose(
            cinn_out[0][0][0], pd_out[0][0][0], atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, input, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, input)        

'''
class PPLCNet_x1_0(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddleclas.PPLCNet_x1_0(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class PPHGNet_small(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddleclas.PPHGNet_small(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])               

class ViT_base_patch16_224(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddleclas.ViT_base_patch16_224(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 224, 224])

class SwinTransformer_base_patch4_window(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        self.net = paddleclas.SwinTransformer_base_patch4_window12_384(pretrained=True)
        self.input = paddle.randn([batch_size, 3, 384, 384])         
'''

class YOLOv3TestBase(TestBase):  
    def __init__(self, config_path, weights_path, batch_size=1):  
        super().__init__()  
        cfg = load_config(config_path)  
        # 加载YOLOv3模型  
        self.net = create(cfg.architecture)  
        # 加载预训练权重  
        self.net.set_state_dict(paddle.load(weights_path))  
          
        # 模拟输入（注意：YOLOv3通常需要固定大小的输入）  
        # 假设YOLOv3的输入尺寸是416x416  
        image_data = paddle.randn([batch_size, 3, 416, 416])
        im_shape = paddle.Tensor(np.array([[416,416]],dtype=np.float32))
        scale_factor = paddle.Tensor(np.array([[1.,1.]],dtype=np.float32))
        self.input = {
                        'image': image_data,
                        'im_shape': im_shape,
                        'scale_factor': scale_factor
                      }
        
class PP_DETR(TestBase):  
    def __init__(self, config_path, weights_path, batch_size=1):  
        super().__init__()  
        cfg = load_config(config_path)  
        # 加载PP_DETR模型  
        self.net = create(cfg.architecture)  
        # 加载预训练权重  
        self.net.set_state_dict(paddle.load(weights_path))  
          
        # 模拟输入（注意：PP_DETR通常需要固定大小的输入）  
        # 假设PP_DETR的输入尺寸是800 x 1333  
        image_data = paddle.randn([batch_size, 3, 800, 1333])
        im_shape = paddle.Tensor(np.array([[800,1333]],dtype=np.float32))
        scale_factor = paddle.Tensor(np.array([[1.,1.]],dtype=np.float32))
        self.input = {
                        'image': image_data,
                        'im_shape': im_shape,
                        'scale_factor': scale_factor
                      }        

class PP_YOLOE(TestBase):
    def __init__(self, config_path, weights_path, batch_size=1):  
        super().__init__()  
        cfg = load_config(config_path)  
        # 加载PP_YOLOE模型  
        self.net = create(cfg.architecture)  
        # 加载预训练权重  
        self.net.set_state_dict(paddle.load(weights_path))  
          
        # 模拟输入（注意：PP_YOLOE通常需要固定大小的输入）  
        # 假设PP_YOLOE的输入尺寸是640 x 640  
        image_data = paddle.randn([batch_size, 3, 640, 640])
        im_shape = paddle.Tensor(np.array([[640,640]],dtype=np.float32))
        scale_factor = paddle.Tensor(np.array([[1.,1.]],dtype=np.float32))
        self.input = {
                        'image': image_data,
                        'im_shape': im_shape,
                        'scale_factor': scale_factor
                      }

class DINO(TestBase):
    def __init__(self, config_path, weights_path, batch_size=1):  
        super().__init__()  
        cfg = load_config(config_path)  
        # 加载DINO模型  
        self.net = create(cfg.architecture)  
        # 加载预训练权重  
        self.net.set_state_dict(paddle.load(weights_path))  
          
        # 模拟输入（注意：DINO通常需要固定大小的输入）  
        # 假设DINO的输入尺寸是800 x 1333  
        image_data = paddle.randn([batch_size, 3, 800, 1333])
        im_shape = paddle.Tensor(np.array([[paddle.to_tensor(800),paddle.to_tensor(1333)]],dtype=np.float32))
        scale_factor = paddle.Tensor(np.array([[1.,1.]],dtype=np.float32))
        self.input = {
                        'image': image_data,
                        # 使用paddle.concat来创建im_shape，将h和w沿axis=1方向合并
                        'im_shape': im_shape,
                        'scale_factor': scale_factor
                      }

class PPLiteSegTest(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        from paddleseg.models.backbones import STDC1
        backbone = STDC1(pretrained=None)
        num_classes = 19  # 你的目标类别数
        self.net = paddleseg.models.PPLiteSeg(num_classes=num_classes, backbone=backbone, pretrained=None)
        self.input = paddle.randn([batch_size, 3, 512, 512])
        
class OCRNetTest(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        from paddleseg.models.backbones import ResNet50_vd
        backbone = ResNet50_vd(pretrained=None)
        num_classes = 19  # 你的目标类别数
        backbone_indices = (2,3) # 输出的backbone层索引
        self.net = paddleseg.models.OCRNet(num_classes=num_classes, backbone=backbone, backbone_indices=backbone_indices, pretrained=None)
        self.input = paddle.randn([batch_size, 3, 512, 512])
        
class SegFormerTest(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        from paddleseg.models.backbones import MixVisionTransformer
        backbone = MixVisionTransformer(pretrained=None)

        # 定义SegFormer模型参数
        num_classes = 19  # 目标类别数
        embedding_dim = 256  # MLP解码器通道维度        
        self.net = paddleseg.models.SegFormer(num_classes=num_classes,
            backbone=backbone,
            embedding_dim=embedding_dim, pretrained=None)
        self.input = paddle.randn([batch_size, 3, 512, 512])
        
class PPMobileSegTest(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        from paddleseg.models.backbones import MobileNetV3_large_x0_5
        backbone = MobileNetV3_large_x0_5(pretrained=None)
        num_classes = 19  # 目标类别数
        self.net = paddleseg.models.PPMobileSeg(num_classes=num_classes, backbone=backbone, pretrained=None)
        self.input = paddle.randn([batch_size, 3, 512, 512])


def preprocess_image(image_batch):
    # Normalize and resize the entire batch
    image_batch = image_batch / 255.
    # Transpose the batch to CHW format
    image_batch = image_batch.transpose(0,1,2)
    # Convert back to HWC format for PaddleOCR compatibility
    image_batch = image_batch.transpose(1,2,0)    
    return image_batch

class PP_OCRv4_Server_Det_Test(TestBaseAno):
    def __init__(self, batch_size=1):
        super().__init__()
        # Initialize input data with random values
        # self.input = paddle.randn([batch_size, 3, 640, 640])
        self.input = np.random.rand(3, 640, 640).astype('float32')
        self.input = preprocess_image(self.input)
        self.model_path = './PaddleOCR/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_teacher.yml'
        # Ensure the model is initialized
        self.init_model()


class PP_OCRv4_Mobile_Det_Test(TestBaseAno):
    def __init__(self, batch_size=1):
        super().__init__()
        # Initialize input data with random values
        # self.input = paddle.randn([batch_size, 3, 640, 640])
        self.input = np.random.rand(3, 640, 640).astype('float32')
        self.input = preprocess_image(self.input)
        self.model_path = './PaddleOCR/configs/det/ch_PP-OCRv4/ch_PP-OCRv4_det_cml.yml'
        # Ensure the model is initialized
        self.init_model()
        
class PP_OCRv4_Server_Ret_Test(TestBaseAno):
    def __init__(self, batch_size=1):
        super().__init__()
        # Initialize input data with random values
        # self.input = paddle.randn([batch_size, 3, 640, 640])
        self.input = np.random.rand(3, 640, 640).astype('float32')
        self.input = preprocess_image(self.input)
        self.model_path = './PaddleOCR/configs/rec/PP-OCRv4/ch_PP-OCRv4_rec_hgnet.yml'
        # Ensure the model is initialized
        self.init_model()
        
class PP_OCRv4_Mobile_Ret_Test(TestBaseAno):
    def __init__(self, batch_size=1):
        super().__init__()
        # Initialize input data with random values
        # self.input = paddle.randn([batch_size, 3, 640, 640])
        self.input = np.random.rand(3, 640, 640).astype('float32')
        self.input = preprocess_image(self.input)
        self.model_path = './PaddleOCR/configs/det/PP-OCRv4/ch_PP-OCRv4_rec_distill.yml'
        # Ensure the model is initialized
        self.init_model()                               

if __name__ == "__main__":
    '''
    print("Test PPLCNet_x1_0 ........")
    model = PPLCNet_x1_0()       
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)

    print("Test PPHGNet_small ........")
    model = PPHGNet_small()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test ViT_base 16 ........")
    model = ViT_base_patch16_224()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test SwinTransformer Base ........")
    model = SwinTransformer_base_patch4_window()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    
    print("Test YOLOv3  ........")
    model = YOLOv3TestBase('configs/yolov3/yolov3_darknet53_270e_coco.yml', 'weights/yolov3_darknet53_270e_coco.pdparams')       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test PP-DETR  ........")
    model = PP_DETR('configs/detr/detr_r50_1x_coco.yml', 'weights/yolov3_darknet53_270e_coco.pdparams')       
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)
    '''
    print("Test PP-YOLOE  ........")
    model = PP_YOLOE('configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml', 'weights/ppyolo_r50vd_dcn_1x_coco.pdparams')       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    '''
    print("Test DINO_r50 ........")
    model = DINO('configs/dino/dino_r50_4scale_1x_coco.yml', 'weights/dino_r50_4scale_1x_coco.pdparams')       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    
    print("Test PPLiteSeg  ........")
    model = PPLiteSegTest()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test OCRNetTest  ........")
    model = OCRNetTest()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test SegFormerTest  ........")
    model = SegFormerTest()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    print("Test PPMobileSegTest  ........")
    model = PPMobileSegTest()       
    # # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)       
    '''
    print("Test PP_OCRv4_Server_Det_Test  ........")
    model = PP_OCRv4_Server_Det_Test(batch_size=1)
    model.benchmark(model.input, use_cinn=False) 
    # model.check_performance()
    
    # image_paths = ['./PaddleOCR/doc/imgs/11.jpg', './PaddleOCR/doc/imgs/12.jpg']
    # Load images
    print("Test PP_OCRv4_Mobile_Det_Test  ........")
    model = PP_OCRv4_Mobile_Det_Test(batch_size=1)
    model.benchmark(model.input, use_cinn=False)
    # model.check_performance()      
    
    print("Test PP_OCRv4_Server_Ret_Test  ........")
    model = PP_OCRv4_Server_Ret_Test(batch_size=1)
    model.benchmark(model.input, use_cinn=False)
    # model.check_performance()
    
    print("Test PP_OCRv4_Mobile_Ret_Test  ........")
    model = PP_OCRv4_Mobile_Ret_Test(batch_size=1)
    model.benchmark(model.input, use_cinn=False)
    # model.check_performance()    
