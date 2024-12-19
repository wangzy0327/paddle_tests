import time
import numpy as np
import paddle
import paddleclas
import ppdet
import paddleseg
import paddleocr.tools.program
import paddleocr.ppocr.modeling.architectures
import paddleocr.ppocr.utils.save_load
import paddlenlp
import ppsci

# cinn_denied_ops = [
#     "arg_max",
#     "concat",
#     "cumsum",
#     "gather",
#     "reduce_sum",
#     "reduce_max",
#     "slice",
#     "strided_slice",
#     "roll",
#     "tile",
#     "transpose2",
#     # "uniform_random",
#     "lookup_table_v2",
#     "matmul_v2",
# ]

cinn_denied_ops = [
    "arg_max",
    "concat",
    "cumsum",
    "gather",
    "lookup_table_v2",
    "reduce_sum",
    "reduce_max",
    "slice",
    "strided_slice",
    "roll",
    "tile",
    "transpose2",
]
paddle.set_flags({
    "FLAGS_prim_all": True,
    "FLAGS_deny_cinn_ops": ";".join(cinn_denied_ops),
})

def to_cinn_net(net, **kwargs):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = True
    return paddle.jit.to_static(
        net,
        build_strategy=build_strategy,
        full_graph=True,
        **kwargs
    )

def benchmark(net, input, repeat=3, warmup=1):
    # warm up
    for i in range(warmup):
        # print("--[benchmark] Warmup %d/%d" % (i+1, warmup))
        net(input)
    paddle.device.synchronize()
    # time
    t = []
    for i in range(repeat):
        # print("--[benchmark] Run %d/%d" % (i+1, repeat))
        t1 = time.time()
        net(input)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("--[benchmark] Run for %d times, the average latency is: %f ms" % (repeat, np.mean(t)))

class TestBase:
    def __init__(self, batch_size=1):
        self.batch_size = batch_size
        self.input = self.init_input()
        self.net = None
        self.cinn_net = None

    def init_model(self):
        raise NotImplementedError

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 224, 224])

    def get_net(self, use_cinn):
        if use_cinn:
            if self.cinn_net is None:
                if self.net is None:
                    self.net = self.init_model()
                    self.net.eval()
                self.cinn_net = to_cinn_net(self.net)
                self.cinn_net.eval()
                self.net = None
            return self.cinn_net
        else:
            if self.net is None:
                self.net = self.init_model()
                self.net.eval()
            return self.net

    def eval(self, use_cinn):
        net = self.get_net(use_cinn)
        return net(self.input)

    def check_cinn_output(self):
        # print("--[check_cinn_output] eval nocinn")
        pd_out = self.eval(use_cinn=False)
        # print("--[check_cinn_output] eval cinn")
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), pd_out.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn, **kwargs):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.get_net(use_cinn)
        benchmark(net, self.input, **kwargs)

class TestResNet18(TestBase):
    def init_model(self):
        return paddle.vision.models.resnet18(pretrained=True)

class TestResNet50(TestBase):
    def init_model(self):
        return paddle.vision.models.resnet50(pretrained=True)

class TestSqueezeNet(TestBase):
    def init_model(self):
        return paddle.vision.models.squeezenet1_1(pretrained=True)

class TestMobileNetV2(TestBase):
    def init_model(self):
        return paddle.vision.models.mobilenet_v2(pretrained=True)

class TestEfficientNetB0_small(TestBase):
    def init_model(self):
        return paddleclas.EfficientNetB0_small(pretrained=True)

class TestPPLCNet(TestBase):
    def init_model(self):
        return paddleclas.PPLCNet_x1_0(pretrained=True)

class TestPPHGNet(TestBase):
    def init_model(self):
        return paddleclas.PPHGNet_small(pretrained=True)

class TestClipViT(TestBase):
    def init_model(self):
        return paddleclas.CLIP_vit_base_patch16_224(pretrained=True)

class TestSwinTransformer(TestBase):
    def init_model(self):
        return paddleclas.SwinTransformer_base_patch4_window7_224(pretrained=True)

class TestFaceDetection(TestBase):
    def init_model(self):
        return ppdet.model_zoo.get_model("face_detection/blazeface_1000e", pretrained=True)

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestYOLOV3MobileNet(TestBase):
    def init_model(self):
        return ppdet.model_zoo.get_model("yolov3/yolov3_mobilenet_v3_large_270e_coco", pretrained=True)

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestDETR(TestBase):
    def init_model(self):
        return ppdet.model_zoo.get_model("detr/detr_r50_1x_coco", pretrained=True)

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
            'pad_mask': paddle.ones([self.batch_size, 224, 224]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestPPYOLOE(TestBase):
    def init_model(self):
        return ppdet.model_zoo.get_model("ppyoloe/ppyoloe_crn_s_300e_coco", pretrained=True)

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 640, 640]),
            'im_shape': paddle.to_tensor([[640, 640] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestDino(TestBase):
    def init_model(self):
        return ppdet.model_zoo.get_model("dino/dino_r50_4scale_1x_coco", pretrained=True)

    def init_input(self):
        return {
            'image': paddle.rand([self.batch_size, 3, 224, 224]),
            'im_shape': paddle.to_tensor([[224, 224] for _ in range(self.batch_size)]),
            'scale_factor': paddle.to_tensor([[1.0, 1.0] for _ in range(self.batch_size)]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['bbox']

class TestPPLiteSeg(TestBase):
    def init_model(self):
        return paddleseg.models.PPLiteSeg(
            num_classes=11, 
            backbone=paddleseg.models.STDC1(), 
            arm_out_chs=[32, 64, 128],
            seg_head_inter_chs=[32, 64, 64],
            pretrained='https://paddleseg.bj.bcebos.com/dygraph/camvid/pp_liteseg_stdc1_camvid_960x720_10k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestOCRNet(TestBase):
    def init_model(self):
        return paddleseg.models.OCRNet(
            num_classes=21,
            backbone=paddleseg.models.HRNet_W18(),
            backbone_indices=[0],
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/ocrnet_hrnetw18_voc12aug_512x512_40k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestSegFormer(TestBase):
    def init_model(self):
        return paddleseg.models.SegFormer(
            num_classes=19,
            backbone=paddleseg.models.MixVisionTransformer_B0(),
            embedding_dim=256,
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/cityscapes/segformer_b0_cityscapes_1024x1024_160k/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestPPMobileSeg(TestBase):
    def init_model(self):
        return paddleseg.models.PPMobileSeg(
            num_classes=150,
            backbone=paddleseg.models.MobileSeg_Base(inj_type='AAMSx8', out_feat_chs=[64, 128, 192]),
            pretrained='https://bj.bcebos.com/paddleseg/dygraph/ade20k/pp_mobileseg_base/model.pdparams'
        )

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]


class TestPPOCRV4ServerDet(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_server_det.yaml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model, model_type=config["Architecture"]["model_type"])
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 224, 224])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']

class TestPPOCRV4MobileDet(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_mobile_det.yaml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model, model_type=config["Architecture"]["model_type"])
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 224, 224])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']

class TestPPOCRV4ServerRec(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_server_rec.yaml')
        out_channels_list = {
            "CTCLabelDecode": 6625,
            "SARLabelDecode": 6627,
            "NRTRLabelDecode": 6628,
        }
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list      
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 48, 320])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']

class TestPPOCRV4MobileRec(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleX/paddlex/repo_apis/PaddleOCR_api/configs/PP-OCRv4_mobile_rec.yaml')
        out_channels_list = {
            "CTCLabelDecode": 6625,
            "SARLabelDecode": 6627,
            "NRTRLabelDecode": 6628,
        }
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list        
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.rand([self.batch_size, 3, 48, 320])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['maps']    

class TestViLayoutXLM(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleOCR/configs/kie/vi_layoutxlm/ser_vi_layoutxlm_xfund_zh.yml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.load('ser_vi_layoutxlm_input.pdtensor')

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestLayoutLM(TestBase):
    def init_model(self):
        config = paddleocr.tools.program.load_config('/opt/PaddleOCR/configs/kie/layoutlm_series/ser_layoutlm_xfund_zh.yml')
        model = paddleocr.ppocr.modeling.architectures.build_model(config['Architecture'])
        paddleocr.ppocr.utils.save_load.load_model(config, model)
        return model

    def init_input(self):
        return paddle.load('ser_layoutlm_input.pdtensor')  


class TestErnie(TestBase):
    def init_model(self):
        return paddlenlp.transformers.ErnieModel.from_pretrained('ernie-3.0-nano-zh')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]

class TestBert(TestBase):
    def init_model(self):
        return paddlenlp.transformers.BertModel.from_pretrained('bert-base-uncased')

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out[0]      

# class TestLlama2(TestBase):
#     def init_model(self):
#         return paddlenlp.transformers.LlamaModel.from_pretrained('meta-llama/Llama-2-7b-chat')

#     def init_input(self):
#         return {
#             'input_ids': paddle.randint(0, 1000, [self.batch_size, 128]),
#             'position_ids': paddle.arange(0, 128, dtype='int64').expand([self.batch_size, -1]),
#             'attention_mask': paddle.ones([self.batch_size, 128]),
#         }

class TestGpt2(TestBase):
    def init_model(self):
        return paddlenlp.transformers.GPTModel.from_pretrained('gpt2-medium-en')
        # return paddlenlp.transformers.GPTForSequenceClassification.from_pretrained('gpt2-medium-en',num_labels=2)

    def init_input(self):
        return paddle.randint(0, 1000, [self.batch_size, 128])    

class TestEulerBeam(TestBase):
    def init_model(self):
        import hydra
        with hydra.initialize_config_dir(version_base=None, config_dir="/opt/PaddleScience/examples/euler_beam/conf"):
            cfg = hydra.compose(config_name="euler_beam.yaml")
        model = ppsci.arch.MLP(**cfg.MODEL)
        ppsci.utils.save_load.load_pretrain(model,
            'https://paddle-org.bj.bcebos.com/paddlescience/models/euler_beam/euler_beam_pretrained.pdparams',
            {"biharmonic": ppsci.equation.Biharmonic(dim=1, q=cfg.q, D=cfg.D)}
        )
        return model

    def init_input(self):
        return {
            'x': paddle.rand([100, 1]),
            'sdf': paddle.rand([100, 1]),
        }

    def eval(self, use_cinn):
        out = super().eval(use_cinn)
        return out['u']

if __name__ == "__main__":
    # print("PPMobileSeg ......")
    # model = TestPPMobileSeg(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("Dino ......")
    # model = TestDino(batch_size=1)
    # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("PPOCRV4ServerDet ......")
    # model = TestPPOCRV4ServerDet(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("PPOCRV4MobileDet ......")
    # model = TestPPOCRV4MobileDet(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True) 
    # print("SwinTransformer......")   
    # model = TestSwinTransformer(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("PP-DETR......")   
    # model = TestDETR(batch_size=1)
    # # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)    
    # print("PPOCRV4ServerRec......")   
    # model = TestPPOCRV4ServerRec(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)  
    # print("PPOCRV4MobileRec......")   
    # model = TestPPOCRV4MobileRec(batch_size=1)
    # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)     
    # print("Blazeface......")   
    # model = TestFaceDetection(batch_size=1)
    # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)  
    # print("PP-Structurev2-vi-layoutxlm")       
    # model = TestViLayoutXLM(batch_size=1)
    # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("PP-Structurev2-vi-layout")       
    # model = TestLayoutLM(batch_size=1)
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)    
    # print("Transformer-Ernie")     
    # model = TestErnie()
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)
    # print("Bert-base-uncased")     
    # model = TestBert()
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)    
    # print("Gpt2")     
    # model = TestGpt2()
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True)  
    print("EulerBeam")     
    model = TestEulerBeam()
    # model.check_cinn_output()
    model.benchmark(use_cinn=False)
    model.benchmark(use_cinn=True)       
