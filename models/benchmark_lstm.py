import os
import time
import paddle  
from paddle import nn  
from paddlenlp.transformers import BertModel, BertTokenizer, ErnieModel, ErnieTokenizer
from paddlenlp.transformers import GPTTokenizer, GPTLMHeadModel
from paddlenlp.transformers import LlamaForCausalLM, LlamaTokenizer
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


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

def convert_to_float16(model):
    for param in model.parameters():
        if param.dtype == paddle.float32:
            param.set_value(param.astype(paddle.float16))    
        elif param.dtype == paddle.bfloat16:
            param.set_value(param.astype(paddle.float16))               

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


def benchmark(net, input_ids, token_type_ids, repeat=5, warmup=3):
    # warm up
    for _ in range(warmup):
        net(input_ids, token_type_ids)
        paddle.device.synchronize()
    # time
    t = []
    for _ in range(repeat):
        t1 = time.time()
        net(input_ids, token_type_ids)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("--[benchmark] Run for %d times, the average latency is: %f ms" % (repeat, np.mean(t))) 
    
def benchmark(net, input_ids, attention_mask, max_new_tokens,num_return_sequences, pad_token, pad_token_id, eos_token_id, repeat=5, warmup=3):
    # warm up
    for _ in range(warmup):
        net.generate(
                input_ids = input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens,  # 生成的最大长度
                num_return_sequences = num_return_sequences,  # 生成的序列数量
                pad_token = pad_token,
                pad_token_id = pad_token_id,
                eos_token_id = eos_token_id,
            )
        paddle.device.synchronize()
    # time
    t = []
    for _ in range(repeat):
        t1 = time.time()
        net.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,  # 生成的最大长度
            num_return_sequences = num_return_sequences,  # 生成的序列数量
            pad_token = pad_token,
            pad_token_id = pad_token_id,
            eos_token_id = eos_token_id,
        )
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


class LSTMModel(nn.Layer):  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  
        super(LSTMModel, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  
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

class GRUModel(nn.Layer):  
    def __init__(self, input_size, hidden_size, num_layers, num_classes):  
        super(GRUModel, self).__init__()  
        self.hidden_size = hidden_size  
        self.num_layers = num_layers  
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  
        self.fc = nn.Linear(hidden_size, num_classes)  
  
    def forward(self, x):  
        # 前向传播GRU  
        # 初始化隐藏状态（可选，PaddlePaddle会默认初始化）  
        # 但为了明确性，这里我们还是显式地写出它（尽管在大多数情况下不需要）  
        # h0 = paddle.zeros([self.num_layers, x.shape[0], self.hidden_size])  
        # 在实际应用中，通常不需要手动初始化h0，因为nn.GRU会自动处理  
  
        out, _ = self.gru(x)  
  
        # 取最后一个时间步的输出  
        out = self.fc(out[:, -1, :])  
        return out    

class TestLSTM(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        seq_len = 1024  
        input_size = 5  
        hidden_size = 20  
        num_layers = 1  
        num_classes = 3
        self.net = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.input = paddle.randn([batch_size, seq_len, input_size])
    pass


class TestGRU(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        seq_len = 1024  
        input_size = 5  
        hidden_size = 20  
        num_layers = 1  
        num_classes = 3
        self.net = LSTMModel(input_size, hidden_size, num_layers, num_classes)
        self.input = paddle.randn([batch_size, seq_len, input_size])
    pass

class TestTransformer(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        max_seq_length = 128  # 最大序列长度
        # model_path = '/home/wzy/AI-ModelScope/bert-base-uncased'
        model_name = 'ernie-3.0-medium-zh'
        # model_name = 'bert-base-uncased'
        self.net = ErnieModel.from_pretrained(model_name)
        self.tokenizer = ErnieTokenizer.from_pretrained(model_name)

        # 随机生成输入数据
        encoded_text = self.tokenizer(text="请输入测试样例", max_seq_length = max_seq_length)
        self.input_ids = paddle.to_tensor([encoded_text['input_ids']])
        self.token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])        

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
        out = net(self.input_ids, self.token_type_ids)
        return out

    def check_cinn_output(self):
        pd_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.last_hidden_state.numpy(), pd_out.last_hidden_state.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, self.input_ids, self.token_type_ids)


class TestGPT(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        max_seq_length = 1024  # 最大序列长度
        # model_path = '/home/wzy/AI-ModelScope/bert-base-uncased'
        # model_name = 'ernie-3.0-medium-zh'
        # model_name = 'gpt-cpm-large-cn'
        # model_name = 'gpt-cpm-large-cn'
        # model_name = 'gpt2-en'
        model_name = 'gpt2-medium-en'
        # model_name = 'gpt-cpm-small-cn-distill'
        self.net = GPTLMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPTTokenizer.from_pretrained(model_name)

        # 随机生成输入数据
        encoded_text = self.tokenizer(text="请输入测试样例", max_seq_length = max_seq_length)
        self.input_ids = paddle.to_tensor([encoded_text['input_ids']])
        self.token_type_ids = paddle.to_tensor([encoded_text['token_type_ids']])        

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
        out = net(self.input_ids, self.token_type_ids)
        return out

    def check_cinn_output(self):
        pd_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.last_hidden_state.numpy(), pd_out.last_hidden_state.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, self.input_ids, self.token_type_ids)
        
        
class TestLlama(TestBase):
    def __init__(self, batch_size=1):
        super().__init__()
        model_name = "meta-llama/Llama-2-7b-chat"
        # model_name = "meta-llama/Meta-Llama-3-8B"
        # 加载 tokenizer
        # self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, dtype = "float16")
        
        # 如果 tokenizer 没有 pad_token，则设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        # self.net = LlamaForCausalLM.from_pretrained(model_name)
        self.net = AutoModelForCausalLM.from_pretrained(model_name)
        
        # 将模型参数转换为 float16
        # convert_to_float16(self.net)
        

        # 随机生成输入数据
        input_text = "请输入测试样例"
        encoded_text = self.tokenizer(input_text, return_tensors="pd", padding=True, truncation=True)
        
        # 生成的最大长度
        self.max_new_tokens = 128
        # 生成的序列数量
        self.num_return_sequences = 1
        
        # 获取输入的 token ids 和 attention mask
        self.input_ids = encoded_text["input_ids"]
        self.attention_mask = encoded_text.get("attention_mask", None)
        
        # 如果 attention_mask 存在且不是 None，将其转换为 float16
        # if self.attention_mask is not None:
        #     self.attention_mask = self.attention_mask.astype(paddle.float16)      

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
        # 执行推理
        with paddle.no_grad():
            outputs = net.generate(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                max_new_tokens=self.max_new_tokens,  # 生成的最大长度
                num_return_sequences=self.num_return_sequences,  # 生成的序列数量
                pad_token = self.tokenizer.pad_token,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return outputs

    def check_cinn_output(self):
        pd_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.last_hidden_state.numpy(), pd_out.last_hidden_state.numpy(), atol=1e-3, rtol=1e-3
        )
        print("--[check_cinn_output] cinn result right.")

    def benchmark(self, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, self.input_ids, self.attention_mask, self.max_new_tokens, self.num_return_sequences, self.tokenizer.pad_token, self.tokenizer.pad_token_id, self.tokenizer.eos_token_id)        

if __name__ == "__main__":
    # print("Test LSTM ........")
    # model = TestLSTM()       
    # # model.check_cinn_output()
    # model.benchmark(use_cinn=False)
    # model.benchmark(use_cinn=True) 
    # print("Test GRU ........")
    # model = TestGRU()       
    # model.benchmark(use_cinn=False)    
    # model.benchmark(use_cinn=True)   
    # print("Test Transformer ErnieModel ........")
    # model = TestTransformer()       
    # model.benchmark(use_cinn=False)    
    # model.benchmark(use_cinn=True)     
    # print("Test Transformer BertModel ........")
    # model = TestTransformer()       
    # model.benchmark(use_cinn=False)        
    # model.benchmark(use_cinn=True)        
    # print("Test GPT Model gpt2-medium-en ........")
    # model = TestGPT()       
    # model.benchmark(use_cinn=False)        
    # model.benchmark(use_cinn=True)     
    print("Test Llama ........")
    model = TestLlama()       
    model.benchmark(use_cinn=False)      