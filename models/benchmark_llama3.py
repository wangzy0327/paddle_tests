import paddle
from paddlenlp.transformers import LlamaForCausalLM, LlamaTokenizer


def convert_to_float16(model):
        for param in model.parameters():
            if param.dtype == paddle.float32:
                param.set_value(param.astype(paddle.float16))

class TestLlama:
    def __init__(self, batch_size=1):
        # 指定模型名称
        model_name = "meta-llama/Llama-2-7b-chat"
        
        # 加载 tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
        # 如果 tokenizer 没有 pad_token，则设置一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.net = LlamaForCausalLM.from_pretrained(model_name)
        
        # 将模型参数转换为 float16
        convert_to_float16(self.net)
        
        # 检查并转换数据类型
        for param in self.net.parameters():
            if param.dtype == paddle.bfloat16:
                param.set_value(param.astype(paddle.float16))
        
        # 设置模型为评估模式
        self.net.eval()
        
        # 随机生成输入数据
        input_text = "请输入测试样例"
        encoded_text = self.tokenizer(input_text, return_tensors="pd", padding=True, truncation=True)
        
        # 获取输入的 token ids 和 attention mask
        self.input_ids = encoded_text["input_ids"]
        self.attention_mask = encoded_text.get("attention_mask", None)
        
        # 如果 attention_mask 存在且不是 None，将其转换为 float16
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.astype(paddle.float16)

# 测试类
if __name__ == "__main__":
    print("Test Llama ........")
    model = TestLlama()
    
    # 执行推理
    with paddle.no_grad():
        outputs = model.net.generate(
            input_ids=model.input_ids,
            attention_mask=model.attention_mask,
            max_new_tokens=50,  # 生成的最大长度
            num_return_sequences=1,  # 生成的序列数量
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
    
    # 解码生成的文本
    generated_texts = [model.tokenizer.decode(output, skip_special_tokens=True) for output in outputs[0].numpy()]
    
    print("打印生成的文本如下：")
    
    # 打印生成的文本
    for text in generated_texts:
        print(text)
