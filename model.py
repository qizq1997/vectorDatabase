from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Any


# 自定义GLM类
class ChatGLM2(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    #top_p = 0.9
    tokenizer: object = None
    model: object = None
    history: list = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "custom_chatglm2"

    # 定义load_model的方法
    def load_model(self, model_path=None):
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 加载模型
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

    # 定义_call方法：进行模型的推理
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(
            tokenizer=self.tokenizer,
            query=prompt,
            history=self.history,
            temperature=self.temperature)

        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        self.history = self.history + [[None, response]]
        return response


if __name__ == '__main__':
    llm = ChatGLM2()
    llm.history = []
    llm.load_model(model_path=r'C:\Users\65197\Downloads\chatglm')
    print(f'llm--->{llm}')
    print(llm("新中国是几几年成立的"))
