# Created in 2025 by Gandecheng
from copy import deepcopy
import json
import ast
import os 
import json
import numpy as np
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import torch
# from peft import PeftModel
from vllm import LLM, SamplingParams
from intent_config import INTENT_CONFIG
from router_config import ROUTER_CONFIG

choices=[
        "Qwen2-VL-7B", 
        "Qwen2-1.5B-Instruct", 
        "Qwen2-14B-Instruct", 
        "Deepseek-R1-Distill-Qwen-7B",
        "Deepseek-R1-Distill-Qwen-14B",
    ]

class intent_router_QwenVL():
    def __init__(self, router_type):
        """
        This class is specific to Qwen2-VL series
        """
        # check router_type
        if router_type!= "Qwen2-VL-7B":
            raise ValueError('This class requires Qwen2-VL')
        # load model and processor
        self.model_dir = ROUTER_CONFIG[router_type]['model_dir']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = AutoProcessor.from_pretrained(self.model_dir)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_dir,
                                                                        device_map = self.device,                  
                                                                        attn_implementation = "flash_attention_2",
                                                                        torch_dtype = torch.bfloat16 if ROUTER_CONFIG[router_type]['torch_dtype'] == 'bf' else torch.float16) 

