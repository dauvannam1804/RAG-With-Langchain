import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

def get_hf_llm(model_name: str = "togethercomputer/RedPajama-INCITE-Chat-3B-v1",
               max_new_token = 1024, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32  # Sử dụng float32 cho CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_token,
        pad_token_id=tokenizer.eos_token_id,
        device_map="cpu"  # Chỉ định sử dụng CPU
    )
    
    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )
    
    return llm