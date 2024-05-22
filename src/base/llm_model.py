# # File code dùng để khai báo hàm khởi tạo mô hình ngôn ngữ lớn.

# import torch
# from transformers import BitsAndBytesConfig
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# nf4_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# def get_hf_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
#                max_new_token = 1024, **kwargs):
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=nf4_config,
#         low_cpu_mem_usage=True
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     model_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_token=max_new_token,
#         pad_token_id=tokenizer.eos_token_id,
#         device_map="auto"
#     )

#     llm = HuggingFacePipeline(
#         pipeline=model_pipeline,
#         model_kwargs=kwargs
#     )

#     return llm


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

def get_hf_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
               max_new_tokens=1024, **kwargs):
    # Tải mô hình và tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Chuyển mô hình sang chế độ đánh giá (eval mode)
    model.eval()

    # Thực hiện lượng tử hóa động
    model_quantized = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # Tạo pipeline với mô hình và tokenizer đã tải
    model_pipeline = pipeline(
        "text-generation",
        model=model_quantized,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        device=-1  # Sử dụng -1 để chỉ định chạy trên CPU
    )

    llm = HuggingFacePipeline(
        pipeline=model_pipeline,
        model_kwargs=kwargs
    )

    return llm