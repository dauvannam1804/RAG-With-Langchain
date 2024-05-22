# File code dùng để tải tự động một số các
# bài báo khoa học dưới dạng file pdf.

import os
import requests

file_links = [
    {
        "title": f"Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762"
    },
    {
        "title": f"BERT Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "url": "https://arxiv.org/pdf/1810.04805"
    },
    {
        "title": f"Denoising Diffusion Probabilistic Models",
        "url": "https://arxiv.org/pdf/2006.11239"
    },
    {
        "title": f"Instruction Tuning for Large Language Models-A Survey",
        "url": "https://arxiv.org/pdf/2308.10792"
    },
    {
        "title": f"Llama 2-Open Foundation and Fine-Tuned Chat Models",
        "url": "https://arxiv.org/pdf/2307.09288"
    }
]

def is_exist(file_link):
    return os.path.exists(f"./{file_link['title']}.pdf")

for file_link in file_links:
    if not is_exist(file_link):
        response = requests.get(file_link['url'], stream=True)
        with open(f"./{file_link['title']}.pdf", 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded {file_link['title']}.pdf")