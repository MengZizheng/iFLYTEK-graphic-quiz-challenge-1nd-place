# 2024 iFLYTEK 大模型图文问答挑战赛 - 第1名方案

## 赛题背景
图像作为一种媒介，在信息传递和沟通中扮演着至关重要的角色。它能够直观地展示复杂的概念、情感和场景，使得信息的传递更为高效和生动。然而，目前的问答系统在图文交互方面仍存在一定局限性。本赛题旨在推动人工智能在理解和处理视觉与文本信息方面的研究与应用，要求参赛者设计能够从图像和相关的文本描述中提取信息并回答相关问题的模型。

## 赛题链接
[赛题页面](https://challenge.xfyun.cn/topic/info?type=graphic-quiz-challenge&option=ssgy)

## 数据集链接
[数据集下载](https://challenge.xfyun.cn/topic/info?type=graphic-quiz-challenge&option=stsj)

## 方案概述
本方案通过微调CLIP（Contrastive Language-Image Pre-training）模型，并使用LoRA（Low-Rank Adaptation）技术来增强模型的能力，从而更好地理解图像和文本的相关性。通过对训练数据集的处理、CLIP模型的加载与微调，我们能够使模型在图文问答任务中表现出色。

### 主要技术方案
1. **微调CLIP模型**：加载OFA-Sys团队提供的`chinese-clip-vit-huge-patch14`模型，并对其进行微调，使其能够更好地理解中文文本与图像之间的对应关系。
2. **LoRA技术**：采用LoRA对模型进行适配，仅微调查询和视觉模块中的重要参数，避免全模型训练，减少训练时间并提高效率。
3. **数据预处理与增强**：处理和加载训练数据集，进行必要的数据增强，并构建适合模型的输入格式。
4. **图文匹配**：通过图像和文本的向量化，执行文本与图像之间的匹配，进一步提升图文问答效果。
5. **基于Qwen2-VL的图文问答**：结合Qwen2-VL模型，利用RAG（Retrieval-Augmented Generation）技术进行精确问答，优化回答质量。

## 1. 数据加载与预处理

在本部分，我们首先从文件中读取训练数据，包含文本查询和图像路径，并对图像数据进行处理，以确保其适合模型输入。

```python
import pandas as pd
import json
from PIL import Image

with open("../xfdata/query.json", "r") as f:
    query = json.load(f)
    df_query = pd.DataFrame(query)
df_train = pd.read_csv("../xfdata/train_annotation.csv", sep="\t")
print(f"Shape of df_query: {df_query.shape}")
print(f"Shape of df_train: {df_train.shape}")
# 所有图片
df_image_all = pd.DataFrame(os.listdir("../xfdata/image"))
df_image_all.columns = ["image"]
```

## 2. 加载CLIP模型与LoRA配置

加载预训练的CLIP模型，并通过LoRA对其进行微调，仅优化特定层，避免过度训练全模型，提升效率。

```python
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from peft import get_peft_model, LoraConfig

# 加载预训练的CLIP模型
model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", cache_dir="../user_data/").to(device)
processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-huge-patch14", cache_dir="../user_data/")

# 设置LoRA配置
lora_config = LoraConfig(
    r=64,
    lora_alpha=96,
    lora_dropout=0.1,
    bias="none",
    target_modules=target_modules
)

# 将LoRA应用于CLIP模型
lora_model = get_peft_model(model, lora_config)
```

## 3. 图像与文本向量化

将图像和文本分别转换为向量表示，确保其可以在后续步骤中进行有效的匹配。我们使用CLIP模型提取图像和文本的嵌入向量。

```python
def get_image_embed(batch):
    with torch.no_grad():
        image_paths = [os.path.join('../xfdata/image', image_name) for image_name in batch["image"]]
        images = [Image.open(image_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGB") for image_path in image_paths]
        pixel_values = processor(text=None, images=images, return_tensors="pt")["pixel_values"].to(device)
        image_embeds = model.get_image_features(pixel_values)
        batch["image_embeds"] = image_embeds
        return batch

def get_text_embed(batch):
    with torch.no_grad():
        inputs = processor(text=batch["text"], images=None, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
        text_embeds = model.get_text_features(**inputs)
        batch["text_embeds"] = text_embeds
        return batch
```

## 4. 图文匹配与问答

通过向量化后的图像和文本，执行图文匹配，并使用Qwen2-VL模型与RAG技术实现图文问答。在此部分，首先使用最相似的图像来回答相关问题，然后根据文本进行优化。

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct", cache_dir="../user_data")
model = Qwen2VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir)

def chat(question, related_image, related_text):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join("../xfdata/image", related_image),
                },
                {"type": "text", "text": f"请根据问题和答案，将答案修改为问题的格式，例如：\n问题：这套裙子是2019年的新款吗？\n答案：是\n润色后的答案：这套裙子是2019年的新款。"}
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

## 5. 匈牙利算法优化

为优化图文匹配结果，我们使用匈牙利算法对文本和图像之间的匹配进行优化，以确保最终结果的准确性。

```python
from scipy.optimize import linear_sum_assignment
import numpy as np

cost_matrix = max_similarity - most_similar_matrix_text2image
row_ind, col_ind = linear_sum_assignment(cost_matrix)
df_submit.loc[df_text2image.index, "answer"] = df_image_not_train.reset_index(drop=True).loc[col_ind, "image"].values
```

## 6. 结果提交

最终，结合以上步骤的结果，我们生成一个符合竞赛要求的`result.json`文件，并将其提交。

```python
submit_json = []
with open("../prediction_result/result.json", "w", encoding="utf-8") as f:
    for line in df_submit.values:
        json_str = {"question": question, "related_image": related_image, "answer": answer}
        submit_json.append(json_str)
    json.dump(submit_json, f, ensure_ascii=False, indent=4)
```