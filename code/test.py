########################################################################################################################
######                                                                                                            ######
######                                          第二部分：开始预测                                                 ######
######                                                                                                            ######
########################################################################################################################


########################################################################################################################
# 1、读取数据
########################################################################################################################

import pandas as pd 
from PIL import Image 
import matplotlib.pyplot as plt 
import os 
import json 


with open("../xfdata/query.json", "r") as f:
    query = json.load(f)
    df_query = pd.DataFrame(query)
df_train = pd.read_csv("../xfdata/train_annotation.csv", sep="\t")
print(f"Shape of df_query: {df_query.shape}")
print(f"Shape of df_train: {df_train.shape}")
# 所有图片
df_image_all = pd.DataFrame(os.listdir("../xfdata/image"))
df_image_all.columns = ["image"]
print(f"Shape of df_image_all: {df_image_all.shape}")
# 所有没在训练集中的图片
df_image_not_train = df_image_all.loc[~df_image_all["image"].isin(df_train["image"])]
print(f"Shape of df_image_not_train: {df_image_not_train.shape}")


########################################################################################################################
# 2、加载CLIP模型
########################################################################################################################

from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import os
import torch 


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_path = "../user_data/models--OFA-Sys--chinese-clip-vit-huge-patch14/snapshots/503e16b560aff94c1922f13a86a7693d36957a4f"
model = torch.load("../user_data/CLIP_LoRA_625_10.pth").to(device)
processor = ChineseCLIPProcessor.from_pretrained(clip_model_path)
print("Loading Done!")


########################################################################################################################
# 3、向量化
########################################################################################################################
from datasets import Dataset
import numpy as np


df_train = pd.read_csv("../xfdata/train_annotation.csv", sep="\t")
model.eval()
dataset_train = Dataset.from_pandas(df_train)
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

# 训练集图片和文本向量化
dataset_train = dataset_train.map(get_image_embed, batched=True, batch_size=256)
dataset_train = dataset_train.map(get_text_embed, batched=True, batch_size=512)
dataset_train.set_format("torch", columns=["image_embeds", "text_embeds"])
image_embeddings = dataset_train["image_embeds"]
text_embeddings = dataset_train["text_embeds"]
image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
image_embeddings = image_embeddings.detach().cpu().numpy()
text_embeddings = text_embeddings.detach().cpu().numpy()

# 未在训练集中的图片向量化
dataset_image_not_train = Dataset.from_pandas(df_image_not_train)
dataset_image_not_train = dataset_image_not_train.map(get_image_embed, batched=True, batch_size=256)
dataset_image_not_train.set_format("torch", columns=["image_embeds"])
image_embeddings_not_train = dataset_image_not_train["image_embeds"]
image_embeddings_not_train = image_embeddings_not_train / image_embeddings_not_train.norm(dim=-1, keepdim=True)
image_embeddings_not_train = image_embeddings_not_train.detach().cpu().numpy()



########################################################################################################################
# 4、文本匹配图片
########################################################################################################################

import re 


# 文搜图的数据
df_text2image = df_query.loc[df_query["related_image"]=="", ["question"]]
# 定义正则表达式模式
pattern = re.compile(r'请匹配到与 (.+) 最相关的图片。')
# 查找所有匹配项
def get_text(x):
    return pattern.findall(x)[0]
df_text2image = df_text2image.map(get_text)


def get_text_embed_text2image(batch):
    """文本向量化"""
    with torch.no_grad():
        inputs = processor(text=batch["question"], images=None, return_tensors="pt", padding=True, truncation=True, max_length=52).to(device)
        text_embeds = model.get_text_features(**inputs)
        batch["text_embeds"] = text_embeds
        return batch

# 文本向量化
dataset_text2image = Dataset.from_pandas(df_text2image)
dataset_text2image = dataset_text2image.map(get_text_embed_text2image, batched=True, batch_size=512)
dataset_text2image.set_format("torch", columns=["text_embeds"])
text_embeddings_query = dataset_text2image["text_embeds"].to(device)
text_embeddings_query = text_embeddings_query / text_embeddings_query.norm(dim=-1, keepdim=True)
text_embeddings_query = text_embeddings_query.detach().cpu().numpy()


########################################################################################################################
# 4、图片匹配文本
########################################################################################################################


# 图片匹配文本的样本
df_image2text = df_query.loc[df_query["related_image"]!=""].copy()

def get_image_embed_image2text(batch):
    """图片向量化"""
    with torch.no_grad():
        image_paths = [os.path.join('../xfdata/image', image_name) for image_name in batch["related_image"]]
        images = [Image.open(image_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGB") for image_path in image_paths]
        pixel_values = processor(text=None, images=images, return_tensors="pt")["pixel_values"].to(device)
        image_embeds = model.get_image_features(pixel_values)
        batch["image_embeds"] = image_embeds
        return batch

# 图片向量化
dataset_image2text = Dataset.from_pandas(df_image2text)
dataset_image2text = dataset_image2text.map(get_image_embed_image2text, batched=True, batch_size=256)
dataset_image2text.set_format("torch", columns=["image_embeds"])
image_embeddings_query = dataset_image2text["image_embeds"]
image_embeddings_query = image_embeddings_query / image_embeddings_query.norm(dim=-1, keepdim=True)
image_embeddings_query = image_embeddings_query.detach().cpu().numpy()

# 对图片寻找最相似的文本
most_similar_index_image2text = (image_embeddings_query @ text_embeddings.T).argmax(axis=1)
df_image2text = df_query.loc[df_query["related_image"]!=""].copy()
df_image2text["answer"] = df_train.loc[most_similar_index_image2text, "text"].values




########################################################################################################################
# 5、图片匹配图片然后匹配到文本
########################################################################################################################


# 图片匹配文本的样本
df_image2image = df_query.loc[df_query["related_image"]!=""].copy()

def get_image_embed_image2text(batch):
    """图片向量化"""
    with torch.no_grad():
        image_paths = [os.path.join('../xfdata/image', image_name) for image_name in batch["related_image"]]
        images = [Image.open(image_path).transpose(Image.FLIP_LEFT_RIGHT).convert("RGB") for image_path in image_paths]
        pixel_values = processor(text=None, images=images, return_tensors="pt")["pixel_values"].to(device)
        image_embeds = model.get_image_features(pixel_values)
        batch["image_embeds"] = image_embeds
        return batch

# 图片向量化
dataset_image2image = Dataset.from_pandas(df_image2image)
dataset_image2image = dataset_image2image.map(get_image_embed_image2text, batched=True, batch_size=256)
dataset_image2image.set_format("torch", columns=["image_embeds"])
image_embeddings_query = dataset_image2image["image_embeds"]
image_embeddings_query = image_embeddings_query / image_embeddings_query.norm(dim=-1, keepdim=True)
image_embeddings_query = image_embeddings_query.detach().cpu().numpy()

# 对图片寻找最相似的图片
most_similar_index_image2image = (image_embeddings_query @ image_embeddings.T).argmax(axis=1)
df_image2image = df_query.loc[df_query["related_image"]!=""].copy()
df_image2image["answer"] = df_train.loc[most_similar_index_image2image, "text"].values

# 对图片寻找最相似的图片
most_similar_index_image2image = (image_embeddings_query @ image_embeddings.T).argmax(axis=1)
df_image2image = df_query.loc[df_query["related_image"]!=""].copy()
df_image2image["answer"] = df_train.loc[most_similar_index_image2image, "text"].values

df_image2text = df_image2image




########################################################################################################################
# 6、使用Qwen2-VL结合RAG检索进行图文问答
########################################################################################################################

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
import torch


model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct", cache_dir="../user_data")
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
# default processer
processor = AutoProcessor.from_pretrained(model_dir)
print("Qwen2-VL Loading Done!")






import Levenshtein

def get_similarity(question, answer):
    """
    计算相似度
    """
    similarity = Levenshtein.ratio(question, answer)
    return similarity


def improve_answer(question, related_image, answer):
    """
    润色答案
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join("../xfdata/image", related_image),
                },
                {"type": "text", "text": f"""请根据问题和答案，将答案修改为问题的格式，例如：
问题：这套裙子是2019年的新款吗？
答案：是
润色后的答案：这套裙子是2019年的新款。
问题：这是什么系列的衣服？
答案：这款衣服属于夏季系列。
润色后的答案：这是夏季系列的衣服。

现在请你进行回答：
问题：{question}
答案: {answer}
润色后的答案："""},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text




def chat(question, related_image, related_text):
    """
    RAG多模态图文问答
    """
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": os.path.join("../xfdata/image", related_image),
            },
            {"type": "text", "text": f"""这件商品的名称是"{related_text}"，请根据商品图片和名称，回答问题，答案尽可能简洁且和问题格式保持一致。
例如：
问题：这件衣服是男款还是女款？
回答：这件衣服是男款。
问题：这是哪一年的衣服？
回答：这是20XX年的衣服。
问题：这件衣服是哪个季节的？
回答：这件衣服是X季的。

现在请你回答：
问题: {question}
回答："""},
        ],
    }
]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    if get_similarity(question, answer) >= 0.35:
        print(question, answer)
        return answer
    else:
        print("Similarity is so low!", question, answer)
        print("Improving Answer...")
        improved_answer = improve_answer(question, related_image, answer)  
        print(question, improved_answer)
        return improved_answer
    

llm_answer = []
with torch.no_grad():
    for idx, line in enumerate(df_image2text.values):
        question, related_image, related_text = line
        if question == "请对给定的图片进行描述。":
            llm_answer.append(related_text)
        else:
            try:
                print(f"[{idx}|{len(df_image2text)}]:")
                llm_answer.append(chat(question, related_image, related_text))
            except:
                llm_answer.append(question)



########################################################################################################################
# 7、匈牙利算法优化
########################################################################################################################

import json 
import pandas as pd 
from scipy.optimize import linear_sum_assignment
import numpy as np


# 创建一个提交副本
df_submit = df_query.copy()
# 对文本寻找最相似的图片
most_similar_matrix_text2image = (text_embeddings_query @ image_embeddings_not_train.T)
max_similarity = np.max(most_similar_matrix_text2image)
cost_matrix = max_similarity - most_similar_matrix_text2image
# 使用匈牙利算法求解
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 填充文本匹配图片的结果
df_submit.loc[df_text2image.index, "answer"] = df_image_not_train.reset_index(drop=True).loc[col_ind, "image"].values
# 填充图片问答的结果
df_submit.loc[df_image2text.index, "answer"] = llm_answer
# 输出为json文件
submit_json = []
with open("../prediction_result/result.json", "w", encoding="utf-8") as f:
    for line in df_submit.values:
        question, related_image, answer = line
        json_str = {"question": question, "related_image": related_image, "answer": answer}
        submit_json.append(json_str)
    json.dump(submit_json, f, ensure_ascii=False, indent=4)