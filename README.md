```python
from diffusers import StableDiffusionXLPipeline
import torch
import re
import os

# 初始化 pipeline（只执行一次）
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1",
    torch_dtype=torch.float16
).to("cuda")

def sanitize_filename(prompt):
    """
    将 prompt 转换为合法的文件名（去掉非法字符）
    
    :param prompt: 原始 prompt
    :return: 合法的文件名
    """
    # 去掉特殊字符，只保留字母、数字、下划线和空格
    sanitized = re.sub(r'[^\w\s-]', '', prompt)
    # 将空格替换为下划线
    sanitized = sanitized.replace(' ', '_')
    # 限制文件名长度（避免过长）
    return sanitized[:50]  # 最多保留 50 个字符

def generate_and_save_image(pipeline, prompt, negative_prompt, seed, save_dir="output_images"):
    """
    生成图片并保存到本地，文件名根据 prompt 生成

    :param pipeline: 已初始化的 StableDiffusionXLPipeline 对象
    :param prompt: 生成图片的正向提示词
    :param negative_prompt: 生成图片的负向提示词
    :param seed: 随机种子
    :param save_dir: 图片保存的目录，默认为 "output_images"
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成图片
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.manual_seed(seed),
    ).images[0]

    # 根据 prompt 生成文件名
    filename = sanitize_filename(prompt) + f"_seed_{seed}.png"
    save_path = os.path.join(save_dir, filename)

    # 保存图片（不调整大小）
    image.save(save_path)
    print(f"Generated and saved: {save_path}")

# 定义所有调用参数
calls = [
    {
        "prompt": "couple ,ZHONGLI, NINGGUANG\(genshin impact\) highres, masterpiece, pack clothes in a bag",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 0
    },
    {
        "prompt": "COUPLE ,KAEDEHARA KAZUHA, NINGGUANG\(genshin impact\) highres, masterpiece, pack clothes in a bag",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 0
    },
    {
        "prompt": "COUPLE ,KAEDEHARA KAZUHA, scaramouche\(genshin impact\) highres, masterpiece, pack clothes in a bag",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 0
    },
    {
        "prompt": "COUPLE ,KAEDEHARA KAZUHA, scaramouche\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 10
    },
    {
        "prompt": "COUPLE ,LYNEY, THOMA\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 10
    },
    {
        "prompt": "COUPLE ,LYNEY, THOMA\(genshin impact\) highres, masterpiece, serve noodles in a bowl",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 10
    },
    {
        "prompt": "TRIPLE ,KAEDEHARA KAZUHA, SCARAMOUCHE, ZHONGLI\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 10
    },
    {
        "prompt": "TRIPLE ,KAEDEHARA KAZUHA, NAHIDA, SCARAMOUCHE\(genshin impact\) highres, masterpiece, pack clothes in a bag",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 12
    },
    {
        "prompt": "TRIPLE ,KAEDEHARA KAZUHA, SCARAMOUCHE, ZHONGLI\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 10
    },
    {
        "prompt": "In a Bar ,TRIPLE ,KAEDEHARA KAZUHA, SCARAMOUCHE, ZHONGLI\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 100
    },
    {
        "prompt": "In a Swimming Pool ,TRIPLE ,KAEDEHARA KAZUHA, SCARAMOUCHE, ZHONGLI\(genshin impact\) highres, masterpiece, drink beverages through a straw",
        "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
        "seed": 8
    }
]

# 批量生成图片
for call in calls:
    generate_and_save_image(pipeline, **call)
```

![couple_ZHONGLI_NINGGUANGgenshin_impact_highres_mas_seed_0](https://github.com/user-attachments/assets/d2353d35-0066-469e-8e4c-748aefa2b73c)

![COUPLE_KAEDEHARA_KAZUHA_NINGGUANGgenshin_impact_hi_seed_0](https://github.com/user-attachments/assets/aa959b84-023c-49a2-8a45-a643bc899476)


![COUPLE_KAEDEHARA_KAZUHA_scaramouchegenshin_impact__seed_0](https://github.com/user-attachments/assets/28d7f2de-c7b2-4267-9e8a-28f31bdbb238)


![COUPLE_KAEDEHARA_KAZUHA_scaramouchegenshin_impact__seed_10](https://github.com/user-attachments/assets/b0d7c9fa-6ce3-413a-a1ce-fc5efcf6563e)


![COUPLE_LYNEY_THOMAgenshin_impact_highres_masterpie_seed_10](https://github.com/user-attachments/assets/2a1a566f-694c-4943-9070-2ce4b5c5ace1)

![TRIPLE_KAEDEHARA_KAZUHA_SCARAMOUCHE_ZHONGLIgenshin_seed_10](https://github.com/user-attachments/assets/071aa7c1-54e1-4d6d-94df-1d86ffc28607)


![TRIPLE_KAEDEHARA_KAZUHA_NAHIDA_SCARAMOUCHEgenshin__seed_12](https://github.com/user-attachments/assets/361d6a62-3e06-47a8-94a5-86004d301f51)

![In_a_Bar_TRIPLE_KAEDEHARA_KAZUHA_SCARAMOUCHE_ZHONG_seed_100](https://github.com/user-attachments/assets/4f42f25a-24b6-40f9-8a33-e78ff93eb1b7)


![In_a_Swimming_Pool_TRIPLE_KAEDEHARA_KAZUHA_SCARAMO_seed_8](https://github.com/user-attachments/assets/36b0827d-d0bb-420a-ac2c-8a25d6bfa249)

- Simgle Image
```python
import os
from tqdm import tqdm
import pandas as pd
from diffusers import StableDiffusionXLPipeline
import torch

# 定义函数生成 prompt
#### 有 性别错位
def gen_one_person_prompt(name, row):
    return "SOLO ,{}, {} \(genshin impact\), masterpiece, {}".format(row["en_location"], name, row["en_action"])

# 定义 name_dict
new_dict = {
    '砂糖': 'SUCROSE', '五郎': 'GOROU', '雷电将军': 'RAIDEN SHOGUN', '七七': 'QIQI', '重云': 'CHONGYUN',
    '荒泷一斗': 'ARATAKI ITTO', '申鹤': 'SHENHE', '赛诺': 'CYNO', '绮良良': 'KIRARA', '优菈': 'EULA',
    '魈': 'XIAO', '行秋': 'XINGQIU', '枫原万叶': 'KAEDEHARA KAZUHA', '凯亚': 'KAEYA', '凝光': 'NING GUANG',
    '安柏': 'AMBER', '柯莱': 'COLLEI', '林尼': 'LYNEY', '胡桃': 'HU TAO', '甘雨': 'GANYU',
    '神里绫华': 'KAMISATO AYAKA', '钟离': 'ZHONGLI', '纳西妲': 'NAHIDA', '云堇': 'YUN JIN',
    '久岐忍': 'KUKI SHINOBU', '迪西娅': 'DEHYA', '珐露珊': 'FARUZAN', '公子 达达利亚': 'TARTAGLIA',
    '琳妮特': 'LYNETTE', '罗莎莉亚': 'ROSARIA', '八重神子': 'YAE MIKO', '迪奥娜': 'DIONA',
    '迪卢克': 'DILUC', '托马': 'THOMA', '神里绫人': 'KAMISATO AYATO', '鹿野院平藏': 'SHIKANOIN HEIZOU',
    '阿贝多': 'ALBEDO', '琴': 'JEAN', '芭芭拉': 'BARBARA', '雷泽': 'RAZOR',
    '珊瑚宫心海': 'SANGONOMIYA KOKOMI', '温迪': 'VENTI', '烟绯': 'YANFEI', '艾尔海森': 'ALHAITHAM',
    '诺艾尔': 'NOELLE', '流浪者 散兵': 'SCARAMOUCHE', '班尼特': 'BENNETT', '芙宁娜': 'FURINA',
    '夏洛蒂': 'CHARLOTTE', '宵宫': 'YOIMIYA', '妮露': 'NILOU', '瑶瑶': 'YAOYAO'
}

# 初始化 Stable Diffusion XL Pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-3.1",
    torch_dtype=torch.float16
).to("cuda")

# 定义 negative prompt
negative_prompt = "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],"

# 假设 dating_df 是一个包含 en_location 和 en_action 列的 DataFrame
# dating_df = pd.read_csv("your_dating_df.csv")  # 替换为你的 DataFrame 加载逻辑

from datasets import load_dataset
dating_df = load_dataset("svjack/dating-actions-en-zh")["train"].to_pandas()

#### en_action
dating_df["en_action"].drop_duplicates().map(lambda x: x.replace("your", "").replace("  ", " ")).values.tolist()

#### en_location
dating_df["en_location"].drop_duplicates().map(lambda x: x.replace("your", "").replace("  ", " ")).values.tolist()

location_prepositions = {
    'home': 'at',       # 通常用 "at home"
    'kitchen': 'in',    # 通常用 "in the kitchen"
    'park': 'in',       # 通常用 "in the park"
    'garage': 'in',     # 通常用 "in the garage"
    'cafe': 'at',       # 通常用 "at the cafe"
    'restaurant': 'at', # 通常用 "at the restaurant"
    'restroom': 'in',   # 通常用 "in the restroom"
    'tea house': 'at',  # 通常用 "at the tea house"
    'supermarket': 'at' # 通常用 "at the supermarket"
}

dating_df["en_action"] = dating_df["en_action"].map(lambda x: x.replace("your", "").replace("  ", " ")).values.tolist()
dating_df["en_location"] = dating_df["en_location"].map(lambda x: x.replace("your", "").replace("  ", " ")).map(
    lambda x: "{} {}".format(location_prepositions[x], x).strip()
).values.tolist()


# 定义 times 参数
times = 3  # 指定 pipeline 执行的次数

# 迭代 dating_df 的每一行
for index, row in tqdm(dating_df.iterrows(), desc="Generating Images", total=len(dating_df)):
    for name, value in new_dict.items():
        # 生成 prompt
        prompt = gen_one_person_prompt(value, row)

        # 创建保存路径
        output_dir = os.path.join("single_output_images", name)  # 路径尾部添加 single
        os.makedirs(output_dir, exist_ok=True)

        # 执行 pipeline 多次
        for i in range(times):
            # 设置随机种子（每次生成图像时 seed 不同）
            seed = index + hash(name) + i  # 使用 index、name 和 i 作为 seed
            generator = torch.manual_seed(seed)

            # 生成图像
            image = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=generator,
            ).images[0]

            # 将 en_location 和 en_action 的值编入文件名
            en_location_clean = row["en_location"].replace(" ", "_").replace("/", "_")  # 清理路径不兼容字符
            en_action_clean = row["en_action"].replace(" ", "_").replace("/", "_")  # 清理路径不兼容字符
            image_path = os.path.join(output_dir, f"{name}_{en_location_clean}_{en_action_clean}_{seed}.png")

            # 保存图像
            image.save(image_path)
        #break
    #break

print("所有图像生成完成！")

```

- COUPLE TRIPLE Image

```bash
python couple_triple_script.py --output_dir c5t7_dir --num_couple 5 --num_triple 7
```

```python
#### pip install -U hf_transfer
import os

# 如果需要设置自定义的 Hugging Face 端点（例如本地服务器），可以取消注释以下行
# os.environ["HF_ENDPOINT"] = "http://localhost:5564"

# 启用 HF 传输加速（可选）
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, logging

# 设置日志级别为调试模式
logging.set_verbosity_debug()

# 初始化 HfApi
hf = HfApi()

# 上传文件
hf.upload_file(
    path_or_fileobj="原神单人图片1.zip",  # 本地文件路径
    path_in_repo="原神单人图片1.zip",    # 文件在仓库中的路径（这里直接放在根目录）
    repo_id="svjack/Genshin-Impact-Novel-Video",  # 目标仓库 ID
    repo_type="dataset"  # 仓库类型（模型仓库）
)
```

- Couple Image validation
```python
#!/usr/bin/env python
# coding: utf-8

from datasets import load_dataset
import json
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

# 1. 加载数据集
ds = load_dataset("svjack/Genshin-Impact-Style-Blended-Couple-with-Tags")

# 2. 转换为 Pandas DataFrame，并移除不需要的列
df = ds["train"].remove_columns(["image"]).to_pandas()

# 3. 处理 tag_json 列
df["tag_json"] = df["tag_json"].map(json.loads).map(lambda d: 
                                   dict(map(lambda t2: (t2[0], json.loads(t2[1])), d.items()))
                                  )

# 4. 定义映射字典
new_dict = {
    '砂糖': 'SUCROSE', '五郎': 'GOROU', '雷电将军': 'RAIDEN SHOGUN', '七七': 'QIQI', '重云': 'CHONGYUN',
    '荒泷一斗': 'ARATAKI ITTO', '申鹤': 'SHENHE', '赛诺': 'CYNO', '绮良良': 'KIRARA', '优菈': 'EULA',
    '魈': 'XIAO', '行秋': 'XINGQIU', '枫原万叶': 'KAEDEHARA KAZUHA', '凯亚': 'KAEYA', '凝光': 'NING GUANG',
    '安柏': 'AMBER', '柯莱': 'COLLEI', '林尼': 'LYNEY', '胡桃': 'HU TAO', '甘雨': 'GANYU',
    '神里绫华': 'KAMISATO AYAKA', '钟离': 'ZHONGLI', '纳西妲': 'NAHIDA', '云堇': 'YUN JIN',
    '久岐忍': 'KUKI SHINOBU', '迪西娅': 'DEHYA', '珐露珊': 'FARUZAN', '公子 达达利亚': 'TARTAGLIA',
    '琳妮特': 'LYNETTE', '罗莎莉亚': 'ROSARIA', '八重神子': 'YAE MIKO', '迪奥娜': 'DIONA',
    '迪卢克': 'DILUC', '托马': 'THOMA', '神里绫人': 'KAMISATO AYATO', '鹿野院平藏': 'SHIKANOIN HEIZOU',
    '阿贝多': 'ALBEDO', '琴': 'JEAN', '芭芭拉': 'BARBARA', '雷泽': 'RAZOR',
    '珊瑚宫心海': 'SANGONOMIYA KOKOMI', '温迪': 'VENTI', '烟绯': 'YANFEI', '艾尔海森': 'ALHAITHAM',
    '诺艾尔': 'NOELLE', '流浪者 散兵': 'SCARAMOUCHE', '班尼特': 'BENNETT', '芙宁娜': 'FURINA',
    '夏洛蒂': 'CHARLOTTE', '宵宫': 'YOIMIYA', '妮露': 'NILOU', '瑶瑶': 'YAOYAO'
}
rev_dict = dict(map(lambda t2: (t2[1].replace(" ", "_"), t2[0]), new_dict.items()))

mapping_dict = {
    "diona": "迪奥娜",
    "clorinde": "克洛琳德",
    "noelle": "诺艾尔",
    "kuki_shinobu": "久岐忍",
    "shikanoin_heizou": "鹿野院平藏",
    "rosaria": "罗莎莉亚",
    "collei": "柯莱",
    "arlecchino": "阿蕾奇诺",
    "kujou_sara": "九条裟罗",
    "nilou": "妮露",
    "kirara": "绮良良",
    "ningguang": "凝光",
    "xiao": "魈",
    "beidou": "北斗",
    "xiangling": "香菱",
    "sayu": "早柚",
    "kaeya": "凯亚",
    "ganyu": "甘雨",
    "arataki_itto": "荒泷一斗",
    "kaedehara_kazuha": "枫原万叶",
    "lisa": "丽莎",
    "sangonomiya_kokomi": "珊瑚宫心海",
    "jean": "琴",
    "yelan": "夜兰",
    "neuvillette": "那维莱特",
    "razor": "雷泽",
    "klee": "可莉",
    "lynette": "琳妮特",
    "wanderer": "流浪者",
    "kaveh": "卡维",
    "lyney": "林尼",
    "alhaitham": "艾尔海森",
    "layla": "莱依拉",
    "fischl": "菲谢尔",
    "gorou": "五郎",
    "kamisato_ayaka": "神里绫华",
    "barbara": "芭芭拉",
    "hu_tao": "胡桃",
    "raiden_shogun": "雷电将军",
    "qiqi": "七七",
    "venti": "温迪",
    "yae_miko": "八重神子",
    "nahida": "纳西妲",
    "sucrose": "砂糖",
    "shenhe": "申鹤",
    "xingqiu": "行秋",
    "xianyun": "闲云",
    "yun_jin": "云堇",
    "navia": "娜维娅",
    "mona": "莫娜",
    "thoma": "托马",
    "yoimiya": "宵宫",
    "wriothesley": "莱欧斯利",
    "faruzan": "珐露珊",
    "kamisato_ayato": "神里绫人",
    "tartaglia": "达达利亚",
    "dehya": "迪希雅",
    "albedo": "阿贝多",
    "keqing": "刻晴",
    "eula": "优菈",
    "cyno": "赛诺",
    "amber": "安柏",
    "tighnari": "提纳里",
    "diluc": "迪卢克",
    "zhongli": "钟离",
    "yanfei": "烟绯",
    "furina": "芙宁娜",
    "chongyun": "重云"
}

def replace_characters_with_chinese_names(tag_json_dict, mapping_dict):
    # 存储成对结果的列表
    paired_results = []

    # 遍历字典中的每个键值对
    for key, value in tag_json_dict.items():
        # 提取 results 中 prediction 为 "Same" 的部分
        same_results = [result for result in value.get("results", []) if result.get("prediction") == "Same"]
        # 提取 characters 部分
        characters = value.get("characters", {})
        # 提取 features 中的 1boy 和 1girl 字段
        features = value.get("features", {})
        is_boy = features.get("1boy", 0) > 0.5  # 假设大于 0.5 表示存在
        is_girl = features.get("1girl", 0) > 0.5  # 假设大于 0.5 表示存在

        # 确定性别标签
        gender_label = []
        if is_boy:
            gender_label.append("boy")
        if is_girl:
            gender_label.append("girl")
        if not gender_label:
            gender_label.append("unknown")  # 如果没有明确的性别标签，标记为 unknown

        # 替换 characters 中的英文标签为中文名称
        chinese_characters = {}
        for tag, score in characters.items():
            # 将 tag 转换为小写并去掉括号部分（如果有）
            tag_normalized = tag.lower().split("_(")[0]
            # 查找映射字典中的中文名称
            chinese_name = mapping_dict.get(tag_normalized, tag)  # 如果找不到映射，保留原标签
            chinese_characters[chinese_name] = score

        # 如果 same_results 或 characters 不为空，则添加到结果中
        if same_results or chinese_characters:
            paired_results.append({
                "same_results": same_results,
                "characters": chinese_characters,
                "gender_label": gender_label
            })

    return paired_results

# 5. 定义 extract_and_check_characters 函数
def extract_and_check_characters(d, rev_dict, mapping_dict):
    # 提取 im_name 中的人物
    im_name_characters = list(filter(lambda t2: t2[0] in d["im_name"], rev_dict.items()))
    im_name_characters = [t2[1] for t2 in im_name_characters]  # 提取中文名称

    # 提取 tag_json 中的人物
    paired_results = replace_characters_with_chinese_names(d["tag_json"], mapping_dict)

    # 检查 im_name 中的人物是否在 tag_json 中
    matched_results = []
    for character in im_name_characters:
        found_in_same_results = False
        found_in_characters = False

        # 检查 same_results
        for pair in paired_results:
            if character in [result["name"] for result in pair["same_results"]]:
                found_in_same_results = True
                break

        # 检查 characters
        for pair in paired_results:
            if character in pair["characters"]:
                found_in_characters = True
                break

        # 记录匹配结果
        matched_results.append({
            "character": character,
            "found_in_same_results": found_in_same_results,
            "found_in_characters": found_in_characters
        })

    return {
        "im_name_characters": im_name_characters,
        "tag_json_characters": paired_results,
        "matched_results": matched_results
    }

# 6. 遍历 DataFrame 的每一行，调用 extract_and_check_characters 函数
results = []  # 用于存储所有行的结果

# 使用 tqdm 显示进度条
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    d = row.to_dict()
    result = extract_and_check_characters(d, rev_dict, mapping_dict)
    
    # 将 result 的结果存储到 d 中
    d.update(result)
    
    # 将更新后的 d 添加到 results 列表中
    results.append(d)

# 将 results 转换为 DataFrame
results_df = pd.DataFrame(results)

# 7. 保存结果到 CSV 文件
results_df.to_csv("results_with_matched_characters.csv", index=False, encoding="utf-8-sig")

print("处理完成，结果已保存到 results_with_matched_characters.csv")

idx = 2
results_df["matched_results"].iloc[idx]

ds["train"][idx]["image"].resize((512, 512))
```


```python
seed = 2
device = 0

import sys
import time
import warnings

sys.path.append('../src')
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.utils import make_image_grid
from functools import reduce

from util import seed_everything, blend
from model import StableMultiDiffusionSDXLPipeline
from ipython_util import dispt


seed_everything(seed)
device = f'cuda:{device}'
print(f'[INFO] Initialized with seed  : {seed}')
print(f'[INFO] Initialized with device: {device}')

background_image = Image.open('../assets/timessquare/timessquare.jpeg')
display(background_image)

smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
)

# Prepare masks.
print('[INFO] Loading masks...')

name = 'timessquare'
mask_all = Image.open(f'../assets/{name}/{name}_full.png').convert('RGBA').resize((1024, 1024))
masks = [Image.open(f'../assets/{name}/{name}_{i}.png').convert('RGBA').resize((1024, 1024)) for i in range(1, 3)]
masks = [(T.ToTensor()(mask)[-1:] > 0.5).float() for mask in masks]
# Background mask is not explicitly specified in the inpainting mode.
dispt(masks, row=1)

masks = torch.stack(masks, dim=0)

# Prepare prompts.
print('[INFO] Loading prompts...')

background_prompt = '1girl, 1boy, times square'
background_prompt = '1boy, 1boy, times square'
#background_prompt = ""

prompts = [
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]

prompts = [
    # Foreground prompts.
    'KAEDEHARA KAZUHA, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
    'SCARAMOUCHE, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
]

negative_prompts = [
    '1girl',
    '1boy',
]

negative_prompts = [
    '',
    '',
]

negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]
background_negative_prompt = negative_prompt_prefix

print('Background Prompt: ' + background_prompt)
print('Background Negative Prompt: ' + background_negative_prompt)
for i, prompt in enumerate(prompts):
    print(f'Prompt{i}: ' + prompt)
for i, prompt in enumerate(negative_prompts):
    print(f'Negative Prompt{i}: ' + prompt)

height, width = masks.shape[-2:]

tic = time.time()
img = smd(
    prompts,
    negative_prompts,
    masks=masks.float(),
    guidance_scale=0,
    # Use larger standard deviation to harmonize the inpainting result!
    mask_stds=8.0,
    mask_strengths=1,
    height=height,
    width=width,
    bootstrap_steps=2,
    #bootstrap_leak_sensitivity=0.1,
    bootstrap_leak_sensitivity=0.2,
    # This is for providing the image input.
    background=background_image,
    background_prompt=background_prompt,
    background_negative_prompt=background_negative_prompt,
)
toc = time.time()
print(f'Elapsed Time: {toc - tic}')
display(img)
```
![inpainting_out (1)](https://github.com/user-attachments/assets/adf868e4-bc5f-4f0d-85fd-0953039ea4b8)


```python
seed = 1
device = 0

import sys
import time
import warnings

sys.path.append('../src')
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.utils import make_image_grid
from functools import reduce

from util import seed_everything, blend
from model import StableMultiDiffusionSDXLPipeline
from ipython_util import dispt
from prompt_util import print_prompts, preprocess_prompts


seed_everything(seed)
device = f'cuda:{device}'
print(f'[INFO] Initialized with seed  : {seed}')
print(f'[INFO] Initialized with device: {device}')

smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
    has_i2t=False,
)

# Prepare masks.
print('[INFO] Loading masks...')

name = 'fantasy_large'
mask_all = Image.open(f'../assets/{name}/{name}_full.png').convert('RGBA')
masks = [Image.open(f'../assets/{name}/{name}_{i}.png').convert('RGBA') for i in range(1, 3)]
masks = [(T.ToTensor()(mask)[-1:] > 0.5).float() for mask in masks]
# Background is simply non-marked set of pixels.
background = reduce(torch.logical_and, [m == 0 for m in masks])
dispt([background] + masks, row=1)

masks = torch.stack([background] + masks, dim=0)
mask_strengths = 1.0
mask_stds = 0.0

###

# Prepare prompts.
print('[INFO] Loading prompts...')

prompts = [
    # Background prompt.
    'purple sky, planets, planets, planets, stars, stars, stars',
    # Foreground prompts.
    'a photo of the dolomites, masterpiece, absurd quality, background, no humans',
    #'1girl, looking at viewer, pretty face, blue hair, fantasy style, witch, magi, robe',
    "KAEDEHARA KAZUHA, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer"
]
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]

negative_prompts = [
    '',
    '',
    '',
]

negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

###

# Preprocess prompts for better results.
quality_name = 'Standard v3.1'
style_name = '(None)'

prompts, negative_prompts = preprocess_prompts(
    prompts, negative_prompts, style_name=style_name, quality_name=quality_name)

###

print_prompts(prompts, negative_prompts, has_background=True)
height, width = masks.shape[-2:]

tic = time.time()
img = smd(
    prompts, negative_prompts, masks=masks.float(),
    mask_stds=mask_stds, mask_strengths=mask_strengths * 1,
    height=height, width=width, bootstrap_steps=2,
    bootstrap_leak_sensitivity=0.1,
    guidance_scale=0,
)
toc = time.time()
print(f'Elapsed Time: {toc - tic}')
display(img)
```

![simple_out](https://github.com/user-attachments/assets/d5f29e45-462f-4b3b-b3d7-e174162b004b)

```python
# 依赖放在函数外部
import sys
import time
import warnings

sys.path.append('src')
warnings.filterwarnings('ignore')

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.utils import make_image_grid
from functools import reduce

from util import seed_everything, blend
from model import StableMultiDiffusionSDXLPipeline
from ipython_util import dispt
from prompt_util import print_prompts, preprocess_prompts

# 初始化 smd（只执行一次）
seed = 1  # 默认 seed
device = 0  # 默认 device
device = f'cuda:{device}'
smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
    has_i2t=False,
)

def generate_image_with_background(
    smd,  # 从外部传入已经初始化的 smd
    background_image_path,  # 背景图片路径
    mask_paths,  # mask 路径列表
    prompts,  # 提示词列表（仅前景提示词）
    negative_prompts,  # 负面提示词列表（仅前景负面提示词）
    background_prompt,  # 背景提示词
    background_negative_prompt=None,  # 背景负面提示词（可选）
    mask_stds=8.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.2,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='Standard v3.1',  # 质量名称
    seed=1,  # 随机种子
    device=0,  # 设备编号
    num_inference_steps = 5,
):
    # 设置随机种子和设备
    if seed >= 0:
        seed_everything(seed)
    device = f'cuda:{device}'
    print(f'[INFO] Initialized with seed  : {seed}')
    print(f'[INFO] Initialized with device: {device}')

    # 加载背景图片
    background_image = Image.open(background_image_path)
    display(background_image)

    # 加载 masks
    print('[INFO] Loading masks...')
    masks = [Image.open(path).convert('RGBA').resize((1024, 1024)) for path in mask_paths]
    masks = [(T.ToTensor()(mask)[-1:] > 0.5).float() for mask in masks]
    masks = torch.stack(masks, dim=0)  # 不包含背景 mask
    dispt(masks, row=1)

    # 处理 prompts
    print('[INFO] Loading prompts...')
    if background_negative_prompt is None:
        background_negative_prompt = 'worst quality, bad quality, normal quality, cropped, framed'

    negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
    negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

    # 预处理 prompts
    prompts, negative_prompts = preprocess_prompts(
        prompts, negative_prompts, style_name=style_name, quality_name=quality_name)

    print('Background Prompt: ' + background_prompt)
    print('Background Negative Prompt: ' + background_negative_prompt)
    for i, prompt in enumerate(prompts):
        print(f'Prompt{i}: ' + prompt)
    for i, prompt in enumerate(negative_prompts):
        print(f'Negative Prompt{i}: ' + prompt)

    height, width = masks.shape[-2:]

    # 生成图像
    tic = time.time()
    img = smd(
        prompts, negative_prompts, masks=masks.float(),
        mask_stds=mask_stds, mask_strengths=mask_strengths,
        height=height, width=width, bootstrap_steps=bootstrap_steps,
        bootstrap_leak_sensitivity=bootstrap_leak_sensitivity,
        guidance_scale=guidance_scale,
        background=background_image,  # 传入背景图片
        background_prompt=background_prompt,  # 传入背景提示词
        background_negative_prompt=background_negative_prompt,  # 传入背景负面提示词
        num_inference_steps = num_inference_steps
    )
    toc = time.time()
    print(f'Elapsed Time: {toc - tic}')
    display(img)
    return img

def generate_image_without_background(
    smd,  # 从外部传入已经初始化的 smd
    mask_paths,  # mask 路径列表（不包含背景 mask）
    prompts,  # 提示词列表（包含背景提示词和前景提示词）
    negative_prompts,  # 负面提示词列表（包含背景负面提示词和前景负面提示词）
    mask_stds=0.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='Standard v3.1',  # 质量名称
    seed=1,  # 随机种子
    device=0,  # 设备编号
    num_inference_steps = 5
):
    """
    生成没有背景的图像。

    参数:
        smd: 已经初始化的 StableMultiDiffusionSDXLPipeline 实例。
        mask_paths: mask 路径列表（不包含背景 mask）。
        prompts: 提示词列表（包含背景提示词和前景提示词）。
        negative_prompts: 负面提示词列表（包含背景负面提示词和前景负面提示词）。
        mask_stds: mask 的标准差，控制 mask 的模糊程度。
        mask_strengths: mask 的强度，控制前景与背景的融合程度。
        bootstrap_steps: 引导步数，控制生成过程中的迭代次数。
        bootstrap_leak_sensitivity: 引导泄漏敏感度，控制前景与背景的泄漏程度。
        guidance_scale: 引导比例，控制生成图像的风格强度。
        style_name: 风格名称，用于提示词预处理。
        quality_name: 质量名称，用于提示词预处理。
        seed: 随机种子，控制生成过程的随机性。
        device: 设备编号，指定使用的 GPU 设备。

    返回:
        生成的图像。
    """
    # 设置随机种子和设备
    if seed >= 0:
        seed_everything(seed)
    device = f'cuda:{device}'
    print(f'[INFO] Initialized with seed  : {seed}')
    print(f'[INFO] Initialized with device: {device}')

    # 加载 masks
    print('[INFO] Loading masks...')
    masks = [Image.open(path).convert('RGBA').resize((1024, 1024)) for path in mask_paths]
    masks = [(T.ToTensor()(mask)[-1:] > 0.5).float() for mask in masks]
    masks = torch.stack(masks, dim=0)  # 不包含背景 mask
    dispt(masks, row=1)

    # 处理 prompts
    print('[INFO] Loading prompts...')
    negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
    negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

    # 预处理 prompts
    prompts, negative_prompts = preprocess_prompts(
        prompts, negative_prompts, style_name=style_name, quality_name=quality_name)

    print('Background Prompt: ' + prompts[0])
    print('Background Negative Prompt: ' + negative_prompts[0])
    for i, prompt in enumerate(prompts[1:]):
        print(f'Foreground Prompt{i}: ' + prompt)
    for i, prompt in enumerate(negative_prompts[1:]):
        print(f'Foreground Negative Prompt{i}: ' + prompt)

    height, width = masks.shape[-2:]

    # 生成图像
    tic = time.time()
    img = smd(
        prompts, negative_prompts, masks=masks.float(),
        mask_stds=mask_stds, mask_strengths=mask_strengths,
        height=height, width=width, bootstrap_steps=bootstrap_steps,
        bootstrap_leak_sensitivity=bootstrap_leak_sensitivity,
        guidance_scale=guidance_scale,
        num_inference_steps = num_inference_steps
    )
    toc = time.time()
    print(f'Elapsed Time: {toc - tic}')
    display(img)
    return img
```

```python
# 调用第一个函数
img = generate_image_with_background(
    smd=smd,
    background_image_path='assets/timessquare/timessquare.jpeg',
    mask_paths=[
        f'assets/timessquare/timessquare_1.png',
        f'assets/timessquare/timessquare_2.png'
    ],
    prompts=[
    # Foreground prompts.
    'KAEDEHARA KAZUHA, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
    'SCARAMOUCHE, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
    ],
    negative_prompts=[
        '',
        '',
    ],
    background_prompt='1boy, 1boy, times square',
    background_negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask_stds=8.0,
    mask_strengths=1.0,
    bootstrap_steps=2,
    bootstrap_leak_sensitivity=0.2,
    guidance_scale=0,
    seed = 0
)
```

![枫散0](https://github.com/user-attachments/assets/9f41cf4c-514e-4cbb-94db-83fae8ea393b)


```python
# 假设 smd 已经初始化
name = 'fantasy_large'

# 准备 mask 路径
mask_paths = [
    f'assets/{name}/{name}_full.png',
    f'assets/{name}/{name}_1.png',
    f'assets/{name}/{name}_2.png'
]

# 准备 prompts
prompts = [
    # Background prompt.
    'purple sky, planets, planets, planets, stars, stars, stars',
    # Foreground prompts.
    'a photo of the dolomites, masterpiece, absurd quality, background, no humans',
    "KAEDEHARA KAZUHA, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer"
]

# 准备 negative prompts
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=0.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='Standard v3.1',  # 质量名称
    seed=1,  # 随机种子
    device=0  # 设备编号
)
```

![枫叶0](https://github.com/user-attachments/assets/f90858c7-9b62-4665-8707-50998b23ab7d)

```python
# 准备 mask 路径
mask_paths = [
    f'assets/fantasy_large/fantasy_large_full.png',
    f'assets/timessquare/timessquare_1.png',
    f'assets/timessquare/timessquare_2.png'
]

# 准备 prompts
prompts = [
    # Background prompt.
    #'purple sky, planets, planets, planets, stars, stars, stars',
    'dimly lit bar, neon lights, glowing cocktails, wooden counter, bustling crowd, jazz music, cozy atmosphere, vintage decor, reflective surfaces, soft shadows',
    # Foreground prompts.
    "KAEDEHARA KAZUHA, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer",
    'SCARAMOUCHE, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
]

# 准备 negative prompts
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=0.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='Standard v3.1',  # 质量名称
    seed=1,  # 随机种子
    device=0  # 设备编号
)
```


![枫散1](https://github.com/user-attachments/assets/487fd73a-e084-4694-9951-36ae46bcdcc6)

```python
# 准备 mask 路径
mask_paths = [
    f'assets/fantasy_large/fantasy_large_full.png',
    f'assets/timessquare/timessquare_1.png',
    f'assets/timessquare/timessquare_2.png'
]

# 准备 prompts
prompts = [
    # Background prompt.
    #'purple sky, planets, planets, planets, stars, stars, stars',
    'a fast food restaurant, brightly lit, with other customers and busy staff, a plate of fries and a soda on the table, relaxed and cheerful atmosphere',
    # Foreground prompts.
    "CHONGYUN, \(genshin impact\) highres, masterpiece, eating a hamburger, looking at viewer",
    'XINGQIU, \(genshin impact\) highres, masterpiece, drink beverages through a straw, looking at viewer',
]

# 准备 negative prompts
negative_prompts = [
    '',
    '',
    '',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=0.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='(None)',  # 质量名称
    seed=2,  # 随机种子
    device=0  # 设备编号
)
```

![行重0](https://github.com/user-attachments/assets/a39b5222-1a30-4af6-9bd2-1adfdda41750)

##### One Can Make Mask Image use Brush by: https://huggingface.co/spaces/svjack/inpaint-mask-maker
##### Or Make Segmentation by https://huggingface.co/spaces/svjack/BRIA-RMBG-2.0 
##### And Overlay Different Colored Mask by https://huggingface.co/spaces/svjack/Layer-Overlay-Tool
##### blue_yellow_green.webp blue.webp yellow.webp green.webp
```python
# 准备 mask 路径
mask_paths = [
    f'blue_yellow_green.webp',
    f'blue.webp',
    f'yellow.webp',
    f'green.webp',
]

# 准备 prompts
prompts = [
    # Background prompt.
    #'purple sky, planets, planets, planets, stars, stars, stars',
    #'a fast food restaurant, brightly lit, with other customers and busy staff, a plate of fries and a soda on the table, relaxed and cheerful atmosphere',
    "dimly lit bar, neon lights",
    # Foreground prompts.
    'solo, SCARAMOUCHE, \(genshin impact\) highres, masterpiece, drink beverages through a straw, laugthing',
    "mirror without people",
    'solo ,XINGQIU, \(genshin impact\) highres, masterpiece, drink beverages through a straw, laugthing',
]

# 准备 negative prompts
negative_prompts = [
    '',
    'Multiple People',
    '',
    'Multiple People',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=0.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='(None)',  # 质量名称
    seed=0,  # 随机种子
    device=0  # 设备编号
)
```

![two_people_with_mirror](https://github.com/user-attachments/assets/0bbbd6c9-9473-450e-b747-2e41a9546fb9)

```python
# 准备 mask 路径
mask_paths = [
    f'blue_yellow.webp',
    f'blue_left.webp',
    f'yellow_right.webp',
]

# 准备 prompts
prompts = [
    # Background prompt.
    'purple sky, planets, planets, planets, stars, stars, stars',
    #'a fast food restaurant, brightly lit, with other customers and busy staff, a plate of fries and a soda on the table, relaxed and cheerful atmosphere',
    #"dimly lit bar, neon lights",
    # Foreground prompts.
    'solo, XINGQIU, \(genshin impact\) highres, masterpiece, eat rice',
    'solo ,RAIDEN SHOGUN, \(genshin impact\) highres, masterpiece, drink water',
]

# 准备 negative prompts
negative_prompts = [
    '',
    '',
    '',
    #'Multiple People',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=1.0,  # mask 标准差
    mask_strengths=1.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='(None)',  # 质量名称
    seed=0,  # 随机种子
    device=0,  # 设备编号
)
```

![行秋_将军](https://github.com/user-attachments/assets/d57aaf90-7fdd-432f-ac65-ac4f9ec14f7f)


```python
# 准备 mask 路径
mask_paths = [
    f'blue_yellow_m.webp',
    f'blue_left_m.webp',
    f'yellow_right_m.webp',
]

# 准备 prompts
prompts = [
    # Background prompt.
    #'purple sky, planets, planets, planets, stars, stars, stars',
    #'a fast food restaurant, brightly lit, with other customers and busy staff, a plate of fries and a soda on the table, relaxed and cheerful atmosphere',
    "dimly lit bar, neon lights",
    # Foreground prompts.
    'solo, FARUZAN, \(genshin impact\) highres, masterpiece, drink water',
    'solo ,RAIDEN SHOGUN, \(genshin impact\) highres, masterpiece, drink water',
]

# 准备 negative prompts
negative_prompts = [
    '',
    '',
    '',
    #'Multiple People',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=2.0,  # mask 标准差
    mask_strengths=3.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='(None)',  # 质量名称
    seed=0,  # 随机种子
    device=0,  # 设备编号
)
```

![百岁珊——将军](https://github.com/user-attachments/assets/05304beb-e546-436d-a22b-9440c1bebff3)

```python
# 准备 mask 路径
mask_paths = [
    f'blue_yellow_m.webp',
    f'blue_left_m.webp',
    f'yellow_right_m.webp',
]

# 准备 prompts
prompts = [
    # Background prompt.
    #'purple sky, planets, planets, planets, stars, stars, stars',
    #'a fast food restaurant, brightly lit, with other customers and busy staff, a plate of fries and a soda on the table, relaxed and cheerful atmosphere',
    "dimly lit bar, neon lights",
    # Foreground prompts.
    'solo, FARUZAN, \(genshin impact\) highres, masterpiece, drink water',
    'solo ,RAIDEN SHOGUN, \(genshin impact\) highres, masterpiece, drink water',
]

# 准备 negative prompts
negative_prompts = [
    '',
    '',
    '',
    #'Multiple People',
]

# 添加负面提示词前缀
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# 调用函数
img = generate_image_without_background(
    smd=smd,
    mask_paths=mask_paths,
    prompts=prompts,
    negative_prompts=negative_prompts,
    mask_stds=2.0,  # mask 标准差
    mask_strengths=3.0,  # mask 强度
    bootstrap_steps=2,  # 引导步数
    bootstrap_leak_sensitivity=0.1,  # 引导泄漏敏感度
    guidance_scale=0,  # 引导比例
    style_name='(None)',  # 风格名称
    quality_name='(None)',  # 质量名称
    seed=0,  # 随机种子
    device=0,  # 设备编号
    num_inference_steps=3,
)
```

![百岁珊——将军-st3](https://github.com/user-attachments/assets/a354bd5e-f567-4982-93aa-3e54a05d3d71)

- Use Script
```bash
python draw_couple_action.py --output_dir "drink_water_couple_dir" --action "drink water" --random --num_combinations=20
```


![砂糖_重云_drink_water](https://github.com/user-attachments/assets/9f1004c9-3d64-4a9e-ac53-c5ba088aa2ce)


![五郎_安柏_drink_water](https://github.com/user-attachments/assets/51e3b660-36b4-4c28-8ac1-ee59c0fbc61a)

```bash
python draw_couple_action.py --num_inference_steps 3 --background_prompt "warm indoor setting during winter" --output_dir "prepare_gift_couple_dir" --action "prepare a gift for the date" --random --num_combinations 3
```

![胡桃_阿贝多_prepare_a_gift_for_the_date](https://github.com/user-attachments/assets/505402a0-d6e0-40dc-b129-be416de128b4)


![流浪者 散兵_妮露_prepare_a_gift_for_the_date](https://github.com/user-attachments/assets/c37bfa21-ca4b-4671-b2da-7a61fce737e7)


<div align="center">  


<h1>StreamMultiDiffusion: Real-Time Interactive Generation</br>with Region-Based Semantic Control</h1>
<h4>🔥🔥🔥 Now Supports Stable Diffusion 3 🔥🔥🔥</h4>

| ![mask](./assets/fantasy_large/fantasy_large_full.png) | ![result](./assets/fantasy_large_sd3_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input (1024x1024) | Generated Image with SD3 (**6.3 sec!**) |

[**Jaerin Lee**](http://jaerinlee.com/) · [**Daniel Sungho Jung**](https://dqj5182.github.io/) · [**Kanggeon Lee**](https://github.com/dlrkdrjs97/) · [**Kyoung Mu Lee**](https://cv.snu.ac.kr/index.php/~kmlee/)


<p align="center">
    <img src="assets/logo_cvlab.png" height=60>
</p>


[![Project](https://img.shields.io/badge/Project-Page-green)](https://jaerinlee.com/research/streammultidiffusion)
[![ArXiv](https://img.shields.io/badge/Arxiv-2403.09055-red)](https://arxiv.org/abs/2403.09055)
[![Github](https://img.shields.io/github/stars/ironjr/StreamMultiDiffusion)](https://github.com/ironjr/StreamMultiDiffusion)
[![X](https://img.shields.io/twitter/url?label=_ironjr_&url=https%3A%2F%2Ftwitter.com%2F_ironjr_)](https://twitter.com/_ironjr_)
[![LICENSE](https://img.shields.io/badge/license-MIT-lightgrey)](https://github.com/ironjr/StreamMultiDiffusion/blob/main/LICENSE)
[![HFPaper](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2403.09055)

[![HFDemoMain](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Main-yellow)](https://huggingface.co/spaces/ironjr/StreamMultiDiffusion)
[![HFDemo1](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SemanticPaletteSD1.5-yellow)](https://huggingface.co/spaces/ironjr/SemanticPalette)
[![HFDemo2](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SemanticPaletteSDXL-yellow)](https://huggingface.co/spaces/ironjr/SemanticPaletteXL)
[![HFDemo3](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SemanticPaletteSD3-yellow)](https://huggingface.co/spaces/ironjr/SemanticPalette3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb)

</div>

<p align="center">
  <img src="./assets/figure_one.png" width=100%>
</p>

tl;dr: StreamMultiDiffusion is a *real-time* *interactive* *multiple*-text-to-image generation from user-assigned *regional* text prompts.
In other words, **you can now draw ✍️ using brushes 🖌️ that paints *meanings* 🧠 in addition to *colors*** 🌈!

<details>
  
<summary>What's the paper about?</summary>
Our paper is mainly about establishing the compatibility between region-based controlling techniques of <a href="https://multidiffusion.github.io/">MultiDiffusion</a> and acceleration techniques of <a href="https://latent-consistency-models.github.io/">LCM</a> and <a href="https://github.com/cumulo-autumn/StreamDiffusion">StreamDiffusion</a>.
To our surprise, these works were not compatible before, limiting the possible applications from both branches of works.
The effect of acceleration and stabilization of multiple region-based text-to-image generation technique is demonstrated using <a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">StableDiffusion v1.5</a> in the video below ⬇️:

https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7

The video means that this project finally lets you work with **large size image generation with fine-grained regional prompt control**.
Previously, this was not feasible at all.
Taking an hour per trial means that you cannot sample multiple times to pick the best generation you want or to tune the generation process to realize your intention.
However, we have decreased the latency **from an hour to a minute**, making the technology workable for creators (hopefully).

</details>

---

- [⭐️ Features](#---features)
- [🚩 Updates](#---updates)
- [🤖 Installation](#---installation)
- [⚡ Usage](#---usage)
  * [Overview](#overview)
  * [Basic Usage (Python)](#basic-usage--python-)
  * [Streaming Generation Process](#streaming-generation-process)
  * [Region-Based Multi-Text-to-Image Generation](#region-based-multi-text-to-image-generation)
  * [Larger Region-Based Multi-Text-to-Image Generation](#larger-region-based-multi-text-to-image-generation)
  * [Image Inpainting with Prompt Separation](#image-inpainting-with-prompt-separation)
  * [Panorama Generation](#panorama-generation)
  * [Basic StableDiffusion](#basic-stablediffusion)
  * [Basic Usage (GUI)](#basic-usage--gui-)
  * [Demo Application (Semantic Palette)](#demo-application--semantic-palette-)
  * [Basic Usage (CLI)](#basic-usage--cli-)
- [💼 Further Information](#---further-information)
  * [User Interface (GUI)](#user-interface--gui-)
  * [Demo Application Architecture](#demo-application-architecture)
- [🙋 FAQ](#---faq)
  * [What is Semantic Palette Anyway?](#what-is--semantic-palette--anyway-)
- [🚨 Notice](#---notice)
- [🌏 Citation](#---citation)
- [🤗 Acknowledgement](#---acknowledgement)
- [📧 Contact](#---contact)

---

## ⭐️ Features


| ![usage1](./assets/feature1.gif) | ![usage2](./assets/feature3.gif) |  ![usage3](./assets/feature2.gif)  |
| :----------------------------: | :----------------------------: | :----------------------------: |

1. **Interactive image generation from scratch with fine-grained region control.** In other words, you paint images using meainings.

2. **Prompt separation.** Be bothered no more by unintentional content mixing when generating two or more objects at the same time!

3. **Real-time image inpainting and editing.** Basically, you draw upon any uploaded photo or a piece of art you want.

---

## 🚩 Updates (NEW!)

![demo_v2](./assets/demo_v2.gif)

- 🔥 June 24, 2024: We have launched our demo of Semantic Palette for vanilla **Stable Diffusion 3** in the Hugging Face 🤗 Space [here](https://huggingface.co/spaces/ironjr/SemanticPalette3)! If you want to run this in your local, we also provided code in this repository: see [here](https://github.com/ironjr/StreamMultiDiffusion/tree/main/demo/semantic_palette_sd3). Make sure to have enough VRAM!
- 🔥 June 22, 2024: We now support [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) powered by [Flash Diffusion](https://huggingface.co/jasperai/flash-sd3)! Installation guide is updated for SD3. See [notebooks](https://github.com/ironjr/StreamMultiDiffusion/tree/main/notebooks) directory for the newly updated Jupyter notebook demo.
- ✅ April 30, 2024: Real-time interactive generation demo is now published at [Hugging Face Space](https://huggingface.co/spaces/ironjr/StreamMultiDiffusion)!
- ✅ April 23, 2024: Real-time interactive generation demo is updated to [version 2](https://github.com/ironjr/StreamMultiDiffusion/tree/main/demo/stream_v2)! We now have fully responsive interface with `gradio.ImageEditor`. Huge thanks to [@pngwn](https://github.com/pngwn) and Hugging Face 🤗 Gradio team for the [great update (4.27)](https://www.gradio.app/changelog#4-27-0)!
- ✅ March 24, 2024: Our new demo app _Semantic Palette SDXL_ is out at [Hugging Face Space](https://huggingface.co/spaces/ironjr/SemanticPaletteXL)! Great thanks to [Cagliostro Research Lab](https://cagliostrolab.net/) for the permission of [Animagine XL 3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1) model used in the demo!
- ✅ March 24, 2024: We now (experimentally) support SDXL with [Lightning LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) in our semantic palette demo! Streaming type with SDXL-Lighning is under development.
- ✅ March 23, 2024: We now support `.safetensors` type models. Please see the instructions in Usage section.
- ✅ March 22, 2024: Our demo app _Semantic Palette_ is now available on [Google Colab](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb)! Huge thanks to [@camenduru](https://github.com/camenduru)!
- ✅ March 22, 2024: The app _Semantic Palette_ is now included in the repository! Run `python src/demo/semantic_palette/app.py --model "your model here"` to run the app from your local machine.
- ✅ March 19, 2024: Our first public demo of _semantic palette_ is out at [Hugging Face Space](https://huggingface.co/spaces/ironjr/SemanticPalette)! We would like to give our biggest thanks to the almighty Hugging Face 🤗 team for their help!
- ✅ March 16, 2024: Added examples and instructions for region-based generation, panorama generation, and inpainting.
- ✅ March 15, 2024: Added detailed instructions in this README for creators.
- ✅ March 14, 2024: We have released our paper, StreamMultiDiffusion on [arXiv](https://arxiv.org/abs/2403.09055).
- ✅ March 13, 2024: Code release!

---

## 🤖 Installation

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create -n smd python=3.10 && conda activate smd
pip install ipykernel
python -m ipykernel install --user --name smd --display-name "smd"

git clone https://github.com/svjack/semantic-draw && cd semantic-draw
pip install -r requirements.txt
pip install -U torch torchvision
pip install git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3
pip install -U transformers
```

### For SD3 (🔥NEW!!!)

We now support Stable Diffusion 3. To enable the feature, in addition to above installation code, enter the following code in your terminal.

```bash
pip install git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3
```

This will allow you to use [Flash Diffusion for SD3](https://huggingface.co/jasperai/flash-sd3). For using SD3 pipelines, please refer to newly updated Jupyter demos in the [notebooks](https://github.com/ironjr/StreamMultiDiffusion/tree/main/notebooks) directory.

## ⚡ Usage

### Overview

StreamMultiDiffusion is served in serveral different forms.

1. The main GUI demo powered by Gradio is available at `demo/stream_v2/app.py`. Just type the below line in your command prompt and open `https://localhost:8000` with any web browser will launch the app.

```bash
cd demo/stream_v2
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

2. The GUI demo _Semantic Palette_ for _SD1.5_ checkpoints is available at `demo/semantic_palette/app.py`. The public version can be found at [![HFDemo1](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SD1.5-yellow)](https://huggingface.co/spaces/ironjr/SemanticPalette) and at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/SemanticPalette-jupyter/blob/main/SemanticPalette_jupyter.ipynb).

```bash
cd demo/semantic_palette
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

3. The GUI demo _Semantic Palette_ for _SDXL_ checkpoints is available at `demo/semantic_palette_sdxl/app.py`. The public version can be found at [![HFDemo2](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-SDXL-yellow)](https://huggingface.co/spaces/ironjr/SemanticPaletteXL).

```bash
cd demo/semantic_palette_sdxl
python app.py --model "your stable diffusion 1.5 checkpoint" --height 512 --width 512 --port 8000
```

4. Jupyter Lab demos are available in the `notebooks` directory. Simply type `jupyter lab` in the command prompt will open a Jupyter server.

5. As a python library by importing the `model` in `src`. For detailed examples and interfaces, please see the Usage section below.


---

### Demo Application (StreamMultiDiffusion)

<p align="center">
  <img src="./assets/demo_v2.gif" width=90%>
</p>

#### Features

- Drawing with _semantic palette_ with streaming interface.
- Fully web-based GUI, powered by Gradio.
- Supports any Stable Diffusion v1.5 checkpoint with option `--model`.
- Supports any-sized canvas (if your VRAM permits!) with opetion `--height`, `--width`.
- Supports 8 semantic brushes.

#### Run

```bash
cd src/demo/stream_v2
python app.py [other options]
```

#### Run with `.safetensors`

We now support `.safetensors` type local models.
You can run the demo app with your favorite checkpoint models as follows:
1. Save `<your model>.safetensors` or a [symbolic link](https://mangohost.net/blog/what-is-the-linux-equivalent-to-symbolic-links-in-windows/) to the actual file to `demo/stream/checkpoints`.
2. Run the demo with your model loaded with `python app.py --model <your model>.safetensors`

Done!

#### Other options

- `--model`: Optional. The path to your custom SDv1.5 checkpoint. Both Hugging Face model repository / local safetensor types are supported. e.g., `--model "KBlueLeaf/kohaku-v2.1"` or `--model "realcartoonPixar_v6.safetensors"` Please note that safetensors models should reside in `src/demo/stream/checkpoints`!
- `--height` (`-H`): Optional. Height of the canvas. Default: 768.
- `--width` (`-W`): Optional. Width of the canvas. Default: 1920.
- `--display_col`: Optional. Number of displays in a row. Useful for buffering the old frames. Default: 2.
- `--display_row`: Optional. Number of displays in a column. Useful for buffering the old frames. Default: 2.
- `--bootstrap_steps`: Optional. The number of bootstrapping steps that separate each of the different semantic regions. Best when 1-3. Larger value means better separation, but less harmony within the image. Default: 1.
- `--seed`: Optional. The default seed of the application. Almost never needed since you can modify the seed value in GUI. Default: 2024.
- `--device`: Optional. The number of GPU card (probably 0-7) you want to run the model. Only for multi-GPU servers. Default: 0.
- `--port`: Optional. The front-end port of the application. If the port is 8000, you can access your runtime through `https://localhost:8000` from any web browser. Default: 8000.


#### Instructions

| ![usage1](./assets/instruction1.png) | ![usage2](./assets/instruction2.png) |
| :----------------------------: | :----------------------------: |
| Upoad a background image | Type some text prompts |
| ![usage3](./assets/instruction3.png) | ![usage4](./assets/instruction4.png) |
| Draw | Press the play button and enjoy 🤩 |

1. (top-left) **Upload a background image.** You can start with a white background image, as well as any other images from your phone camera or other AI-generated artworks. You can also entirely cover the image editor with specific semantic brush to draw background image simultaneously from the text prompt.

2. (top-right) **Type some text prompts.** Click each semantic brush on the semantic palette on the left of the screen and type in text prompts in the interface below. This will create a new semantic brush for you.

3. (bottom-left) **Draw.** Select appropriate layer (*important*) that matches the order of the semantic palette. That is, ***Layer n*** corresponds to ***Prompt n***. I am not perfectly satisfied with the interface of the drawing interface. Importing professional Javascript-based online drawing tools instead of the default `gr.ImageEditor` will enable more responsive interface. We have released our code with MIT License, so please feel free to fork this repo and build a better user interface upon it. 😁

4. (bottom-right) **Press the play button and enjoy!** The buttons literally mean 'toggle stream/run single/run batch (4)'.

---

### Demo Application (Semantic Palette)

<div>

<p align="center">
  <img src="./assets/demo_semantic_draw_large.gif" width=90%>
</p>

</div>

Our first demo _[Semantic Palette](https://huggingface.co/spaces/ironjr/SemanticPalette)_ is now available in your local machine.

#### Features

- Fully web-based GUI, powered by Gradio.
- Supports any Stable Diffusion v1.5 checkpoint with option `--model`.
- Supports any-sized canvas (if your VRAM permits!) with opetion `--height`, `--width`.
- Supports 5 semantic brushes. If you want more brushes, you can use our python interface directly. Please see our Jupyter notebook references in the `notebooks` directory.

#### Run

```bash
cd src/demo/semantic_palette
python app.py [other options]
```

#### Run with `.safetensors`

We now support `.safetensors` type local models.
You can run the demo app with your favorite checkpoint models as follows:
1. Save `<your model>.safetensors` or a [symbolic link](https://mangohost.net/blog/what-is-the-linux-equivalent-to-symbolic-links-in-windows/) to the actual file to `demo/semantic_palette/checkpoints`.
2. Run the demo with your model loaded with `python app.py --model <your model>.safetensors`

Done!

#### Other options

- `--model`: Optional. The path to your custom SDv1.5 checkpoint. Both Hugging Face model repository / local safetensor types are supported. e.g., `--model "KBlueLeaf/kohaku-v2.1"` or `--model "realcartoonPixar_v6.safetensors"` Please note that safetensors models should reside in `src/demo/semantic_palette/checkpoints`!
- `--height` (`-H`): Optional. Height of the canvas. Default: 768.
- `--width` (`-W`): Optional. Width of the canvas. Default: 1920.
- `--bootstrap_steps`: Optional. The number of bootstrapping steps that separate each of the different semantic regions. Best when 1-3. Larger value means better separation, but less harmony within the image. Default: 1.
- `--seed`: Optional. The default seed of the application. Almost never needed since you can modify the seed value in GUI. Default: -1 (random).
- `--device`: Optional. The number of GPU card (probably 0-7) you want to run the model. Only for multi-GPU servers. Default: 0.
- `--port`: Optional. The front-end port of the application. If the port is 8000, you can access your runtime through `https://localhost:8000` from any web browser. Default: 8000.

#### Instructions

Instructions on how to use the app to create your images: Please see this twitter [thread](https://twitter.com/_ironjr_/status/1770245714591064066).

#### Tips

I have provided more tips in using the app in another twitter [thread](https://twitter.com/_ironjr_/status/1770716860948025539).

---

### Basic Usage (Python)

The main python modules in our project is two-fold: (1) [`model.StableMultiDiffusionPipeline`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/stablemultidiffusion_pipeline.py) for single-call generation (might be more preferable for CLI users), and (2) [`model.StreamMultiDiffusion`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/streammultidiffusion.py) for streaming application such as the [one](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/app.py) in the main figure of this README page.
We provide minimal examples for the possible applications below.


### Streaming Generation Process

With [multi-prompt stream batch](https://arxiv.org/abs/2403.09055), our modification to the [original stream batch architecture](https://github.com/cumulo-autumn/StreamDiffusion) by [@cumulo_autumn](https://twitter.com/cumulo_autumn), we can stream this multi-prompt text-to-image generation process to generate images for ever.

**Result:**

| ![mask](./assets/zeus/prompt.png) | ![result](./assets/athena_stream.gif) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Stream |

**Code:**

```python
import torch
from util import seed_everything, Streamer
from model import StreamMultiDiffusion

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
import time
import imageio # This is not included in our requirements.txt!
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0
height = 768
width = 512

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StreamMultiDiffusion(
    device,
    hf_key='ironjr/BlazingDriveV11m',
    height=height,
    width=width,
    cfg_type='none',
    autoflush=True,
    use_tiny_vae=True,
    mask_type='continuous',
    bootstrap_steps=2,
    bootstrap_mix_steps=1.5,
    seed=seed,
)

# Load the masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/zeus/prompt_p{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])

# Register a background, prompts, and masks (this can be called multiple times).
smd.update_background(Image.new(size=(width, height), mode='RGB', color=(255, 255, 255)))
smd.update_single_layer(
    idx=0,
    prompt='a photo of Mount Olympus',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=background,
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)
smd.update_single_layer(
    idx=1,
    prompt='1girl, looking at viewer, lifts arm, smile, happy, Greek goddess Athena',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=masks[0],
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)
smd.update_single_layer(
    idx=2,
    prompt='a small, sitting owl',
    negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
    mask=masks[1],
    mask_strength=1.0,
    mask_std=0.0,
    prompt_strength=1.0,
)


# Generate images... forever.
# while True:
#     image = smd()
#     image.save(f'{str(int(time.time() % 100000))}.png') # This will take up your hard drive pretty much soon.
#     display(image) # If `from IPython.display import display` is called.
#
#     You can also intercept the process in the middle of the generation by updating other background, prompts or masks.
#     smd.update_single_layer(
#         idx=2,
#         prompt='a small, sitting owl',
#         negative_prompt='worst quality, bad quality, normal quality, cropped, framed',
#         mask=masks[1],
#         mask_strength=1.0,
#         mask_std=0.0,
#         prompt_strength=1.0,
#     )

# Or make a video/gif from your generation stream (requires `imageio`)
frames = []
for _ in range(50):
    image = smd()
    frames.append(image)
imageio.mimsave('my_beautiful_creation.gif', frames, loop=0)
```

---

### Region-Based Multi-Text-to-Image Generation

We support arbitrary-sized image generation from arbitrary number of prompt-mask pairs.
The first example is a simple example of generation 
Notice that **our generation results also obeys strict prompt separation**.


**Result:**

| ![mask](./assets/timessquare/timessquare_full.png) | ![result](./assets/timessquare_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Image (10 sec) |

<p align="center">
    No more unwanted prompt mixing! Brown boy and pink girl generated simultaneously without a problem.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(
    device,
    hf_key='ironjr/BlazingDriveV11m',
)

# Load prompts.
prompts = [
    # Background prompt.
    '1girl, 1boy, times square',
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]
negative_prompts = [
    '',
    '1girl', # (Optional) The first prompt is a boy so we don't want a girl.
    '1boy', # (Optional) The first prompt is a girl so we don't want a boy.
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (768, 768) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
)
image.save('my_beautiful_creation.png')
```

---

### (🔥NEW!!!) Region-Based Multi-Text-to-Image Generation with Stable Diffusion 3

We support arbitrary-sized image generation from arbitrary number of prompt-mask pairs using custom SDXL models.
This is powered by [SDXL-Lightning LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) and our stabilization trick for MultiDiffusion in conjunction with Lightning-type sampling algorithm.

**Result:**

| ![mask](./assets/fantasy_large/fantasy_large_full.png) | ![result](./assets/fantasy_large_sd3_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Image (**6.3 sec!**) |

<p align="center">
    1024x1024 image generated with <a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3</a> accelerated by <a href="https://huggingface.co/jasperai/flash-sd3">Flash Diffusion</a>.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusion3Pipeline
from util import seed_everything
from prompt_util import print_prompts, preprocess_prompts

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 1
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
    has_i2t=False,
)

# Load prompts.
prompts = [
    # Background prompt.
    'blue sky with large words "Stream" on it',
    # Foreground prompts.
    'a photo of the dolomites, masterpiece, absurd quality, background, no humans',
    'a photo of Gandalf the Gray staring at the viewer',
]
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Preprocess prompts for better results.
prompts, negative_prompts = preprocess_prompts(
    prompts,
    negative_prompts,
    style_name='(None)',
    quality_name='Standard v3.1',
)

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/fantasy_large/fantasy_large_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (1024, 1024) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
    guidance_scale=0,
)
image.save('my_beautiful_creation.png')
```

---

### Region-Based Multi-Text-to-Image Generation with Custom SDXL

We support arbitrary-sized image generation from arbitrary number of prompt-mask pairs using custom SDXL models.
This is powered by [SDXL-Lightning LoRA](https://huggingface.co/ByteDance/SDXL-Lightning) and our stabilization trick for MultiDiffusion in conjunction with Lightning-type sampling algorithm.

#### Known Issue:

SDXL-Lightning support is currently experimental, so there can be additional issues I have not yet noticed.
Please open an issue or a pull request if you find any.
These are the currently known SDXL-Lightning-specific issues compared to SD1.5 models.
- The model tends to be less obedient to the text prompts. SDXL-Lightning-specific prompt engineering may be required. The problem is less severe in custom models, such as [this](https://huggingface.co/cagliostrolab/animagine-xl-3.1).
- The vanilla SDXL-Lightning model produces NaNs when used as a FP16 variant. Please use `dtype=torch.float32` option for initializing `StableMultiDiffusionSDXLPipeline` if you want the _vanilla_ version of the SDXL-Lightning. This is _not_ a problem when using a custom checkpoint. You can use `dtype=torch.float16`.

**Result:**

| ![mask](./assets/fantasy_large/fantasy_large_full.png) | ![result](./assets/fantasy_large_generation.png) |
| :----------------------------: | :----------------------------: |
| Semantic Brush Input | Generated Image (12 sec) |

<p align="center">
    1024x1024 image generated with <a href="https://huggingface.co/ByteDance/SDXL-Lightning">SDXL-Lightning LoRA</a> and <a href="https://huggingface.co/cagliostrolab/animagine-xl-3.1">Animagine XL 3.1</a> checkpoint.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionSDXLPipeline
from util import seed_everything
from prompt_util import print_prompts, preprocess_prompts

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 0
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionSDXLPipeline(
    device,
    hf_key='cagliostrolab/animagine-xl-3.1',
    has_i2t=False,
)

# Load prompts.
prompts = [
    # Background prompt.
    'purple sky, planets, planets, planets, stars, stars, stars',
    # Foreground prompts.
    'a photo of the dolomites, masterpiece, absurd quality, background, no humans',
    '1girl, looking at viewer, pretty face, blue hair, fantasy style, witch, magi, robe',
]
negative_prompts = [
    '1girl, 1boy, humans, humans, humans',
    '1girl, 1boy, humans, humans, humans',
    '',
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

# Preprocess prompts for better results.
prompts, negative_prompts = preprocess_prompts(
    prompts,
    negative_prompts,
    style_name='(None)',
    quality_name='Standard v3.1',
)

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/fantasy_large/fantasy_large_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (1024, 1024) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
    guidance_scale=0,
)
image.save('my_beautiful_creation.png')
```

---

### *Larger* Region-Based Multi-Text-to-Image Generation

The below code reproduces the results in the [second video](https://github.com/ironjr/MagicDraw/assets/12259041/9dda9740-58ba-4a96-b8c1-d40765979bd7) of this README page.
The original MultiDiffusion pipeline using 50 step DDIM sampler takes roughly an hour to run the code, but we have reduced in down to **a minute**.

**Result:**

| ![mask](./assets/irworobongdo/irworobongdo_full.png) |
| :----------------------------: |
| Semantic Brush Input |
|  ![result](./assets/irworobongdo_generation.png) |
| Generated Image (59 sec) |

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from functools import reduce
from io import BytesIO
from PIL import Image


seed = 2024
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Load prompts.
prompts = [
    # Background prompt.
    'clear deep blue sky',
    # Foreground prompts.
    'summer mountains',
    'the sun',
    'the moon',
    'a giant waterfall',
    'a giant waterfall',
    'clean deep blue lake',
    'a large tree',
    'a large tree',
]
negative_prompts = ['worst quality, bad quality, normal quality, cropped, framed'] * len(prompts)

# Load masks.
masks = []
for i in range(1, 9):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/irworobongdo/irworobongdo_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
# In this example, background is simply set as non-marked regions.
background = reduce(torch.logical_and, [m == 0 for m in masks])
masks = torch.stack([background] + masks, dim=0).float()

height, width = masks.shape[-2:] # (768, 1920) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    mask_stds=0,
    height=height,
    width=width,
    bootstrap_steps=2,
)
image.save('my_beautiful_creation.png')
```

---

### Image Inpainting with Prompt Separation

Our pipeline also enables editing and inpainting existing images.
We also support *any* SD 1.5 checkpoint models.
One exceptional advantage of ours is that we provide an easy separation of prompt
You can additionally trade-off between prompt separation and overall harmonization by changing the argument `bootstrap_steps` from 0 (full mixing) to 5 (full separation).
We recommend `1-3`.
The following code is a minimal example of performing prompt separated multi-prompt image inpainting using our pipeline on a custom model.

**Result:**

| ![mask](./assets/timessquare/timessquare.jpeg) | ![mask](./assets/timessquare/timessquare_full.png) | ![result](./assets/timessquare_inpainting.png) |
| :----------------------------: | :----------------------------: | :----------------------------: |
| Images to Inpaint | Semantic Brush Input | Inpainted Image (9 sec) |

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline
from util import seed_everything

# The following packages are imported only for loading the images.
import torchvision.transforms as T
import requests
from io import BytesIO
from PIL import Image


seed = 2
device = 0

# Load the module.
seed_everything(seed)
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(
    device,
    hf_key='ironjr/BlazingDriveV11m',
)

# Load the background image you want to start drawing.
#   Although it works for any image, we recommend to use background that is generated
#   or at least modified by the same checkpoint model (e.g., preparing it by passing
#   it to the same checkpoint for an image-to-image pipeline with denoising_strength 0.2)
#   for the maximally harmonized results!
#   However, in this example, we choose to use a real-world image for the demo.
url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare.jpeg'
response = requests.get(url)
background_image = Image.open(BytesIO(response.content)).convert('RGB')

# Load prompts and background prompts (explicitly).
background_prompt = '1girl, 1boy, times square'
prompts = [
    # Foreground prompts.
    '1boy, looking at viewer, brown hair, casual shirt',
    '1girl, looking at viewer, pink hair, leather jacket',
]
negative_prompts = [
    '1girl',
    '1boy',
]
negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]
background_negative_prompt = negative_prompt_prefix

# Load masks.
masks = []
for i in range(1, 3):
    url = f'https://raw.githubusercontent.com/ironjr/StreamMultiDiffusion/main/assets/timessquare/timessquare_{i}.png'
    response = requests.get(url)
    mask = Image.open(BytesIO(response.content)).convert('RGBA')
    mask = (T.ToTensor()(mask)[-1:] > 0.5).float()
    masks.append(mask)
masks = torch.stack(masks, dim=0).float()
height, width = masks.shape[-2:] # (768, 768) in this example.

# Sample an image.
image = smd(
    prompts,
    negative_prompts,
    masks=masks,
    mask_strengths=1,
    # Use larger standard deviation to harmonize the inpainting result (Recommended: 8-32)!
    mask_stds=16.0,
    height=height,
    width=width,
    bootstrap_steps=2,
    bootstrap_leak_sensitivity=0.1,
    # This is for providing the image input.
    background=background_image,
    background_prompt=background_prompt,
    background_negative_prompt=background_negative_prompt,
)
image.save('my_beautiful_inpainting.png')
```

---

### Panorama Generation

Our [`model.StableMultiDiffusionPipeline`](https://github.com/ironjr/StreamMultiDiffusion/blob/main/src/model/stablemultidiffusion_pipeline.py) supports x10 faster generation of irregularly large size images such as panoramas.
For example, the following code runs in 10s with a single 2080 Ti GPU.

**Result:**

<p align="center">
  <img src="./assets/panorama_generation.png" width=100%>
</p>
<p align="center">
    512x3072 image generated in 10 seconds.
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline

device = 0

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Sample a panorama image.
smd.sample_panorama('A photo of Alps', height=512, width=3072)
image.save('my_panorama_creation.png')
```

---

### Basic StableDiffusion

We also support standard single-prompt single-tile sampling of StableDiffusion checkpoint for completeness.
This behaves exactly the same as calling [`diffuser`](https://huggingface.co/docs/diffusers/en/index)'s [`StableDiffusionPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py).

**Result:**

<p align="left">
  <img src="./assets/dolomites_generation.png" width=50%>
</p>

**Code:**

```python
import torch
from model import StableMultiDiffusionPipeline

device = 0

# Load the module.
device = torch.device(f'cuda:{device}')
smd = StableMultiDiffusionPipeline(device)

# Sample an image.
image = smd.sample('A photo of the dolomites')
image.save('my_creation.png')
```


---


## 💼 Further Information

We have provided detailed explanation of the application design and the expected usages in appendices of our [paper](https://arxiv.org/abs/2403.09055).
This section is a summary of its contents.
Although we expect everything to work fine, there may be unexpected bugs or missed features in the implementation.
We are always welcoming issues and pull requests from you to improve this project! 🤗


### User Interface (GUI)

<p align="center">
  <img src="./assets/user_interface.png" width=90%>
</p>

| No. | Component Name | Description |
| --- | -------------- | ----------- |
| 1 | *Semantic palette* | Creates and manages text prompt-mask pairs, a.k.a., _semantic brushes_. |
| 2 | Create new _semantic brush_ btn. | Creates a new text prompt-mask pair. |
| 3 | Main drawing pad | User draws at each semantic layers with a brush tool. |
| 4 | Layer selection | Each layer corresponds to each of the prompt mask in the *semantic palette*. |
| 5 | Background image upload | User uploads background image to start drawing. |
| 6 | Drawing tools | Using brushes and erasers to interactively edit the prompt masks. |
| 7 | Play button | Switches between streaming/step-by-step mode. |
| 8 | Display | Generated images are streamed through this component. |
| 9 | Mask alpha control | Changes the mask alpha value before quantization. Controls local content blending (simply means that you can use nonbinary masks for fine-grained controls), but extremely sensitive. Recommended: >0.95 |
| 10 | Mask blur std. dev. control | Changes the standard deviation of the quantized mask of the current semantic brush. Less sensitive than mask alpha control. |
| 11 | Seed control | Changes the seed of the application. May not be needed, since we generate infinite stream of images. |
| 12 | Prompt edit | User can interactively change the positive/negative prompts at need. |
| 13 | Prompt strength control | Prompt embedding mix ratio between the current & the background. Helps global content blending. Recommended: >0.75 |
| 14 | Brush name edit | Adds convenience by changing the name of the brush. Does not affect the generation. Just for preference. |

### Demo Application Architecture

There are two types of transaction data between the front-end and the back-end (`model.streammultidiffusion_pipeline.StreamMultiDiffusion`) of the application: a (1) background image object and a (2) list of text prompt-mask pairs.
We choose to call a pair of the latter as a _semantic brush_.
Despite its fancy name, a _semantic brush_ is just a pair of a text prompt and a regional mask assigned to the prompt, possibly with additional mask-controlling parameters.
Users interact with the application by registering and updating these two types of data to control the image generation stream.
The interface is summarized in the image below ⬇️:

<p align="center">
  <img src="./assets/app_design.png" width=90%>
</p>


---

## 🙋 FAQ

### What is _Semantic Palette_ Anyway?

**Semantic palette** basically means that you paint things with semantics, i.e., text prompts, just like how you may use brush tools in commercial image editing software, such as Adobe Photoshop, etc.
Our acceleration technique for region-based controlled image generation allows users to edit their prompt masks similarly to drawing.
We couldn't find a good preexisting name for this type of user interface, so we named it as _semantic palette_, hoping for it to make sense to you. 😄

### Can it run realistic models / anime-style models?

Of course. Both types of models are supported.
For realistic models and SDXL-type models, using `--bootstrap_steps=2` or `3` produces better (non-cropped) images.

https://github.com/ironjr/StreamMultiDiffusion/assets/12259041/9a6bb02b-7dca-4dd0-a1bc-6153dde1571d


## 🌏 Citation

Please cite us if you find our project useful!

```latex
@article{lee2024streammultidiffusion,
    title={{StreamMultiDiffusion:} Real-Time Interactive Generation with Region-Based Semantic Control},
    author={Lee, Jaerin and Jung, Daniel Sungho and Lee, Kanggeon and Lee, Kyoung Mu},
    journal={arXiv preprint arXiv:2403.09055},
    year={2024}
}
```

## 🚨 Notice

Please note that we do not host separate pages for this project other than our [official project page](https://jaerinlee.com/research/streammultidiffusion) and [hugging face](https://huggingface.co/spaces/ironjr/SemanticPalette) [space demos](https://huggingface.co/spaces/ironjr/SemanticPaletteXL).
For example [https://streammultidiffusion.net](https://streammultidiffusion.net/) is not related to us!
(By the way thanks for hosting)

We do welcome anyone who wants to use our framework/code/app for any personal or commercial purpose (we have opened the code here for free with MIT License).
However, **we'd be much happier if you cite us in any format in your application**.
We are very open to discussion, so if you find any issue with this project, including commercialization of the project, please contact [us](https://twitter.com/_ironjr_) or post an issue.


## 🤗 Acknowledgement

Our code is based on the projects: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), [MultiDiffusion](https://multidiffusion.github.io/), and [Latent Consistency Model](https://latent-consistency-models.github.io/). Thank you for sharing such amazing works!
We also give our huge thanks to [@br_d](https://twitter.com/br_d) and [@KBlueleaf](https://twitter.com/KBlueleaf) for the wonderful models [BlazingDriveV11m](https://civitai.com/models/121083?modelVersionId=236210) and [Kohaku V2](https://civitai.com/models/136268/kohaku-v2)!


## 📧 Contact

If you have any questions, please email `jarin.lee@gmail.com`.
