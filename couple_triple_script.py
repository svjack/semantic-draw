'''
python couple_triple_script.py --output_dir c5t7_dir --num_couple 5 --num_triple 7
'''

import torch
import re
import os
import random
import argparse
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
from datasets import load_dataset

# 初始化 Stable Diffusion XL Pipeline
def initialize_pipeline():
    return StableDiffusionXLPipeline.from_pretrained(
        "cagliostrolab/animagine-xl-3.1",
        torch_dtype=torch.float16
    ).to("cuda")

# 角色字典
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

# 加载 dating_df 数据集
def load_dating_data():
    dating_df = load_dataset("svjack/dating-actions-en-zh")["train"].to_pandas()
    dating_df["en_action"] = dating_df["en_action"].map(lambda x: x.replace("your", "").replace("  ", " ")).values.tolist()
    dating_df["en_location"] = dating_df["en_location"].map(lambda x: x.replace("your", "").replace("  ", " ")).values.tolist()
    location_prepositions = {
        'home': 'at', 'kitchen': 'in', 'park': 'in', 'garage': 'in', 'cafe': 'at',
        'restaurant': 'at', 'restroom': 'in', 'tea house': 'at', 'supermarket': 'at'
    }
    dating_df["en_location"] = dating_df["en_location"].map(
        lambda x: "{} {}".format(location_prepositions.get(x.split()[-1], "at"), x).strip()
    ).values.tolist()
    return dating_df

# 生成合法的文件名
def sanitize_filename(prompt):
    sanitized = re.sub(r'[^\w\s-]', '', prompt)
    sanitized = sanitized.replace(' ', '_')
    return sanitized[:50]

# 生成并保存图片
def generate_and_save_image(pipeline, prompt, negative_prompt, seed, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.manual_seed(seed),
    ).images[0]
    filename = sanitize_filename(prompt) + f"_seed_{seed}.png"
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)
    print(f"Generated and saved: {save_path}")

# 生成 COUPLE with location 类型的 prompt
def generate_couple_with_location_prompt(location, character1, character2, action):
    return f"{location} ,COUPLE ,{character1}, {character2} \(genshin impact\) highres, masterpiece, {action}"

# 生成 TRIPLE with location 类型的 prompt
def generate_triple_with_location_prompt(location, character1, character2, character3, action):
    return f"{location} ,TRIPLE ,{character1}, {character2}, {character3} \(genshin impact\) highres, masterpiece, {action}"

# 随机生成 prompt
def generate_random_prompts(num_couple_with_location, num_triple_with_location, characters, actions, locations):
    prompts = []
    for _ in range(num_couple_with_location):
        character1, character2 = random.sample(characters, 2)
        action = random.choice(actions)
        location = random.choice(locations)
        prompt = generate_couple_with_location_prompt(location, character1, character2, action)
        prompts.append({
            "prompt": prompt,
            "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
            "seed": random.randint(0, 1000)
        })
    for _ in range(num_triple_with_location):
        character1, character2, character3 = random.sample(characters, 3)
        action = random.choice(actions)
        location = random.choice(locations)
        prompt = generate_triple_with_location_prompt(location, character1, character2, character3, action)
        prompts.append({
            "prompt": prompt,
            "negative_prompt": "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],",
            "seed": random.randint(0, 1000)
        })
    return prompts

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Generate images with COUPLE and TRIPLE prompts.")
    parser.add_argument("--output_dir", type=str, default="output_images", help="Directory to save generated images.")
    parser.add_argument("--num_couple", type=int, default=3, help="Number of COUPLE with location prompts to generate.")
    parser.add_argument("--num_triple", type=int, default=2, help="Number of TRIPLE with location prompts to generate.")
    args = parser.parse_args()

    # 初始化
    pipeline = initialize_pipeline()
    dating_df = load_dating_data()
    characters = list(new_dict.values())
    actions = dating_df["en_action"].drop_duplicates().tolist()
    locations = dating_df["en_location"].drop_duplicates().tolist()

    # 生成 prompt
    random_prompts = generate_random_prompts(args.num_couple, args.num_triple, characters, actions, locations)

    # 打乱 random_prompts 的顺序
    random.shuffle(random_prompts)

    # 使用 tqdm 显示进度条
    for call in tqdm(random_prompts, desc="Generating images"):
        generate_and_save_image(pipeline, **call, save_dir=args.output_dir)

if __name__ == "__main__":
    main()