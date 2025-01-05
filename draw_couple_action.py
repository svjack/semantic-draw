import os
import sys
import time
import warnings
import argparse
from tqdm import tqdm
import torch
import torchvision.transforms as T
from PIL import Image
from diffusers.utils import make_image_grid
from functools import reduce
import random

# 添加项目路径
sys.path.append('src')
warnings.filterwarnings('ignore')

# 导入自定义模块
from util import seed_everything, blend
from model import StableMultiDiffusionSDXLPipeline
from ipython_util import dispt
from prompt_util import print_prompts, preprocess_prompts

# 定义 new_dict
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

# 定义 generate_image_without_background 函数
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
    num_inference_steps=5
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
        num_inference_steps: 推理步数，控制生成图像的精细程度。

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
    #negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
    #negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

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
        num_inference_steps=num_inference_steps
    )
    toc = time.time()
    print(f'Elapsed Time: {toc - tic}')
    #display(img)
    return img

def main(args):
    # 创建保存图片的目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 smd（只执行一次）
    seed_everything(args.seed)
    device = f'cuda:{args.device}'
    smd = StableMultiDiffusionSDXLPipeline(
        device,
        hf_key='cagliostrolab/animagine-xl-3.1',
        has_i2t=False,
    )

    # 将 action 转换为文件名友好的格式（替换空格为下划线）
    action_safe = args.action.replace(" ", "_")

    # 获取所有角色名
    characters = list(new_dict.keys())

    # 如果启用随机组合
    if args.random:
        # 随机生成组合
        random_combinations = []
        for _ in range(args.num_combinations):  # 生成指定数量的随机组合
            name1, name2 = random.sample(characters, 2)  # 随机选择两个不同的角色
            random_combinations.append((name1, name2))
        combinations = random_combinations
    else:
        # 生成所有可能的组合
        combinations = [(k1, k2) for k1 in characters for k2 in characters if k1 != k2]

    # 定义 negative_prompts
    negative_prompts = [
        '',
        '',
        '',
        #'Multiple People',
    ]

    # 添加负面提示词前缀
    negative_prompt_prefix = 'worst quality, bad quality, normal quality, cropped, framed'
    negative_prompts = [negative_prompt_prefix + ', ' + p for p in negative_prompts]

    # 遍历组合
    for name1, name2 in tqdm(combinations, desc="Generating images"):
        # 替换 prompts 中的角色和动作
        prompts = [
            args.background_prompt,  # 使用命令行参数中的背景提示词
            f'solo, {new_dict[name1]}, \(genshin impact\) highres, masterpiece, {args.action}',
            f'solo, {new_dict[name2]}, \(genshin impact\) highres, masterpiece, {args.action}',
        ]

        # 生成图片
        img = generate_image_without_background(
            smd=smd,
            mask_paths=args.mask_paths,  # 使用命令行参数中的 mask 路径
            prompts=prompts,
            negative_prompts=negative_prompts,
            mask_stds=args.mask_stds,  # mask 标准差
            mask_strengths=args.mask_strengths,  # mask 强度
            bootstrap_steps=args.bootstrap_steps,  # 引导步数
            bootstrap_leak_sensitivity=args.bootstrap_leak_sensitivity,  # 引导泄漏敏感度
            guidance_scale=args.guidance_scale,  # 引导比例
            style_name=args.style_name,  # 风格名称
            quality_name=args.quality_name,  # 质量名称
            seed=args.seed,  # 随机种子
            device=args.device,  # 设备编号
            num_inference_steps=args.num_inference_steps,
        )

        # 保存图片，文件名包含 action
        img_path = os.path.join(args.output_dir, f'{name1}_{name2}_{action_safe}.png')
        img.save(img_path)

    print(f"All images have been saved to {args.output_dir}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Generate images with different character combinations.")
    parser.add_argument('--output_dir', type=str, default='generated_images', help='Directory to save generated images.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for image generation.')
    parser.add_argument('--device', type=int, default=0, help='Device ID for GPU.')
    parser.add_argument('--mask_stds', type=float, default=2.0, help='Mask standard deviation.')
    parser.add_argument('--mask_strengths', type=float, default=3.0, help='Mask strength.')
    parser.add_argument('--bootstrap_steps', type=int, default=2, help='Bootstrap steps.')
    parser.add_argument('--bootstrap_leak_sensitivity', type=float, default=0.1, help='Bootstrap leak sensitivity.')
    parser.add_argument('--guidance_scale', type=float, default=0, help='Guidance scale.')
    parser.add_argument('--style_name', type=str, default='(None)', help='Style name for prompts.')
    parser.add_argument('--quality_name', type=str, default='(None)', help='Quality name for prompts.')
    parser.add_argument('--num_inference_steps', type=int, default=3, help='Number of inference steps.')
    parser.add_argument('--action', type=str, default='drink water', help='Action to be performed by characters.')
    parser.add_argument('--random', action='store_true', help='Enable random combinations.')
    parser.add_argument('--num_combinations', type=int, default=10, help='Number of random combinations to generate.')
    parser.add_argument('--background_prompt', type=str, default='dimly lit bar, neon lights', help='Background prompt for the image.')
    parser.add_argument('--mask_paths', type=str, nargs=3, default=['blue_yellow_m.webp', 'blue_left_m.webp', 'yellow_right_m.webp'], help='Paths to the mask images.')
    args = parser.parse_args()

    # 运行主函数
    main(args)
