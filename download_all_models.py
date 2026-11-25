import os
from modelscope import snapshot_download

from model_cache import MODEL_CACHE_DIR

# 🚨 锁定你指定的缓存路径
CACHE_DIR = str(MODEL_CACHE_DIR)
os.makedirs(CACHE_DIR, exist_ok=True)

# 你的终极模型清单
models_to_download = [
    # 1. 陪伴/通用聊天 (约 15GB)
    "Qwen/Qwen2.5-7B-Instruct",
    
    # 2. 工作模式 (约 15GB)
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    
    # 3. 代码模式 (约 6GB)
    "Qwen/Qwen2.5-Coder-3B-Instruct",
    
    # 4. 小模型模式 (约 3GB)
    "Qwen/Qwen2.5-1.5B-Instruct",
    
    # 5. 图片、视频和手写识别 (约 7GB)
    "Qwen/Qwen2.5-VL-3B-Instruct",
    
    # 6. 本地绘图 (SD-Turbo，魔搭上的镜像ID)
    "AI-ModelScope/sd-turbo" 
]

print(f"📦 准备将模型下载到: {CACHE_DIR}")
print(f"🔢 总计 {len(models_to_download)} 个模型，预计总大小约 45GB+，请耐心等待...\n")

for model_id in models_to_download:
    print(f"==================================================")
    print(f"⬇️ 开始下载: {model_id}")
    print(f"==================================================")
    try:
        # 这里的 cache_dir 参数是核心，它会把模型死死锁在你的 D 盘
        model_path = snapshot_download(model_id, cache_dir=CACHE_DIR)
        print(f"✅ 下载完成！保存在: {model_path}\n")
    except Exception as e:
        print(f"❌ 下载 {model_id} 失败，报错信息: {e}\n")
        print("⚠️ 建议：检查网络后，重新运行此脚本，它会自动断点续传。\n")

print("🎉 所有模型下载任务处理完毕！")