from modelscope import snapshot_download

print("开始下载 DeepSeek 7B (工作模式)...")
snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
print("DeepSeek 7B 下载完成！")

print("开始下载 Qwen 1.5B (游戏模式)...")
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')
print("Qwen 1.5B 下载完成！")