"""下载嵌入模型到本地"""
import os
# 使用 hf-mirror 镜像加速（国内访问）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 下载 bge-base-zh-v1.5 模型
print("📥 开始下载 bge-base-zh-v1.5...")
snapshot_download(
    repo_id="BAAI/bge-base-zh-v1.5",
    local_dir="models/bge-base-zh-v1.5",
    local_dir_use_symlinks=False,
)
print("✅ 模型下载完成！位置：models/bge-base-zh-v1.5")
