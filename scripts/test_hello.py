"""测试模型是否接通"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.model import get_llm

def main():
    llm = get_llm()
    response = llm.invoke("你好，请用一句话介绍你自己。")
    print("模型回复：", response.content)

if __name__ == "__main__":
    main()
