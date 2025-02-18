from openai import OpenAI
import json

# API配置
API_KEY = ""
BASE_URL = "https://openrouter.ai/api/v1"


def print_model_info(model_info):
    """格式化打印模型信息"""
    print(f"{'='*50}")
    print(f"模型ID: {model_info.id}")
    print(f"创建时间: {model_info.created}")
    print(f"所属组织: {model_info.owned_by}")
    if hasattr(model_info, 'permission'):
        print("\n权限信息:")
        for perm in model_info.permission:
            print(f"  - 允许创建: {perm.allow_create_engine}")
            print(f"  - 允许采样: {perm.allow_sampling}")
            print(f"  - 允许日志概率: {perm.allow_logprobs}")
            print(f"  - 允许搜索索引: {perm.allow_search_indices}")
            print(f"  - 允许视图: {perm.allow_view}")
    print(f"{'='*50}")

def main():
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        # 获取可用模型列表
        print("正在获取可用模型列表...\n")
        models = client.models.list()

        # 打印模型总数
        print(f"找到 {len(models.data)} 个可用模型")

        # 详细打印每个模型的信息
        for model in models.data:
            print_model_info(model)

    except Exception as e:
        print(f"获取模型信息时发生错误: {e}")

if __name__ == "__main__":
    main()
