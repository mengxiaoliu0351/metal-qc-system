# -*- coding: utf-8 -*-
"""
全局配置：路径、VLM 后端、门控阈值、演示模式
=============================================
现场演示策略：
  MODE = "cache_first"  -> 优先读缓存（0秒出结果），无缓存走 API 并写入缓存
  MODE = "realtime"     -> 全部实时推理（API 2-5 秒）
断网兜底：VLM_BACKEND="local" 走本机 Ollama（Qwen2-VL-2B，30-60秒/图）
"""
import os

# ---------- 路径（自动定位项目根，不写死绝对路径）----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
DEMO_CACHE_DIR = os.path.join(RESULTS_DIR, "demo_cache")   # 预缓存样例结果
SAMPLE_DIR = os.path.join(ROOT, "data", "demo_samples")    # 演示样例图
KB_CSV = os.path.join(RESULTS_DIR, "cognition", "standard_library.csv")

# ---------- 演示模式 ----------
MODE = "cache_first"          # cache_first（现场演示）/ realtime（实时）
VLM_BACKEND = "api"           # api（豆包/通义）/ local（Ollama）
VLM_PROVIDER = "dashscope"    # doubao（火山方舟）/ dashscope（通义）

# ---------- VLM API（OpenAI 兼容格式）----------
DOUBAO_API_KEY = os.environ.get("DOUBAO_API_KEY", "PASTE_YOUR_KEY_HERE")
DOUBAO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DOUBAO_MODEL = "doubao-vision-pro-32k"     # 换成你火山方舟开通的视觉模型ID

DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "PASTE_YOUR_DASHSCOPE_KEY_HERE")   # 队友在阿里云DashScope申请自己的key，用环境变量配置，不要写进代码
# 公共endpoint（已实测连通）
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen3-vl-plus"   # VLM慢链路：视觉模型（已实测200）
DASHSCOPE_CHAT_MODEL = "qwen-plus" # AI助手：纯文本问答（更稳定）

# ---------- 本地 Ollama（兜底）----------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:2b"

# ---------- 门控 ----------
# 0.2345 = train/good(80张)上 median(0.1900) + 1.5×MAD(0.0297)，test集准确率96.52%
# 反馈自动调整时会覆盖此值，存在 results/perception/adjusted_threshold.json
THRESHOLD = 0.2345

# ---------- 感知层 ----------
CLIP_MODEL = "ViT-B-16"
PERCEPTION_WEIGHTS = os.path.join(ROOT, "weights",
                                  "CLIP-ViT-B-16-open_clip_model.safetensors")
