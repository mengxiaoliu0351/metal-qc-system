# 金属表面缺陷双链路门控智能质检辅助系统

> 《人工智能应用实践》课程项目 · 山东大学人工智能学院
> 团队：刘孟霄、李丰润、侯亚琳

## 项目简介

面向中小机械加工厂"多品种、小批量"的金属零件离线抽检场景，做人在回路的质检辅助（最终判定仍由人工确认签发）。

- **感知层**：WinCLIP（CLIP ViT-B-16），用正常件 patch 特征建记忆库，输出异常分数 + 热力图
- **门控层**：每个产品独立阈值（正常件 median + 1.5×MAD，holdout 独立标定，无数据泄露）
- **双链路**：高异常分 → 快链路直出不合格；低异常分 → 慢链路升级 VLM 深度复核
- **认知层**：Qwen-Vision（DashScope API）做缺陷描述与行业标准对照
- **Web**：FastAPI + 静态 HTML 单页应用，支持单图/批量检测、历史记录、AI 助手、PDF 报告

当前内置三个真实产品库（MVTec）：金属螺母（test 准确率 96.5%）、金属网格（96.2%，缺陷 100% 检出）、金属螺丝（约 82% 检出）。

## 环境要求

- Python 3.10+（推荐 conda 环境）
- **无需 GPU**：CLIP 小模型 CPU 可跑；VLM 走云端 API
- 依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

## 配置（队友必做）

### 1. VLM 密钥（不要写进代码）

在 [阿里云 DashScope](https://dashscope.aliyun.com) 申请自己的 API Key，用环境变量配置：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "你的key"
# Linux / macOS
export DASHSCOPE_API_KEY="你的key"
```

代码从环境变量读取，仓库里的 `src/config.py` 只有占位符，**不要把真实 key 提交到 git**。

### 2. CLIP 模型权重

将 `CLIP-ViT-B-16-open_clip_model.safetensors` 放到 `weights/` 目录（约 570MB，open_clip 可自动下载，或从 Release 获取）。

### 3. 数据与建库

数据集（MVTec-AD）因版权和体积不随仓库分发。需要某产品时，按 `data/mvtec/<产品>/train/good/` 组织合格件图片，运行 `src/experiments/` 下的建库脚本（如 `add_screw.py`、`add_grid.py`）生成记忆库与阈值，配置写入 `results/perception/banks/products.json`。

## 运行

```bash
cd src
uvicorn web.server:app --host 0.0.0.0 --port 8000
```

浏览器打开 http://localhost:8000

## 目录结构

```
├── src/
│   ├── config.py            # 配置（路径自动定位，key 走环境变量）
│   ├── perception/          # WinCLIP 异常检测
│   ├── gating/              # 置信度门控路由
│   ├── cognition/           # VLM 认知分析
│   ├── web/                 # FastAPI 后端 + static/index.html 前端
│   ├── interface/           # 报告/PDF/反馈工具（被 web 引用）
│   └── experiments/         # 建库、导出、验证脚本
├── weights/                 # 模型权重（不入库）
├── data/                    # 数据集（不入库）
└── results/                 # 记忆库、历史、结果（不入库）
```

## 协作约定

- 不要提交：`weights/`、`data/`、`results/`、`.env`、真实 API key
- 建库脚本、实验结果可放 `experiments/` 但不要提交大文件
- 改完代码本地跑通 `uvicorn` 再提交
