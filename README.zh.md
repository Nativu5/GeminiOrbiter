# Gemini-FastAPI

[![Python 3.12](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)


[ [English](README.md) | 中文 ]

将 Gemini 网页端模型封装为兼容 OpenAI API 的 API Server。基于 [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) 实现。

**✅ 无需 API Key，免费通过 API 调用 Gemini 网页端模型！**

## 功能特性

- 🔐 **无需 Google API Key**：只需网页 Cookie，即可免费通过 API 调用 Gemini 模型。
- 🔍 **内置 Google 搜索**：API 已内置 Gemini 网页端的搜索能力，模型响应更加准确。
- 💾 **会话持久化**：基于 LMDB 存储，支持多轮对话历史记录。
- 🖼️ **多模态支持**：可处理文本、图片及文件上传。
- 🔧 **灵活配置**：基于 YAML 配置文件，支持通过环境变量动态覆盖配置项。

## 快速开始

**如需 Docker 部署，请参见下方 [Docker 部署](#docker-部署) 部分。**

### 前置条件

- Python 3.12
- 拥有网页版 Gemini 访问权限的 Google 账号
- 从 Gemini 网页获取的 `secure_1psid` 和 `secure_1psidts` Cookie

### 安装

#### 使用 uv (推荐)

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
uv sync
```

#### 使用 pip

```bash
git clone https://github.com/Nativu5/Gemini-FastAPI.git
cd Gemini-FastAPI
pip install -e .
```

### 配置

编辑 `config/config.yaml` 并提供至少一组凭证：
```yaml
gemini:
  clients:
    - id: "client-a"
      secure_1psid: "YOUR_SECURE_1PSID_HERE"
      secure_1psidts: "YOUR_SECURE_1PSIDTS_HERE"
```

> [!NOTE]
> 详细说明请参见下方 [配置](#配置说明) 部分。

### 启动服务

```bash
# 使用 uv
uv run python run.py

# 直接用 Python
python run.py
```

服务默认启动在 `http://localhost:8000`。

## Docker 部署

### 直接运行

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/cache:/app/.venv/lib/python3.12/site-packages/gemini_webapi/utils/temp \
  -e CONFIG_SERVER__API_KEY="your-api-key-here" \
  -e CONFIG_GEMINI__CLIENTS__0__ID="client-a" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid" \
  -e CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts" \
  ghcr.io/nativu5/gemini-fastapi
```

### 使用 Docker Compose

创建 `docker-compose.yml` 文件：

```yaml
services:
  gemini-fastapi:
    image: ghcr.io/nativu5/gemini-fastapi:latest
    ports:
      - "8000:8000"
    volumes:
      # - ./config:/app/config  # Uncomment to use a custom config file
      - ./data:/app/data
      - ./cache:/app/.venv/lib/python3.12/site-packages/gemini_webapi/utils/temp
    environment:
      - CONFIG_SERVER__HOST=0.0.0.0
      - CONFIG_SERVER__PORT=8000
      - CONFIG_SERVER__API_KEY=${API_KEY}
      - CONFIG_GEMINI__CLIENTS__0__ID=client-a
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID=${SECURE_1PSID}
      - CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS=${SECURE_1PSIDTS}
    restart: on-failure:3 # 避免过多重试
```

然后运行：

```bash
docker compose up -d
```

> [!IMPORTANT]
> 请务必挂载 `/app/data` 卷以保证对话数据在容器重启后持久化。
> 建议同时挂载 `gemini_webapi/utils/temp` 目录以保存刷新后的 Cookie。

## 配置说明

服务器读取 `config/config.yaml` 配置文件。

各项配置说明请参见 [`config/config.yaml`](https://github.com/Nativu5/Gemini-FastAPI/blob/main/config/config.yaml) 文件中的注释。

### 环境变量覆盖

> [!TIP]
> 该功能适用于 Docker 部署和生产环境，可将敏感信息与配置文件分离。

你可以通过带有 `CONFIG_` 前缀的环境变量覆盖任意配置项，嵌套键用双下划线（`__`）分隔，例如：

```bash
# 覆盖服务器设置
export CONFIG_SERVER__API_KEY="your-secure-api-key"

# 覆盖 Gemini Cookie
export CONFIG_GEMINI__CLIENTS__0__ID="client-a"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSID="your-secure-1psid"
export CONFIG_GEMINI__CLIENTS__0__SECURE_1PSIDTS="your-secure-1psidts"

# 覆盖对话存储大小限制
export CONFIG_STORAGE__MAX_SIZE=268435456  # 256 MB
```

### 客户端 ID 与会话重用

会话在保存时会绑定创建它的客户端 ID。请在配置中保持这些 `id` 值稳定，
这样在更新 Cookie 列表时依然可以复用旧会话。

### Gemini 凭据

> [!WARNING]
> 请妥善保管这些凭据，切勿提交到版本控制。这些 Cookie 可访问你的 Google 账号。

使用 Gemini-FastAPI 需提取 Gemini 会话 Cookie：

1. 在无痕/隐私窗口打开 [Gemini](https://gemini.google.com/) 并登录
2. 打开开发者工具（F12）
3. 进入 **Application** → **Storage** → **Cookies**
4. 查找并复制以下值：
   - `__Secure-1PSID`
   - `__Secure-1PSIDTS`

> [!TIP]
> 详细操作请参考 [HanaokaYuzu/Gemini-API 认证指南](https://github.com/HanaokaYuzu/Gemini-API?tab=readme-ov-file#authentication)。

## 鸣谢

- [HanaokaYuzu/Gemini-API](https://github.com/HanaokaYuzu/Gemini-API) - 底层 Gemini Web API 客户端
- [zhiyu1998/Gemi2Api-Server](https://github.com/zhiyu1998/Gemi2Api-Server) - 本项目最初基于此仓库，经过深度重构与工程化改进，现已成为独立项目，并增加了多轮会话复用等新特性。在此表示特别感谢。

## 免责声明

本项目与 Google 或 OpenAI 无关，仅供学习和研究使用。本项目使用了逆向工程 API，可能不符合 Google 服务条款。使用风险自负。
