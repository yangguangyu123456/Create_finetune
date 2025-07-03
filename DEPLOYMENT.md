# 煤矿安全问答模型Web服务部署指南

## 概述

本项目提供了一个基于Flask的Web服务，用于部署煤矿安全问答模型。支持通过Web界面和API接口进行交互。

## 功能特性

- 🌐 **Web界面**: 提供友好的网页界面进行问答
- 🔌 **REST API**: 支持HTTP API调用
- 📦 **批量推理**: 支持一次处理多个问题
- 🚀 **异步加载**: 模型在后台加载，不阻塞服务启动
- 📊 **状态监控**: 提供健康检查和状态查询接口
- 🔧 **参数配置**: 支持自定义生成参数

## 环境要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU加速)
- 内存: 8GB+ (使用4bit量化时)
- 显存: 4GB+ (使用GPU时)

## 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 如果使用GPU，确保安装了正确版本的PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取安装命令
```

## 快速启动

### 方法1: 使用启动脚本 (Windows)

```bash
# 双击运行或在命令行执行
start_server.bat
```

### 方法2: 手动启动

```bash
# 基本启动（使用默认参数）
python app.py

# 自定义参数启动
python app.py --base_model ./DeepSeek-R1-Distill-Qwen-1.5B --port 8080 --host 0.0.0.0

# 如果有微调模型
python app.py --base_model ./DeepSeek-R1-Distill-Qwen-1.5B --peft_model ./path/to/lora/model
```

## 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--base_model` | `./DeepSeek-R1-Distill-Qwen-1.5B` | 基础模型路径 |
| `--peft_model` | `None` | LoRA微调模型路径 |
| `--use_4bit` | `True` | 是否使用4bit量化 |
| `--host` | `0.0.0.0` | 服务器主机地址 |
| `--port` | `5000` | 服务器端口 |
| `--debug` | `False` | 是否启用调试模式 |

## 使用方式

### 1. Web界面

启动服务后，在浏览器中访问:

```
http://localhost:5000
```

界面功能:
- 实时状态显示
- 交互式问答
- 美观的聊天界面
- 自动滚动和动画效果

### 2. API接口

#### 状态检查

```bash
# 检查模型状态
curl http://localhost:5000/api/status

# 健康检查
curl http://localhost:5000/api/health
```

#### 单个问题

```bash
curl -X POST http://localhost:5000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "煤矿井下作业的安全要求有哪些？"}'
```

#### 批量推理

```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{"questions": ["瓦斯爆炸的条件是什么？", "煤矿防治水的主要措施有哪些？"]}'
```

### 3. Python客户端

使用提供的API客户端:

```python
from api_example import CoalSafetyAPIClient

# 创建客户端
client = CoalSafetyAPIClient("http://localhost:5000")

# 等待服务就绪
client.wait_for_ready()

# 提问
result = client.ask_question("煤矿井下作业的安全要求有哪些？")
print(result['answer'])
```

## API接口详细说明

### GET /api/status

检查模型加载状态

**响应示例:**
```json
{
  "status": "ready",
  "message": "模型已就绪",
  "model_info": {
    "base_model": "./DeepSeek-R1-Distill-Qwen-1.5B",
    "peft_model": null,
    "use_4bit": true
  }
}
```

### POST /api/ask

单个问题推理

**请求参数:**
```json
{
  "question": "问题内容",
  "max_length": 512,      // 可选，最大生成长度
  "temperature": 0.7,     // 可选，生成温度
  "top_p": 0.9           // 可选，top-p采样
}
```

**响应示例:**
```json
{
  "success": true,
  "question": "煤矿井下作业的安全要求有哪些？",
  "answer": "煤矿井下作业需要遵守以下安全要求...",
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /api/batch

批量推理

**请求参数:**
```json
{
  "questions": ["问题1", "问题2", "问题3"]
}
```

**响应示例:**
```json
{
  "success": true,
  "results": [
    {
      "question": "问题1",
      "answer": "回答1"
    }
  ],
  "total": 3,
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /api/health

健康检查

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true,
  "gpu_available": true,
  "gpu_count": 1
}
```

## 性能优化建议

### 1. 硬件配置

- **GPU**: 推荐使用NVIDIA RTX 3060或更高配置
- **内存**: 至少8GB，推荐16GB+
- **存储**: 使用SSD存储模型文件

### 2. 模型配置

- **4bit量化**: 启用`--use_4bit`减少显存占用
- **批量大小**: 根据硬件配置调整批量推理大小
- **生成参数**: 调整`temperature`和`top_p`平衡质量和速度

### 3. 服务配置

- **多线程**: Flask默认启用多线程支持
- **负载均衡**: 可使用nginx等反向代理
- **缓存**: 考虑添加Redis缓存常见问题

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认有足够的内存/显存
   - 查看日志中的详细错误信息

2. **服务无法访问**
   - 检查防火墙设置
   - 确认端口未被占用
   - 检查host参数设置

3. **推理速度慢**
   - 启用GPU加速
   - 使用4bit量化
   - 减少max_length参数

4. **内存不足**
   - 启用4bit量化
   - 减少批量大小
   - 关闭其他占用内存的程序

### 日志查看

服务运行时会输出详细日志，包括:
- 模型加载进度
- 请求处理信息
- 错误详情

## 生产环境部署

### 1. 使用WSGI服务器

```bash
# 安装gunicorn
pip install gunicorn

# 启动服务
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 2. 使用Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

### 3. 反向代理配置

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 安全注意事项

1. **访问控制**: 在生产环境中添加身份验证
2. **HTTPS**: 使用SSL/TLS加密传输
3. **输入验证**: 对用户输入进行严格验证
4. **资源限制**: 设置请求频率和大小限制
5. **日志审计**: 记录所有API调用

## 许可证

本项目遵循相应的开源许可证，请查看LICENSE文件了解详情。

## 支持

如有问题或建议，请通过以下方式联系:
- 提交Issue
- 发送邮件
- 查看文档