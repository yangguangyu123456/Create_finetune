#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤矿安全问答模型Web服务部署
基于Flask框架提供HTTP API接口
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch

# 导入推理类
from inference import CoalSafetyInference

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量存储模型实例
model_instance = None

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>煤矿安全问答系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .chat-container {
            padding: 30px;
            min-height: 400px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        #questionInput {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        #questionInput:focus {
            border-color: #667eea;
        }
        
        #askButton {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        #askButton:hover {
            transform: translateY(-2px);
        }
        
        #askButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 0.5s ease-in;
        }
        
        .user-message {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .bot-message {
            background: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .status {
            text-align: center;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .status.success {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .status.error {
            background: #ffebee;
            color: #c62828;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 煤矿安全问答系统</h1>
            <p>基于DeepSeek-R1-Distill-Qwen-1.5B微调模型</p>
        </div>
        
        <div class="chat-container">
            <div id="status" class="status">正在初始化模型...</div>
            
            <div class="input-group">
                <input type="text" id="questionInput" placeholder="请输入您的煤矿安全相关问题..." disabled>
                <button id="askButton" disabled>提问</button>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>AI正在思考中...</p>
            </div>
            
            <div id="chatHistory"></div>
        </div>
    </div>
    
    <script>
        let isModelReady = false;
        
        // 检查模型状态
        async function checkModelStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const statusDiv = document.getElementById('status');
                const questionInput = document.getElementById('questionInput');
                const askButton = document.getElementById('askButton');
                
                if (data.status === 'ready') {
                    statusDiv.textContent = '✅ 模型已就绪，可以开始提问';
                    statusDiv.className = 'status success';
                    questionInput.disabled = false;
                    askButton.disabled = false;
                    isModelReady = true;
                } else {
                    statusDiv.textContent = '⏳ 模型正在加载中...';
                    statusDiv.className = 'status';
                }
            } catch (error) {
                const statusDiv = document.getElementById('status');
                statusDiv.textContent = '❌ 无法连接到服务器';
                statusDiv.className = 'status error';
            }
        }
        
        // 发送问题
        async function askQuestion() {
            if (!isModelReady) return;
            
            const questionInput = document.getElementById('questionInput');
            const question = questionInput.value.trim();
            
            if (!question) {
                alert('请输入问题');
                return;
            }
            
            // 显示用户问题
            addMessage(question, 'user');
            
            // 清空输入框
            questionInput.value = '';
            
            // 显示加载状态
            const loading = document.getElementById('loading');
            const askButton = document.getElementById('askButton');
            loading.style.display = 'block';
            askButton.disabled = true;
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question: question })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    addMessage(data.answer, 'bot');
                } else {
                    addMessage('抱歉，处理您的问题时出现错误：' + data.error, 'bot');
                }
            } catch (error) {
                addMessage('抱歉，网络连接出现问题，请稍后重试。', 'bot');
            } finally {
                loading.style.display = 'none';
                askButton.disabled = false;
            }
        }
        
        // 添加消息到聊天历史
        function addMessage(text, type) {
            const chatHistory = document.getElementById('chatHistory');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const prefix = type === 'user' ? '🤔 您的问题：' : '🤖 AI回答：';
            messageDiv.innerHTML = `<strong>${prefix}</strong><br>${text}`;
            
            chatHistory.appendChild(messageDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }
        
        // 事件监听
        document.getElementById('askButton').addEventListener('click', askQuestion);
        document.getElementById('questionInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
        
        // 初始化
        checkModelStatus();
        setInterval(checkModelStatus, 5000); // 每5秒检查一次状态
    </script>
</body>
</html>
"""


def initialize_model(base_model_path: str, peft_model_path: str = None, use_4bit: bool = True):
    """初始化模型"""
    global model_instance
    try:
        logger.info("开始初始化模型...")
        model_instance = CoalSafetyInference(
            base_model_path=base_model_path,
            peft_model_path=peft_model_path,
            use_4bit=use_4bit
        )
        logger.info("模型初始化完成")
        return True
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return False


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status', methods=['GET'])
def get_status():
    """获取模型状态"""
    if model_instance is None:
        return jsonify({
            'status': 'loading',
            'message': '模型正在加载中'
        })
    else:
        return jsonify({
            'status': 'ready',
            'message': '模型已就绪',
            'model_info': {
                'base_model': model_instance.base_model_path,
                'peft_model': model_instance.peft_model_path,
                'use_4bit': model_instance.use_4bit
            }
        })


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """处理问答请求"""
    if model_instance is None:
        return jsonify({
            'success': False,
            'error': '模型尚未加载完成，请稍后重试'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': '请提供问题内容'
            }), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({
                'success': False,
                'error': '问题不能为空'
            }), 400
        
        # 获取可选参数
        max_length = data.get('max_length', 512)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        logger.info(f"收到问题: {question[:50]}...")
        
        # 生成回答
        answer = model_instance.generate_response(
            question=question,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        logger.info(f"生成回答完成: {answer[:50]}...")
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"处理问题时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch', methods=['POST'])
def batch_inference():
    """批量推理接口"""
    if model_instance is None:
        return jsonify({
            'success': False,
            'error': '模型尚未加载完成，请稍后重试'
        }), 503
    
    try:
        data = request.get_json()
        if not data or 'questions' not in data:
            return jsonify({
                'success': False,
                'error': '请提供问题列表'
            }), 400
        
        questions = data['questions']
        if not isinstance(questions, list) or not questions:
            return jsonify({
                'success': False,
                'error': '问题列表格式错误或为空'
            }), 400
        
        logger.info(f"收到批量推理请求，共{len(questions)}个问题")
        
        # 批量推理
        results = model_instance.batch_inference(questions)
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"批量推理时出错: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_instance is not None,
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="煤矿安全问答模型Web服务")
    parser.add_argument(
        '--base_model',
        type=str,
        default='./DeepSeek-R1-Distill-Qwen-1.5B',
        help='基础模型路径'
    )
    parser.add_argument(
        '--peft_model',
        type=str,
        default=None,
        help='LoRA模型路径'
    )
    parser.add_argument(
        '--use_4bit',
        action='store_true',
        default=True,
        help='是否使用4bit量化'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器主机地址'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='服务器端口'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式'
    )
    
    args = parser.parse_args()
    
    # 初始化模型
    logger.info("启动Web服务...")
    success = initialize_model(
        base_model_path=args.base_model,
        peft_model_path=args.peft_model,
        use_4bit=args.use_4bit
    )
    
    if not success:
        logger.error("模型初始化失败，服务将在模型加载完成前不可用")
    
    # 启动Flask应用
    logger.info(f"启动Web服务，地址: http://{args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    )