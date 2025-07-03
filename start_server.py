#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
煤矿安全问答模型Web服务启动脚本
替代批处理文件，提供更好的跨平台支持和错误处理
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("[错误] 需要Python 3.8或更高版本")
        print(f"[错误] 当前版本: {sys.version}")
        return False
    print(f"[信息] Python版本检查通过: {sys.version.split()[0]}")
    return True


def install_requirements():
    """安装依赖包"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("[警告] requirements.txt文件不存在，跳过依赖安装")
        return True
    
    print("[信息] 检查并安装依赖包...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            print("[信息] 依赖包安装完成")
            return True
        else:
            print("[警告] 依赖包安装可能不完整")
            print(f"[警告] 错误信息: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[警告] 依赖包安装超时")
        return False
    except Exception as e:
        print(f"[警告] 依赖包安装出错: {e}")
        return False


def check_model_files(base_model_path):
    """检查模型文件"""
    model_path = Path(base_model_path)
    if not model_path.exists():
        print(f"[错误] 基础模型路径不存在: {base_model_path}")
        return False, None
    
    # 检查必要的模型文件
    required_files = ["config.json", "tokenizer.json"]
    missing_files = []
    
    for file_name in required_files:
        if not (model_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"[警告] 缺少模型文件: {', '.join(missing_files)}")
    
    # 检查是否有微调模型
    peft_model_path = "./output/"
    if Path("model.safetensors").exists():
        peft_model_path = "."
        print("[信息] 检测到微调模型，将使用微调后的模型")
    elif (model_path / "adapter_model.safetensors").exists():
        peft_model_path = str(model_path)
        print("[信息] 检测到LoRA适配器，将使用微调后的模型")
    else:
        print("[信息] 未检测到微调模型，将使用原始模型")
    
    return True, peft_model_path


def start_server(base_model, peft_model, port, host, use_4bit, debug):
    """启动Web服务"""
    print("[信息] 启动Web服务...")
    print(f"[信息] 服务启动后，请访问 http://{host}:{port} 使用问答系统")
    print("[信息] 按 Ctrl+C 停止服务")
    print("=" * 50)
    
    # 构建启动命令
    cmd = [
        sys.executable, "app.py",
        "--base_model", base_model,
        "--port", str(port),
        "--host", host
    ]
    
    if peft_model:
        cmd.extend(["--peft_model", peft_model])
    
    if use_4bit:
        cmd.append("--use_4bit")
    
    if debug:
        cmd.append("--debug")
    
    try:
        # 启动服务
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[信息] 服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 服务启动失败: {e}")
        return False
    except FileNotFoundError:
        print("\n[错误] 找不到app.py文件")
        return False
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="煤矿安全问答模型Web服务启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python start_server.py                          # 使用默认参数启动
  python start_server.py --port 8080             # 指定端口
  python start_server.py --base_model ./model    # 指定模型路径
  python start_server.py --no-install            # 跳过依赖安装
  python start_server.py --debug                 # 启用调试模式
        """
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default="./DeepSeek-R1-Distill-Qwen-1.5B",
        help="基础模型路径 (默认: ./DeepSeek-R1-Distill-Qwen-1.5B)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="服务器端口 (默认: 5000)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="禁用4bit量化"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="跳过依赖包安装"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制启动（忽略检查错误）"
    )
    
    args = parser.parse_args()
    
    print("===================================")
    print("煤矿安全问答模型Web服务启动脚本")
    print("===================================")
    
    # 1. 检查Python版本
    if not check_python_version():
        if not args.force:
            sys.exit(1)
    
    # 2. 安装依赖包
    if not args.no_install:
        if not install_requirements():
            if not args.force:
                print("[错误] 依赖包安装失败，使用 --force 强制启动或 --no-install 跳过安装")
                sys.exit(1)
    
    # 3. 检查模型文件
    model_ok, peft_model = check_model_files(args.base_model)
    if not model_ok:
        if not args.force:
            print("[错误] 模型文件检查失败，使用 --force 强制启动")
            sys.exit(1)
    
    # 4. 检查app.py文件
    if not Path("app.py").exists():
        print("[错误] 找不到app.py文件")
        sys.exit(1)
    
    # 5. 启动服务
    use_4bit = not args.no_4bit
    success = start_server(
        base_model=args.base_model,
        peft_model=peft_model,
        port=args.port,
        host=args.host,
        use_4bit=use_4bit,
        debug=args.debug
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()