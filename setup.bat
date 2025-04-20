@echo off
chcp 65001 >nul
echo 正在配置Fabry-Pérot腔OAM模式分析器环境...

REM 检查Python是否已安装
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 未检测到Python，请先安装Python 3.13或更高版本
    pause
    exit /b 1
)

REM 安装依赖库
echo 正在安装依赖库...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo 安装依赖库失败，请检查网络连接或手动运行: pip install -r requirements.txt
    pause
    exit /b 1
)

REM 启动程序
echo 环境配置完成，正在启动程序...
python fp_oam_simulator.py
if %ERRORLEVEL% NEQ 0 (
    echo 程序运行出错
    pause
    exit /b 1
)

pause 