import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不需要GUI
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def plot_stability_diagram(g1=None, g2=None, save_path='stability_diagram.png'):
    """绘制腔稳定性图并保存到文件"""
    plt.figure(figsize=(10, 8))
    
    # 创建g1-g2网格
    g_range = np.linspace(-2, 2, 500)
    G1, G2 = np.meshgrid(g_range, g_range)
    stability = G1 * G2
    
    # 绘制稳定区域（0 <= g1*g2 <= 1）
    plt.contourf(G1, G2, stability, levels=[-np.inf, 0, 1, np.inf], 
               colors=['red', 'lightgreen', 'red'], alpha=0.3)
    
    # 添加稳定性曲线
    plt.contour(G1, G2, stability, levels=[0, 1], colors='blue', linewidths=2)
    
    # 标记几种经典腔型
    special_cases = {
        '平行平面镜腔': (1, 1),
        '共焦腔': (0, 0),
        '半球腔': (1, 0),
        '球面对称腔': (0.5, 0.5),
        '同心腔': (-1, -1),
    }
    
    for name, (g1_val, g2_val) in special_cases.items():
        plt.plot(g1_val, g2_val, 'ko', markersize=6)
        plt.annotate(name, (g1_val, g2_val), xytext=(10, 5), textcoords='offset points')
    
    # 如果当前有计算过的参数，标记当前腔点
    if g1 is not None and g2 is not None:
        plt.plot(g1, g2, 'ro', markersize=8)
        stability_product = g1 * g2
        stability_status = "稳定" if 0 <= stability_product <= 1 else "不稳定"
        plt.annotate(f'当前腔点 (g1={g1:.3f}, g2={g2:.3f})\n稳定性: {stability_status}', 
                  (g1, g2), xytext=(20, 10), textcoords='offset points',
                  bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                  arrowprops=dict(arrowstyle="->"))
    
    # 添加g1=0、g2=0轴线
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    # 添加g1=g2对角线
    plt.plot([-2, 2], [-2, 2], 'k--', alpha=0.5)
    
    # 设置图表属性
    plt.xlabel('g1 = 1 - L/R1')
    plt.ylabel('g2 = 1 - L/R2')
    plt.title('腔稳定性图')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 添加稳定性解释
    legend_elements = [
        patches.Patch(facecolor='lightgreen', edgecolor='blue', alpha=0.3, label='稳定区域 (0 <= g1*g2 <= 1)'),
        patches.Patch(facecolor='red', edgecolor='blue', alpha=0.3, label='不稳定区域')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300)
    print(f"腔稳定性图已保存到：{save_path}")

def calculate_cavity_stability(L, R1, R2):
    """计算腔的稳定性参数"""
    g1 = 1 - L/R1 if R1 != 0 else float('inf')
    g2 = 1 - L/R2 if R2 != 0 else float('inf')
    
    print(f"当前腔参数:")
    print(f"腔长 L = {L*1000:.2f} mm")
    print(f"曲率半径 R1 = {R1*1000:.2f} mm" if R1 != float('inf') else "曲率半径 R1 = ∞ (平面镜)")
    print(f"曲率半径 R2 = {R2*1000:.2f} mm" if R2 != float('inf') else "曲率半径 R2 = ∞ (平面镜)")
    print(f"g1 = {g1:.4f}" if g1 != float('inf') else "g1 = 1 (平面镜)")
    print(f"g2 = {g2:.4f}" if g2 != float('inf') else "g2 = 1 (平面镜)")
    
    # 计算稳定性条件
    if g1 == float('inf') or g2 == float('inf'):
        stability_product = 1 * (g2 if g1 == float('inf') else g1)
    else:
        stability_product = g1 * g2
    
    print(f"g1*g2 = {stability_product:.4f}")
    
    stability_status = "稳定" if 0 <= stability_product <= 1 else "不稳定"
    print(f"腔稳定性: {stability_status}")
    
    return g1, g2

if __name__ == "__main__":
    # 允许用户输入参数或使用默认值
    try:
        L = float(input("请输入腔长 L (mm，默认10): ") or "10") * 1e-3  # 转换为m
    except ValueError:
        print("输入无效，使用默认值。")
        L = 10e-3  # 默认腔长 (m)
    
    try:
        R1 = float(input("请输入曲率半径 R1 (mm，默认20): ") or "20") * 1e-3  # 转换为m
    except ValueError:
        print("输入无效，使用默认值。")
        R1 = 20e-3  # 默认曲率半径1 (m)
    
    try:
        R2 = float(input("请输入曲率半径 R2 (mm，默认20): ") or "20") * 1e-3  # 转换为m
    except ValueError:
        print("输入无效，使用默认值。")
        R2 = 20e-3  # 默认曲率半径2 (m)
    
    # 计算稳定性参数
    g1, g2 = calculate_cavity_stability(L, R1, R2)
    
    # 绘制并保存稳定性图
    plot_stability_diagram(g1 if g1 != float('inf') else 1, 
                         g2 if g2 != float('inf') else 1)
    
    print("\n程序执行完成。请查看生成的稳定性图像。")
    input("按回车键继续...") 