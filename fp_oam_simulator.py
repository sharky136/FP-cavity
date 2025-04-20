import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QPushButton, QGridLayout, QGroupBox,
                            QTableWidget, QTableWidgetItem, QMessageBox, QFileDialog,
                            QTabWidget, QComboBox, QSlider, QSpinBox, QDoubleSpinBox,
                            QCheckBox)
from PyQt5.QtCore import Qt
import pandas as pd
from scipy import constants
import matplotlib
import matplotlib.patches as patches

# 设置matplotlib中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

class FPResonatorSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fabry-Pérot腔OAM模式分析器")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建参数输入面板
        params_panel = QGroupBox("参数输入")
        params_layout = QGridLayout(params_panel)
        
        # 设置参数输入控件
        self.params = {}
        self.param_fields = {}
        
        # 波长
        row = 0
        params_layout.addWidget(QLabel("入射光波长 (λ, nm):"), row, 0)
        self.param_fields['wavelength'] = QLineEdit("795")
        params_layout.addWidget(self.param_fields['wavelength'], row, 1)
        
        # 腔长
        row += 1
        params_layout.addWidget(QLabel("基准腔长 (L, mm):"), row, 0)
        self.param_fields['cavity_length'] = QLineEdit("10")
        params_layout.addWidget(self.param_fields['cavity_length'], row, 1)
        
        # 腔长变化范围
        row += 1
        params_layout.addWidget(QLabel("腔长变化范围 (nm):"), row, 0)
        self.param_fields['length_variation'] = QLineEdit("900")
        params_layout.addWidget(self.param_fields['length_variation'], row, 1)
        
        # 反射率1
        row += 1
        params_layout.addWidget(QLabel("镜面反射率(左) (R1):"), row, 0)
        self.param_fields['reflectivity1'] = QLineEdit("0.9")
        params_layout.addWidget(self.param_fields['reflectivity1'], row, 1)
        
        # 反射率2
        row += 1
        params_layout.addWidget(QLabel("镜面反射率(右) (R2):"), row, 0)
        self.param_fields['reflectivity2'] = QLineEdit("0.9")
        params_layout.addWidget(self.param_fields['reflectivity2'], row, 1)
        
        # 曲率半径1
        row += 1
        params_layout.addWidget(QLabel("曲率半径(左) (Ra1, mm):"), row, 0)
        self.param_fields['radius1'] = QLineEdit("10")
        params_layout.addWidget(self.param_fields['radius1'], row, 1)
        
        # 曲率半径2
        row += 1
        params_layout.addWidget(QLabel("曲率半径(右) (Ra2, mm):"), row, 0)
        self.param_fields['radius2'] = QLineEdit("10")
        params_layout.addWidget(self.param_fields['radius2'], row, 1)
        
        # OAM模式最大值
        row += 1
        params_layout.addWidget(QLabel("最大OAM模式数 (l_max):"), row, 0)
        self.param_fields['l_max'] = QLineEdit("5")
        params_layout.addWidget(self.param_fields['l_max'], row, 1)
        
        # 分辨率设置
        row += 1
        params_layout.addWidget(QLabel("计算点数 (分辨率):"), row, 0)
        self.param_fields['resolution'] = QSpinBox()
        self.param_fields['resolution'].setRange(500, 5000)
        self.param_fields['resolution'].setValue(1000)
        self.param_fields['resolution'].setSingleStep(100)
        params_layout.addWidget(self.param_fields['resolution'], row, 1)
        
        # 按钮组
        row += 1
        button_layout = QHBoxLayout()
        
        self.calculate_btn = QPushButton("计算")
        self.calculate_btn.clicked.connect(self.calculate)
        button_layout.addWidget(self.calculate_btn)
        
        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_params)
        button_layout.addWidget(self.reset_btn)
        
        self.export_btn = QPushButton("导出数据")
        self.export_btn.clicked.connect(self.export_data)
        button_layout.addWidget(self.export_btn)
        
        params_layout.addLayout(button_layout, row, 0, 1, 2)
        
        # 创建计算结果显示区域
        results_group = QGroupBox("计算结果")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget(7, 2)
        self.results_table.setHorizontalHeaderLabels(["物理量", "数值"])
        self.results_table.setItem(0, 0, QTableWidgetItem("自由光谱范围 (FSR)"))
        self.results_table.setItem(1, 0, QTableWidgetItem("精细度 (Finesse)"))
        self.results_table.setItem(2, 0, QTableWidgetItem("基模束腰半径"))
        self.results_table.setItem(3, 0, QTableWidgetItem("模式间距 (Δν)"))
        self.results_table.setItem(4, 0, QTableWidgetItem("线宽 (Δλ)"))
        self.results_table.setItem(5, 0, QTableWidgetItem("g因子 (g1)"))
        self.results_table.setItem(6, 0, QTableWidgetItem("g因子 (g2)"))
        
        results_layout.addWidget(self.results_table)
        
        # 创建左侧面板布局
        left_panel = QVBoxLayout()
        left_panel.addWidget(params_panel)
        left_panel.addWidget(results_group)
        
        # 创建右侧显示区域（使用选项卡）
        self.tab_widget = QTabWidget()
        
        # 创建透射率曲线选项卡
        transmission_tab = QWidget()
        transmission_layout = QVBoxLayout(transmission_tab)
        
        # 曲线控制选项
        curve_controls = QGroupBox("曲线显示选项")
        curve_controls_layout = QGridLayout(curve_controls)
        
        # 添加模式选择复选框
        self.mode_checkboxes = []
        for i in range(6):  # 默认支持0-5的模式
            checkbox = QCheckBox(f"显示l={i}模式")
            if i <= 3:  # 默认只显示前4个模式
                checkbox.setChecked(True)
            self.mode_checkboxes.append(checkbox)
            curve_controls_layout.addWidget(checkbox, i//3, i%3)
        
        # 添加曲线范围控制
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("显示范围:"))
        
        self.min_slider = QDoubleSpinBox()
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(0)
        self.min_slider.setSuffix("%")
        self.min_slider.valueChanged.connect(self.update_plot_range)
        range_layout.addWidget(self.min_slider)
        
        range_layout.addWidget(QLabel("至"))
        
        self.max_slider = QDoubleSpinBox()
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(100)
        self.max_slider.setSuffix("%")
        self.max_slider.valueChanged.connect(self.update_plot_range)
        range_layout.addWidget(self.max_slider)
        
        curve_controls_layout.addLayout(range_layout, 2, 0, 1, 3)
        
        # 添加刷新曲线按钮
        self.refresh_btn = QPushButton("刷新曲线")
        self.refresh_btn.clicked.connect(self.refresh_plot)
        curve_controls_layout.addWidget(self.refresh_btn, 3, 0, 1, 3)
        
        transmission_layout.addWidget(curve_controls)
        
        # 添加绘图区域和工具栏
        self.transmission_figure = Figure(figsize=(6, 6), dpi=100)
        self.transmission_canvas = FigureCanvas(self.transmission_figure)
        self.transmission_toolbar = NavigationToolbar(self.transmission_canvas, transmission_tab)
        
        transmission_layout.addWidget(self.transmission_toolbar)
        transmission_layout.addWidget(self.transmission_canvas)
        
        # 创建腔稳定性选项卡
        stability_tab = QWidget()
        stability_layout = QVBoxLayout(stability_tab)
        
        # 添加腔稳定性图形和工具栏
        self.stability_figure = Figure(figsize=(6, 6), dpi=100)
        self.stability_canvas = FigureCanvas(self.stability_figure)
        self.stability_toolbar = NavigationToolbar(self.stability_canvas, stability_tab)
        
        stability_layout.addWidget(self.stability_toolbar)
        stability_layout.addWidget(self.stability_canvas)
        
        # 创建OAM模式选项卡
        oam_tab = QWidget()
        oam_layout = QVBoxLayout(oam_tab)
        
        # 添加模式选择器
        mode_selector_layout = QHBoxLayout()
        mode_selector_layout.addWidget(QLabel("选择OAM模式:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("l = 0")
        self.mode_selector.currentIndexChanged.connect(self.update_mode_display)
        mode_selector_layout.addWidget(self.mode_selector)
        
        oam_layout.addLayout(mode_selector_layout)
        
        # 添加OAM模式图形和工具栏
        self.mode_figure = Figure(figsize=(6, 6), dpi=100)
        self.mode_canvas = FigureCanvas(self.mode_figure)
        self.mode_toolbar = NavigationToolbar(self.mode_canvas, oam_tab)
        
        oam_layout.addWidget(self.mode_toolbar)
        oam_layout.addWidget(self.mode_canvas)
        
        # 添加选项卡
        self.tab_widget.addTab(transmission_tab, "透射率曲线")
        self.tab_widget.addTab(stability_tab, "腔稳定性图")
        self.tab_widget.addTab(oam_tab, "OAM模式分布")
        
        # 将面板添加到主布局
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.tab_widget, 2)
        
        # 存储计算数据
        self.cavity_lengths = None
        self.transmissions = None
        self.beam_waist = None
        self.l_max = 0
        self.plot_range = [0, 1]  # 初始范围为0-100%
        self.g1 = None
        self.g2 = None
        
        # 绘制初始腔稳定性图
        self.plot_stability_diagram()
        
    def calculate(self):
        """执行计算并更新显示"""
        try:
            # 获取参数
            wavelength = float(self.param_fields['wavelength'].text()) * 1e-9  # nm转m
            base_cavity_length = float(self.param_fields['cavity_length'].text()) * 1e-3  # mm转m
            length_variation = float(self.param_fields['length_variation'].text()) * 1e-9  # nm转m
            reflectivity1 = float(self.param_fields['reflectivity1'].text())
            reflectivity2 = float(self.param_fields['reflectivity2'].text())
            radius1 = float(self.param_fields['radius1'].text()) * 1e-3  # mm转m
            radius2 = float(self.param_fields['radius2'].text()) * 1e-3  # mm转m
            self.l_max = int(self.param_fields['l_max'].text())
            resolution = self.param_fields['resolution'].value()
            
            # 参数验证
            if not (400e-9 <= wavelength <= 2000e-9):
                QMessageBox.warning(self, "参数错误", "波长必须在400-2000 nm范围内")
                return
                
            if base_cavity_length <= 0:
                QMessageBox.warning(self, "参数错误", "腔长必须为正数")
                return
                
            if not (0 <= reflectivity1 < 1) or not (0 <= reflectivity2 < 1):
                QMessageBox.warning(self, "参数错误", "反射率必须在[0,1)范围内")
                return
                
            if radius1 <= 0 or radius2 <= 0:
                QMessageBox.warning(self, "参数错误", "曲率半径必须为正数")
                return
            
            # 计算物理量
            c = constants.c  # 光速
            n = 1.0  # 空气折射率
            
            # 创建腔长变化数组（单位：nm）
            self.cavity_lengths = np.linspace(0, length_variation*1e9, resolution)  # 转换为nm显示
            
            # 计算g参数
            self.g1 = 1 - base_cavity_length / radius1
            self.g2 = 1 - base_cavity_length / radius2
            
            # 检查腔的稳定性
            if not (0 <= self.g1*self.g2 <= 1):
                QMessageBox.warning(self, "参数错误", "腔参数不满足稳定条件: 0 <= g1*g2 <= 1")
                return
            
            # 自由光谱范围 (FSR) in Hz
            fsr = c / (2 * n * base_cavity_length)
            
            # 精细度
            r1 = np.sqrt(reflectivity1)
            r2 = np.sqrt(reflectivity2)
            finesse = np.pi * np.sqrt(r1 * r2) / (1 - r1 * r2)
            
            # 基模束腰半径 (m)
            if (self.g1 + self.g2 - 2*self.g1*self.g2) == 0:  # 避免除零
                self.beam_waist = np.sqrt((base_cavity_length * wavelength) / np.pi)
            else:
                self.beam_waist = np.sqrt((wavelength * base_cavity_length) / np.pi) * \
                            np.sqrt(self.g1*self.g2*(1-self.g1*self.g2) / ((self.g1+self.g2-2*self.g1*self.g2)**2))
            
            # 模式间距
            mode_spacing = fsr / finesse
            
            # 线宽
            linewidth = (wavelength**2 / c) * mode_spacing
            
            # 更新结果表
            self.results_table.setItem(0, 1, QTableWidgetItem(f"{fsr/1e9:.4g} GHz"))
            self.results_table.setItem(1, 1, QTableWidgetItem(f"{finesse:.4g}"))
            self.results_table.setItem(2, 1, QTableWidgetItem(f"{self.beam_waist*1e6:.4g} μm"))
            self.results_table.setItem(3, 1, QTableWidgetItem(f"{mode_spacing/1e9:.4g} GHz"))
            self.results_table.setItem(4, 1, QTableWidgetItem(f"{linewidth*1e9:.4g} nm"))
            self.results_table.setItem(5, 1, QTableWidgetItem(f"{self.g1:.4g}"))
            self.results_table.setItem(6, 1, QTableWidgetItem(f"{self.g2:.4g}"))
            
            # 初始化透射率数组
            self.transmissions = np.zeros((self.l_max+1, resolution))
            
            # 计算不同OAM模式在不同腔长下的透射率
            t1 = 1 - reflectivity1
            t2 = 1 - reflectivity2
            
            for l in range(self.l_max+1):
                for i, delta_length in enumerate(self.cavity_lengths):
                    # 计算当前腔长
                    cavity_length = base_cavity_length + delta_length*1e-9  # 转换回m
                    
                    # 计算相位
                    if self.g1*self.g2 >= 0:
                        phi = np.arccos(np.sqrt(self.g1*self.g2))
                    else:
                        phi = np.arccos(-np.sqrt(abs(self.g1*self.g2)))
                    
                    # p = 0 (默认值)
                    p = 0
                    
                    # 计算相位差
                    delta = (4*np.pi/wavelength) * n * cavity_length + 2*(2*p + abs(l) + 1) * phi
                    
                    # 计算透射率
                    numerator = t1 * t2
                    denominator = (1 - np.sqrt(reflectivity1 * reflectivity2))**2 + \
                                4 * np.sqrt(reflectivity1 * reflectivity2) * (np.sin(delta/2))**2
                    
                    self.transmissions[l, i] = numerator / denominator
            
            # 更新模式选择器和复选框
            self.mode_selector.clear()
            for l in range(self.l_max+1):
                self.mode_selector.addItem(f"l = {l}")
                
            # 更新模式复选框
            for i, checkbox in enumerate(self.mode_checkboxes):
                if i <= self.l_max:
                    checkbox.setVisible(True)
                else:
                    checkbox.setVisible(False)
            
            # 绘制透射率曲线
            self.plot_transmission_curves()
            
            # 更新腔稳定性图
            self.plot_stability_diagram()
            
            # 绘制初始OAM模式分布
            self.update_mode_display()
            
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请检查输入值是否有效: {str(e)}")
    
    def plot_transmission_curves(self):
        """绘制透射率曲线"""
        if self.cavity_lengths is None or self.transmissions is None:
            return
            
        self.transmission_figure.clear()
        ax = self.transmission_figure.add_subplot(111)
        
        # 确定显示范围
        min_idx = int(self.plot_range[0] * len(self.cavity_lengths))
        max_idx = int(self.plot_range[1] * len(self.cavity_lengths))
        if min_idx == max_idx:
            max_idx = min_idx + 1
        
        x_display = self.cavity_lengths[min_idx:max_idx]
        
        # 创建颜色映射
        colors = plt.cm.jet(np.linspace(0, 1, self.l_max+1))
        
        # 绘制选中的OAM模式的透射率曲线
        for l in range(self.l_max+1):
            if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked():
                y_display = self.transmissions[l, min_idx:max_idx]
                ax.plot(x_display, y_display, 
                      label=f'l = {l}', color=colors[l], linewidth=1.5)
        
        ax.set_xlabel('腔长变化 (nm)')
        ax.set_ylabel('透射率')
        ax.set_title('不同OAM模式的透射率随腔长变化曲线')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # 优化显示
        ax.set_xlim(x_display.min(), x_display.max())
        y_min = np.min([self.transmissions[l, min_idx:max_idx].min() 
                      for l in range(self.l_max+1) 
                      if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked()])
        y_max = np.max([self.transmissions[l, min_idx:max_idx].max() 
                      for l in range(self.l_max+1) 
                      if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked()])
        margin = (y_max - y_min) * 0.05
        ax.set_ylim(max(0, y_min - margin), min(1, y_max + margin))
        
        self.transmission_figure.tight_layout()
        self.transmission_canvas.draw()
    
    def plot_stability_diagram(self):
        """绘制腔稳定性图"""
        self.stability_figure.clear()
        ax = self.stability_figure.add_subplot(111)
        
        # 创建g1-g2网格
        g_range = np.linspace(-2, 2, 500)
        G1, G2 = np.meshgrid(g_range, g_range)
        stability = G1 * G2
        
        # 绘制稳定区域（0 ≤ g1*g2 ≤ 1）
        cs = ax.contourf(G1, G2, stability, levels=[-np.inf, 0, 1, np.inf], 
                       colors=['red', 'lightgreen', 'red'], alpha=0.3)
        
        # 添加稳定性曲线
        ax.contour(G1, G2, stability, levels=[0, 1], colors='blue', linewidths=2)
        
        # 标记几种经典腔型
        special_cases = {
            '平行平面镜腔': (1, 1),
            '共焦腔': (0, 0),
            '半球腔': (1, 0),
            '球面对称腔': (0.5, 0.5),
            '同心腔': (-1, -1),
        }
        
        for name, (g1, g2) in special_cases.items():
            ax.plot(g1, g2, 'ko', markersize=6)
            ax.annotate(name, (g1, g2), xytext=(10, 5), textcoords='offset points')
        
        # 如果当前有计算过的参数，标记当前腔点
        if self.g1 is not None and self.g2 is not None:
            ax.plot(self.g1, self.g2, 'ro', markersize=8)
            stability_product = self.g1 * self.g2
            stability_status = "稳定" if 0 <= stability_product <= 1 else "不稳定"
            ax.annotate(f'当前腔点 (g1={self.g1:.3f}, g2={self.g2:.3f})\n稳定性: {stability_status}', 
                      (self.g1, self.g2), xytext=(20, 10), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                      arrowprops=dict(arrowstyle="->"))
        
        # 添加g1=0、g2=0轴线
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # 添加g1=g2对角线
        ax.plot([-2, 2], [-2, 2], 'k--', alpha=0.5)
        
        # 设置图表属性
        ax.set_xlabel('g1 = 1 - L/R1')
        ax.set_ylabel('g2 = 1 - L/R2')
        ax.set_title('腔稳定性图')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 添加稳定性解释
        legend_elements = [
            patches.Patch(facecolor='lightgreen', edgecolor='blue', alpha=0.3, label='稳定区域 (0 <= g1*g2 <= 1)'),
            patches.Patch(facecolor='red', edgecolor='blue', alpha=0.3, label='不稳定区域')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        self.stability_figure.tight_layout()
        self.stability_canvas.draw()
    
    def update_plot_range(self):
        """更新曲线显示范围"""
        if self.min_slider.value() < self.max_slider.value():
            self.plot_range = [self.min_slider.value()/100, self.max_slider.value()/100]
        else:
            self.min_slider.setValue(self.max_slider.value())
            self.plot_range = [self.max_slider.value()/100, self.max_slider.value()/100]
    
    def refresh_plot(self):
        """刷新曲线显示"""
        self.update_plot_range()
        self.plot_transmission_curves()
    
    def update_mode_display(self):
        """更新OAM模式空间分布显示"""
        if self.beam_waist is None:
            return
            
        # 获取当前选择的模式
        current_mode = self.mode_selector.currentIndex()
        
        # 清除图形
        self.mode_figure.clear()
        
        # 创建子图
        ax1 = self.mode_figure.add_subplot(121)
        ax2 = self.mode_figure.add_subplot(122, projection='polar')
        
        # 创建坐标网格
        x = np.linspace(-5*self.beam_waist*1e6, 5*self.beam_waist*1e6, 200)  # μm
        y = np.linspace(-5*self.beam_waist*1e6, 5*self.beam_waist*1e6, 200)  # μm
        X, Y = np.meshgrid(x, y)
        
        # 计算径向距离和角度
        R = np.sqrt(X**2 + Y**2)
        Theta = np.arctan2(Y, X)
        
        # 计算LG模式强度分布
        l = current_mode  # 当前模式的OAM值
        p = 0  # 径向指数，默认为0
        
        # 归一化半径
        rho = R / (self.beam_waist*1e6)
        
        # 计算相关联的拉盖尔多项式
        if p == 0 and l == 0:
            laguerre = np.ones_like(rho)
        else:
            from scipy.special import eval_genlaguerre
            # L_p^|l|(x) 是广义拉盖尔多项式
            laguerre = eval_genlaguerre(p, abs(l), 2*(rho**2))
        
        # 计算强度分布
        field = np.sqrt(2/np.pi) * np.exp(-rho**2) * rho**abs(l) * laguerre * np.exp(1j*l*Theta)
        intensity = np.abs(field)**2
        
        # 在直角坐标系中绘制强度分布
        im = ax1.pcolormesh(X, Y, intensity, cmap='inferno', shading='auto')
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        ax1.set_title(f'OAM模式 l={l} 强度分布')
        ax1.set_aspect('equal')
        self.mode_figure.colorbar(im, ax=ax1)
        
        # 在极坐标系中绘制相位分布
        phase = np.angle(field)
        ax2.pcolormesh(Theta, R, phase, cmap='hsv', shading='auto')
        ax2.set_title(f'OAM模式 l={l} 相位分布')
        
        self.mode_figure.tight_layout()
        self.mode_canvas.draw()
    
    def reset_params(self):
        """重置参数为默认值"""
        self.param_fields['wavelength'].setText("795")
        self.param_fields['cavity_length'].setText("10")
        self.param_fields['length_variation'].setText("900")
        self.param_fields['reflectivity1'].setText("0.9")
        self.param_fields['reflectivity2'].setText("0.9")
        self.param_fields['radius1'].setText("10")
        self.param_fields['radius2'].setText("10")
        self.param_fields['l_max'].setText("5")
        self.param_fields['resolution'].setValue(1000)
        
        # 重置曲线控制
        self.min_slider.setValue(0)
        self.max_slider.setValue(100)
        
        # 清除结果和图表
        for i in range(7):
            self.results_table.setItem(i, 1, QTableWidgetItem(""))
        
        self.transmission_figure.clear()
        self.transmission_canvas.draw()
        
        self.mode_figure.clear()
        self.mode_canvas.draw()
        
        # 重置存储的数据
        self.cavity_lengths = None
        self.transmissions = None
        self.beam_waist = None
        self.g1 = None
        self.g2 = None
        
        # 更新腔稳定性图
        self.plot_stability_diagram()
    
    def export_data(self):
        """导出计算数据为CSV文件"""
        if self.cavity_lengths is None or self.transmissions is None:
            QMessageBox.warning(self, "导出错误", "请先执行计算")
            return
        
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV文件 (*.csv);;所有文件 (*)", options=options)
        
        if filename:
            # 创建数据框
            data = {'腔长变化 (nm)': self.cavity_lengths}
            
            l_max = self.transmissions.shape[0]
            for l in range(l_max):
                data[f'透射率 (l={l})'] = self.transmissions[l]
            
            # 添加参数信息
            params_data = {
                '波长 (nm)': float(self.param_fields['wavelength'].text()),
                '基准腔长 (mm)': float(self.param_fields['cavity_length'].text()),
                '腔长变化范围 (nm)': float(self.param_fields['length_variation'].text()),
                '反射率1': float(self.param_fields['reflectivity1'].text()),
                '反射率2': float(self.param_fields['reflectivity2'].text()),
                '曲率半径1 (mm)': float(self.param_fields['radius1'].text()),
                '曲率半径2 (mm)': float(self.param_fields['radius2'].text()),
                'g因子1': self.g1 if self.g1 is not None else None,
                'g因子2': self.g2 if self.g2 is not None else None
            }
            
            # 保存数据
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            # 保存参数信息到单独的文件
            params_df = pd.DataFrame([params_data])
            params_filename = filename.replace('.csv', '_params.csv')
            params_df.to_csv(params_filename, index=False)
            
            QMessageBox.information(self, "导出成功", f"数据已保存到 {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FPResonatorSimulator()
    window.show()
    sys.exit(app.exec_()) 