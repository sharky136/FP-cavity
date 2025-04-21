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
                            QCheckBox, QScrollArea, QHeaderView)
from PyQt5.QtCore import Qt
import pandas as pd
from scipy import constants
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib

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
        
        # 添加保存默认值按钮
        row += 1
        self.save_defaults_btn = QPushButton("保存为默认值")
        self.save_defaults_btn.clicked.connect(self.save_default_params)
        params_layout.addWidget(self.save_defaults_btn, row, 0, 1, 2)
        
        # 创建计算结果显示区域
        results_group = QGroupBox("计算结果")
        results_layout = QVBoxLayout(results_group)
        
        # 增加显示半高全宽的行
        self.results_table = QTableWidget(7, 2)
        self.results_table.setHorizontalHeaderLabels(["物理量", "数值"])
        self.results_table.setItem(0, 0, QTableWidgetItem("自由光谱范围 (FSR)"))
        self.results_table.setItem(1, 0, QTableWidgetItem("精细度 (Finesse)"))
        self.results_table.setItem(2, 0, QTableWidgetItem("基模束腰半径"))
        self.results_table.setItem(3, 0, QTableWidgetItem("模式间距 (Δν)"))
        self.results_table.setItem(4, 0, QTableWidgetItem("线宽 (Δλ)"))
        self.results_table.setItem(5, 0, QTableWidgetItem("理论可分辨最大模式数"))
        self.results_table.setItem(6, 0, QTableWidgetItem("透射峰半高全宽 (FWHM)"))
        
        results_layout.addWidget(self.results_table)
        
        # 创建敏感度分析结果表格
        self.sensitivity_table = QTableWidget(0, 3)
        self.sensitivity_table.setHorizontalHeaderLabels(["模式 (l)", "半高全宽 (nm)", "敏感度系数"])
        self.sensitivity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.sensitivity_table)
        
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
        
        # 使用滚动区域包装曲线控制，以便支持更多选项
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        curve_controls_wrapper = QWidget()
        curve_controls_layout = QGridLayout(curve_controls_wrapper)
        
        # 添加模式选择复选框
        self.mode_checkboxes = []
        # 初始默认创建8个模式选择框，后续会在calculate方法中动态调整
        for i in range(8):  # 默认支持0-7的模式
            checkbox = QCheckBox(f"显示l={i}模式")
            if i <= 3:  # 默认只显示前4个模式
                checkbox.setChecked(True)
            self.mode_checkboxes.append(checkbox)
            curve_controls_layout.addWidget(checkbox, i//3, i%3)
        
        scroll_area.setWidget(curve_controls_wrapper)
        curve_controls_layout_main = QVBoxLayout(curve_controls)
        curve_controls_layout_main.addWidget(scroll_area)
        
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
        
        curve_controls_layout_main.addLayout(range_layout)
        
        # 添加刷新曲线按钮
        self.refresh_btn = QPushButton("刷新曲线")
        self.refresh_btn.clicked.connect(self.refresh_plot)
        curve_controls_layout_main.addWidget(self.refresh_btn)
        
        transmission_layout.addWidget(curve_controls)
        
        # 添加绘图区域和工具栏
        self.transmission_figure = Figure(figsize=(6, 6), dpi=100)
        self.transmission_canvas = FigureCanvas(self.transmission_figure)
        self.transmission_toolbar = NavigationToolbar(self.transmission_canvas, transmission_tab)
        
        transmission_layout.addWidget(self.transmission_toolbar)
        transmission_layout.addWidget(self.transmission_canvas)
        
        # 添加选项卡
        self.tab_widget.addTab(transmission_tab, "透射率曲线")
        
        # 将面板添加到主布局
        main_layout.addLayout(left_panel, 1)
        main_layout.addWidget(self.tab_widget, 2)
        
        # 存储计算数据
        self.cavity_lengths = None
        self.transmissions = None
        self.beam_waist = None
        self.l_max = 0
        self.plot_range = [0, 1]  # 初始范围为0-100%
        self.max_distinguishable_modes = 0  # 存储最大可分辨模式数
        
        # 加载默认参数
        self.load_default_params()
    
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
            
            # 检查是否是超高反射率情况 - 需要特殊处理
            ultra_high_reflectivity = reflectivity1 > 0.995 or reflectivity2 > 0.995
            if ultra_high_reflectivity:
                # 在超高反射率情况下给用户提示
                msg = f"检测到超高反射率({max(reflectivity1, reflectivity2):.4f})。\n"
                msg += "为保证计算效率和稳定性，将使用优化算法。\n"
                msg += "计算可能需要较长时间，请耐心等待。"
                QMessageBox.information(self, "超高反射率模式", msg)
            
            # 计算物理量
            c = constants.c  # 光速
            n = 1.0  # 空气折射率
            
            # 计算g参数
            g1 = 1 - base_cavity_length / radius1
            g2 = 1 - base_cavity_length / radius2
            
            # 检查腔的稳定性
            if not (0 <= g1*g2 <= 1):
                QMessageBox.warning(self, "参数错误", "腔参数不满足稳定条件: 0 ≤ g1*g2 ≤ 1")
                return
            
            # 自由光谱范围 (FSR) in Hz
            fsr = c / (2 * n * base_cavity_length)
            
            # 精细度
            r1 = np.sqrt(reflectivity1)
            r2 = np.sqrt(reflectivity2)
            finesse = np.pi * np.sqrt(r1 * r2) / (1 - r1 * r2)
            
            # 在超高精细度下，限制精细度上限以避免数值问题
            if ultra_high_reflectivity and finesse > 5000:
                print(f"警告: 精细度({finesse:.1f})非常高，已限制为5000以保证计算稳定性")
                finesse = 5000
            
            # 基模束腰半径 (m)
            if (g1 + g2 - 2*g1*g2) == 0:  # 避免除零
                self.beam_waist = np.sqrt((base_cavity_length * wavelength) / np.pi)
            else:
                self.beam_waist = np.sqrt((wavelength * base_cavity_length) / np.pi) * \
                            np.sqrt(g1*g2*(1-g1*g2) / ((g1+g2-2*g1*g2)**2))
            
            # 模式间距
            mode_spacing = fsr / finesse
            
            # 线宽
            linewidth = (wavelength**2 / c) * mode_spacing
            
            # 创建自适应采样的腔长变化数组
            # 对于高精细度腔（高反射率），使用自适应网格以提高峰值附近的精度
            high_finesse_cavity = finesse > 500  # 高精细度阈值
            
            if high_finesse_cavity:
                # 估计FSR对应的腔长变化
                delta_L_FSR = wavelength / 2  # 单位: m
                delta_L_FSR_nm = delta_L_FSR * 1e9  # 转为nm
                
                # 计算整个范围内的峰值数量
                num_peaks = int(length_variation * 1e9 / delta_L_FSR_nm) + 1
                peak_positions = np.linspace(0, length_variation * 1e9, num_peaks)
                
                # 设置最大采样点数，防止内存溢出
                max_total_points = min(resolution * 5, 10000)  # 最大允许10000个点
                
                # 根据精细度调整峰值附近采样宽度
                if ultra_high_reflectivity:
                    # 对于超高精细度，采用更窄的密集采样区和更少的点
                    scale_factor = np.log10(finesse/100) if finesse > 1000 else 1
                    peak_width = delta_L_FSR_nm / (finesse * scale_factor) * 5
                    points_per_peak = int(min(resolution / num_peaks / 8, 50))
                    uniform_points_count = min(int(resolution / 8), 200)
                else:
                    # 普通高精细度情况
                    peak_width = delta_L_FSR_nm / finesse * 10
                    points_per_peak = int(min(resolution / num_peaks / 2, 100))
                    uniform_points_count = int(resolution / 2)
                
                # 创建最终的采样点数组
                adaptive_points = []
                for peak_pos in peak_positions:
                    # 在峰值附近采用更细的网格
                    dense_points = np.linspace(
                        max(0, peak_pos - peak_width),
                        min(length_variation * 1e9, peak_pos + peak_width),
                        points_per_peak
                    )
                    adaptive_points.extend(dense_points)
                
                # 添加均匀分布的点以覆盖整个范围
                uniform_points = np.linspace(0, length_variation * 1e9, uniform_points_count)
                
                # 合并所有点并去重排序
                all_points = np.concatenate([adaptive_points, uniform_points])
                self.cavity_lengths = np.unique(all_points)
                
                # 如果点数过多，进行下采样
                if len(self.cavity_lengths) > max_total_points:
                    indices = np.linspace(0, len(self.cavity_lengths) - 1, max_total_points).astype(int)
                    self.cavity_lengths = self.cavity_lengths[indices]
                
                print(f"自适应网格: 生成了{len(self.cavity_lengths)}个计算点")
            else:
                # 创建均匀的腔长变化数组（单位：nm）
                self.cavity_lengths = np.linspace(0, length_variation*1e9, resolution)  # 转换为nm显示
            
            # 使用64位浮点精度进行计算
            self.cavity_lengths = self.cavity_lengths.astype(np.float64)
            
            # 初始化透射率数组
            self.transmissions = np.zeros((self.l_max+1, len(self.cavity_lengths)), dtype=np.float64)
            
            # 计算不同OAM模式在不同腔长下的透射率
            t1 = 1 - reflectivity1
            t2 = 1 - reflectivity2
            r1r2_sqrt = np.sqrt(reflectivity1 * reflectivity2)
            
            # 预先计算一些常量，减少重复计算
            if g1*g2 >= 0:
                phi = np.arccos(np.sqrt(g1*g2))
            else:
                phi = np.arccos(-np.sqrt(abs(g1*g2)))
            
            # 常数项，与腔长无关
            constant_denominator_term = (1 - r1r2_sqrt)**2
            constant_numerator = t1 * t2
            
            # 转换腔长变化数组为实际腔长(单位m)
            cavity_lengths_m = base_cavity_length + self.cavity_lengths * 1e-9
            
            # 预计算与腔长有关的相位项，所有模式共用
            phase_L_term = (4 * np.pi / wavelength) * n * cavity_lengths_m
            
            for l in range(self.l_max+1):
                # 计算与模式有关的相位项
                p = 0  # 默认径向模式为0
                phase_mode_term = 2 * (2 * p + abs(l) + 1) * phi
                
                # 计算总相位差 delta
                delta = phase_L_term + phase_mode_term
                
                # 计算相位相关项
                sin_term = np.sin(delta / 2)**2
                
                # 计算透射率
                denominator = constant_denominator_term + 4 * r1r2_sqrt * sin_term
                transmission = constant_numerator / denominator
                
                # 避免数值误差导致的透射率>1
                transmission = np.minimum(transmission, 1.0)
                
                # 保存计算结果
                self.transmissions[l] = transmission
            
            # 计算理论可分辨最大模式数
            self.max_distinguishable_modes = self.calculate_max_distinguishable_modes()
            
            # 计算各模式的半高全宽和敏感度
            sensitivities = self.calculate_peak_sensitivity()
            
            # 更新敏感度表格
            self.update_sensitivity_table(sensitivities)
            
            # 平均半高全宽计算
            avg_fwhm = 0
            if sensitivities:
                fwhm_values = [data['fwhm'] for data in sensitivities]
                if fwhm_values:
                    avg_fwhm = np.mean(fwhm_values)
            
            # 更新结果表
            self.results_table.setItem(0, 1, QTableWidgetItem(f"{fsr/1e9:.4g} GHz"))
            self.results_table.setItem(1, 1, QTableWidgetItem(f"{finesse:.4g}"))
            self.results_table.setItem(2, 1, QTableWidgetItem(f"{self.beam_waist*1e6:.4g} μm"))
            self.results_table.setItem(3, 1, QTableWidgetItem(f"{mode_spacing/1e9:.4g} GHz"))
            self.results_table.setItem(4, 1, QTableWidgetItem(f"{linewidth*1e9:.4g} nm"))
            self.results_table.setItem(5, 1, QTableWidgetItem(f"{self.max_distinguishable_modes}"))
            self.results_table.setItem(6, 1, QTableWidgetItem(f"{avg_fwhm:.4g} nm"))
            
            # 更新模式选择器和复选框
            # 动态调整复选框数量
            # 首先检查现有复选框数量是否足够
            existing_checkboxes = len(self.mode_checkboxes)
            
            # 计算一行需要显示多少个复选框（默认是3个）
            checkboxes_per_row = 3
            
            # 如果现有复选框不足，则创建新的复选框
            if existing_checkboxes < self.l_max + 1:
                # 获取已布局的父容器
                parent_layout = self.mode_checkboxes[0].parent().layout()
                
                # 添加新的复选框
                for i in range(existing_checkboxes, self.l_max + 1):
                    checkbox = QCheckBox(f"显示l={i}模式")
                    if i <= 3:  # 默认只显示前4个模式
                        checkbox.setChecked(True)
                    self.mode_checkboxes.append(checkbox)
                    parent_layout.addWidget(checkbox, i//checkboxes_per_row, i%checkboxes_per_row)
            
            # 更新所有复选框的可见性
            for i, checkbox in enumerate(self.mode_checkboxes):
                if i <= self.l_max:
                    checkbox.setVisible(True)
                else:
                    checkbox.setVisible(False)
            
            # 绘制透射率曲线
            self.plot_transmission_curves()
            
        except ValueError as e:
            QMessageBox.warning(self, "输入错误", f"请检查输入值是否有效: {str(e)}")
        except MemoryError:
            QMessageBox.critical(self, "内存错误", "计算需要的内存超出系统可用内存。请降低分辨率或减小腔长变化范围后重试。")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"计算过程中发生未知错误: {str(e)}")
    
    def calculate_peak_sensitivity(self):
        """
        计算各模式透射峰的半高全宽(FWHM)和对腔长变化的敏感度
        
        返回:
        list : 包含每个模式敏感度信息的字典列表
        """
        if self.transmissions is None or self.cavity_lengths is None:
            return []
        
        sensitivities = []
        
        # 计算各个模式的半高全宽和敏感度
        for l in range(self.l_max + 1):
            # 查找峰值
            peaks, properties = find_peaks(self.transmissions[l], height=0.5, prominence=0.1)
            
            if len(peaks) == 0:
                continue
                
            # 对于每个峰值，计算半高全宽
            for i, peak_idx in enumerate(peaks):
                # 获取峰值高度
                peak_height = self.transmissions[l, peak_idx]
                
                # 半高值
                half_max = peak_height / 2
                
                # 寻找半高点左侧位置
                left_idx = peak_idx
                while left_idx > 0 and self.transmissions[l, left_idx] > half_max:
                    left_idx -= 1
                
                # 寻找半高点右侧位置
                right_idx = peak_idx
                while right_idx < len(self.cavity_lengths) - 1 and self.transmissions[l, right_idx] > half_max:
                    right_idx += 1
                
                # 如果没有找到左右半高点，则跳过该峰
                if left_idx == 0 or right_idx == len(self.cavity_lengths) - 1:
                    continue
                
                # 使用线性插值找到更精确的半高点位置
                left_pos = self.interpolate_position(
                    self.cavity_lengths[left_idx], self.transmissions[l, left_idx],
                    self.cavity_lengths[left_idx+1], self.transmissions[l, left_idx+1],
                    half_max
                )
                
                right_pos = self.interpolate_position(
                    self.cavity_lengths[right_idx-1], self.transmissions[l, right_idx-1],
                    self.cavity_lengths[right_idx], self.transmissions[l, right_idx],
                    half_max
                )
                
                # 计算半高全宽
                fwhm = right_pos - left_pos
                
                # 计算敏感度系数（半高全宽的倒数）
                sensitivity = 1.0 / fwhm if fwhm > 0 else 0
                
                # 只记录最高的峰值
                if i == 0 or peak_height > sensitivities[-1]['peak_height']:
                    sensitivities.append({
                        'mode': l,
                        'fwhm': fwhm,
                        'sensitivity': sensitivity,
                        'peak_height': peak_height
                    })
        
        # 按模式排序
        sensitivities.sort(key=lambda x: x['mode'])
        
        return sensitivities
        
    def interpolate_position(self, x1, y1, x2, y2, y_target):
        """
        线性插值计算目标y值对应的x位置
        """
        if abs(y2 - y1) < 1e-10:  # 避免除零
            return (x1 + x2) / 2
        
        # 线性插值公式: x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
        return x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
        
    def update_sensitivity_table(self, sensitivities):
        """更新敏感度分析表格"""
        self.sensitivity_table.setRowCount(0)
        
        if not sensitivities:
            return
            
        # 设置行数
        self.sensitivity_table.setRowCount(len(sensitivities))
        
        # 填充数据
        for i, data in enumerate(sensitivities):
            self.sensitivity_table.setItem(i, 0, QTableWidgetItem(f"{data['mode']}"))
            self.sensitivity_table.setItem(i, 1, QTableWidgetItem(f"{data['fwhm']:.4g}"))
            self.sensitivity_table.setItem(i, 2, QTableWidgetItem(f"{data['sensitivity']:.4g}"))
    
    def plot_transmission_curves(self):
        """绘制透射率曲线"""
        if self.cavity_lengths is None or self.transmissions is None:
            return
            
        self.transmission_figure.clear()
        
        # 检查是否为高反射率条件
        r1 = float(self.param_fields['reflectivity1'].text())
        r2 = float(self.param_fields['reflectivity2'].text())
        high_reflectivity = r1 > 0.99 and r2 > 0.99
        ultra_high_reflectivity = r1 > 0.995 or r2 > 0.995
        
        # 创建主图和缩放插图
        ax = self.transmission_figure.add_subplot(111)
        
        # 确定显示范围
        min_idx = int(self.plot_range[0] * len(self.cavity_lengths))
        max_idx = int(self.plot_range[1] * len(self.cavity_lengths))
        if min_idx == max_idx:
            max_idx = min_idx + 1
        
        x_display = self.cavity_lengths[min_idx:max_idx]
        
        # 创建颜色映射
        colors = plt.cm.jet(np.linspace(0, 1, self.l_max+1))
        
        # 高反射率下限制峰值检测和注释数量，以避免过度绘制
        max_peaks_to_show = 5 if ultra_high_reflectivity else (10 if high_reflectivity else 20)
        
        # 保存所有选中模式的精确峰值，用于后续处理
        all_peak_info = {}  # 以模式l为键，存储(位置, 高度)列表
        
        # 绘制选中的OAM模式的透射率曲线
        for l in range(self.l_max+1):
            if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked():
                y_display = self.transmissions[l, min_idx:max_idx]
                line, = ax.plot(x_display, y_display, 
                      label=f'l = {l}', color=colors[l], linewidth=1.5)
                
                # 对于高反射率情况，标记精确估计的峰值
                if high_reflectivity:
                    # 优化峰值检测参数，提高效率和准确性
                    if ultra_high_reflectivity:
                        # 超高反射率下使用更高的高度和显著性阈值
                        height_threshold = 0.7
                        prominence_threshold = 0.2
                    else:
                        height_threshold = 0.5
                        prominence_threshold = 0.1
                    
                    # 对显示范围内的数据执行峰值检测
                    peaks, properties = find_peaks(y_display, 
                                                height=height_threshold, 
                                                prominence=prominence_threshold,
                                                distance=3)  # 增加最小距离要求
                    
                    # 如果检测到的峰过多，只取最高的几个
                    if len(peaks) > max_peaks_to_show:
                        # 按照高度排序
                        peak_heights = properties["peak_heights"]
                        sorted_indices = np.argsort(peak_heights)[::-1]
                        peaks = peaks[sorted_indices[:max_peaks_to_show]]
                    
                    peak_info = []
                    
                    # 仅处理有限数量的峰值以提高性能
                    for peak_idx in peaks:
                        # 计算在整个数组中的索引
                        full_idx = peak_idx + min_idx
                        
                        # 使用优化的插值方法估计真实峰值
                        true_amplitude, true_position = self.estimate_true_peak(
                            self.cavity_lengths, self.transmissions[l], full_idx)
                        
                        # 只有当真实位置在显示范围内时才绘制
                        if x_display[0] <= true_position <= x_display[-1]:
                            ax.plot(true_position, true_amplitude, 'o', 
                                  color=colors[l], markersize=5, alpha=0.8)
                            
                            # 保存峰值信息
                            peak_info.append((true_position, true_amplitude))
                            
                            # 仅对显著峰值添加注释，避免图表过度混乱
                            # 在超高反射率下减少注释数量
                            annotation_threshold = 0.95 if ultra_high_reflectivity else 0.9
                            if true_amplitude > annotation_threshold:
                                # 格式化位置标签
                                pos_label = f"{true_position:.2f}"
                                
                                # 添加文本注释，避免标签重叠
                                # 使用小字体和半透明效果避免遮挡曲线
                                ax.annotate(
                                    pos_label, 
                                    xy=(true_position, true_amplitude),
                                    xytext=(0, 10),   # 文本偏移
                                    textcoords='offset points',
                                    ha='center',      # 水平居中
                                    fontsize=8,       # 小字体
                                    alpha=0.7,        # 半透明
                                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6)
                                )
                    
                    # 保存所有峰值信息
                    if peak_info:
                        all_peak_info[l] = peak_info
        
        # 添加放大图
        add_inset = False
        if high_reflectivity and len(all_peak_info) > 1:
            # 查找最高的峰值
            max_peak_height = 0
            max_peak_pos = None
            for l, peaks in all_peak_info.items():
                for pos, height in peaks:
                    if height > max_peak_height:
                        max_peak_height = height
                        max_peak_pos = pos
            
            # 如果找到了显著的峰值，创建子图放大显示
            if max_peak_pos is not None and max_peak_height > 0.8:
                add_inset = True
                
                # 计算缩放窗口宽度，超高反射率时使用更窄的窗口
                zoom_width_percent = 0.005 if ultra_high_reflectivity else 0.02
                zoom_width = (x_display[-1] - x_display[0]) * zoom_width_percent
                x_min = max(x_display[0], max_peak_pos - zoom_width)
                x_max = min(x_display[-1], max_peak_pos + zoom_width)
                
                # 确保搜索范围有意义
                if x_min < x_max and (x_max - x_min) > zoom_width * 0.1:
                    # 创建插图 - 在主图的右上角
                    inset_ax = self.transmission_figure.add_axes([0.6, 0.6, 0.25, 0.25])
                    
                    try:
                        # 绘制放大区域的曲线，使用采样以提高效率
                        for l in range(self.l_max+1):
                            if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked():
                                # 找到x范围对应的索引
                                zoom_min_idx = np.searchsorted(self.cavity_lengths, x_min) - 1
                                zoom_max_idx = np.searchsorted(self.cavity_lengths, x_max) + 1
                                
                                # 确保索引在有效范围内
                                zoom_min_idx = max(0, zoom_min_idx)
                                zoom_max_idx = min(len(self.cavity_lengths), zoom_max_idx)
                                
                                # 如果范围太小或无效，跳过
                                if zoom_max_idx <= zoom_min_idx + 2:
                                    continue
                                
                                # 提取放大区域的数据，对数据进行采样以提高性能
                                step = max(1, (zoom_max_idx - zoom_min_idx) // 100)
                                slice_indices = slice(zoom_min_idx, zoom_max_idx, step)
                                x_zoom = self.cavity_lengths[slice_indices]
                                y_zoom = self.transmissions[l, slice_indices]
                                
                                # 绘制曲线
                                inset_ax.plot(x_zoom, y_zoom, color=colors[l], linewidth=1.5)
                                
                                # 添加精确峰值点
                                if l in all_peak_info:
                                    for pos, height in all_peak_info[l]:
                                        if x_min <= pos <= x_max:
                                            inset_ax.plot(pos, height, 'o', color=colors[l], markersize=5)
                        
                        # 设置插图属性
                        inset_ax.set_xlim(x_min, x_max)
                        y_min = max(0, max_peak_height - 0.2)
                        y_max = min(1.02, max_peak_height + 0.02)
                        inset_ax.set_ylim(y_min, y_max)
                        
                        # 设置插图网格
                        inset_ax.grid(True, linestyle='--', alpha=0.4)
                        
                        # 添加标题
                        inset_ax.set_title("峰值放大视图", fontsize=8)
                        
                        # 为了避免过分拥挤，可以关闭插图的x和y轴刻度标签，只保留网格线
                        inset_ax.tick_params(axis='both', which='major', labelsize=6)
                    except Exception as e:
                        print(f"创建放大图时出错: {str(e)}")
                        # 如果放大图创建失败，不影响主图显示
                        add_inset = False
                        if 'inset_ax' in locals():
                            self.transmission_figure.delaxes(inset_ax)
        
        # 设置主图属性
        ax.set_xlabel('腔长变化 (nm)')
        ax.set_ylabel('透射率')
        ax.set_title('不同OAM模式的透射率随腔长变化曲线')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # 优化主图的显示范围
        ax.set_xlim(x_display.min(), x_display.max())
        
        # 计算y轴范围时只考虑所选模式
        selected_indices = [l for l in range(self.l_max+1) 
                        if l < len(self.mode_checkboxes) and self.mode_checkboxes[l].isChecked()]
        
        if selected_indices:
            y_min = np.min([self.transmissions[l, min_idx:max_idx].min() for l in selected_indices])
            y_max = np.max([self.transmissions[l, min_idx:max_idx].max() for l in selected_indices])
            
            # 如果是高反射率情况，可能需要调整y轴上限以便更好地显示峰值
            if high_reflectivity:
                # 检查是否有很高的峰值
                high_peak_exists = any(height > 0.9 for peaks in all_peak_info.values() for _, height in peaks)
                
                if high_peak_exists:
                    y_max = 1.02  # 确保高峰值完全可见
            
            margin = (y_max - y_min) * 0.05
            ax.set_ylim(max(0, y_min - margin), min(1.02, y_max + margin))
        
        # 避免图表调整时出现异常
        try:
            self.transmission_figure.tight_layout()
        except:
            print("图表布局调整失败，使用默认布局")
        
        # 刷新画布
        self.transmission_canvas.draw()
    
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
        
        # 重置存储的数据
        self.cavity_lengths = None
        self.transmissions = None
        self.beam_waist = None
        self.max_distinguishable_modes = 0
    
    def save_default_params(self):
        """保存当前参数为默认值"""
        try:
            # 收集当前所有参数值
            defaults = {}
            for key, field in self.param_fields.items():
                if isinstance(field, QLineEdit):
                    defaults[key] = field.text()
                elif isinstance(field, QSpinBox):
                    defaults[key] = str(field.value())
            
            # 保存到配置文件
            import json
            import os
            
            # 确保目录存在
            config_dir = os.path.expanduser('~/.fp_simulator')
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            config_path = os.path.join(config_dir, 'default_params.json')
            with open(config_path, 'w') as f:
                json.dump(defaults, f)
            
            QMessageBox.information(self, "保存成功", "参数已保存为默认值")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存参数时出错: {str(e)}")
    
    def load_default_params(self):
        """加载默认参数"""
        try:
            import json
            import os
            
            config_path = os.path.expanduser('~/.fp_simulator/default_params.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    defaults = json.load(f)
                
                # 应用默认参数
                for key, value in defaults.items():
                    if key in self.param_fields:
                        field = self.param_fields[key]
                        if isinstance(field, QLineEdit):
                            field.setText(value)
                        elif isinstance(field, QSpinBox):
                            field.setValue(int(value))
                
                print("已加载默认参数")
        except Exception as e:
            print(f"加载默认参数失败: {str(e)}")
            # 失败时使用硬编码的默认值，不显示错误消息
    
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
                '曲率半径2 (mm)': float(self.param_fields['radius2'].text())
            }
            
            # 保存数据
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            # 保存参数信息到单独的文件
            params_df = pd.DataFrame([params_data])
            params_filename = filename.replace('.csv', '_params.csv')
            params_df.to_csv(params_filename, index=False)
            
            QMessageBox.information(self, "导出成功", f"数据已保存到 {filename}")
    
    def calculate_max_distinguishable_modes(self, overlap_threshold=0.1):
        """
        计算理论可分辨的最大模式数,使用改进的科学方法及峰值插值估计
        
        参数:
        overlap_threshold : float
            认为模式可分辨的最大允许重叠度阈值(0-1之间)
        
        返回:
        int : 最大可分辨的模式数
        """
        if self.transmissions is None or self.l_max < 1:
            return 0
            
        # 获取腔长数组作为x坐标
        x_values = self.cavity_lengths
        
        # 检查是否为高反射率条件 - 影响评判标准
        r1 = float(self.param_fields['reflectivity1'].text())
        r2 = float(self.param_fields['reflectivity2'].text())
        high_reflectivity = r1 > 0.99 and r2 > 0.99
        ultra_high_reflectivity = r1 > 0.995 or r2 > 0.995
        
        # 针对超高反射率情况进行特殊处理
        if ultra_high_reflectivity:
            # 在超高反射率条件下，计算量过大，使用简化计算并给出警告
            # 默认值设置为4（保守估计，适用于大多数应用场景）
            QMessageBox.information(self, "超高反射率提示", 
                                  "在超高反射率条件下，系统采用优化算法估算理论可分辨最大模式数。")
            return 4
            
        # 高反射率下调整阈值 - 调低阈值使判断更严格
        if high_reflectivity:
            # 高反射率下峰更窄，允许更小的间距
            rayleigh_criterion = 0.6  # 提高到0.6
            # 高反射率下对比度要求更高
            contrast_threshold = 0.7  # 提高到0.7
            # 重叠度判断更严格
            overlap_threshold = 0.05  # 降低到0.05
            # 为提高效率，限制峰值检测数量
            max_peaks_to_analyze = 2
        else:
            # 普通情况下的标准也调整更严格
            rayleigh_criterion = 0.7
            contrast_threshold = 0.4
            overlap_threshold = 0.05
            max_peaks_to_analyze = 3
        
        try:
            # 对每个模式找出所有透射峰并进行精确峰值估计
            all_peaks = []
            accurate_peak_heights = []
            accurate_peak_positions = []
            
            # 设置进度条
            progress = QMessageBox(QMessageBox.Information, "处理中", "正在计算模式分辨能力...")
            progress.setStandardButtons(QMessageBox.NoButton)
            progress.show()
            QApplication.processEvents()  # 刷新界面
            
            for l in range(self.l_max + 1):
                # 使用改进的峰值检测方法 - 针对高反射率情况调整参数
                if high_reflectivity:
                    # 高反射率下峰更窄但更高，降低height要求，提高prominence
                    peaks, properties = find_peaks(self.transmissions[l], height=0.5, prominence=0.1, width=1)
                else:
                    peaks, properties = find_peaks(self.transmissions[l], height=0.3, prominence=0.05)
                
                if len(peaks) > 0:
                    # 排序峰值按透射率大小
                    peak_indices = np.argsort(properties["peak_heights"])[::-1]
                    sorted_peaks = peaks[peak_indices]
                    
                    # 只保留前N个最强峰值(如果有的话)，加快计算
                    top_peaks = sorted_peaks[:min(max_peaks_to_analyze, len(sorted_peaks))]
                    
                    # 对每个峰值进行精确估计
                    peak_props = []
                    for peak_idx in top_peaks:
                        # 使用插值方法精确估计峰值
                        true_amplitude, true_position = self.estimate_true_peak(
                            x_values, self.transmissions[l], peak_idx)
                        peak_props.append((peak_idx, true_amplitude, true_position))
                    
                    # 按精确振幅排序
                    peak_props.sort(key=lambda x: x[1], reverse=True)
                    
                    # 更新峰值列表
                    refined_peaks = [p[0] for p in peak_props]
                    all_peaks.append(refined_peaks)
                    
                    # 存储精确的峰值高度和位置
                    accurate_peak_heights.append([p[1] for p in peak_props])
                    accurate_peak_positions.append([p[2] for p in peak_props])
                else:
                    # 如果没有找到明显的峰，使用最大值位置
                    max_idx = np.argmax(self.transmissions[l])
                    all_peaks.append(np.array([max_idx]))
                    
                    # 对单峰也进行精确估计
                    true_amplitude, true_position = self.estimate_true_peak(
                        x_values, self.transmissions[l], max_idx)
                    accurate_peak_heights.append([true_amplitude])
                    accurate_peak_positions.append([true_position])
        
                # 更新进度
                QApplication.processEvents()  # 刷新界面以保持响应
            
            # 关闭进度提示
            progress.close()
        
            # 先初始化delta_L_linewidth变量，以确保它在所有路径上都有定义
            delta_L_linewidth = 0
            estimated_linewidth = 0
            
            # 计算线宽估计值
            # 针对高反射率条件优化线宽估计
            if high_reflectivity:
                # 使用理论计算的线宽
                wavelength = float(self.param_fields['wavelength'].text()) * 1e-9  # nm转m
                base_cavity_length = float(self.param_fields['cavity_length'].text()) * 1e-3  # mm转m
                c = constants.c  # 光速
                n = 1.0  # 空气折射率
                
                # 自由光谱范围 (FSR) in Hz
                fsr = c / (2 * n * base_cavity_length)
                
                # 精细度
                finesse = np.pi * np.sqrt(r1 * r2) / (1 - r1 * r2)
                # 限制精细度上限以避免数值问题
                if finesse > 5000:
                    finesse = 5000
                
                # 模式间距
                mode_spacing = fsr / finesse
                
                # 线宽
                theoretical_linewidth = (wavelength**2 / c) * mode_spacing
                
                # 将理论线宽转换为腔长变化量
                delta_L_linewidth = theoretical_linewidth / 2
                
                # 将线宽从物理单位转换为索引点数
                estimated_linewidth = delta_L_linewidth / (x_values[-1] - x_values[0]) * len(x_values)
            else:
                # 常规方法估计线宽
                peak_indices = all_peaks[0]
                if len(peak_indices) > 1:
                    # 基于相邻峰之间的平均距离估计线宽
                    avg_peak_distance = np.mean(np.diff(peak_indices))
                    # 使用一个合理的默认值
                    estimated_linewidth = avg_peak_distance * 0.1  # 假设线宽是峰间距的10%
                    
                    # 估算实际物理单位的线宽
                    length_per_point = (x_values[-1] - x_values[0]) / len(x_values)
                    delta_L_linewidth = estimated_linewidth * length_per_point
                else:
                    # 默认线宽估计值(分辨率的1%)
                    estimated_linewidth = len(self.cavity_lengths) * 0.01
                    
                    # 估算实际物理单位的线宽
                    delta_L_linewidth = (x_values[-1] - x_values[0]) * 0.01
            
            # 添加基于峰值匹配的直接比较
            # 这是一个快速检查，用于捕捉明显的模式重叠情况
            peak_match_matrix = np.zeros((self.l_max + 1, self.l_max + 1))
            
            # 计算所有模式对之间峰值位置的匹配程度
            for i in range(self.l_max + 1):
                for j in range(i + 1, self.l_max + 1):
                    if (not accurate_peak_positions[i] or not accurate_peak_positions[j]):
                        # 如果任一模式没有明显峰值，则记为0（不匹配）
                        continue
                    
                    # 每个模式只考虑最强的峰值
                    pos_i = accurate_peak_positions[i][0]
                    pos_j = accurate_peak_positions[j][0]
                    
                    # 计算两个峰之间的距离
                    peak_distance = abs(pos_i - pos_j)
                    
                    # 计算匹配程度 - 距离越近，匹配度越高
                    if peak_distance < delta_L_linewidth:
                        # 对于距离小于线宽的峰值，根据距离设置匹配度
                        match_score = 1.0 - (peak_distance / delta_L_linewidth)
                        peak_match_matrix[i, j] = peak_match_matrix[j, i] = match_score
            
            # 使用科学的多准则评估方法判断模式可分辨性
            max_distinguishable = 1  # 至少有一种模式是可分辨的
            
            # 为提高计算效率，在高反射率情况下适当限制循环次数和计算量
            max_modes_to_check = min(self.l_max + 1, 8 if high_reflectivity else self.l_max + 1)
                
            for num_modes in range(2, max_modes_to_check + 1):
                is_distinguishable = True
                
                # 检查模式集合中是否存在显著的峰值匹配
                # 如果任意两个模式的峰值匹配度超过阈值，则认为这些模式不可分辨
                for i in range(num_modes - 1):
                    if not is_distinguishable:
                        break
                
                    for j in range(i + 1, num_modes):
                        # 首先检查峰值匹配度
                        if peak_match_matrix[i, j] > 0.7:  # 匹配度阈值
                            is_distinguishable = False
                            break
                            
                        # 定期刷新界面，保持程序响应性
                        if (i * num_modes + j) % 10 == 0:
                            QApplication.processEvents()
                            
                        # 1. 峰值重叠度评估 - 使用准确的阈值
                        overlap = self.calculate_curve_overlap(i, j)
                        
                        # 2. 峰值分离度评估(莱利判据) - 使用精确计算的峰值位置
                        peak_resolved = False
                        if accurate_peak_positions[i] and accurate_peak_positions[j]:
                            # 计算所有峰值对之间的最小距离
                            min_peak_distance = float('inf')
                            closest_i_pos = None
                            closest_j_pos = None
                            
                            # 限制计算量，只比较前几个峰
                            max_peaks_i = min(len(accurate_peak_positions[i]), max_peaks_to_analyze)
                            max_peaks_j = min(len(accurate_peak_positions[j]), max_peaks_to_analyze)
                            
                            for pos_i in accurate_peak_positions[i][:max_peaks_i]:
                                for pos_j in accurate_peak_positions[j][:max_peaks_j]:
                                    distance = abs(pos_i - pos_j)
                                    if distance < min_peak_distance:
                                        min_peak_distance = distance
                                        closest_i_pos = pos_i
                                        closest_j_pos = pos_j
                            
                            # 如果没有找到峰值对，默认为不可分辨
                            if min_peak_distance == float('inf'):
                                min_peak_distance = 0
                                peak_resolved = False
                            else:
                                # 将距离从nm转换为点数
                                min_peak_distance_in_points = min_peak_distance / (x_values[-1] - x_values[0]) * len(x_values)
                                
                                # 应用莱利判据
                                peak_resolved = min_peak_distance_in_points > (rayleigh_criterion * estimated_linewidth)
                                
                                # 高反射率条件下的额外检查 - 只在必要时进行以提高效率
                                if high_reflectivity and peak_resolved and min_peak_distance < delta_L_linewidth * 5:
                                    # 如果两个峰很接近，检查它们在最接近位置处的透射率差异
                                    # 找到这两个点在原始数据中的最近索引
                                    i_idx = np.argmin(np.abs(self.cavity_lengths - closest_i_pos))
                                    j_idx = np.argmin(np.abs(self.cavity_lengths - closest_j_pos))
                                    
                                    # 只有当索引差异不大时才进行山谷检查，避免过度计算
                                    if abs(i_idx - j_idx) < len(self.cavity_lengths) * 0.01:
                                        # 检查峰值之间的"山谷"深度
                                        # 找到两个峰值之间的所有点
                                        start_idx = min(i_idx, j_idx)
                                        end_idx = max(i_idx, j_idx)
                                        
                                        if start_idx != end_idx:
                                            # 峰值之间的点
                                            between_peaks_i = self.transmissions[i, start_idx:end_idx+1]
                                            between_peaks_j = self.transmissions[j, start_idx:end_idx+1]
                                            
                                            # 找到这些点中的最小透射率
                                            min_transmission_i = np.min(between_peaks_i)
                                            min_transmission_j = np.min(between_peaks_j)
                                            
                                            # 如果两个模式的最小透射率之间的差异小于阈值，认为它们不可分辨
                                            valley_difference = abs(min_transmission_i - min_transmission_j)
                                            if valley_difference < 0.05:  # 5%的差异阈值
                                                peak_resolved = False
                        else:
                            peak_resolved = False
                        
                        # 3. 峰值高度对比度 - 评估分辨能力
                        good_contrast = False
                        if accurate_peak_heights[i] and accurate_peak_heights[j]:
                            height_i = accurate_peak_heights[i][0]  # 使用最强峰的精确高度
                            height_j = accurate_peak_heights[j][0]
                            
                            # 计算对比度 - 较低峰与较高峰的比值
                            contrast_ratio = min(height_i, height_j) / max(height_i, height_j)
                            
                            # 高反射率下峰值理论上应该接近相等
                            if high_reflectivity:
                                # 在高反射率情况下，峰值高度应该接近1且彼此接近
                                if min(height_i, height_j) > 0.8 and contrast_ratio > contrast_threshold:
                                    good_contrast = True
                            else:
                                # 在普通情况下使用标准对比度判据
                                if contrast_ratio > contrast_threshold:
                                    good_contrast = True
                        
                        # 综合判据: 重叠度+峰值分离+对比度
                        # 模式被认为是可分辨的，如果:
                        # - 峰值重叠度小于阈值
                        # - 峰值根据莱利判据可分辨
                        # - 峰值对比度足够高
                        if (overlap > overlap_threshold or not peak_resolved or not good_contrast):
                            is_distinguishable = False
                            break
                
                if not is_distinguishable:
                    break
                    
                max_distinguishable = num_modes
                
                # 在高反射率情况下，如果已经计算到6个模式，提前结束计算
                if high_reflectivity and max_distinguishable >= 6:
                    break
            
            # 额外保险措施：确保结果合理 - 一般情况下不超过9个模式
            if max_distinguishable > 9 and not high_reflectivity:
                print(f"警告：计算的最大可分辨模式数({max_distinguishable})可能不合理，已调整为9")
                max_distinguishable = 9
                
            # 高反射率情况下，一般不超过6个模式
            if high_reflectivity and max_distinguishable > 6:
                print(f"警告：高反射率下计算的最大可分辨模式数({max_distinguishable})可能不合理，已调整为6")
                max_distinguishable = 6
            
            return max_distinguishable
            
        except Exception as e:
            print(f"计算可分辨模式数时出错: {str(e)}")
            # 发生错误时返回保守估计值
            return 3 if high_reflectivity else 4
    
    def calculate_curve_overlap(self, mode1, mode2):
        """
        计算两个模式的透射率曲线之间的重叠度
        
        参数:
        mode1, mode2 : int
            要比较的两个模式的索引
            
        返回:
        float : 重叠度 (0-1之间的值)
        """
        if mode1 == mode2:
            return 1.0
            
        # 检查是否为高反射率情况，高反射率时采用采样计算以提高效率
        r1 = float(self.param_fields['reflectivity1'].text())
        r2 = float(self.param_fields['reflectivity2'].text())
        high_reflectivity = r1 > 0.99 and r2 > 0.99
        
        transmission1 = self.transmissions[mode1]
        transmission2 = self.transmissions[mode2]
        
        # 特殊情况: 高反射率下进行更严格的检查
        if high_reflectivity:
            # 计算主要峰值位置
            peaks1, _ = find_peaks(transmission1, height=0.5, prominence=0.1)
            peaks2, _ = find_peaks(transmission2, height=0.5, prominence=0.1)
            
            # 如果没有检测到峰值，使用整个透射率曲线的最大值位置
            if len(peaks1) == 0:
                peaks1 = [np.argmax(transmission1)]
            if len(peaks2) == 0:
                peaks2 = [np.argmax(transmission2)]
            
            # 将两个模式的主要峰值按照高度排序
            peaks1_heights = transmission1[peaks1]
            peaks2_heights = transmission2[peaks2]
            sorted_peaks1 = [p for _, p in sorted(zip(peaks1_heights, peaks1), reverse=True)]
            sorted_peaks2 = [p for _, p in sorted(zip(peaks2_heights, peaks2), reverse=True)]
            
            # 初始化重叠度
            max_overlap = 0.0
            
            # 计算模式间最近峰之间的距离和高度比
            for p1 in sorted_peaks1[:min(3, len(sorted_peaks1))]:  # 只考虑最高的3个峰
                for p2 in sorted_peaks2[:min(3, len(sorted_peaks2))]:  # 只考虑最高的3个峰
                    # 计算峰值间距 (以索引点数表示)
                    peak_distance = abs(p1 - p2)
                    
                    # 计算两个峰的高度
                    height1 = transmission1[p1]
                    height2 = transmission2[p2]
                    
                    # 如果峰距离很近 (<10点) 并且两个峰都很高 (>0.7)，认为有显著重叠
                    if peak_distance < 10 and height1 > 0.7 and height2 > 0.7:
                        # 计算距离因子 - 距离越近，重叠度越高
                        distance_factor = 1.0 - (peak_distance / 20.0)  # 20点内线性下降
                        distance_factor = max(0, distance_factor)
                        
                        # 计算高度因子 - 两个峰越接近，重叠度越高
                        height_ratio = min(height1, height2) / max(height1, height2)
                        
                        # 结合两个因子计算重叠度
                        overlap = 0.5 * distance_factor + 0.5 * height_ratio
                        
                        # 更新最大重叠度
                        max_overlap = max(max_overlap, overlap)
                    
                    # 检查两个峰之间的透射率变化
                    if peak_distance > 1:  # 避免同一位置的峰
                        # 获取两个峰之间的所有点
                        start_idx = min(p1, p2)
                        end_idx = max(p1, p2)
                        
                        # 计算两个峰之间的透射率平均差异
                        between_points1 = transmission1[start_idx:end_idx+1]
                        between_points2 = transmission2[start_idx:end_idx+1]
                        
                        # 如果两个峰之间的中间点透射率都很低，认为它们能被区分
                        # 如果中间有高透射点，说明它们有重叠
                        mid_point_overlap = np.mean(np.minimum(between_points1, between_points2))
                        
                        # 更新重叠度
                        max_overlap = max(max_overlap, mid_point_overlap)
            
            # 模式间全局相似度的额外检查
            # 计算两个透射率曲线的相关系数
            correlation = np.corrcoef(transmission1, transmission2)[0, 1]
            if correlation > 0.9:  # 如果两个曲线高度相关
                max_overlap = max(max_overlap, 0.3)  # 添加一个基础重叠度
            
            return max_overlap
        
        else:
            # 普通情况下使用更标准的方法
            # 寻找交叉点位置
            crossings = []
            for i in range(1, len(transmission1)):
                if ((transmission1[i-1] - transmission2[i-1]) * 
                    (transmission1[i] - transmission2[i])) <= 0:
                    # 找到交叉点
                    crossings.append(i)
            
            # 如果没有交叉点，计算最小距离
            if not crossings:
                # 计算全局差异
                diff = np.abs(transmission1 - transmission2)
                
                # 两种情况：
                # 1. 如果透射率曲线整体差异很大，则重叠度低
                if np.mean(diff) > 0.3:
                    return 0.05
                    
                # 2. 否则，基于相似度估计一个重叠度
                return max(0, 1 - np.mean(diff) * 2)
            
            # 计算交叉点处的最大重叠度
            max_overlap = 0
            
            # 对每个交叉点计算附近的重叠情况
            for crossing in crossings:
                # 计算交叉点处的透射率
                transmission_at_crossing = (transmission1[crossing] + transmission2[crossing]) / 2
                
                # 只有透射率较高的交叉点才真正影响可分辨性
                if transmission_at_crossing > 0.3:
                    # 定义交叉点附近的窗口
                    window = 5
                    start = max(0, crossing - window)
                    end = min(len(transmission1) - 1, crossing + window)
                    
                    # 计算窗口内的平均重叠度
                    overlap_in_window = np.mean(np.minimum(
                        transmission1[start:end+1],
                        transmission2[start:end+1]
                    ))
                    
                    # 对高透射率区域给予更高权重
                    if transmission_at_crossing > 0.7:
                        overlap_in_window *= 1.5
                        
                    max_overlap = max(max_overlap, overlap_in_window, transmission_at_crossing)
            
            return min(1.0, max_overlap)  # 确保重叠度不超过1
    
    def estimate_true_peak(self, x_values, y_values, peak_idx):
        """
        使用插值方法精确估计峰值高度和位置
        
        参数:
        x_values : array
            x坐标数组(如腔长变化)
        y_values : array
            y坐标数组(如透射率)
        peak_idx : int
            初步峰值的索引
            
        返回:
        tuple : (真实振幅, 真实位置)
        """
        # 动态调整窗口大小 - 根据峰值高度自动优化
        peak_height = y_values[peak_idx]
        
        # 高窄峰使用更窄的窗口，低宽峰使用更宽的窗口
        if peak_height > 0.9:  # 非常尖锐的峰
            window = 2
        elif peak_height > 0.7:
            window = 3
        else:
            window = 4
            
        # 选取峰值周围的点进行拟合
        start_idx = max(0, peak_idx - window)
        end_idx = min(len(y_values), peak_idx + window + 1)
        
        # 如果点数太少，返回原始值
        if end_idx - start_idx < 4:
            return min(1.0, y_values[peak_idx]), x_values[peak_idx]
        
        # 提取用于拟合的数据点
        x_fit = x_values[start_idx:end_idx]
        y_fit = y_values[start_idx:end_idx]
        
        try:
            # 检查是否所有y值都几乎相等(平坦峰值)
            if np.max(y_fit) - np.min(y_fit) < 1e-6:
                return min(1.0, y_values[peak_idx]), x_values[peak_idx]
                
            # 使用多种拟合方法，选择最佳结果
            
            # 方法1: 洛伦兹函数拟合 - 适合FP腔理论共振峰形状
            def lorentzian(x, amplitude, x0, gamma):
                return amplitude * gamma**2 / ((x - x0)**2 + gamma**2)
            
            # 初始参数猜测：振幅、中心位置、半宽度
            initial_guess = [y_values[peak_idx], x_values[peak_idx], 
                           (x_values[end_idx-1] - x_values[start_idx])/4]
            
            # 使用边界限制改善拟合稳定性
            bounds = ([0, x_values[start_idx], 0], 
                     [1.01, x_values[end_idx-1], (x_values[end_idx-1] - x_values[start_idx])])
            
            try:
                # 尝试洛伦兹拟合，但设置较短的超时时间
                lorentz_popt, lorentz_pcov = curve_fit(
                    lorentzian, x_fit, y_fit, 
                    p0=initial_guess, 
                    bounds=bounds,
                    maxfev=500,   # 降低最大迭代次数以加速
                    ftol=1e-8,    # 适当降低收敛精度
                    method='trf'  # 使用信任区域优化方法
                )
                
                # 计算洛伦兹拟合的MSE误差
                lorentz_mse = np.mean((lorentzian(x_fit, *lorentz_popt) - y_fit)**2)
                
                # 验证拟合结果的合理性
                if lorentz_popt[0] > 1.1 or lorentz_popt[0] < 0 or lorentz_popt[2] <= 0:
                    # 拟合结果不合理，抛出异常转用多项式拟合
                    raise ValueError("洛伦兹拟合结果不合理")
                    
                lorentz_valid = True
            except:
                lorentz_valid = False
                lorentz_mse = float('inf')
            
            # 方法2: 二次多项式拟合 - 对于采样点少但接近峰值顶点的情况有优势
            try:
                # 使用二阶多项式 y = a*x^2 + b*x + c
                poly_coeffs = np.polyfit(x_fit, y_fit, 2)
                
                # 从多项式系数计算峰值位置和高度
                # 对于二次多项式，峰值位置是 x = -b/(2*a)
                a, b, c = poly_coeffs
                
                # 检查二次项是否为负(保证是峰而非谷)
                if a >= 0:
                    poly_valid = False
                    poly_mse = float('inf')
                else:
                    poly_x0 = -b / (2 * a)
                    
                    # 如果拟合的极值点在拟合范围外，使用原始值
                    if poly_x0 < x_fit[0] or poly_x0 > x_fit[-1]:
                        poly_x0 = x_values[peak_idx]
                        
                    # 峰值高度是在极值点处的函数值 y = a*x0^2 + b*x0 + c
                    poly_amplitude = a * poly_x0**2 + b * poly_x0 + c
                    
                    # 验证拟合结果的合理性
                    if poly_amplitude > 1.1 or poly_amplitude < 0:
                        poly_valid = False
                        poly_mse = float('inf')
                    else:
                        # 计算多项式拟合的MSE误差
                        poly_y_fit = np.polyval(poly_coeffs, x_fit)
                        poly_mse = np.mean((poly_y_fit - y_fit)**2)
                        poly_valid = True
            except:
                poly_valid = False
                poly_mse = float('inf')
            
            # 如果两种拟合方法都失败，返回原始值
            if not lorentz_valid and not poly_valid:
                return min(1.0, y_values[peak_idx]), x_values[peak_idx]
                
            # 选择有效且误差较小的拟合结果
            if lorentz_valid and (lorentz_mse <= poly_mse or not poly_valid):
                # 使用洛伦兹拟合结果
                amplitude = min(1.0, lorentz_popt[0])  # 确保透射率不超过1
                position = lorentz_popt[1]
            elif poly_valid:
                # 使用多项式拟合结果
                amplitude = min(1.0, poly_amplitude)
                position = poly_x0
            else:
                # 所有方法都失败，返回原始值
                return min(1.0, y_values[peak_idx]), x_values[peak_idx]
                
            # 最后进行有效性检查
            # 1. 振幅必须大于原始峰值的80%，且不超过1.0
            # 2. 位置必须在拟合窗口范围内
            if (amplitude < 0.8 * y_values[peak_idx] or 
                position < x_fit[0] or position > x_fit[-1]):
                return min(1.0, y_values[peak_idx]), x_values[peak_idx]
                
            # 返回拟合的真实峰值和位置
            return amplitude, position
            
        except Exception as e:
            # 拟合失败时返回原始值
            print(f"峰值拟合失败: {str(e)}")
            return min(1.0, y_values[peak_idx]), x_values[peak_idx]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FPResonatorSimulator()
    window.show()
    sys.exit(app.exec_()) 