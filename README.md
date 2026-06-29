# 题目二 PyCharm 可运行仿真项目

## 怎么运行

1. 用 PyCharm 打开这个文件夹：
   `题目二_PyCharm可运行仿真项目`

2. 在 PyCharm 终端安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行图形界面：

```bash
python app.py
```

也可以直接运行：

```bash
python main.py
```

`main.py` 默认同样打开图形界面。

## 图形界面说明

界面只显示两张图：

- 原图
- 处理图片

导入图片或选择示例图片后，左边自动显示原图，右边等待检测按钮触发显示。左侧检测按钮包括：

- `地形区分`
- `HSV 色彩分类`
- `纹理特征提取`
- `植被/草地检测`
- `砂石路面检测`
- `障碍物检测`
- `危险区域剔除`
- `安全可通行区域`
- `最终处理结果`

点击哪个按钮，右边就显示哪个处理结果。例如点击 `障碍物检测`，右边就显示障碍物检测图。

界面上的标题、状态说明、图像标题等描述文字均居中显示。

## 批量仿真与结果分析

如果要一次处理所有示例图片并生成结果，可以运行：

```bash
python main.py --batch --no-show
```

如果只想生成植被遮挡、纹理干扰等多工况验证报告，可以运行：

```bash
python main.py --conditions
```

程序会输出：

- `results/summary.png`：所有示例图片最终结果总览
- `results/<类别>/*_analysis.png`：HSV、纹理、掩膜、最终结果等详细分析图
- `results/<类别>/*_final.png`：原图与最终可通行区域对比图
- `results/conditions/`：原始工况、植被遮挡工况、纹理干扰工况的对比图
- `results/condition_metrics.csv`：多工况量化指标
- `results/condition_analysis.txt`：多工况结果分析文字

## 文件夹说明

- `data/forest`：森林场景
- `data/gravel`：砂石路面/林道场景
- `data/grassland`：草地/杂草场景
- `results`：运行后自动保存结果
- `main.py`：图像处理与批量仿真主程序
- `app.py`：图形界面程序
- `requirements.txt`：依赖库
- `图片来源.txt`：图片来源说明

## 算法覆盖内容

- HSV 色彩像素分类：对整张图区分绿色植被、干草、低饱和砂石/土壤区域
- 纹理特征提取：使用 Laplacian 纹理强度识别高纹理干扰区域
- 形态学滤波：开运算去噪、闭运算补洞
- 连通域分析：过滤小面积噪声，保留连续目标区域
- 障碍物检测：结合暗区域、非地面边缘和高纹理区域进行危险区域剔除
- 安全区域划分：输出绿色半透明叠加的可通行区域
