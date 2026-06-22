# 题目二 PyCharm 可运行仿真项目

## 怎么运行

1. 用 PyCharm 打开这个文件夹：
   `题目二_PyCharm可运行仿真项目`

2. 在 PyCharm 终端安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行：

```bash
python main.py
```

运行后会自动处理 `data` 里的 6 张图片，并在 `results` 文件夹里生成结果图。

## 文件夹说明

- `data/forest`：森林场景，两张图
- `data/gravel`：砂石路面/林道场景，两张图
- `data/grassland`：草地/杂草场景，两张图
- `results`：运行后自动保存结果
- `main.py`：主程序
- `requirements.txt`：依赖库
- `图片来源.txt`：图片来源说明

## 结果怎么看

每张图片会生成一张分析图，包含：

- 原图
- HSV-H 色调通道
- 植被掩膜
- 砂石/土壤候选区域
- 障碍物候选区域
- 最终可通行区域结果

颜色含义：

- 绿色：可通行候选区域
- 黄色：植被/需谨慎区域
- 红色：障碍物或危险区域

如果你不想弹出窗口，只想保存结果，可以运行：

```bash
python main.py --no-show
```
