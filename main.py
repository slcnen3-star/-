# -*- coding: utf-8 -*-
"""
题目二：自然复杂场景下安全地形可通行区域仿真

本程序使用传统图像处理方法，不需要训练深度学习模型。
核心思想是：根据颜色、纹理、边缘和区域面积，判断图像中哪些区域更可能安全可通行。

运行方式：
    1. 在 PyCharm 中打开本文件夹
    2. 在终端执行：pip install -r requirements.txt
    3. 运行 main.py

运行结果：
    - 每张输入图片会生成一张 2x3 详细分析图
    - 每张输入图片会生成一张“原图 + 最终结果”对比图
    - results/summary.png 是六张图片的最终效果总览

最终结果颜色说明：
    绿色：可通行候选区域
    黄色：植被或不确定区域，需要谨慎通行
    红色：障碍物或危险区域
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
except ImportError as exc:
    print("缺少依赖库：", exc)
    print("请先在 PyCharm 终端运行：pip install -r requirements.txt")
    raise SystemExit(1)


# 当前项目所在目录。
BASE_DIR = Path(__file__).resolve().parent

# 输入图片目录，里面分为 forest、gravel、grassland 三个类别。
DATA_DIR = BASE_DIR / "data"

# 结果输出目录，程序运行后会自动生成结果图片。
RESULT_DIR = BASE_DIR / "results"

# 三类测试场景。英文名对应文件夹名，中文名用于显示在结果图标题中。
CATEGORIES = {
    "forest": "森林",
    "gravel": "砂石路面",
    "grassland": "草地/杂草",
}


def setup_matplotlib_font() -> None:
    """设置 Matplotlib 中文字体，防止结果图中的中文标题显示成方框。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def imread_unicode(path: Path):
    """读取图片。

    OpenCV 的 cv2.imread 在部分 Windows 环境下读取中文路径会失败。
    所以这里使用 np.fromfile 先读取二进制数据，再用 cv2.imdecode 解码成图像。
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片：{path}")
    return img


def resize_by_width(img, width=900):
    """把图片按指定宽度缩放，并保持原始宽高比例。"""
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def clean_mask(mask, kernel_size=7, min_area=350):
    """清理二值掩膜图。

    掩膜图只有黑白两种值：
    - 白色：算法认为属于目标区域
    - 黑色：算法认为不属于目标区域

    处理步骤：开运算去噪、闭运算补洞、连通域分析过滤小区域。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 开运算 = 先腐蚀再膨胀，主要用于去除小噪点。
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 闭运算 = 先膨胀再腐蚀，主要用于填补目标区域中的小孔洞。
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 连通域分析会给每一块白色连通区域编号，并统计每块区域的面积。
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, num):
        # stats[i, cv2.CC_STAT_AREA] 是第 i 个连通区域的像素面积。
        # 小于 min_area 的区域通常是噪声，所以丢弃。
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned


def preprocess(bgr):
    """图像预处理。

    输出：
    - blur：降噪后的彩色图像
    - hsv：HSV 色彩空间图像，用于按颜色识别草地、砂石、土壤等
    - gray：灰度图，用于边缘检测和纹理分析
    """
    bgr = resize_by_width(bgr, width=900)

    # 高斯滤波用于平滑噪声，减少零散纹理对阈值判断的影响。
    blur = cv2.GaussianBlur(bgr, (5, 5), 0)

    # HSV 比 RGB 更适合做颜色分割。
    # H 表示色调，S 表示饱和度，V 表示明度。
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 灰度图用于 Canny 边缘检测和 Laplacian 纹理检测。
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    return blur, hsv, gray


def excess_green_mask(bgr):
    """使用 ExG（Excess Green，超绿指数）辅助识别植被。

    HSV 对绿色很好用，但遇到阴影、曝光变化时容易漏检。
    ExG = 2G - R - B，绿色越明显，ExG 越大。
    它可以补充 HSV，让草地/树叶识别更稳定。
    """
    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = 2 * g - r - b
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def adaptive_texture_mask(texture, roi):
    """根据当前图像自适应生成高纹理区域。

    原来直接用 Otsu 阈值，有时会把大面积草地都判为高纹理。
    这里改成只取 ROI 内纹理强度排名靠前的一部分，减少草地误判。
    """
    roi_values = texture[roi > 0]
    if roi_values.size == 0:
        threshold_value = 128
    else:
        # 取 78 分位数：只有纹理最复杂的约 22% 区域才算“高纹理”。
        threshold_value = np.percentile(roi_values, 78)
    return (texture > threshold_value).astype(np.uint8) * 255


def build_masks(bgr, hsv, gray, category):
    """根据颜色、纹理和边缘生成不同区域的掩膜。

    这是本程序的核心识别部分。它没有训练模型，而是用人工设定的规则：
    - 绿色 HSV 范围：识别草地、树叶、灌木等植被
    - ExG 超绿指数：补充识别受光照影响的绿色植被
    - 黄草 HSV 范围：补充识别干草、黄草、暖色草地
    - 低饱和度或棕黄色范围：识别砂石路、土路、裸露地面
    - Laplacian 纹理强度：判断区域是否纹理过于复杂
    - 暗区域 + Canny 边缘：提取障碍物候选区域
    """
    h, w = gray.shape

    # 拆分 HSV 三个通道，方便后面写阈值条件。
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # ROI 是感兴趣区域。图像上方通常是天空、远处树冠或背景，
    # 对“脚下区域是否能走”帮助较小，所以只分析下方 75%。
    roi = np.zeros((h, w), dtype=np.uint8)
    roi[int(h * 0.25):, :] = 255

    # near_ground_roi 是“近处地面约束”。
    # 前面的 roi 用来做颜色/纹理分析，范围可以稍大；但最终标成绿色的可通行区域，
    # 应该更偏向画面下方，因为远处山体、远处树林虽然颜色相似，却不是机器人当前能走到的区域。
    near_ground_roi = np.zeros((h, w), dtype=np.uint8)
    near_start_ratio = 0.36 if category == "grassland" else 0.45
    near_ground_roi[int(h * near_start_ratio):, :] = 255

    # 绿色植被识别：这个范围识别草地、树叶、灌木等绿色区域。
    green_hsv = cv2.inRange(hsv, np.array([28, 25, 30]), np.array([96, 255, 255]))

    # ExG 作为补充，能识别一些 HSV 阈值下容易漏掉的绿色区域。
    exg = excess_green_mask(bgr)

    # 黄草/干草识别：很多草地并不是鲜绿色，而是黄色、棕黄色或偏暖色。
    # 这是对原算法的主要改进之一。
    yellow_grass = (
        (h_ch >= 10) & (h_ch <= 45) & (s_ch >= 25) & (s_ch <= 230) & (v_ch >= 55) & (v_ch <= 250)
    ).astype(np.uint8) * 255

    # vegetation 主要表示绿色植被。
    vegetation = cv2.bitwise_or(green_hsv, exg)
    vegetation = cv2.bitwise_and(vegetation, roi)
    vegetation = clean_mask(vegetation, kernel_size=5, min_area=450)

    # grass_candidate 表示“草地候选”，包括绿色草地和黄草/干草。
    # 这个掩膜专门用于草地场景，避免只认绿色、不认黄草。
    grass_candidate = cv2.bitwise_or(vegetation, yellow_grass)
    grass_candidate = cv2.bitwise_and(grass_candidate, roi)
    grass_candidate = clean_mask(grass_candidate, kernel_size=7, min_area=650)

    # 砂石/土壤识别规则 1：低饱和度区域。
    # 砂石路、灰色路面、泥土路常常颜色不鲜艳，所以 S 较低。
    low_saturation_surface = ((s_ch < 105) & (v_ch > 45) & (v_ch < 240)).astype(np.uint8) * 255

    # 砂石/土壤识别规则 2：棕黄色区域。
    brown_soil_surface = (
        (h_ch >= 4) & (h_ch <= 35) & (s_ch >= 25) & (s_ch <= 185) & (v_ch >= 40) & (v_ch <= 235)
    ).astype(np.uint8) * 255

    # 合并两类地面候选区域。
    gravel_or_soil = cv2.bitwise_or(low_saturation_surface, brown_soil_surface)
    gravel_or_soil = cv2.bitwise_and(gravel_or_soil, roi)

    # 纹理检测：Laplacian 可以检测局部灰度变化。
    # 灰度变化越强，说明纹理越复杂。
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = cv2.convertScaleAbs(lap)
    texture = cv2.GaussianBlur(texture, (9, 9), 0)
    high_texture = adaptive_texture_mask(texture, roi)

    # 暗区域可能对应树干、石块、深阴影等障碍物。
    dark = ((v_ch < 55) & (s_ch < 210)).astype(np.uint8) * 255

    # Canny 边缘检测用于提取物体边界，例如石块边缘、树干边界、路边界。
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    # 草地场景中，花草边缘非常多，如果直接把边缘当障碍物，会出现大片红色误检。
    # 所以草地场景要把 grass_candidate 也作为“地面候选”排除掉。
    if category == "grassland":
        surface_for_edge = cv2.bitwise_or(gravel_or_soil, grass_candidate)
    else:
        surface_for_edge = gravel_or_soil

    non_surface_edge = cv2.bitwise_and(edges, cv2.bitwise_not(surface_for_edge))

    # 障碍物候选 = 非地面边缘 + 暗区域。
    obstacle = cv2.bitwise_or(non_surface_edge, dark)
    obstacle = cv2.bitwise_and(obstacle, roi)

    # 对草地场景再做一次保护：已经被识别为草地候选的区域，不轻易标红为障碍物。
    if category == "grassland":
        obstacle = cv2.bitwise_and(obstacle, cv2.bitwise_not(grass_candidate))

    obstacle = clean_mask(obstacle, kernel_size=5, min_area=500)

    # 不同场景下，“可通行”的判断标准稍微不同。
    if category == "grassland":
        # 草地场景：草地候选区域中，纹理不是特别复杂的部分标为绿色。
        # 高纹理草丛、花草混杂区域标为黄色，表示需要谨慎。
        safe_candidate = cv2.bitwise_and(grass_candidate, cv2.bitwise_not(high_texture))
        caution_candidate = cv2.bitwise_and(grass_candidate, high_texture)
    else:
        # 森林和砂石路场景：砂石/土壤区域更像道路或可行走地面。
        safe_candidate = gravel_or_soil
        caution_candidate = vegetation

    # 最终可通行区域必须满足：位于 ROI 内、属于候选可通行区域、不是障碍物区域。
    safe = cv2.bitwise_and(safe_candidate, roi)
    safe = cv2.bitwise_and(safe, near_ground_roi)
    safe = cv2.bitwise_and(safe, cv2.bitwise_not(obstacle))
    safe = clean_mask(safe, kernel_size=9, min_area=700)

    # 谨慎区域：不能和 safe 或 obstacle 重叠。
    caution = cv2.bitwise_and(caution_candidate, roi)
    caution = cv2.bitwise_and(caution, cv2.bitwise_not(safe))
    caution = cv2.bitwise_and(caution, cv2.bitwise_not(obstacle))
    caution = clean_mask(caution, kernel_size=7, min_area=500)

    return {
        "roi": roi,
        "vegetation": vegetation,
        "grass_candidate": grass_candidate,
        "yellow_grass": yellow_grass,
        "gravel_or_soil": gravel_or_soil,
        "texture": texture,
        "high_texture": high_texture,
        "obstacle": obstacle,
        "safe": safe,
        "caution": caution,
    }


def overlay_result(bgr, safe, caution, obstacle):
    """把识别结果叠加到原图上，生成最终可视化结果。

    颜色说明：
    - 绿色：可通行区域
    - 黄色：谨慎通行区域
    - 红色：障碍物或危险区域
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = rgb.copy()

    green = np.array([0, 220, 90], dtype=np.uint8)
    yellow = np.array([255, 205, 45], dtype=np.uint8)
    red = np.array([245, 50, 50], dtype=np.uint8)

    # alpha 混合：不是直接覆盖颜色，而是把结果颜色半透明叠加到原图上。
    out[caution > 0] = (0.62 * out[caution > 0] + 0.38 * yellow).astype(np.uint8)
    out[safe > 0] = (0.55 * out[safe > 0] + 0.45 * green).astype(np.uint8)
    out[obstacle > 0] = (0.52 * out[obstacle > 0] + 0.48 * red).astype(np.uint8)
    return out


def analyze_image(image_path: Path, category: str):
    """处理单张图片，并返回中间结果与最终叠加图。"""
    bgr = imread_unicode(image_path)
    bgr, hsv, gray = preprocess(bgr)
    masks = build_masks(bgr, hsv, gray, category)
    overlay = overlay_result(bgr, masks["safe"], masks["caution"], masks["obstacle"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return hsv, masks, overlay, rgb


def save_analysis_figure(image_path: Path, category: str, label: str) -> tuple[Path, Path]:
    """保存单张图片的两种结果图。

    1. *_analysis.png：2x3 详细分析图，用来给老师解释算法过程。
    2. *_final.png：原图 + 最终结果对比图，用来快速查看识别效果。
    """
    hsv, masks, overlay, rgb = analyze_image(image_path, category)

    category_result_dir = RESULT_DIR / category
    category_result_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = category_result_dir / f"{image_path.stem}_analysis.png"
    final_path = category_result_dir / f"{image_path.stem}_final.png"

    h_channel = hsv[:, :, 0]

    # 草地场景显示“草地候选掩膜”，其他场景显示“植被掩膜”。
    # 这样结果图更符合当前场景重点。
    if category == "grassland":
        vegetation_panel_title = "草地候选掩膜"
        vegetation_panel = masks["grass_candidate"]
    else:
        vegetation_panel_title = "植被掩膜"
        vegetation_panel = masks["vegetation"]

    panels = [
        ("原图", rgb, None),
        ("HSV-H 色调通道", h_channel, "gray"),
        (vegetation_panel_title, vegetation_panel, "gray"),
        ("砂石/土壤候选", masks["gravel_or_soil"], "gray"),
        ("障碍物候选", masks["obstacle"], "gray"),
        ("最终可通行结果", overlay, None),
    ]

    # 详细分析图：展示每一步中间结果，方便答辩时讲清楚算法流程。
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"{label} - {image_path.name}", fontsize=15)
    for ax, (title, img, cmap) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.text(
        0.5,
        0.02,
        "颜色说明：绿色=可通行候选区域；黄色=植被/需谨慎区域；红色=障碍物或危险区域。",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(analysis_path, dpi=160)
    plt.close(fig)

    # 最终对比图：只显示原图和最终结果，便于快速观察识别效果。
    final_fig, final_axes = plt.subplots(1, 2, figsize=(10, 4.8))
    final_fig.suptitle(f"{label} - {image_path.name}", fontsize=14)
    final_axes[0].imshow(rgb)
    final_axes[0].set_title("原图", fontsize=11)
    final_axes[0].axis("off")
    final_axes[1].imshow(overlay)
    final_axes[1].set_title("最终可通行区域", fontsize=11)
    final_axes[1].axis("off")
    final_fig.text(
        0.5,
        0.02,
        "绿色=可通行；黄色=谨慎通行；红色=障碍/危险。",
        ha="center",
        fontsize=10,
    )
    final_fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    final_fig.savefig(final_path, dpi=170)
    plt.close(final_fig)
    return analysis_path, final_path


def build_summary(result_paths: list[tuple[str, Path]]) -> Path:
    """把六张图片的最终结果汇总成一张总览图。"""
    out_path = RESULT_DIR / "summary.png"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle("题目二：安全地形可通行区域仿真结果总览", fontsize=17)
    for ax, (title, path) in zip(axes.ravel(), result_paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    return out_path


def collect_images() -> list[tuple[str, str, Path]]:
    """收集 data 目录下所有待处理图片。"""
    items: list[tuple[str, str, Path]] = []
    for category, label in CATEGORIES.items():
        folder = DATA_DIR / category
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((category, label, path))
    return items


def main():
    """程序入口：收集图片、逐张处理、保存结果。"""
    parser = argparse.ArgumentParser(description="题目二：安全地形可通行区域仿真")
    parser.add_argument("--no-show", action="store_true", help="只保存结果，不弹出图像窗口")
    args = parser.parse_args()

    setup_matplotlib_font()
    RESULT_DIR.mkdir(exist_ok=True)

    images = collect_images()
    if not images:
        print("没有找到图片，请检查 data/forest、data/gravel、data/grassland 目录。")
        return

    # result_paths 用来保存每张最终结果图路径，最后生成 summary.png。
    result_paths: list[tuple[str, Path]] = []
    for category, label, image_path in images:
        analysis_path, final_path = save_analysis_figure(image_path, category, label)
        result_paths.append((f"{label}: {image_path.name}", final_path))
        print(f"已处理：{image_path.name}")
        print(f"  详细分析图：{analysis_path}")
        print(f"  最终结果图：{final_path}")

    summary_path = build_summary(result_paths)
    print(f"\n全部完成。总览图：{summary_path}")
    print(f"单张分析图和最终结果图保存在：{RESULT_DIR}")

    # 默认会弹出结果窗口；如果命令行加 --no-show，则只保存不弹窗。
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()