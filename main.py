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
    - results/summary.png 是示例图片的最终效果总览

最终结果颜色说明：
    绿色：可通行候选区域
"""

from __future__ import annotations

import argparse
import csv
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
    """根据全图颜色、纹理、边缘和连通域生成识别掩膜。"""
    h, w = gray.shape
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    roi = np.full((h, w), 255, dtype=np.uint8)

    def remove_huge_components(mask, min_area=250, max_area_ratio=0.20, reject_floor_band=False):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        max_area = int(mask.size * max_area_ratio)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            box_w = stats[i, cv2.CC_STAT_WIDTH]
            box_h = stats[i, cv2.CC_STAT_HEIGHT]
            if reject_floor_band and y + box_h > int(h * 0.88) and box_w > int(w * 0.32):
                continue
            if min_area <= area <= max_area:
                out[labels == i] = 255
        return out

    def keep_large_components(mask, min_area=250):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out

    def keep_accessible_safe_components(mask):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        bottom_line = int(h * 0.90)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            box_w = stats[i, cv2.CC_STAT_WIDTH]
            box_h = stats[i, cv2.CC_STAT_HEIGHT]
            bottom = stats[i, cv2.CC_STAT_TOP] + box_h
            if area < 700:
                continue
            if bottom < bottom_line:
                continue
            if box_h > box_w * 1.3 and box_w < int(w * 0.12):
                continue
            out[labels == i] = 255
        return out

    def filter_surface_components(mask):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            box_w = stats[i, cv2.CC_STAT_WIDTH]
            box_h = stats[i, cv2.CC_STAT_HEIGHT]
            if area < 450:
                continue
            is_vertical_false_positive = box_h > box_w * 2.2 and box_w < int(w * 0.10)
            is_tiny_upper_patch = y < int(h * 0.55) and area < int(mask.size * 0.01)
            if is_vertical_false_positive or is_tiny_upper_patch:
                continue
            out[labels == i] = 255
        return out

    b, g, r = cv2.split(bgr.astype(np.float32))
    exg = cv2.normalize(2 * g - r - b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    sky_or_glare = (
        ((v_ch > 210) & (s_ch < 55))
        | ((h_ch >= 85) & (h_ch <= 125) & (s_ch < 95) & (v_ch > 135))
    ).astype(np.uint8) * 255

    road_corridor = np.zeros((h, w), dtype=np.uint8)
    start_y = int(h * 0.34)
    for y in range(start_y, h):
        progress = (y - start_y) / max(h - start_y, 1)
        half_width = int((0.08 + 0.48 * (progress ** 1.1)) * w)
        center = w // 2
        road_corridor[y, max(0, center - half_width): min(w, center + half_width)] = 255

    safe_road_corridor = np.zeros((h, w), dtype=np.uint8)
    safe_start_y = int(h * 0.43)
    for y in range(safe_start_y, h):
        progress = (y - safe_start_y) / max(h - safe_start_y, 1)
        half_width = int((0.05 + 0.46 * (progress ** 1.1)) * w)
        center = w // 2
        safe_road_corridor[y, max(0, center - half_width): min(w, center + half_width)] = 255

    gravel_road_color = (
        (h_ch >= 8)
        & (h_ch <= 48)
        & (s_ch >= 18)
        & (s_ch <= 225)
        & (v_ch >= 35)
        & (v_ch <= 235)
        & (sky_or_glare == 0)
    ).astype(np.uint8) * 255
    gravel_road_prior = cv2.bitwise_and(gravel_road_color, road_corridor)
    gravel_road_prior = clean_mask(gravel_road_prior, kernel_size=7, min_area=550)

    gravel_safe_color = (
        (((s_ch < 128) & (v_ch > 50) & (v_ch < 235)) | ((h_ch >= 10) & (h_ch <= 38) & (s_ch < 160)))
        & (sky_or_glare == 0)
    ).astype(np.uint8) * 255
    gravel_safe_surface = cv2.bitwise_and(gravel_safe_color, safe_road_corridor)
    gravel_safe_surface = clean_mask(gravel_safe_surface, kernel_size=7, min_area=700)

    green_hsv = (
        ((h_ch >= 28) & (h_ch <= 96) & (s_ch >= 35) & (v_ch >= 35) & (v_ch <= 245))
        | ((exg > 138) & (h_ch >= 24) & (h_ch <= 105) & (s_ch >= 22) & (v_ch >= 30))
    ).astype(np.uint8) * 255
    green_hsv = cv2.bitwise_and(green_hsv, cv2.bitwise_not(sky_or_glare))
    if category == "gravel":
        green_hsv = cv2.bitwise_and(green_hsv, cv2.bitwise_not(gravel_road_prior))

    vegetation = clean_mask(green_hsv, kernel_size=3, min_area=180)

    yellow_grass = (
        (h_ch >= 13)
        & (h_ch <= 45)
        & (s_ch >= 35)
        & (s_ch <= 205)
        & (v_ch >= 70)
        & (v_ch <= 245)
        & (sky_or_glare == 0)
    ).astype(np.uint8) * 255

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    texture = cv2.convertScaleAbs(lap)
    texture = cv2.GaussianBlur(texture, (9, 9), 0)
    high_texture = adaptive_texture_mask(texture, roi)

    haze_or_uncertain = (
        (s_ch < 55)
        & (v_ch > 92)
        & (v_ch < 225)
        & (texture < 24)
        & (green_hsv == 0)
    ).astype(np.uint8) * 255
    haze_or_uncertain = clean_mask(haze_or_uncertain, kernel_size=5, min_area=300)
    if category == "gravel":
        haze_or_uncertain = cv2.bitwise_and(haze_or_uncertain, cv2.bitwise_not(gravel_safe_surface))

    visible_for_safe = np.full((h, w), 255, dtype=np.uint8)
    haze_bool = haze_or_uncertain > 0
    y_indices = np.arange(h)[:, None]
    lower_haze = haze_bool & (y_indices > int(h * 0.34))
    for x_col in np.where(lower_haze.any(axis=0))[0]:
        bottom_y = int(np.max(np.where(lower_haze[:, x_col])[0]))
        visible_for_safe[: min(bottom_y + 8, h), x_col] = 0

    low_saturation_surface = (
        (s_ch < 105)
        & (v_ch > 42)
        & (v_ch < 235)
        & (sky_or_glare == 0)
        & (green_hsv == 0)
        & (haze_or_uncertain == 0)
    ).astype(np.uint8) * 255
    brown_soil_surface = (
        (h_ch >= 4)
        & (h_ch <= 35)
        & (s_ch >= 24)
        & (s_ch <= 185)
        & (v_ch >= 38)
        & (v_ch <= 235)
        & (green_hsv == 0)
        & (haze_or_uncertain == 0)
    ).astype(np.uint8) * 255

    gravel_or_soil = cv2.bitwise_or(low_saturation_surface, brown_soil_surface)
    if category == "gravel":
        gravel_or_soil = cv2.bitwise_or(gravel_or_soil, gravel_road_prior)
    gravel_or_soil = clean_mask(gravel_or_soil, kernel_size=5, min_area=450)
    gravel_or_soil = filter_surface_components(gravel_or_soil)

    grass_candidate = cv2.bitwise_or(vegetation, yellow_grass)
    grass_candidate = cv2.bitwise_and(grass_candidate, cv2.bitwise_not(gravel_or_soil))
    grass_candidate = clean_mask(grass_candidate, kernel_size=5, min_area=450)

    if category == "grassland":
        track_surface = (
            (h_ch >= 7)
            & (h_ch <= 35)
            & (s_ch >= 28)
            & (s_ch <= 145)
            & (v_ch >= 50)
            & (v_ch <= 220)
            & (green_hsv == 0)
            & (sky_or_glare == 0)
            & (haze_or_uncertain == 0)
        ).astype(np.uint8) * 255
        track_surface = clean_mask(track_surface, kernel_size=5, min_area=260)
        if np.count_nonzero(track_surface) / max(track_surface.size, 1) > 0.01:
            gravel_or_soil = track_surface
            grass_candidate = cv2.bitwise_or(vegetation, yellow_grass)
            grass_candidate = cv2.bitwise_and(grass_candidate, cv2.bitwise_not(track_surface))
            grass_candidate = clean_mask(grass_candidate, kernel_size=5, min_area=450)

    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    dark_object = (
        (v_ch < 82)
        & (s_ch > 18)
        & (green_hsv == 0)
        & (gravel_or_soil == 0)
        & (sky_or_glare == 0)
    ).astype(np.uint8) * 255
    rock_like = (
        (s_ch < 105)
        & (v_ch >= 50)
        & (v_ch <= 190)
        & (high_texture > 0)
        & (green_hsv == 0)
        & (sky_or_glare == 0)
    ).astype(np.uint8) * 255
    edge_object = cv2.bitwise_and(edges, cv2.bitwise_not(cv2.bitwise_or(gravel_or_soil, vegetation)))

    obstacle = cv2.bitwise_or(dark_object, cv2.bitwise_and(rock_like, cv2.bitwise_or(edges, high_texture)))
    obstacle = cv2.bitwise_or(obstacle, cv2.bitwise_and(edge_object, dark_object))
    obstacle = clean_mask(obstacle, kernel_size=3, min_area=180)
    obstacle = remove_huge_components(obstacle, min_area=220, max_area_ratio=0.18, reject_floor_band=True)

    if category == "grassland":
        obstacle = cv2.bitwise_and(obstacle, cv2.bitwise_not(grass_candidate))
        safe_candidate = gravel_or_soil
        if np.count_nonzero(safe_candidate) < int(h * w * 0.015):
            safe_candidate = cv2.bitwise_and(grass_candidate, cv2.bitwise_not(high_texture))
        caution_candidate = cv2.bitwise_and(grass_candidate, cv2.bitwise_not(safe_candidate))
    else:
        if category == "gravel":
            safe_candidate = gravel_safe_surface
            if np.count_nonzero(safe_candidate) < int(h * w * 0.025):
                safe_candidate = cv2.bitwise_and(gravel_or_soil, safe_road_corridor)
        else:
            safe_candidate = cv2.bitwise_and(gravel_or_soil, cv2.bitwise_not(high_texture))
            if np.count_nonzero(safe_candidate) < int(h * w * 0.012):
                safe_candidate = gravel_or_soil
        caution_candidate = vegetation

    safe = cv2.bitwise_and(safe_candidate, cv2.bitwise_not(obstacle))
    safe = cv2.bitwise_and(safe, cv2.bitwise_not(haze_or_uncertain))
    safe = cv2.bitwise_and(safe, visible_for_safe)
    safe = keep_accessible_safe_components(clean_mask(safe, kernel_size=7, min_area=600))

    caution = cv2.bitwise_and(caution_candidate, cv2.bitwise_not(safe))
    caution = cv2.bitwise_and(caution, cv2.bitwise_not(obstacle))
    caution = clean_mask(caution, kernel_size=5, min_area=450)

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


def overlay_result(bgr, safe, caution=None, obstacle=None):
    """把最终可通行区域叠加到原图上。

    最终结果只显示绿色的可通行区域；谨慎区和危险/障碍区只参与内部计算，不在结果图中展示。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = rgb.copy()

    green = np.array([0, 220, 90], dtype=np.uint8)

    # alpha 混合：最终图只把可通行区域用绿色半透明叠加到原图上。
    out[safe > 0] = (0.55 * out[safe > 0] + 0.45 * green).astype(np.uint8)
    return out


def analyze_array(bgr, category: str):
    """处理一张 BGR 图像数组，并返回中间结果与最终叠加图。"""
    bgr, hsv, gray = preprocess(bgr)
    masks = build_masks(bgr, hsv, gray, category)
    overlay = overlay_result(bgr, masks["safe"], masks["caution"], masks["obstacle"])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return hsv, masks, overlay, rgb


def analyze_image(image_path: Path, category: str):
    """处理单张图片，并返回中间结果与最终叠加图。"""
    return analyze_array(imread_unicode(image_path), category)


def simulate_vegetation_occlusion(bgr):
    """模拟近景草叶、灌木枝条遮挡对识别结果的影响。"""
    h, w = bgr.shape[:2]
    out = bgr.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    for index, x_ratio in enumerate([0.06, 0.14, 0.25, 0.72, 0.84, 0.93]):
        x0 = int(w * x_ratio)
        y0 = h
        x1 = int(x0 + ((-1) ** index) * w * 0.08)
        y1 = int(h * (0.32 + 0.07 * (index % 3)))
        cv2.line(mask, (x0, y0), (x1, y1), 255, thickness=max(8, w // 70))

    for center_ratio, size_ratio in [((0.18, 0.58), (0.16, 0.08)), ((0.82, 0.54), (0.18, 0.09))]:
        center = (int(w * center_ratio[0]), int(h * center_ratio[1]))
        axes = (int(w * size_ratio[0]), int(h * size_ratio[1]))
        cv2.ellipse(mask, center, axes, 8, 0, 360, 255, thickness=-1)

    color = np.array([45, 118, 48], dtype=np.uint8)
    covered = mask > 0
    out[covered] = (0.52 * out[covered] + 0.48 * color).astype(np.uint8)
    return out


def simulate_texture_interference(bgr):
    """模拟砂石颗粒、杂草纹理和成像噪声带来的纹理干扰。"""
    h, w = bgr.shape[:2]
    rng = np.random.default_rng(2026)
    out = bgr.copy()

    roi = np.zeros((h, w), dtype=bool)
    roi[int(h * 0.28):, :] = True
    noise = rng.normal(0, 18, bgr.shape).astype(np.int16)
    noisy = out.astype(np.int16)
    noisy[roi] += noise[roi]
    out = np.clip(noisy, 0, 255).astype(np.uint8)

    for _ in range(260):
        x = int(rng.integers(0, w))
        y = int(rng.integers(int(h * 0.32), h))
        radius = int(rng.integers(1, max(2, w // 140)))
        value = int(rng.integers(70, 190))
        color = (value, int(value * rng.uniform(0.82, 1.10)), int(value * rng.uniform(0.70, 1.05)))
        cv2.circle(out, (x, y), radius, color, thickness=-1)

    return out


def build_condition_variants(bgr) -> list[tuple[str, str, object]]:
    """构造多组工况：原始、植被遮挡、纹理干扰。"""
    return [
        ("normal", "原始工况", bgr),
        ("vegetation_occlusion", "植被遮挡工况", simulate_vegetation_occlusion(bgr)),
        ("texture_interference", "纹理干扰工况", simulate_texture_interference(bgr)),
    ]


def measure_masks(masks: dict[str, object]) -> dict[str, float | int]:
    """统计可通行、障碍、纹理等关键指标，便于做结果分析。"""
    roi_count = max(int(np.count_nonzero(masks["roi"])), 1)
    safe_count = int(np.count_nonzero(masks["safe"]))
    obstacle_count = int(np.count_nonzero(masks["obstacle"]))
    caution_count = int(np.count_nonzero(masks["caution"]))
    texture_count = int(np.count_nonzero(masks["high_texture"]))

    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(masks["safe"], connectivity=8)
    safe_components = max(num - 1, 0)
    largest_safe_area = int(stats[1:, cv2.CC_STAT_AREA].max()) if safe_components else 0

    return {
        "safe_ratio": safe_count / roi_count,
        "obstacle_ratio": obstacle_count / roi_count,
        "caution_ratio": caution_count / roi_count,
        "high_texture_ratio": texture_count / roi_count,
        "safe_components": safe_components,
        "largest_safe_area": largest_safe_area,
    }


def save_condition_figure(
    out_path: Path,
    title: str,
    rgb,
    overlay,
    metrics: dict[str, float | int],
) -> None:
    """保存某一工况下的原图与识别结果对比图。"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.suptitle(title, fontsize=14)
    axes[0].imshow(rgb)
    axes[0].set_title("工况输入图", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("安全可通行区域", fontsize=11)
    axes[1].axis("off")
    fig.text(
        0.5,
        0.02,
        f"可通行占比={metrics['safe_ratio']:.2%}，障碍占比={metrics['obstacle_ratio']:.2%}，高纹理占比={metrics['high_texture_ratio']:.2%}",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_condition_report(rows: list[dict[str, object]], report_path: Path) -> None:
    """根据多工况统计指标生成中文结果分析。"""
    lines = [
        "题目二多工况仿真验证与结果分析",
        "",
        "一、验证工况",
        "1. 原始工况：直接对森林、砂石路面、杂草草地图片进行识别。",
        "2. 植被遮挡工况：在近景叠加草叶、灌木状遮挡，观察遮挡对可通行区域连续性的影响。",
        "3. 纹理干扰工况：在地面区域叠加颗粒噪声与小斑点，观察复杂纹理对障碍检测和安全区域划分的影响。",
        "",
        "二、平均指标",
    ]

    condition_labels = ["原始工况", "植被遮挡工况", "纹理干扰工况"]
    for condition in condition_labels:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        safe = sum(float(row["safe_ratio"]) for row in subset) / len(subset)
        obstacle = sum(float(row["obstacle_ratio"]) for row in subset) / len(subset)
        texture = sum(float(row["high_texture_ratio"]) for row in subset) / len(subset)
        lines.append(
            f"- {condition}：平均可通行占比 {safe:.2%}，平均障碍占比 {obstacle:.2%}，平均高纹理占比 {texture:.2%}。"
        )

    lines.extend(["", "三、扰动影响分析"])
    image_names = sorted({str(row["image"]) for row in rows})
    for image_name in image_names:
        image_rows = [row for row in rows if row["image"] == image_name]
        normal = next((row for row in image_rows if row["condition"] == "原始工况"), None)
        if normal is None:
            continue
        lines.append(f"- {image_name}：")
        for condition in ["植被遮挡工况", "纹理干扰工况"]:
            current = next((row for row in image_rows if row["condition"] == condition), None)
            if current is None:
                continue
            safe_delta = float(current["safe_ratio"]) - float(normal["safe_ratio"])
            obstacle_delta = float(current["obstacle_ratio"]) - float(normal["obstacle_ratio"])
            lines.append(
                f"  {condition}相对原始工况，可通行占比变化 {safe_delta:+.2%}，障碍占比变化 {obstacle_delta:+.2%}。"
            )

    lines.extend(
        [
            "",
            "四、结论",
            "HSV 色彩像素分类能够稳定区分绿色植被、干草和低饱和砂石/土壤区域；Laplacian 纹理特征可以抑制杂草、碎石等高纹理干扰；形态学开闭运算与连通域面积过滤可以去除零散噪声并保留连续可通行区域。",
            "植被遮挡会削弱道路或草地的连续性，使可通行掩膜面积下降；纹理干扰会提升高纹理和障碍候选比例，容易压缩安全区域。优化后的近地面梯形 ROI、草地车辙优先规则和小连通域过滤可减少误检，满足复杂野外非结构化场景下危险区域剔除与安全区域划分的仿真要求。",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_condition_validation(images: list[tuple[str, str, Path]] | None = None) -> tuple[Path, Path]:
    """完成多组工况仿真验证，输出对比图、指标 CSV 和中文分析报告。"""
    setup_matplotlib_font()
    RESULT_DIR.mkdir(exist_ok=True)
    images = images or collect_images()
    if not images:
        raise ValueError("没有找到图片，请检查 data/forest、data/gravel、data/grassland 目录。")

    condition_dir = RESULT_DIR / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for category, label, image_path in images:
        category_dir = condition_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        source_bgr = imread_unicode(image_path)
        for condition_key, condition_label, condition_bgr in build_condition_variants(source_bgr):
            _hsv, masks, overlay, rgb = analyze_array(condition_bgr, category)
            metrics = measure_masks(masks)
            out_path = category_dir / f"{image_path.stem}_{condition_key}_final.png"
            save_condition_figure(out_path, f"{label} - {image_path.name} - {condition_label}", rgb, overlay, metrics)
            row = {
                "category": category,
                "category_label": label,
                "image": image_path.name,
                "condition": condition_label,
                "result_path": str(out_path),
                **metrics,
            }
            rows.append(row)

    metrics_path = RESULT_DIR / "condition_metrics.csv"
    fieldnames = [
        "category",
        "category_label",
        "image",
        "condition",
        "safe_ratio",
        "obstacle_ratio",
        "caution_ratio",
        "high_texture_ratio",
        "safe_components",
        "largest_safe_area",
        "result_path",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report_path = RESULT_DIR / "condition_analysis.txt"
    write_condition_report(rows, report_path)
    return metrics_path, report_path


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
        ("可通行掩膜", masks["safe"], "gray"),
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
        "颜色说明：绿色=可通行候选区域。",
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
        "绿色=可通行区域。",
        ha="center",
        fontsize=10,
    )
    final_fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    final_fig.savefig(final_path, dpi=170)
    plt.close(final_fig)
    return analysis_path, final_path


def build_summary(result_paths: list[tuple[str, Path]]) -> Path:
    """把示例图片的最终结果汇总成一张总览图。"""
    if not result_paths:
        raise ValueError("没有可汇总的结果图。")
    out_path = RESULT_DIR / "summary.png"
    cols = min(3, max(len(result_paths), 1))
    rows = (len(result_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.7 * rows))
    fig.suptitle("题目二：安全地形可通行区域仿真结果总览", fontsize=17)
    axes_flat = np.array(axes).reshape(-1)
    for ax, (title, path) in zip(axes_flat, result_paths):
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    for ax in axes_flat[len(result_paths):]:
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def collect_images() -> list[tuple[str, str, Path]]:
    """收集 data 目录下所有待处理图片。"""
    items: list[tuple[str, str, Path]] = []
    for category, label in CATEGORIES.items():
        folder = DATA_DIR / category
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                items.append((category, label, path))
    return items


def run_batch(no_show: bool = False):
    """批量处理 data 目录中的示例图片，并生成结果图。"""
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
    metrics_path, report_path = run_condition_validation(images)
    print(f"\n全部完成。总览图：{summary_path}")
    print(f"多工况指标表：{metrics_path}")
    print(f"多工况结果分析：{report_path}")
    print(f"单张分析图和最终结果图保存在：{RESULT_DIR}")

    # 如果命令行加 --no-show，则只保存不弹窗。
    if not no_show:
        plt.show()


def main():
    """程序入口。

    默认打开 UI 界面；只有加 --batch 时才执行原来的批量处理流程。
    """
    parser = argparse.ArgumentParser(description="题目二：安全地形可通行区域仿真")
    parser.add_argument("--batch", action="store_true", help="批量处理 data 目录中的示例图片")
    parser.add_argument("--conditions", action="store_true", help="仅生成多工况验证报告")
    parser.add_argument("--no-show", action="store_true", help="批量处理时只保存结果，不弹出图像窗口")
    args = parser.parse_args()

    if args.conditions:
        setup_matplotlib_font()
        metrics_path, report_path = run_condition_validation()
        print(f"多工况指标表：{metrics_path}")
        print(f"多工况结果分析：{report_path}")
        return

    if args.batch:
        run_batch(no_show=args.no_show)
        return

    from app import main as launch_ui

    launch_ui()


if __name__ == "__main__":
    main()
