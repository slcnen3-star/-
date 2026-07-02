# -*- coding: utf-8 -*-
"""
题目二图形界面入口。

运行方式：
    python app.py

界面固定显示两张图：左边原图，右边为当前按钮对应的处理结果。
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError:
    messagebox.showerror("缺少依赖", "缺少 Pillow，请先运行：pip install -r requirements.txt")
    raise SystemExit(1)

import matplotlib

matplotlib.use("Agg")

import numpy as np

import main as engine


COLORS = {
    "bg": "#eef3f8",
    "panel": "#ffffff",
    "panel_soft": "#f6f8fb",
    "sidebar": "#172235",
    "sidebar_soft": "#22324a",
    "text": "#142033",
    "muted": "#64748b",
    "muted_light": "#b9c7d8",
    "line": "#d7e0ea",
    "blue": "#2563eb",
}


class ImagePanel(tk.Frame):
    """固定尺寸图片面板，标题、占位文字和图片都居中显示。"""

    def __init__(self, master, title: str, placeholder: str):
        super().__init__(master, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.title_var = tk.StringVar(value=title)
        tk.Label(
            self,
            textvariable=self.title_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Microsoft YaHei", 12, "bold"),
            anchor="center",
            justify="center",
            padx=14,
            pady=8,
        ).grid(row=0, column=0, sticky="ew")

        self.placeholder = placeholder
        self.canvas = tk.Canvas(self, bg=COLORS["panel_soft"], highlightthickness=0, width=440, height=330)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _event: self.refresh())

    def set_title(self, title: str) -> None:
        self.title_var.set(title)

    def set_placeholder(self, placeholder: str) -> None:
        self.placeholder = placeholder
        if self.image is None:
            self.refresh()

    def set_image(self, image: Image.Image | None) -> None:
        self.image = image
        self.refresh()

    def refresh(self) -> None:
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 180)
        h = max(self.canvas.winfo_height(), 140)

        if self.image is None:
            if self.placeholder:
                self.canvas.create_text(
                    w / 2,
                    h / 2,
                    text=self.placeholder,
                    fill=COLORS["muted"],
                    font=("Microsoft YaHei", 12),
                    justify="center",
                    width=max(w - 40, 120),
                )
            self.photo = None
            return

        img = self.image.copy()
        img.thumbnail((max(w - 18, 80), max(h - 18, 80)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(w / 2, h / 2, image=self.photo, anchor="center")


class TerrainApp:
    """安全地形可通行区域仿真 UI。"""

    PROCESS_BUTTONS = [
        ("terrain", "地形区分"),
        ("hsv", "HSV 色彩分类"),
        ("texture", "纹理特征提取"),
        ("vegetation", "植被/草地检测"),
        ("surface", "砂石路面检测"),
        ("obstacle", "障碍物检测"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("")
        self.root.geometry("1320x780")
        self.root.minsize(1120, 700)
        self.root.configure(bg=COLORS["bg"])

        engine.setup_matplotlib_font()
        engine.RESULT_DIR.mkdir(exist_ok=True)

        self.images = engine.collect_images()
        self.current_item: tuple[str, str, Path] | None = None
        self.analysis_cache: tuple[Path, str, object, dict[str, object], object, object] | None = None
        self.status_var = tk.StringVar(value="请先导入图片。")
        self.original_panel: ImagePanel | None = None
        self.process_panel: ImagePanel | None = None
        self.last_processed_image: Image.Image | None = None
        self.last_processed_title = ""
        self.last_processed_key = ""

        self._build_style()
        self._build_layout()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Primary.TButton",
            background=COLORS["blue"],
            foreground="#ffffff",
            font=("Microsoft YaHei", 11, "bold"),
            padding=(16, 9),
            borderwidth=0,
            anchor="center",
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "Soft.TButton",
            background="#e7eef7",
            foreground=COLORS["text"],
            font=("Microsoft YaHei", 10),
            padding=(14, 8),
            borderwidth=0,
            anchor="center",
        )
        style.map("Soft.TButton", background=[("active", "#d6e2f0")])

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        content = tk.Frame(self.root, bg=COLORS["bg"])
        content.grid(row=0, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)
        content.rowconfigure(2, weight=0)

        self._build_header(content)
        self._build_image_area(content)
        self._build_bottom_controls(content)

    def _build_bottom_controls(self, parent: tk.Frame) -> None:
        controls = tk.Frame(parent, bg=COLORS["bg"], padx=22)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        controls.columnconfigure(0, weight=1)

        bar = tk.Frame(controls, bg=COLORS["bg"])
        bar.grid(row=0, column=0)

        def group(master: tk.Frame, title: str, column: int, columns: int) -> tk.Frame:
            box = tk.Frame(
                master,
                bg=COLORS["panel"],
                highlightthickness=1,
                highlightbackground=COLORS["line"],
                padx=10,
                pady=8,
            )
            box.grid(row=0, column=column, padx=8, sticky="n")
            tk.Label(
                box,
                text=title,
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=("Microsoft YaHei", 9, "bold"),
                anchor="center",
                justify="center",
            ).grid(row=0, column=0, columnspan=columns, sticky="ew", pady=(0, 6))
            for index in range(columns):
                box.columnconfigure(index, weight=1, uniform=f"{title}_button")
            return box

        file_group = group(bar, "图像操作", 0, 1)
        analysis_group = group(bar, "检测分析", 1, 3)
        result_group = group(bar, "结果区域", 2, 1)

        button_width = 12
        ttk.Button(file_group, text="导入", style="Primary.TButton", width=button_width, command=self.import_image).grid(
            row=1, column=0, sticky="ew", pady=(0, 5)
        )
        ttk.Button(file_group, text="保存", style="Soft.TButton", width=button_width, command=self.save_processed_image).grid(
            row=2, column=0, sticky="ew", pady=(0, 5)
        )
        for index, (process_key, text) in enumerate(self.PROCESS_BUTTONS):
            row, col = divmod(index, 3)
            ttk.Button(
                analysis_group,
                text=text,
                style="Soft.TButton",
                width=button_width,
                command=lambda key=process_key: self.show_process_result(key),
            ).grid(row=row + 1, column=col, sticky="ew", padx=4, pady=3)

        ttk.Button(
            result_group,
            text="安全可通行区域",
            style="Soft.TButton",
            width=button_width,
            command=lambda: self.show_process_result("safe"),
        ).grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Button(
            result_group,
            text="打开文件夹",
            style="Soft.TButton",
            width=button_width,
            command=self.open_results_folder,
        ).grid(row=2, column=0, sticky="ew")

    def _build_header(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg=COLORS["bg"], padx=22, pady=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="安全地形可通行区域仿真平台",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Microsoft YaHei", 20, "bold"),
            anchor="center",
            justify="center",
        ).grid(row=0, column=0, sticky="ew")

    def _build_image_area(self, parent: tk.Frame) -> None:
        preview = tk.Frame(parent, bg=COLORS["bg"])
        preview.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 10))
        preview.columnconfigure(0, weight=1, uniform="image_grid")
        preview.columnconfigure(1, weight=1, uniform="image_grid")
        preview.rowconfigure(0, weight=1)

        self.original_panel = ImagePanel(preview, "原图", "")
        self.original_panel.grid(row=0, column=0, sticky="nsew", padx=7, pady=7)

        self.process_panel = ImagePanel(preview, "处理图片", "")
        self.process_panel.grid(row=0, column=1, sticky="nsew", padx=7, pady=7)

    def import_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*"),
            ],
        )
        if not file_path:
            return

        path = Path(file_path)
        category = self._guess_category(path)
        label = engine.CATEGORIES[category]
        self.current_item = (category, label, path)
        self._load_current_original()
        self.status_var.set(f"已导入：{label} / {path.name}。请点击下方检测按钮。")

    def _guess_category(self, path: Path) -> str:
        """在 UI 不显示类别选择的情况下，内部自动估计地形类别。"""
        text = str(path).lower()
        for key in engine.CATEGORIES:
            if key in text:
                return key
        if "森林" in text:
            return "forest"
        if "草" in text:
            return "grassland"
        if "砂" in text or "石" in text or "路" in text:
            return "gravel"

        bgr = engine.imread_unicode(path)
        bgr, hsv, gray = engine.preprocess(bgr)
        h, w = gray.shape
        roi = np.zeros((h, w), dtype=np.uint8)
        roi[int(h * 0.35):, :] = 255
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        green = (h_ch >= 28) & (h_ch <= 96) & (s_ch > 25) & (v_ch > 30) & (roi > 0)
        dry_grass = (h_ch >= 10) & (h_ch <= 45) & (s_ch >= 25) & (v_ch >= 55) & (roi > 0)
        low_sat = (s_ch < 105) & (v_ch > 45) & (v_ch < 240) & (roi > 0)
        roi_count = max(int(np.count_nonzero(roi)), 1)
        grass_ratio = (np.count_nonzero(green) + np.count_nonzero(dry_grass)) / roi_count
        surface_ratio = np.count_nonzero(low_sat) / roi_count

        if grass_ratio > 0.48 and surface_ratio < 0.45:
            return "grassland"
        if surface_ratio > 0.42:
            return "gravel"
        return "forest"

    def _load_current_original(self) -> None:
        self.analysis_cache = None
        self.last_processed_image = None
        self.last_processed_title = ""
        self.last_processed_key = ""
        if self.process_panel is not None:
            self.process_panel.set_title("处理图片")
            self.process_panel.set_placeholder("")
            self.process_panel.set_image(None)
        if self.current_item is None:
            return
        _category, _label, path = self.current_item
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return
        if self.original_panel is not None:
            self.original_panel.set_title("原图")
            self.original_panel.set_image(image)

    def _require_current_item(self) -> tuple[str, str, Path] | None:
        if self.current_item is None:
            messagebox.showinfo("提示", "请先导入图片。")
            return None
        return self.current_item

    def show_process_result(self, process_key: str) -> None:
        if self._require_current_item() is None:
            return
        self._run_worker(lambda: self._process_result_worker(process_key))

    def process_all(self) -> None:
        if not self.images:
            messagebox.showinfo("提示", "data 文件夹中没有找到图片。")
            return
        self._run_worker(self._process_all_worker)

    def _run_worker(self, target) -> None:
        threading.Thread(target=target, daemon=True).start()

    def _get_analysis(self):
        category, _label, path = self.current_item
        if self.analysis_cache is not None and self.analysis_cache[0] == path and self.analysis_cache[1] == category:
            return self.analysis_cache[2:]
        hsv, masks, overlay, rgb = engine.analyze_image(path, category)
        self.analysis_cache = (path, category, hsv, masks, overlay, rgb)
        return hsv, masks, overlay, rgb

    def _process_result_worker(self, process_key: str) -> None:
        category, label, path = self.current_item
        title = self._process_title(process_key)
        self._set_status(f"正在生成：{title} / {path.name}")
        try:
            hsv, masks, overlay, rgb = self._get_analysis()
            image = self._make_process_image(process_key, category, hsv, masks, overlay, rgb)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("处理失败", str(exc)))
            self._set_status("处理失败，请检查图片或依赖环境。")
            return

        self.root.after(0, lambda: self._set_process_image(title, image, label, path))

    def _process_title(self, process_key: str) -> str:
        if process_key == "safe":
            return "安全可通行区域"
        return dict(self.PROCESS_BUTTONS).get(process_key, "处理图片")

    def _set_process_image(self, title: str, image: Image.Image, label: str, path: Path) -> None:
        self.last_processed_image = image.copy()
        self.last_processed_title = title
        self.last_processed_key = next((key for key, text in self.PROCESS_BUTTONS if text == title), "")
        if title == "安全可通行区域":
            self.last_processed_key = "safe"
        if self.process_panel is not None:
            self.process_panel.set_title(title)
            self.process_panel.set_image(image)
        self.status_var.set(f"已显示：{title} / {label} / {path.name}。")

    def _make_process_image(self, process_key: str, category: str, hsv, masks, overlay, rgb) -> Image.Image:
        if process_key == "terrain":
            return Image.fromarray(self._terrain_classification(rgb, masks, category))
        if process_key == "hsv":
            h_channel = np.clip(hsv[:, :, 0].astype(np.float32) / 179 * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(h_channel).convert("RGB")
        if process_key == "texture":
            return Image.fromarray(masks["texture"]).convert("RGB")
        if process_key == "vegetation":
            key = "grass_candidate" if category == "grassland" else "vegetation"
            return Image.fromarray(self._mask_overlay(rgb, masks[key], (20, 190, 80)))
        if process_key == "surface":
            return Image.fromarray(self._mask_overlay(rgb, masks["gravel_or_soil"], (230, 170, 60)))
        if process_key == "obstacle":
            return Image.fromarray(self._mask_overlay(rgb, masks["obstacle"], (230, 40, 40)))
        if process_key == "safe":
            return Image.fromarray(self._safe_danger_overlay(rgb, masks["safe"], masks["obstacle"]))
        return Image.fromarray(overlay)

    def _terrain_classification(self, rgb, masks: dict[str, object], category: str):
        out = rgb.copy()
        vegetation_key = "grass_candidate" if category == "grassland" else "vegetation"

        layers = [
            (masks[vegetation_key] > 0, np.array([25, 190, 85], dtype=np.uint8)),
            (masks["gravel_or_soil"] > 0, np.array([225, 170, 65], dtype=np.uint8)),
            (masks["obstacle"] > 0, np.array([230, 50, 50], dtype=np.uint8)),
        ]
        for selected, color in layers:
            out[selected] = (0.48 * out[selected] + 0.52 * color).astype(np.uint8)
        return out

    def _mask_overlay(self, rgb, mask, color: tuple[int, int, int]):
        out = rgb.copy()
        selected = mask > 0
        color_array = np.array(color, dtype=np.uint8)
        out[selected] = (0.50 * out[selected] + 0.50 * color_array).astype(np.uint8)
        return out

    def _safe_danger_overlay(self, rgb, safe_mask, danger_mask):
        out = rgb.copy()
        safe = safe_mask > 0
        danger = danger_mask > 0
        green = np.array([0, 220, 90], dtype=np.uint8)
        red = np.array([230, 45, 45], dtype=np.uint8)
        out[safe] = (0.45 * out[safe] + 0.55 * green).astype(np.uint8)
        out[danger] = (0.40 * out[danger] + 0.60 * red).astype(np.uint8)
        return out

    def save_processed_image(self) -> None:
        item = self._require_current_item()
        if item is None:
            return
        if self.last_processed_image is None:
            messagebox.showinfo("提示", "请先点击一个检测按钮生成处理图像。")
            return
        _category, _label, path = item
        save_dir = engine.RESULT_DIR / "saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        key = self.last_processed_key or "processed"
        out_path = save_dir / f"{path.stem}_{key}.png"
        self.last_processed_image.save(out_path)
        self.status_var.set(f"已保存：{out_path}")
        messagebox.showinfo("保存成功", f"处理后的图像已保存到：\n{out_path}")

    def _process_all_worker(self) -> None:
        result_paths: list[tuple[str, Path]] = []
        total = len(self.images)
        try:
            for index, (category, label, path) in enumerate(self.images, start=1):
                self._set_status(f"正在批量验证：{index}/{total}  {label} / {path.name}")
                _analysis_path, final_path = engine.save_analysis_figure(path, category, label)
                result_paths.append((f"{label}: {path.name}", final_path))
            summary_path = engine.build_summary(result_paths)
            metrics_path, report_path = engine.run_condition_validation()
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("处理失败", str(exc)))
            self._set_status("批量验证失败，请检查图片或依赖环境。")
            return

        self._set_status(f"批量验证完成：{summary_path.name}、{metrics_path.name}、{report_path.name}")

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def open_results_folder(self) -> None:
        engine.RESULT_DIR.mkdir(exist_ok=True)
        os.startfile(str(engine.RESULT_DIR))


def smoke_test() -> None:
    """不打开窗口，只验证 UI 依赖和算法调用是否正常。"""
    engine.setup_matplotlib_font()
    images = engine.collect_images()
    if not images:
        raise SystemExit("没有找到测试图片")
    category, _label, path = images[0]
    hsv, masks, overlay, rgb = engine.analyze_image(path, category)
    app_like = TerrainApp.__new__(TerrainApp)
    app_like._mask_overlay = TerrainApp._mask_overlay.__get__(app_like, TerrainApp)
    app_like._safe_danger_overlay = TerrainApp._safe_danger_overlay.__get__(app_like, TerrainApp)
    app_like._terrain_classification = TerrainApp._terrain_classification.__get__(app_like, TerrainApp)
    for key, _title in [*TerrainApp.PROCESS_BUTTONS, ("safe", "安全可通行区域")]:
        image = TerrainApp._make_process_image(app_like, key, category, hsv, masks, overlay, rgb)
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise SystemExit(f"{key} 处理图生成失败")
    for category, _label, path in images:
        if category == "grassland" and path.name == "0001.jpg_wh860.jpg":
            _hsv, grass_masks, _overlay, _rgb = engine.analyze_image(path, category)
            grass = grass_masks["grass_candidate"] > 0
            road = grass_masks["gravel_or_soil"] > 0
            overlap = (grass & road).sum() / max(road.sum(), 1)
            if overlap > 0.25:
                raise SystemExit(f"草地检测与土路候选重叠过高：{overlap:.2%}")
    print("UI smoke test passed")


def main() -> None:
    if "--smoke-test" in sys.argv:
        smoke_test()
        return
    print("正在启动图形界面，请查看屏幕上的新窗口...")
    root = tk.Tk()
    root.deiconify()
    root.lift()
    root.focus_force()
    root.attributes("-topmost", True)
    root.after(900, lambda: root.attributes("-topmost", False))
    TerrainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
