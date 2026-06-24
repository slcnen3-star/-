# -*- coding: utf-8 -*-
"""
题目二图形界面入口。

运行方式：
    python app.py

这个界面调用 main.py 中已经写好的图像识别函数，不重复实现算法。
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
    "green": "#16a36a",
    "yellow": "#d99a16",
    "red": "#d64545",
    "blue": "#2563eb",
}


class ImagePanel(tk.Frame):
    """固定尺寸的图片展示面板，图片变化时不会撑大或缩小布局。"""

    def __init__(self, master, title: str):
        super().__init__(master, bg=COLORS["panel"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.title = title
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.placeholder_id: int | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        tk.Label(
            self,
            text=title,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Microsoft YaHei", 11, "bold"),
            anchor="w",
            padx=14,
            pady=9,
        ).grid(row=0, column=0, sticky="ew")

        self.canvas = tk.Canvas(
            self,
            bg=COLORS["panel_soft"],
            highlightthickness=0,
            width=360,
            height=210,
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", lambda _event: self.refresh())

    def set_image(self, image: Image.Image | None) -> None:
        self.image = image
        self.refresh()

    def refresh(self) -> None:
        self.canvas.delete("all")
        w = max(self.canvas.winfo_width(), 160)
        h = max(self.canvas.winfo_height(), 120)

        if self.image is None:
            self.canvas.create_text(
                w / 2,
                h / 2,
                text="等待导入图片",
                fill=COLORS["muted"],
                font=("Microsoft YaHei", 11),
            )
            self.photo = None
            return

        img = self.image.copy()
        img.thumbnail((max(w - 18, 80), max(h - 18, 80)), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(w / 2, h / 2, image=self.photo, anchor="center")


class TerrainApp:
    """安全地形可通行区域仿真 UI。"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("题目二：安全地形可通行区域仿真平台")
        self.root.geometry("1360x820")
        self.root.minsize(1180, 720)
        self.root.configure(bg=COLORS["bg"])

        engine.setup_matplotlib_font()
        engine.RESULT_DIR.mkdir(exist_ok=True)

        self.images = engine.collect_images()
        self.current_item: tuple[str, str, Path] | None = None
        self.status_var = tk.StringVar(value="请点击“导入并识别图片”，选择图片后才会开始处理。")
        self.detail_path: Path | None = None
        self.final_path: Path | None = None
        self.preview_panels: dict[str, ImagePanel] = {}

        self._build_style()
        self._build_layout()
        self._fill_image_list()

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
        )
        style.map("Primary.TButton", background=[("active", "#1d4ed8")])
        style.configure(
            "Soft.TButton",
            background="#e7eef7",
            foreground=COLORS["text"],
            font=("Microsoft YaHei", 10),
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Soft.TButton", background=[("active", "#d6e2f0")])

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = tk.Frame(self.root, bg=COLORS["sidebar"], width=310)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        content = tk.Frame(self.root, bg=COLORS["bg"])
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(2, weight=1)

        self._build_sidebar(sidebar)
        self._build_header(content)
        self._build_preview_area(content)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="地形识别",
            bg=COLORS["sidebar"],
            fg="#ffffff",
            font=("Microsoft YaHei", 22, "bold"),
            anchor="w",
            padx=24,
            pady=0,
        ).pack(fill="x", pady=(26, 10))
        tk.Label(
            parent,
            text="中间过程与最终结果展示",
            bg=COLORS["sidebar"],
            fg=COLORS["muted_light"],
            font=("Microsoft YaHei", 10),
            anchor="w",
            padx=24,
        ).pack(fill="x")

        actions = tk.Frame(parent, bg=COLORS["sidebar"], padx=22, pady=22)
        actions.pack(fill="x")
        ttk.Button(actions, text="导入并识别图片", style="Primary.TButton", command=self.import_and_process).pack(
            fill="x", pady=(0, 10)
        )
        ttk.Button(actions, text="分析示例图片", style="Soft.TButton", command=self.process_current).pack(
            fill="x", pady=(0, 10)
        )
        ttk.Button(actions, text="打开结果文件夹", style="Soft.TButton", command=self.open_results_folder).pack(fill="x")

        status_box = tk.Frame(parent, bg=COLORS["sidebar_soft"], padx=14, pady=12)
        status_box.pack(fill="x", padx=22, pady=(0, 20))
        tk.Label(
            status_box,
            textvariable=self.status_var,
            bg=COLORS["sidebar_soft"],
            fg="#e7eef7",
            font=("Microsoft YaHei", 9),
            anchor="w",
            justify="left",
            wraplength=240,
        ).pack(fill="x")

        list_wrap = tk.Frame(parent, bg=COLORS["sidebar"], padx=22)
        list_wrap.pack(fill="both", expand=True)
        tk.Label(
            list_wrap,
            text="示例图片",
            bg=COLORS["sidebar"],
            fg="#ffffff",
            font=("Microsoft YaHei", 11, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        self.image_list = tk.Listbox(
            list_wrap,
            bg="#101a27",
            fg="#edf4fb",
            selectbackground=COLORS["blue"],
            selectforeground="#ffffff",
            activestyle="none",
            relief="flat",
            highlightthickness=0,
            font=("Microsoft YaHei", 10),
            height=10,
        )
        self.image_list.pack(fill="both", expand=True)
        self.image_list.bind("<<ListboxSelect>>", lambda _event: self._on_select_image())

        legend = tk.Frame(parent, bg=COLORS["sidebar"], padx=24, pady=20)
        legend.pack(fill="x")
        self._legend_row(legend, COLORS["green"], "绿色：可通行")
        self._legend_row(legend, COLORS["yellow"], "黄色：谨慎通行")
        self._legend_row(legend, COLORS["red"], "红色：障碍/危险")

    def _legend_row(self, parent: tk.Frame, color: str, text: str) -> None:
        row = tk.Frame(parent, bg=COLORS["sidebar"])
        row.pack(fill="x", pady=4)
        tk.Frame(row, width=14, height=14, bg=color).pack(side="left")
        tk.Label(row, text=text, bg=COLORS["sidebar"], fg=COLORS["muted_light"], font=("Microsoft YaHei", 10)).pack(
            side="left", padx=10
        )

    def _build_header(self, parent: tk.Frame) -> None:
        header = tk.Frame(parent, bg=COLORS["bg"], padx=26, pady=20)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        title_box = tk.Frame(header, bg=COLORS["bg"])
        title_box.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_box,
            text="安全地形可通行区域仿真平台",
            bg=COLORS["bg"],
            fg=COLORS["text"],
            font=("Microsoft YaHei", 21, "bold"),
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_box,
            text="导入一张图片后，界面会同时显示原图、处理中间结果和最终可通行区域。",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=("Microsoft YaHei", 10),
            anchor="w",
        ).pack(anchor="w", pady=(6, 0))

    def _build_preview_area(self, parent: tk.Frame) -> None:
        preview = tk.Frame(parent, bg=COLORS["bg"])
        preview.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 22))
        for col in range(3):
            preview.columnconfigure(col, weight=1, uniform="image_grid")
        for row in range(2):
            preview.rowconfigure(row, weight=1, uniform="image_grid")

        panel_defs = [
            ("original", "原图"),
            ("h_channel", "HSV-H 色调通道"),
            ("vegetation", "植被/草地候选"),
            ("surface", "砂石/土壤候选"),
            ("obstacle", "障碍物候选"),
            ("overlay", "最终可通行区域"),
        ]
        for index, (key, title) in enumerate(panel_defs):
            row, col = divmod(index, 3)
            panel = ImagePanel(preview, title)
            panel.grid(row=row, column=col, sticky="nsew", padx=7, pady=7)
            self.preview_panels[key] = panel

    def _fill_image_list(self) -> None:
        self.image_list.delete(0, tk.END)
        for _category, _label, path in self.images:
            self.image_list.insert(tk.END, path.name)

    def _on_select_image(self) -> None:
        selection = self.image_list.curselection()
        if not selection:
            return
        self.current_item = self.images[selection[0]]
        category, label, path = self.current_item
        self.status_var.set(f"已选择示例图片：{path.name}。点击“分析示例图片”后开始处理。")
        self._clear_panels()
        self._load_original(path)

    def _load_original(self, path: Path) -> None:
        image = Image.open(path).convert("RGB")
        self.preview_panels["original"].set_image(image)

    def _clear_panels(self) -> None:
        for panel in self.preview_panels.values():
            panel.set_image(None)

    def import_and_process(self) -> None:
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
        self.image_list.selection_clear(0, tk.END)
        self.status_var.set(f"已导入：{path.name}，正在识别并生成中间过程图。")
        self._clear_panels()
        self._load_original(path)
        self.process_current()

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

        green = ((h_ch >= 28) & (h_ch <= 96) & (s_ch > 25) & (v_ch > 30) & (roi > 0))
        dry_grass = ((h_ch >= 10) & (h_ch <= 45) & (s_ch >= 25) & (v_ch >= 55) & (roi > 0))
        low_sat = ((s_ch < 105) & (v_ch > 45) & (v_ch < 240) & (roi > 0))
        roi_count = max(int(np.count_nonzero(roi)), 1)
        grass_ratio = (np.count_nonzero(green) + np.count_nonzero(dry_grass)) / roi_count
        surface_ratio = np.count_nonzero(low_sat) / roi_count

        if grass_ratio > 0.48 and surface_ratio < 0.45:
            return "grassland"
        if surface_ratio > 0.42:
            return "gravel"
        return "forest"

    def process_current(self) -> None:
        if self.current_item is None:
            messagebox.showinfo("提示", "请先导入图片，或在左侧选择一张示例图片。")
            return
        self._run_worker(self._process_current_worker)

    def process_all(self) -> None:
        if not self.images:
            messagebox.showinfo("提示", "data 文件夹中没有找到图片。")
            return
        self._run_worker(self._process_all_worker)

    def _run_worker(self, target) -> None:
        threading.Thread(target=target, daemon=True).start()

    def _process_current_worker(self) -> None:
        category, label, path = self.current_item
        self._set_status(f"正在分析：{label} / {path.name}")
        try:
            _hsv, _masks, overlay, _rgb = engine.analyze_image(path, category)
            self.detail_path, self.final_path = engine.save_analysis_figure(path, category, label)
            panel_images = self._make_panel_images(category, _hsv, _masks, overlay, _rgb)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("处理失败", str(exc)))
            self._set_status("处理失败，请检查图片或依赖环境。")
            return

        self.root.after(0, lambda: self._show_panel_images(panel_images))
        self._set_status(f"分析完成：{path.name}，已显示原图、中间过程和最终结果。")

    def _make_panel_images(self, category: str, hsv, masks, overlay, rgb) -> dict[str, Image.Image]:
        h_channel = hsv[:, :, 0]
        h_channel = np.clip(h_channel.astype(np.float32) / 179 * 255, 0, 255).astype(np.uint8)
        vegetation_key = "grass_candidate" if category == "grassland" else "vegetation"
        return {
            "original": Image.fromarray(rgb),
            "h_channel": Image.fromarray(h_channel).convert("RGB"),
            "vegetation": Image.fromarray(masks[vegetation_key]).convert("RGB"),
            "surface": Image.fromarray(masks["gravel_or_soil"]).convert("RGB"),
            "obstacle": Image.fromarray(masks["obstacle"]).convert("RGB"),
            "overlay": Image.fromarray(overlay),
        }

    def _show_panel_images(self, images: dict[str, Image.Image]) -> None:
        for key, image in images.items():
            self.preview_panels[key].set_image(image)

    def _process_all_worker(self) -> None:
        result_paths: list[tuple[str, Path]] = []
        total = len(self.images)
        try:
            for index, (category, label, path) in enumerate(self.images, start=1):
                self._set_status(f"正在生成全部结果：{index}/{total}  {label} / {path.name}")
                _analysis_path, final_path = engine.save_analysis_figure(path, category, label)
                result_paths.append((f"{label}: {path.name}", final_path))
            summary_path = engine.build_summary(result_paths)
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("处理失败", str(exc)))
            self._set_status("生成失败，请检查图片或依赖环境。")
            return

        self._set_status(f"全部结果生成完成：{summary_path}")

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def open_results_folder(self) -> None:
        engine.RESULT_DIR.mkdir(exist_ok=True)
        os.startfile(engine.RESULT_DIR)


def smoke_test() -> None:
    """不打开窗口，只验证 UI 依赖和算法调用是否正常。"""
    engine.setup_matplotlib_font()
    images = engine.collect_images()
    if not images:
        raise SystemExit("没有找到测试图片")
    category, _label, path = images[0]
    bgr = engine.imread_unicode(path)
    bgr, hsv, gray = engine.preprocess(bgr)
    masks = engine.build_masks(bgr, hsv, gray, category)
    if masks["safe"].shape[:2] != gray.shape[:2]:
        raise SystemExit("掩膜尺寸检查失败")
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
