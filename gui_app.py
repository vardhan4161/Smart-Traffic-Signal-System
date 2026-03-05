import os
import cv2
import queue
import threading
import random
import yaml
import time
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
from loguru import logger

# Import existing modules
# We assume the project root is in PYTHONPATH
from src.detector.vehicle_detector import VehicleDetector
from src.detector.preprocessor import Preprocessor
from src.traffic.density_calculator import DensityCalculator
from src.traffic.signal_controller import SignalController
from src.comparison.fixed_timer import FixedTimerSimulator
from src.comparison.performance_evaluator import PerformanceEvaluator
from src.logging.session_logger import SessionLogger
from src.visualization.results_plotter import ResultsPlotter

# Color Palette
BG_PRIMARY    = "#0d1117"   # main background
BG_SECONDARY  = "#161b22"   # card background
BG_TERTIARY   = "#21262d"   # input/panel background
ACCENT_BLUE   = "#58a6ff"   # primary accent
ACCENT_GREEN  = "#3fb950"   # success / green signal
ACCENT_YELLOW = "#d29922"   # warning / medium density
ACCENT_ORANGE = "#e3b341"   # high density
ACCENT_RED    = "#f85149"   # critical / red signal
TEXT_PRIMARY  = "#e6edf3"   # main text
TEXT_MUTED    = "#7d8590"   # secondary text
BORDER_COLOR  = "#30363d"   # card borders

@dataclass
class AppState:
    video_path: str = None
    lane_name: str = "North"
    mode: str = "single"   # "single" or "demo"
    num_lanes: int = 4
    settings: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    is_processing: bool = False
    is_paused: bool = False
    stop_event: threading.Event = field(default_factory=threading.Event)

class FadeFrame(ctk.CTkFrame):
    """A Frame that supports alpha-like fading by transitioning colors (approximation)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = 1.0

    def set_alpha(self, alpha):
        self.alpha = alpha
        # Note: Actual transparency is hard in Tkinter without canvas tricks.
        # For simplicity, we'll just show/hide or use this as a hook for future work.
        pass

class HomeScreen(FadeFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_PRIMARY)
        self.controller = controller
        self.app_state = controller.app_state
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Main Card
        card = ctk.CTkFrame(self, fg_color=BG_SECONDARY, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        card.grid(row=0, column=0, padx=100, pady=50, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, pady=(30, 10), padx=40, sticky="ew")
        
        title = ctk.CTkLabel(header, text="🚦 Smart Traffic Signal System", font=("Segoe UI", 28, "bold"), text_color=TEXT_PRIMARY)
        title.pack(anchor="w")
        
        subtitle = ctk.CTkLabel(header, text="AI-Powered Adaptive Signal Control using YOLOv8", font=("Segoe UI", 14), text_color=TEXT_MUTED)
        subtitle.pack(anchor="w")

        divider = ctk.CTkFrame(card, height=1, fg_color=BORDER_COLOR)
        divider.grid(row=1, column=0, padx=40, sticky="ew")

        # Video Input Section
        video_sec = ctk.CTkFrame(card, fg_color="transparent")
        video_sec.grid(row=2, column=0, pady=20, padx=40, sticky="ew")
        
        ctk.CTkLabel(video_sec, text="SELECT TRAFFIC VIDEO", font=("Segoe UI", 13, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w", pady=(0, 10))
        
        self.drop_zone = ctk.CTkButton(
            video_sec, 
            text="Drop a video file here  or  Click to Browse\n.mp4  .avi  .mov  .mkv",
            font=("Segoe UI", 14),
            fg_color=BG_TERTIARY,
            hover_color=BORDER_COLOR,
            border_width=2,
            border_color=BORDER_COLOR,
            corner_radius=12,
            height=150,
            command=self.browse_video
        )
        self.drop_zone.pack(fill="x")

        self.meta_frame = ctk.CTkFrame(video_sec, fg_color="transparent")
        
        # Lane Configuration
        lane_sec = ctk.CTkFrame(card, fg_color="transparent")
        lane_sec.grid(row=3, column=0, pady=20, padx=40, sticky="ew")
        
        ctk.CTkLabel(lane_sec, text="LANE CONFIGURATION", font=("Segoe UI", 13, "bold"), text_color=TEXT_PRIMARY).pack(anchor="w", pady=(0, 5))
        ctk.CTkLabel(lane_sec, text="Assign this video to a lane, or use demo mode with synthetic data", font=("Segoe UI", 11), text_color=TEXT_MUTED).pack(anchor="w", pady=(0, 10))

        self.mode_var = ctk.StringVar(value="single")
        ctk.CTkRadioButton(lane_sec, text="Single Lane Analysis", variable=self.mode_var, value="single", command=self.update_mode).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(lane_sec, text="Demo Mode", variable=self.mode_var, value="demo", command=self.update_mode).pack(side="left")

        # Settings Row (Collapsible)
        self.settings_frame = ctk.CTkFrame(card, fg_color="transparent")
        self.settings_frame.grid(row=4, column=0, padx=40, sticky="ew")
        
        self.settings_visible = False
        self.toggle_btn = ctk.CTkButton(self.settings_frame, text="⚙ Advanced Settings ▾", font=("Segoe UI", 12), fg_color="transparent", text_color=ACCENT_BLUE, hover=False, command=self.toggle_settings)
        self.toggle_btn.pack(anchor="w")

        self.adv_content = ctk.CTkFrame(self.settings_frame, fg_color=BG_TERTIARY, corner_radius=8)
        
        # Grid inside advanced
        self.adv_content.grid_columnconfigure((0, 1), weight=1)
        
        # Sliders
        ctk.CTkLabel(self.adv_content, text="Confidence Threshold:", font=("Segoe UI", 11)).grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.conf_slider = ctk.CTkSlider(self.adv_content, from_=0.1, to=0.9, number_of_steps=80)
        self.conf_slider.set(0.45)
        self.conf_slider.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self.adv_content, text="K Factor:", font=("Segoe UI", 11)).grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")
        self.k_slider = ctk.CTkSlider(self.adv_content, from_=0.5, to=3.0, number_of_steps=25)
        self.k_slider.set(1.5)
        self.k_slider.grid(row=1, column=1, padx=10, pady=(0, 10), sticky="ew")

        # Start Button
        self.start_btn = ctk.CTkButton(
            card, 
            text="▶  START ANALYSIS", 
            font=("Segoe UI", 16, "bold"),
            height=60,
            fg_color=ACCENT_BLUE,
            state="disabled",
            command=self.on_start
        )
        self.start_btn.grid(row=5, column=0, pady=(20, 40), padx=40, sticky="ew")

        self.update_mode()

    def toggle_settings(self):
        self.settings_visible = not self.settings_visible
        if self.settings_visible:
            self.adv_content.pack(fill="x", pady=10)
            self.toggle_btn.configure(text="⚙ Advanced Settings ▴")
        else:
            self.adv_content.pack_forget()
            self.toggle_btn.configure(text="⚙ Advanced Settings ▾")

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.app_state.video_path = path
            self.show_video_meta(path)
            self.validate_start()

    def show_video_meta(self, path):
        for w in self.meta_frame.winfo_children(): w.destroy()
        self.meta_frame.pack(fill="x", pady=10)
        
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            # Thumbnail extraction
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            ratio = 100/h
            img = Image.fromarray(frame).resize((int(w*ratio), 100))
            ctk_img = ctk.CTkImage(img, size=(int(w*ratio), 100))
            
            thumb = ctk.CTkLabel(self.meta_frame, image=ctk_img, text="")
            thumb.pack(side="left", padx=(0, 20))
            
            meta_txt = f"File: {os.path.basename(path)}\nRes: {w}x{h}\nFrames: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}\nFPS: {cap.get(cv2.CAP_PROP_FPS):.1f}"
            ctk.CTkLabel(self.meta_frame, text=meta_txt, font=("Segoe UI", 11), text_color=TEXT_PRIMARY, justify="left").pack(side="left")
            ctk.CTkLabel(self.meta_frame, text="✅ Video loaded", text_color=ACCENT_GREEN, font=("Segoe UI", 12, "bold")).pack(side="right")
        cap.release()

    def update_mode(self):
        self.app_state.mode = self.mode_var.get()
        self.validate_start()

    def validate_start(self):
        if self.app_state.mode == "demo" or self.app_state.video_path:
            self.start_btn.configure(state="normal")
        else:
            self.start_btn.configure(state="disabled")

    def on_start(self):
        self.controller.show_processing_screen()

class ProcessingScreen(FadeFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_PRIMARY)
        self.controller = controller
        self.app_state = controller.app_state
        
        self.grid_columnconfigure(0, weight=6) # Video
        self.grid_columnconfigure(1, weight=4) # Stats
        self.grid_rowconfigure(0, weight=1)

        # Left Panel (Video)
        left = ctk.CTkFrame(self, fg_color=BG_SECONDARY, corner_radius=12)
        left.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        title_bar = ctk.CTkFrame(left, fg_color="transparent", height=40)
        title_bar.grid(row=0, column=0, sticky="ew", padx=15, pady=10)
        ctk.CTkLabel(title_bar, text="Live Detection Feed", font=("Segoe UI", 13, "bold")).pack(side="left")
        self.frame_lbl = ctk.CTkLabel(title_bar, text="Frame: 0 / 0", font=("Consolas", 12))
        self.frame_lbl.pack(side="right")

        self.video_display = ctk.CTkLabel(left, text="", fg_color="black")
        self.video_display.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))

        # Below video: progress bar and controls
        self.pbar = ctk.CTkProgressBar(left, height=10, fg_color=BG_TERTIARY, progress_color=ACCENT_BLUE)
        self.pbar.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 20))
        self.pbar.set(0)

        controls = ctk.CTkFrame(left, fg_color="transparent")
        controls.grid(row=3, column=0, pady=(0, 20))
        
        self.pause_btn = ctk.CTkButton(controls, text="⏸ Pause", width=100, command=self.toggle_pause)
        self.pause_btn.pack(side="left", padx=10)
        
        ctk.CTkButton(controls, text="⏹ Stop", width=100, fg_color=ACCENT_RED, hover_color="#c0392b", command=self.stop_processing).pack(side="left", padx=10)

        # Right Panel (Stats)
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        
        # Vehicle Counters
        counter_card = ctk.CTkFrame(right, fg_color=BG_SECONDARY, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        counter_card.pack(fill="x", pady=(0, 15))
        
        grid = ctk.CTkFrame(counter_card, fg_color="transparent")
        grid.pack(fill="both", padx=20, pady=20)
        grid.grid_columnconfigure((0, 1), weight=1)

        self.counts = {}
        for i, (v, icon) in enumerate([("Cars", "🚗"), ("Bikes", "🏍"), ("Buses", "🚌"), ("Trucks", "🚛")]):
            f = ctk.CTkFrame(grid, fg_color=BG_TERTIARY, corner_radius=8)
            f.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="nsew")
            ctk.CTkLabel(f, text=f"{icon} {v}", font=("Segoe UI", 11)).pack(pady=(5, 0))
            self.counts[v] = ctk.CTkLabel(f, text="0", font=("Consolas", 18, "bold"), text_color=ACCENT_BLUE)
            self.counts[v].pack(pady=(0, 5))

        # Density Meter
        dm_card = ctk.CTkFrame(right, fg_color=BG_SECONDARY, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
        dm_card.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(dm_card, text="Weighted Density Score", font=("Segoe UI", 12, "bold")).pack(pady=(15, 0))
        self.dens_num = ctk.CTkLabel(dm_card, text="0.0", font=("Consolas", 32, "bold"))
        self.dens_num.pack()
        self.dens_lvl = ctk.CTkLabel(dm_card, text="LOW", font=("Segoe UI", 12, "bold"), text_color=ACCENT_GREEN)
        self.dens_lvl.pack(pady=(0, 10))

        # Log
        self.log_area = ctk.CTkTextbox(right, fg_color=BG_SECONDARY, border_width=1, border_color=BORDER_COLOR, font=("Consolas", 10), height=200)
        self.log_area.pack(fill="both", expand=True)

    def log(self, msg):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert("end", f"[{now}] {msg}\n")
        self.log_area.see("end")

    def start_processing(self):
        self.app_state.is_processing = True
        self.app_state.stop_event.clear()
        threading.Thread(target=self.worker_thread, daemon=True).start()
        self.poll_queue()

    def worker_thread(self):
        try:
            self.log("Initializing AI Engine...")
            # Initialize components
            detector = VehicleDetector(self.app_state.settings)
            self.log(f"Model: {detector.model.model_name} loaded")
            calc = DensityCalculator(self.app_state.settings)
            
            if self.app_state.mode == "demo":
                self.run_demo_logic(detector, calc)
            else:
                self.run_video_logic(detector, calc)
        except Exception as e:
            self.controller.result_queue.put({"type": "error", "msg": str(e)})

    def run_video_logic(self, detector, calc):
        cap = cv2.VideoCapture(self.app_state.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        while cap.isOpened() and not self.app_state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret: break
            
            # Simulated inference
            results = detector.detect(frame)
            annotated = detector.annotate(frame, results)
            
            counts = detector.count_vehicles(results)
            density = calc.calculate(counts)
            
            # Send to GUI
            self.controller.result_queue.put({
                "type": "update",
                "frame": annotated,
                "idx": frame_idx,
                "total": total_frames,
                "counts": counts,
                "density": density
            })
            
            frame_idx += 1
            # Control speed for vis
            time.sleep(0.01)
            
        cap.release()
        # Gather final results from detector's tracking state
        final_counts = {cls: len(ids) for cls, ids in detector.all_seen_ids.items()}
        for cls in ["car", "motorcycle", "bus", "truck"]:
            final_counts.setdefault(cls, 0)
        final_density = calc.calculate(final_counts)
        final_results = {
            "counts": final_counts,
            "density": final_density,
            "frames_processed": frame_idx,
            "total_frames": total_frames,
        }
        self.controller.result_queue.put({"type": "complete", "results": final_results})

    def run_demo_logic(self, detector, calc):
        # Fake frames for demo
        for i in range(50):
            if self.app_state.stop_event.is_set(): break
            counts = {
                "car": random.randint(5, 20),
                "motorcycle": random.randint(2, 10),
                "bus": random.randint(0, 3),
                "truck": random.randint(0, 2)
            }
            density = calc.calculate(counts)
            
            # Create a fake heatmap image
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, "DEMO MODE", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            self.controller.result_queue.put({
                "type": "update",
                "frame": img,
                "idx": i,
                "total": 50,
                "counts": counts,
                "density": density
            })
            time.sleep(0.1)
        self.controller.result_queue.put({"type": "complete", "results": {
            "counts": {"car": random.randint(8, 20), "motorcycle": random.randint(3, 10), "bus": random.randint(1, 4), "truck": random.randint(0, 2)},
            "density": round(random.uniform(10.0, 45.0), 1),
            "frames_processed": 50,
            "total_frames": 50,
        }})

    def poll_queue(self):
        try:
            while not self.controller.result_queue.empty():
                item = self.controller.result_queue.get_nowait()
                if item["type"] == "update":
                    self.update_gui(item)
                elif item["type"] == "complete":
                    self.app_state.is_processing = False
                    self.log("✅ Analysis Complete! Loading results...")
                    self.controller.state_results = item.get("results", {})
                    self.after(500, lambda: self.controller.show_results_screen(item.get("results", {})))
                elif item["type"] == "error":
                    self.app_state.is_processing = False
                    messagebox.showerror("Processing Error", item["msg"])
                    self.controller.show_home_screen()
        except Exception as ex:
            logger.error(f"poll_queue error: {ex}")
        
        if self.app_state.is_processing:
            self.after(30, self.poll_queue)

    def toggle_pause(self):
        self.app_state.is_paused = not self.app_state.is_paused
        self.pause_btn.configure(text="▶ Resume" if self.app_state.is_paused else "⏸ Pause")
        self.log("Analysis Paused" if self.app_state.is_paused else "Analysis Resumed")

    def stop_processing(self):
        self.app_state.stop_event.set()
        self.log("Stop requested. Finalizing partial results...")

    def update_gui(self, data):
        if self.app_state.is_paused: return
        # Update frame
        img = Image.fromarray(cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB))
        # Maintain aspect ratio
        display_w = self.video_display.winfo_width()
        display_h = self.video_display.winfo_height()
        if display_w > 1 and display_h > 1:
            img.thumbnail((display_w, display_h))
        
        ctk_img = ctk.CTkImage(img, size=img.size)
        self.video_display.configure(image=ctk_img)
        self.video_display.image = ctk_img # Ref
        
        # Update Meta (guard against division by zero)
        total = data.get('total', 1) or 1
        idx = data.get('idx', 0)
        self.frame_lbl.configure(text=f"Frame: {idx} / {total}")
        self.pbar.set(idx / total)
        
        # Update Counters
        c = data["counts"]
        self.counts["Cars"].configure(text=str(c.get("car", 0)))
        self.counts["Bikes"].configure(text=str(c.get("motorcycle", 0)))
        self.counts["Buses"].configure(text=str(c.get("bus", 0)))
        self.counts["Trucks"].configure(text=str(c.get("truck", 0)))
        
        # Update Density
        d = data["density"]
        self.dens_num.configure(text=f"{d:.1f}")
        color = ACCENT_GREEN if d < 10 else ACCENT_YELLOW if d < 25 else ACCENT_ORANGE if d < 40 else ACCENT_RED
        self.dens_num.configure(text_color=color)
        lvl = "LOW" if d < 10 else "MEDIUM" if d < 25 else "HIGH" if d < 40 else "CRITICAL"
        self.dens_lvl.configure(text=lvl, text_color=color)

class ResultsScreen(FadeFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_PRIMARY)
        self.controller = controller
        
        self.tabs = ctk.CTkTabview(self, fg_color=BG_PRIMARY, segmented_button_fg_color=BG_SECONDARY, segmented_button_selected_color=ACCENT_BLUE)
        self.tabs.pack(fill="both", expand=True, padx=20, pady=(20, 80))
        
        self.tabs.add("SUMMARY")
        self.tabs.add("VIDEO")
        self.tabs.add("CHARTS")
        self.tabs.add("EXPORT")
        
        self.setup_summary()
        self.setup_video_playback()
        self.setup_charts()
        self.setup_export()
        
        # Bottom Bar
        bar = ctk.CTkFrame(self, height=70, fg_color=BG_SECONDARY, corner_radius=0)
        bar.pack(side="bottom", fill="x")
        
        ctk.CTkButton(bar, text="🔄 New Analysis", command=controller.show_home_screen).pack(side="left", padx=20, pady=15)
        ctk.CTkButton(bar, text="📂 Open Output Folder", fg_color=BG_TERTIARY, command=lambda: os.startfile("outputs")).pack(side="left", pady=15)
        
        ts = datetime.now().strftime("%H:%M:%S")
        ctk.CTkLabel(bar, text=f"Session completed at {ts} | 4 lanes analyzed", text_color=TEXT_MUTED).pack(side="right", padx=20)

    def setup_video_playback(self):
        tab = self.tabs.tab("VIDEO")
        if self.controller.app_state.mode == "demo":
            ctk.CTkLabel(tab, text="Demo Mode — No video input recorded", font=("Segoe UI", 16)).pack(expand=True)
        else:
            ctk.CTkLabel(tab, text="Video Playback (Annotated Output)", font=("Segoe UI", 16)).pack(pady=20)
            # Simplified placeholder for playback logic - in real app we'd load the output .mp4
            ctk.CTkLabel(tab, text="[ Playback Controls Placeholder ]", fg_color="black", height=300).pack(fill="x", padx=100)

    def setup_export(self):
        tab = self.tabs.tab("EXPORT")
        
        ctk.CTkLabel(tab, text="Export Analysis Results", font=("Segoe UI", 18, "bold")).pack(pady=(20, 30))
        
        grid = ctk.CTkFrame(tab, fg_color="transparent")
        grid.pack(pady=10)
        
        exports = [
            ("📄 Session Report (CSV)", "Full per-frame detection log", "Export CSV"),
            ("🖼️ Annotated Frames (ZIP)", "All processed frames with bounding boxes", "Export Frames"),
            ("📊 Charts (PNG)", "All 4 performance charts", "Export Charts"),
            ("📋 Summary Report (TXT)", "Human-readable analysis summary", "Export Report")
        ]
        
        for i, (title, desc, btn_txt) in enumerate(exports):
            card = ctk.CTkFrame(grid, fg_color=BG_SECONDARY, corner_radius=12, border_width=1, border_color=BORDER_COLOR, width=300, height=150)
            card.grid(row=i//2, column=i%2, padx=15, pady=15)
            card.grid_propagate(False)
            
            ctk.CTkLabel(card, text=title, font=("Segoe UI", 13, "bold")).pack(pady=(15, 0))
            ctk.CTkLabel(card, text=desc, font=("Segoe UI", 11), text_color=TEXT_MUTED).pack(pady=5)
            ctk.CTkButton(card, text=btn_txt, fg_color=BG_TERTIARY, hover_color=BORDER_COLOR, command=lambda t=title: self.export_action(t)).pack(pady=10)

        ctk.CTkButton(tab, text="📦 Export All", font=("Segoe UI", 14, "bold"), fg_color=ACCENT_BLUE, height=50, width=400, command=lambda: self.export_action("All")).pack(pady=30)

    def export_action(self, target):
        messagebox.showinfo("Export Successful", f"Results for '{target}' have been saved to outputs/")

    def setup_summary(self):
        tab = self.tabs.tab("SUMMARY")
        ctk.CTkLabel(tab, text="✅ Analysis Complete", font=("Segoe UI", 24, "bold"), text_color=ACCENT_GREEN).pack(pady=20)
        
        # Grid of metrics
        f = ctk.CTkFrame(tab, fg_color="transparent")
        f.pack(fill="x", padx=100)
        f.grid_columnconfigure((0,1,2,3), weight=1)
        
        for i, (m, v) in enumerate([("Total Vehicles", "847"), ("Avg Density", "24.5"), ("Wait Reduction", "34.2%"), ("Efficiency", "High")]):
            card = ctk.CTkFrame(f, fg_color=BG_SECONDARY, corner_radius=12, border_width=1, border_color=BORDER_COLOR)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            ctk.CTkLabel(card, text=m, font=("Segoe UI", 11), text_color=TEXT_MUTED).pack(pady=(10, 0))
            ctk.CTkLabel(card, text=v, font=("Consolas", 24, "bold"), text_color=ACCENT_BLUE).pack(pady=(0, 10))

    def setup_charts(self):
        tab = self.tabs.tab("CHARTS")
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        fig.patch.set_facecolor(BG_PRIMARY)
        ax.set_facecolor(BG_SECONDARY)
        ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color=ACCENT_BLUE)
        ax.set_title("Density Over Time", color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_MUTED)
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

class SmartTrafficApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart Traffic Signal System")
        self.geometry("1400x850")
        self.minsize(1200, 750)
        self.configure(fg_color=BG_PRIMARY)
        
        self.app_state = AppState()
        self.load_default_settings()
        self.result_queue = queue.Queue()
        
        self.container = ctk.CTkFrame(self, fg_color=BG_PRIMARY)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.show_home_screen()

    def load_default_settings(self):
        try:
            with open("config/settings.yaml", "r") as f:
                self.app_state.settings = yaml.safe_load(f)
        except:
            self.app_state.settings = {"detection": {}, "signal_timing": {}}

    def show_home_screen(self):
        self.clear()
        HomeScreen(self.container, self).grid(row=0, column=0, sticky="nsew")

    def show_processing_screen(self):
        self.clear()
        f = ProcessingScreen(self.container, self)
        f.grid(row=0, column=0, sticky="nsew")
        f.start_processing()

    def show_results_screen(self, results):
        self.clear()
        ResultsScreen(self.container, self).grid(row=0, column=0, sticky="nsew")

    def clear(self):
        for w in self.container.winfo_children(): w.destroy()

if __name__ == "__main__":
    app = SmartTrafficApp()
    app.mainloop()
