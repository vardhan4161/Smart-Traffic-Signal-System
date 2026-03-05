import tkinter as tk
from tkinter import ttk
from loguru import logger
from typing import Dict, Any

class GUIDashboard:
    """A simple Tkinter dashboard for monitoring traffic signal states."""
    
    def __init__(self):
        """Initialize the main window and layout."""
        self.root = tk.Tk()
        self.root.title("🚦 Smart Traffic Signal Monitor")
        self.root.geometry("900x650")
        self.root.configure(bg="#1a1a2e")
        
        self.lane_cards = {}
        self.build_layout()
        logger.info("GUI Dashboard initialized.")

    def build_layout(self):
        """Create the header and grid of lane status cards."""
        # Header
        header = tk.Label(self.root, text="🚦 Smart Traffic Signal Control System", 
                          font=("Helvetica", 24, "bold"), fg="white", bg="#1a1a2e", pady=20)
        header.pack()

        # Grid for Lane Cards
        grid_frame = tk.Frame(self.root, bg="#1a1a2e")
        grid_frame.pack(expand=True, fill="both", padx=40, pady=20)

        # 2x2 Grid setup
        lanes = ["North", "South", "East", "West"]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for lane, pos in zip(lanes, positions):
            card = tk.Frame(grid_frame, bg="#16213e", bd=2, relief="groove", padx=20, pady=20)
            card.grid(row=pos[0], column=pos[1], padx=15, pady=15, sticky="nsew")
            
            lane_label = tk.Label(card, text=lane, font=("Helvetica", 18, "bold"), fg="#e94560", bg="#16213e")
            lane_label.pack(anchor="w")
            
            # Indicators
            density_label = tk.Label(card, text="Density: --", font=("Helvetica", 12), fg="white", bg="#16213e")
            density_label.pack(anchor="w", pady=5)
            
            green_label = tk.Label(card, text="Green Time: --s", font=("Helvetica", 12), fg="#08d9d6", bg="#16213e")
            green_label.pack(anchor="w", pady=5)
            
            # Signal visual indicator
            canvas = tk.Canvas(card, width=40, height=40, bg="#16213e", highlightthickness=0)
            canvas.pack(anchor="e")
            circle = canvas.create_oval(5, 5, 35, 35, fill="gray")
            
            self.lane_cards[lane] = {
                "density": density_label,
                "green": green_label,
                "canvas": canvas,
                "circle": circle
            }
            
            grid_frame.grid_columnconfigure(pos[1], weight=1)
            grid_frame.grid_rowconfigure(pos[0], weight=1)

        # Bottom Panel
        bottom_panel = tk.Frame(self.root, bg="#1a1a2e", pady=20)
        bottom_panel.pack(fill="x")
        
        self.status_label = tk.Label(bottom_panel, text="System Ready | Cycle: 0", 
                                     font=("Helvetica", 12), fg="#95a5a6", bg="#1a1a2e")
        self.status_label.pack()
        
        btn_frame = tk.Frame(bottom_panel, bg="#1a1a2e")
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Run Simulation", command=self.on_run, width=15, 
                  bg="#0f3460", fg="white", font=("Helvetica", 10, "bold")).pack(side="left", padx=10)
        tk.Button(btn_frame, text="Export Report", command=self.on_export, width=15,
                  bg="#533483", fg="white", font=("Helvetica", 10, "bold")).pack(side="left", padx=10)

    def update_lane(self, lane_name: str, data: Dict[str, Any]):
        """Update a lane card with new simulation data."""
        if lane_name in self.lane_cards:
            card = self.lane_cards[lane_name]
            card["density"].config(text=f"Density: {data.get('density', 0.0):.1f} ({data.get('level', 'LOW')})")
            card["green"].config(text=f"Green Time: {data.get('green_time', 0.0)}s")
            
            # Animate signal
            level = data.get("level", "LOW")
            color = "#ff4d4d" if level == "CRITICAL" else "#ff9f43" if level == "HIGH" else "#00d2d3" if level == "MEDIUM" else "#1dd1a1"
            card["canvas"].itemconfig(card["circle"], fill=color)

    def on_run(self):
        """Placeholder for run action."""
        logger.info("Run Simulation clicked")
        self.status_label.config(text="Running Simulation...")

    def on_export(self):
        """Placeholder for export action."""
        logger.info("Export Report clicked")

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()
