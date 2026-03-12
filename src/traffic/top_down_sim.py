import customtkinter as ctk
import random
import time
import threading
from typing import Dict, List, Optional, Any
from loguru import logger

# Constants for the simulation
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 800
ROAD_WIDTH = 120
VEHICLE_SIZE = (20, 30)  # (width, length) relative to orientation
STOP_DISTANCE = 40       # Distance from center where vehicles stop

# Colors
COLOR_ROAD = "#2c3e50"
COLOR_LINE = "#ecf0f1"
COLOR_CAR = "#3498db"
COLOR_BUS = "#e67e22"
COLOR_PEDESTRIAN = "#9b59b6"
COLOR_AMBULANCE = "#ffffff"

class VehicleEntity:
    def __init__(self, canvas: ctk.CTkCanvas, lane: str, v_type: str = "car"):
        self.canvas = canvas
        self.lane = lane
        self.v_type = v_type
        self.speed = random.uniform(2, 4)
        self.stopped = False
        self.id = None
        
        # Initial position and velocity
        self.x, self.y = self._get_spawn_pos()
        self.vx, self.vy = self._get_velocity()
        
        # Create shape
        color = COLOR_CAR
        if v_type == "bus": color = COLOR_BUS
        elif v_type == "ambulance": color = COLOR_AMBULANCE
        elif v_type == "person": color = COLOR_PEDESTRIAN
        
        w, l = VEHICLE_SIZE
        if lane in ["East", "West"]: w, l = l, w
        
        self.id = canvas.create_rectangle(
            self.x - w/2, self.y - l/2, 
            self.x + w/2, self.y + l/2, 
            fill=color, outline="white", width=1
        )

    def _get_spawn_pos(self):
        offset = 30 # Lane offset from center
        if self.lane == "North": return (CANVAS_WIDTH/2 - offset, -50)
        if self.lane == "South": return (CANVAS_WIDTH/2 + offset, CANVAS_HEIGHT + 50)
        if self.lane == "West":  return (-50, CANVAS_HEIGHT/2 + offset)
        if self.lane == "East":  return (CANVAS_WIDTH + 50, CANVAS_HEIGHT/2 - offset)
        return (0,0)

    def _get_velocity(self):
        s = self.speed
        if self.lane == "North": return (0, s)
        if self.lane == "South": return (0, -s)
        if self.lane == "West":  return (s, 0)
        if self.lane == "East":  return (-s, 0)
        return (0,0)

    def move(self, is_green: bool):
        # Stop line logic
        at_stop_line = False
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        
        if self.lane == "North" and self.y < cy - STOP_DISTANCE and self.y + self.speed >= cy - STOP_DISTANCE: at_stop_line = True
        if self.lane == "South" and self.y > cy + STOP_DISTANCE and self.y - self.speed <= cy + STOP_DISTANCE: at_stop_line = True
        if self.lane == "West"  and self.x < cx - STOP_DISTANCE and self.x + self.speed >= cx - STOP_DISTANCE: at_stop_line = True
        if self.lane == "East"  and self.x > cx + STOP_DISTANCE and self.x - self.speed <= cx + STOP_DISTANCE: at_stop_line = True
        
        if at_stop_line and not is_green:
            self.stopped = True
        elif is_green:
            self.stopped = False
            
        if not self.stopped:
            self.canvas.move(self.id, self.vx, self.vy)
            self.x += self.vx
            self.y += self.vy

    def is_off_screen(self):
        return (self.x < -100 or self.x > CANVAS_WIDTH + 100 or 
                self.y < -100 or self.y > CANVAS_HEIGHT + 100)

class TopDownSimulator(ctk.CTkFrame):
    def __init__(self, parent, config, signal_callback=None):
        super().__init__(parent, fg_color="#0d1117")
        self.config = config
        self.signal_callback = signal_callback
        
        self.canvas = ctk.CTkCanvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, 
                                     bg=COLOR_ROAD, highlightthickness=0)
        self.canvas.pack(expand=True, padx=20, pady=20)
        
        self.vehicles: Dict[str, List[VehicleEntity]] = {l: [] for l in ["North", "South", "East", "West"]}
        self.signals: Dict[str, str] = {l: "red" for l in ["North", "South", "East", "West"]}
        self.running = False
        
        self._draw_static_elements()
        self._draw_lights()

    def _draw_static_elements(self):
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        w = ROAD_WIDTH
        # Vertical Road
        self.canvas.create_rectangle(cx-w/2, 0, cx+w/2, CANVAS_HEIGHT, fill="#34495e", outline="")
        # Horizontal Road
        self.canvas.create_rectangle(0, cy-w/2, CANVAS_WIDTH, cy+w/2, fill="#34495e", outline="")
        # Intersection Box
        self.canvas.create_rectangle(cx-w/2, cy-w/2, cx+w/2, cy+w/2, fill="#2c3e50", outline=COLOR_LINE)
        
        # Lane Lines
        self.canvas.create_line(cx, 0, cx, cy-w/2, fill=COLOR_LINE, dash=(10,10))
        self.canvas.create_line(cx, cy+w/2, cx, CANVAS_HEIGHT, fill=COLOR_LINE, dash=(10,10))
        self.canvas.create_line(0, cy, cx-w/2, cy, fill=COLOR_LINE, dash=(10,10))
        self.canvas.create_line(cx+w/2, cy, CANVAS_WIDTH, cy, fill=COLOR_LINE, dash=(10,10))

    def _draw_lights(self):
        self.light_objs = {}
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        w = ROAD_WIDTH/2 + 20
        positions = {
            "North": (cx - 50, cy - w),
            "South": (cx + 50, cy + w),
            "East":  (cx + w, cy - 50),
            "West":  (cx - w, cy + 50)
        }
        for lane, (x, y) in positions.items():
            self.light_objs[lane] = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="red", outline="white")

    def set_light(self, lane: str, state: str):
        color = "red"
        if state.upper() == "GREEN": color = "#2ecc71"
        elif state.upper() == "YELLOW": color = "#f1c40f"
        self.canvas.itemconfig(self.light_objs[lane], fill=color)
        self.signals[lane] = color

    def spawn_vehicle(self, lane: str, v_type: str = "car"):
        # Prevent overlaps at spawn
        if self.vehicles[lane]:
            last = self.vehicles[lane][-1]
            dist = 0
            if lane == "North": dist = last.y - (-50)
            elif lane == "South": dist = (CANVAS_HEIGHT+50) - last.y
            elif lane == "West": dist = last.x - (-50)
            elif lane == "East": dist = (CANVAS_WIDTH+50) - last.x
            if abs(dist) < 60: return
            
        v = VehicleEntity(self.canvas, lane, v_type)
        self.vehicles[lane].append(v)

    def update_sim(self):
        if not self.running: return
        
        # 1. Random Spawning
        for lane in self.vehicles:
            if random.random() < 0.02: # ~2% chance per frame
                v_type = random.choices(["car", "bus", "person", "ambulance"], weights=[70, 15, 10, 5])[0]
                self.spawn_vehicle(lane, v_type)
        
        # 2. Movement logic
        for lane, v_list in self.vehicles.items():
            is_green = self.signals[lane] == "#2ecc71" or self.signals[lane] == "#f1c40f"
            
            for i, v in enumerate(v_list):
                # Distance to vehicle in front
                can_move = True
                if i > 0:
                    front = v_list[i-1]
                    dist = 0
                    if lane == "North": dist = front.y - v.y
                    elif lane == "South": dist = v.y - front.y
                    elif lane == "West": dist = front.x - v.x
                    elif lane == "East": dist = v.x - front.x
                    
                    if 0 < dist < 50:
                        can_move = False
                
                if can_move:
                    v.move(is_green)
                else:
                    v.stopped = True
            
            # Remove off-screen
            self.vehicles[lane] = [v for v in v_list if not v.is_off_screen() or (v.canvas.delete(v.id) and False)]

        # 3. Report densities for adaptive logic
        if self.signal_callback:
            counts = {l: self._count_lane(l) for l in self.vehicles}
            self.signal_callback(counts)

        self.after(30, self.update_sim)

    def _count_lane(self, lane: str) -> Dict[str, int]:
        c = {"car": 0, "bus": 0, "person": 0, "truck": 0, "motorcycle": 0}
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        for v in self.vehicles[lane]:
            # Only count vehicles waiting at or approaching intersection
            in_zone = False
            if lane == "North" and v.y < cy: in_zone = True
            elif lane == "South" and v.y > cy: in_zone = True
            elif lane == "West" and v.x < cx: in_zone = True
            elif lane == "East" and v.x > cx: in_zone = True
            
            if in_zone:
                t = v.v_type
                if t == "ambulance": c["truck"] += 15 # Emergency boost
                elif t in c: c[t] += 1
                else: c["car"] += 1
        return c

    def start(self):
        self.running = True
        self.update_sim()

    def stop(self):
        self.running = False
        for l in self.vehicles:
            for v in self.vehicles[l]:
                self.canvas.delete(v.id)
            self.vehicles[l].clear()
