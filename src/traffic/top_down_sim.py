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
STOP_DISTANCE = 50       # Distance from center where vehicles stop

# Colors - Modern Premium Palette
COLOR_GRASS = "#1b5e20"
COLOR_SIDEWALK = "#455a64"
COLOR_ROAD = "#263238"
COLOR_LINE = "#ffffff"
COLOR_ZEBRA = "#cfd8dc"
COLOR_CAR = "#00b0ff"
COLOR_BUS = "#ff9100"
COLOR_PEDESTRIAN = "#e040fb"
COLOR_AMBULANCE = "#f44336"
COLOR_HEADLIGHT = "#fffde7"
COLOR_TAILLIGHT = "#ff1744"

class VehicleEntity:
    def __init__(self, canvas: ctk.CTkCanvas, lane: str, v_type: str = "car"):
        self.canvas = canvas
        self.lane = lane
        self.v_type = v_type
        self.speed = random.uniform(2.5, 4.5)
        self.stopped = False
        
        # Position & Rendering
        self.x, self.y = self._get_spawn_pos()
        self.vx, self.vy = 0, 0
        self.id = None
        self.lights = []
        self.particles = []
        
        # Physics state
        self.current_speed = self.speed
        self.target_speed = self.speed
        
        self._render()

    def _render(self):
        color = COLOR_CAR
        if self.v_type == "bus": color = COLOR_BUS
        elif self.v_type == "ambulance": color = COLOR_AMBULANCE
        elif self.v_type == "person": color = COLOR_PEDESTRIAN
        
        w, l = VEHICLE_SIZE
        if self.lane in ["East", "West"]: w, l = l, w
        
        # Rounded shape (Polygon)
        points = self._get_shape_points(self.x, self.y, w, l)
        self.id = self.canvas.create_polygon(points, fill=color, outline="white", width=1, smooth=True)
        
        # Add Headlights
        lx1, ly1, lx2, ly2 = self._get_light_pos(w, l, "front")
        self.lights.append(self.canvas.create_oval(lx1-3, ly1-3, lx1+3, ly1+3, fill=COLOR_HEADLIGHT, outline=""))
        self.lights.append(self.canvas.create_oval(lx2-3, ly2-3, lx2+3, ly2+3, fill=COLOR_HEADLIGHT, outline=""))
        
        # Add Taillights
        tx1, ty1, tx2, ty2 = self._get_light_pos(w, l, "back")
        self.lights.append(self.canvas.create_oval(tx1-2, ty1-2, tx1+2, ty1+2, fill=COLOR_TAILLIGHT, outline=""))
        self.lights.append(self.canvas.create_oval(tx2-2, ty2-2, tx2+2, ty2+2, fill=COLOR_TAILLIGHT, outline=""))

    def _get_shape_points(self, x, y, w, l):
        return [x-w/2, y-l/2, x+w/2, y-l/2, x+w/2, y+l/2, x-w/2, y+l/2]

    def _get_light_pos(self, w, l, side="front"):
        if self.lane == "North":
            y = (self.y + l/2) if side == "front" else (self.y - l/2)
            return (self.x - w/3, y, self.x + w/3, y)
        if self.lane == "South":
            y = (self.y - l/2) if side == "front" else (self.y + l/2)
            return (self.x - w/3, y, self.x + w/3, y)
        if self.lane == "West":
            x = (self.x + w/2) if side == "front" else (self.x - w/2)
            return (x, self.y - l/3, x, self.y + l/3)
        if self.lane == "East":
            x = (self.x - w/2) if side == "front" else (self.x + w/2)
            return (x, self.y - l/3, x, self.y + l/3)
        return (0,0,0,0)

    def _get_spawn_pos(self):
        offset = 30
        if self.lane == "North": return (CANVAS_WIDTH/2 - offset, -50)
        if self.lane == "South": return (CANVAS_WIDTH/2 + offset, CANVAS_HEIGHT + 50)
        if self.lane == "West":  return (-50, CANVAS_HEIGHT/2 + offset)
        if self.lane == "East":  return (CANVAS_WIDTH + 50, CANVAS_HEIGHT/2 - offset)
        return (0,0)

    def move(self, is_green: bool):
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        
        # Stop line logic
        at_stop_line = False
        if self.lane == "North" and cy - STOP_DISTANCE - 10 < self.y < cy - STOP_DISTANCE: at_stop_line = True
        if self.lane == "South" and cy + STOP_DISTANCE < self.y < cy + STOP_DISTANCE + 10: at_stop_line = True
        if self.lane == "West"  and cx - STOP_DISTANCE - 10 < self.x < cx - STOP_DISTANCE: at_stop_line = True
        if self.lane == "East"  and cx + STOP_DISTANCE < self.x < cx + STOP_DISTANCE + 10: at_stop_line = True
        
        if at_stop_line and not is_green:
            self.target_speed = 0
        else:
            self.target_speed = self.speed
            
        # Smooth Braking/Acceleration
        lerp_factor = 0.08
        self.current_speed += (self.target_speed - self.current_speed) * lerp_factor
        
        if self.current_speed > 0.1:
            if self.lane == "North": self.vx, self.vy = 0, self.current_speed
            elif self.lane == "South": self.vx, self.vy = 0, -self.current_speed
            elif self.lane == "West": self.vx, self.vy = self.current_speed, 0
            elif self.lane == "East": self.vx, self.vy = -self.current_speed, 0
            
            self.canvas.move(self.id, self.vx, self.vy)
            for lid in self.lights: self.canvas.move(lid, self.vx, self.vy)
            self.x += self.vx
            self.y += self.vy
            
            # Exhaust Smoke
            if random.random() < 0.2:
                ex_x, ex_y = self._get_exhaust_pos()
                p = self.canvas.create_oval(ex_x-2, ex_y-2, ex_x+2, ex_y+2, fill="#78909c", outline="")
                self.particles.append({"id": p, "age": 0})

    def _get_exhaust_pos(self):
        w, l = VEHICLE_SIZE
        if self.lane == "North": return (self.x, self.y - l/2)
        if self.lane == "South": return (self.x, self.y + l/2)
        if self.lane == "West": return (self.x - w/2, self.y)
        if self.lane == "East": return (self.x + w/2, self.y)
        return (self.x, self.y)

    def is_off_screen(self):
        return (self.x < -100 or self.x > CANVAS_WIDTH + 100 or 
                self.y < -100 or self.y > CANVAS_HEIGHT + 100)

    def cleanup(self):
        self.canvas.delete(self.id)
        for lid in self.lights: self.canvas.delete(lid)
        for p_dict in self.particles: self.canvas.delete(p_dict["id"])

class TopDownSimulator(ctk.CTkFrame):
    def __init__(self, parent, config, signal_callback=None):
        super().__init__(parent, fg_color="#0d1117")
        self.config = config
        self.signal_callback = signal_callback
        
        self.canvas = ctk.CTkCanvas(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, 
                                     bg=COLOR_GRASS, highlightthickness=0)
        self.canvas.pack(expand=True, padx=20, pady=20)
        
        self.vehicles: Dict[str, List[VehicleEntity]] = {l: [] for l in ["North", "South", "East", "West"]}
        self.signals: Dict[str, str] = {l: "red" for l in ["North", "South", "East", "West"]}
        self.running = False
        
        self._draw_static_elements()
        self._draw_lights()
        self._draw_hud_base()

    def _draw_static_elements(self):
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        w = ROAD_WIDTH
        sw = w + 40
        
        # Sidewalks
        self.canvas.create_rectangle(cx-sw/2, 0, cx+sw/2, CANVAS_HEIGHT, fill=COLOR_SIDEWALK, outline="")
        self.canvas.create_rectangle(0, cy-sw/2, CANVAS_WIDTH, cy+sw/2, fill=COLOR_SIDEWALK, outline="")

        # Road
        self.canvas.create_rectangle(cx-w/2, 0, cx+w/2, CANVAS_HEIGHT, fill=COLOR_ROAD, outline="")
        self.canvas.create_rectangle(0, cy-w/2, CANVAS_WIDTH, cy+w/2, fill=COLOR_ROAD, outline="")
        
        # Center Box
        self.canvas.create_rectangle(cx-w/2, cy-w/2, cx+w/2, cy+w/2, fill="#37474f", outline=COLOR_LINE)
        
        # Zebra Crosswalks
        for i in range(5):
            self.canvas.create_rectangle(cx-w/2 + i*25, cy-w/2-30, cx-w/2 + i*25 + 15, cy-w/2-5, fill=COLOR_ZEBRA, outline="")
            self.canvas.create_rectangle(cx-w/2 + i*25, cy+w/2+5, cx-w/2 + i*25 + 15, cy+w/2+30, fill=COLOR_ZEBRA, outline="")
            self.canvas.create_rectangle(cx-w/2-30, cy-w/2 + i*25, cx-w/2-5, cy-w/2 + i*25 + 15, fill=COLOR_ZEBRA, outline="")
            self.canvas.create_rectangle(cx+w/2+5, cy-w/2 + i*25, cx+w/2+30, cy-w/2 + i*25 + 15, fill=COLOR_ZEBRA, outline="")

        # Dashed Lines
        self.canvas.create_line(cx, 0, cx, cy-w/2-35, fill=COLOR_LINE, dash=(10,10), width=2)
        self.canvas.create_line(cx, cy+w/2+35, cx, CANVAS_HEIGHT, fill=COLOR_LINE, dash=(10,10), width=2)
        self.canvas.create_line(0, cy, cx-w/2-35, cy, fill=COLOR_LINE, dash=(10,10), width=2)
        self.canvas.create_line(cx+w/2+35, cy, CANVAS_WIDTH, cy, fill=COLOR_LINE, dash=(10,10), width=2)

    def _draw_hud_base(self):
        self.canvas.create_rectangle(10, 10, 200, 90, fill=COLOR_ROAD, outline=COLOR_SIDEWALK, width=2)
        self.canvas.create_text(105, 30, text="SIMULATION HUD", fill=COLOR_ZEBRA, font=("Arial", 10, "bold"))
        self.sim_status_id = self.canvas.create_text(105, 60, text="INITIALIZING", fill=COLOR_SIDEWALK, font=("Arial", 12, "bold"))

    def _draw_lights(self):
        self.light_objs = {}
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        w = ROAD_WIDTH/2 + 25
        pos = {"North": (cx-50, cy-w), "South": (cx+50, cy+w), "East": (cx+w, cy-50), "West": (cx-w, cy+50)}
        for lane, (x, y) in pos.items():
            self.light_objs[lane] = self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="red", outline="white", width=2)

    def set_light(self, lane: str, state: str):
        color = "red"
        status = "STOPPED"
        if state.upper() == "GREEN": 
            color = "#2ecc71"
            status = "FLOWING"
        elif state.upper() == "YELLOW": 
            color = "#f1c40f"
            status = "TRANSITION"
            
        self.canvas.itemconfig(self.light_objs[lane], fill=color)
        self.signals[lane] = color
        self.canvas.itemconfig(self.sim_status_id, text=status, fill=color)

    def update_sim(self):
        if not self.running: return
        
        # Spawning
        for lane in self.vehicles:
            if random.random() < 0.03:
                v_type = random.choices(["car", "bus", "person", "ambulance"], weights=[70, 15, 10, 5])[0]
                self._spawn_safely(lane, v_type)
        
        # Logic
        for lane, v_list in self.vehicles.items():
            is_green = self.signals[lane] == "#2ecc71" or self.signals[lane] == "#f1c40f"
            for i, v in enumerate(v_list):
                can_move = True
                if i > 0:
                    front = v_list[i-1]
                    dist = abs(front.y - v.y) if lane in ["North", "South"] else abs(front.x - v.x)
                    if dist < 60: can_move = False
                
                if can_move: v.move(is_green)
                else: v.target_speed = 0; v.move(is_green)
                
                # Particles
                for p_dict in v.particles[:]:
                    p_dict["age"] += 1
                    if p_dict["age"] > 12:
                        self.canvas.delete(p_dict["id"])
                        v.particles.remove(p_dict)
                    else:
                        scale = 1.0 - (p_dict["age"] / 12.0)
                        self.canvas.scale(p_dict["id"], v.x, v.y, scale, scale)

            # Cleanup
            self.vehicles[lane] = [v for v in v_list if not v.is_off_screen() or (v.cleanup() and False)]

        if self.signal_callback:
            self.signal_callback({l: self._count_lane(l) for l in self.vehicles})
        self.after(30, self.update_sim)

    def _spawn_safely(self, lane: str, v_type: str):
        if self.vehicles[lane]:
            last = self.vehicles[lane][-1]
            dist = abs(last.y - (-50)) if lane == "North" else abs((CANVAS_HEIGHT+50) - last.y) if lane == "South" else abs(last.x - (-50)) if lane == "West" else abs((CANVAS_WIDTH+50) - last.x)
            if dist < 80: return
        self.vehicles[lane].append(VehicleEntity(self.canvas, lane, v_type))

    def _count_lane(self, lane: str) -> Dict[str, int]:
        c = {"car": 0, "bus": 0, "person": 0, "truck": 0, "motorcycle": 0}
        cx, cy = CANVAS_WIDTH/2, CANVAS_HEIGHT/2
        for v in self.vehicles[lane]:
            in_zone = False
            if lane == "North" and v.y < cy: in_zone = True
            elif lane == "South" and v.y > cy: in_zone = True
            elif lane == "West" and v.x < cx: in_zone = True
            elif lane == "East" and v.x > cx: in_zone = True
            if in_zone:
                t = v.v_type
                if t == "ambulance": c["truck"] += 15
                elif t in c: c[t] += 1
                else: c["car"] += 1
        return c

    def start(self):
        self.running = True
        self.update_sim()

    def stop(self):
        self.running = False
        for l in self.vehicles:
            for v in self.vehicles[l]: v.cleanup()
            self.vehicles[l].clear()
