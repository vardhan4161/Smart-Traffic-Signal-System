import customtkinter as ctk
import random
from typing import Dict, List, Optional
from loguru import logger

# ─── Canvas & Road Geometry ────────────────────────────────────────────────────
CANVAS_W = 800
CANVAS_H = 800
CX, CY   = CANVAS_W / 2, CANVAS_H / 2

LANE_W   = 38           # width of one lane
N_LANES  = 2            # incoming lanes per arm
ROAD_W   = LANE_W * N_LANES * 2   # 152 px total road width

STOP_D   = ROAD_W // 2 + 18   # stop-line distance from centre

# ─── Color palette ─────────────────────────────────────────────────────────────
C_BG        = "#0a1628"
C_ROAD      = "#1e2a3a"
C_ROAD_EDGE = "#253545"
C_SIDEWALK  = "#2c3e50"
C_LINE_MAIN = "#ffd700"
C_LINE_DASH = "#b0bec5"
C_ZEBRA     = "#eceff1"
C_BOX       = "#162030"
C_HUD_BG    = "#0d1b2a"
C_HUD_TXT   = "#e0e0e0"

C_CAR       = "#29b6f6"
C_BUS       = "#ffa726"
C_TRUCK     = "#ab47bc"
C_AMBULANCE = "#ef5350"
C_PERSON    = "#66bb6a"
C_HEADLIGHT = "#fff9c4"
C_TAILLIGHT = "#ff1744"

C_RED       = "#c62828"
C_YELLOW    = "#f9a825"
C_GREEN     = "#2e7d32"
GLOW_RED    = "#ff5252"
GLOW_GREEN  = "#69f0ae"
GLOW_YELLOW = "#ffff00"


def lane_offset(lane_idx: int) -> float:
    return -(LANE_W * (lane_idx + 0.5))


# ─── VehicleEntity ─────────────────────────────────────────────────────────────
class VehicleEntity:
    _EMOJI = {"car": "🚗", "bus": "🚌", "truck": "🚛",
              "ambulance": "🚑"}
    _COLOR = {"car": C_CAR, "bus": C_BUS, "truck": C_TRUCK,
              "ambulance": C_AMBULANCE}

    def __init__(self, canvas: ctk.CTkCanvas, direction: str,
                 v_type: str = "car", lane_idx: int = 0):
        self.canvas    = canvas
        self.direction = direction
        self.v_type    = v_type
        self.lane_idx  = lane_idx

        base = 3.5 if v_type != "bus" else 2.8
        self.speed         = random.uniform(base - 0.5, base + 1.0)
        self.current_speed = self.speed
        self.target_speed  = self.speed

        self.body_id  = None
        self.emoji_id = None
        self.lights: List[int] = []
        self.particles: List[dict] = []

        self.x, self.y = self._spawn_pos()
        self.vx = self.vy = 0.0
        self._draw()

    def _spawn_pos(self):
        off = lane_offset(self.lane_idx)
        if self.direction == "North":  return (CX + off,         -60)
        if self.direction == "South":  return (CX - off, CANVAS_H + 60)
        if self.direction == "West":   return (-60,       CY - off)
        if self.direction == "East":   return (CANVAS_W + 60, CY + off)
        return (0, 0)

    def _body_wl(self):
        w, l = 18, 30
        if self.v_type == "bus":   w, l = 20, 44
        if self.v_type == "truck": w, l = 22, 38
        if self.direction in ("East", "West"): w, l = l, w
        return w, l

    def _poly_pts(self, x, y, w, l):
        r = min(w, l) * 0.25
        return [x-w/2+r, y-l/2,  x+w/2-r, y-l/2,
                x+w/2,   y-l/2+r, x+w/2,   y+l/2-r,
                x+w/2-r, y+l/2,  x-w/2+r, y+l/2,
                x-w/2,   y+l/2-r, x-w/2,   y-l/2+r]

    def _front_back(self, w, l):
        if self.direction == "North": return (self.x, self.y + l/2, self.x, self.y - l/2)
        if self.direction == "South": return (self.x, self.y - l/2, self.x, self.y + l/2)
        if self.direction == "West":  return (self.x + l/2, self.y, self.x - l/2, self.y)
        return (self.x - l/2, self.y, self.x + l/2, self.y)  # East

    def _draw(self):
        color = self._COLOR.get(self.v_type, C_CAR)
        w, l  = self._body_wl()
        pts   = self._poly_pts(self.x, self.y, w, l)
        self.body_id  = self.canvas.create_polygon(pts, fill=color, outline="#ffffff", width=1, smooth=True)
        font_sz = 12 if self.v_type not in ("bus", "truck") else 15
        self.emoji_id = self.canvas.create_text(
            self.x, self.y, text=self._EMOJI.get(self.v_type, "🚗"),
            font=("Segoe UI Emoji", font_sz))
        fx, fy, bx, by = self._front_back(w, l)
        side = w / 3
        if self.direction in ("North", "South"):
            self.lights += [
                self.canvas.create_oval(fx-side-2, fy-2, fx-side+2, fy+2, fill=C_HEADLIGHT, outline=""),
                self.canvas.create_oval(fx+side-2, fy-2, fx+side+2, fy+2, fill=C_HEADLIGHT, outline=""),
                self.canvas.create_oval(bx-side-2, by-2, bx-side+2, by+2, fill=C_TAILLIGHT, outline=""),
                self.canvas.create_oval(bx+side-2, by-2, bx+side+2, by+2, fill=C_TAILLIGHT, outline=""),
            ]
        else:
            self.lights += [
                self.canvas.create_oval(fx-2, fy-side-2, fx+2, fy-side+2, fill=C_HEADLIGHT, outline=""),
                self.canvas.create_oval(fx-2, fy+side-2, fx+2, fy+side+2, fill=C_HEADLIGHT, outline=""),
                self.canvas.create_oval(bx-2, by-side-2, bx+2, by-side+2, fill=C_TAILLIGHT, outline=""),
                self.canvas.create_oval(bx-2, by+side-2, bx+2, by+side+2, fill=C_TAILLIGHT, outline=""),
            ]

    def move(self, is_green: bool, can_move: bool = True):
        if self.direction == "North":   dist = (CY - STOP_D) - self.y
        elif self.direction == "South": dist = self.y - (CY + STOP_D)
        elif self.direction == "West":  dist = (CX - STOP_D) - self.x
        else:                           dist = self.x - (CX + STOP_D)

        if not can_move:
            self.target_speed = 0.0
        elif not is_green and 0.0 < dist < 160:
            self.target_speed = 0.0
            if dist < 4:
                self.current_speed = 0.0
                if self.direction == "North":   self.y = CY - STOP_D - 2
                elif self.direction == "South": self.y = CY + STOP_D + 2
                elif self.direction == "West":  self.x = CX - STOP_D - 2
                else:                           self.x = CX + STOP_D + 2
        else:
            self.target_speed = self.speed

        self.current_speed += (self.target_speed - self.current_speed) * 0.12

        if self.current_speed > 0.05:
            if self.direction == "North":   self.vx, self.vy =  0,  self.current_speed
            elif self.direction == "South": self.vx, self.vy =  0, -self.current_speed
            elif self.direction == "West":  self.vx, self.vy =  self.current_speed, 0
            else:                           self.vx, self.vy = -self.current_speed, 0

            for item in [self.body_id, self.emoji_id] + self.lights:
                self.canvas.move(item, self.vx, self.vy)
            self.x += self.vx
            self.y += self.vy

            if random.random() < 0.15:
                w, l = self._body_wl()
                _, _, bx, by = self._front_back(w, l)
                p = self.canvas.create_oval(bx-3, by-3, bx+3, by+3, fill="#607d8b", outline="")
                self.particles.append({"id": p, "age": 0})

    def is_off_screen(self) -> bool:
        m = 120
        return self.x < -m or self.x > CANVAS_W + m or self.y < -m or self.y > CANVAS_H + m

    def cleanup(self):
        for item in [self.body_id, self.emoji_id] + self.lights:
            self.canvas.delete(item)
        for p in self.particles:
            self.canvas.delete(p["id"])


# ─── TopDownSimulator ──────────────────────────────────────────────────────────
class TopDownSimulator(ctk.CTkFrame):
    DIRECTIONS = ["North", "South", "East", "West"]

    def __init__(self, parent, config, signal_callback=None):
        super().__init__(parent, fg_color=C_BG)
        self.config          = config
        self.signal_callback = signal_callback
        self.running         = False
        self._frame          = 0

        self.vehicles: Dict[str, List[VehicleEntity]] = {d: [] for d in self.DIRECTIONS}
        self.signals:  Dict[str, str]                 = {d: "red" for d in self.DIRECTIONS}

        self.canvas = ctk.CTkCanvas(self, width=CANVAS_W, height=CANVAS_H,
                                    bg=C_BG, highlightthickness=0)
        self.canvas.pack(expand=True, padx=10, pady=10)

        self._draw_scene()
        self._draw_traffic_lights()
        self._draw_hud()

    # ─── Static scene ──────────────────────────────────────────────────────
    def _draw_scene(self):
        c = self.canvas
        # Decorative grass blobs
        for _ in range(30):
            rx, ry = random.uniform(10, CANVAS_W-10), random.uniform(10, CANVAS_H-10)
            if abs(rx - CX) > ROAD_W/2 + 20 and abs(ry - CY) > ROAD_W/2 + 20:
                r = random.uniform(4, 12)
                c.create_oval(rx-r, ry-r, rx+r, ry+r,
                              fill=random.choice(["#0d1f0d","#112211","#0f1e0f"]), outline="")
        # Sidewalks
        sw = ROAD_W + 28
        c.create_rectangle(CX-sw/2, 0,        CX+sw/2, CANVAS_H, fill=C_SIDEWALK, outline="")
        c.create_rectangle(0,       CY-sw/2,  CANVAS_W, CY+sw/2, fill=C_SIDEWALK, outline="")
        # Road
        c.create_rectangle(CX-ROAD_W/2, 0,           CX+ROAD_W/2, CANVAS_H, fill=C_ROAD, outline="")
        c.create_rectangle(0,           CY-ROAD_W/2, CANVAS_W,     CY+ROAD_W/2, fill=C_ROAD, outline="")
        # Road edge highlights
        for dx in [-ROAD_W/2, ROAD_W/2-2]:
            c.create_line(CX+dx, 0, CX+dx, CANVAS_H, fill=C_ROAD_EDGE, width=3)
        for dy in [-ROAD_W/2, ROAD_W/2-2]:
            c.create_line(0, CY+dy, CANVAS_W, CY+dy, fill=C_ROAD_EDGE, width=3)
        # Intersection box
        c.create_rectangle(CX-ROAD_W/2, CY-ROAD_W/2, CX+ROAD_W/2, CY+ROAD_W/2,
                           fill=C_BOX, outline="#ffd700", width=2)
        for i in range(-4, 5):
            off = i * 30
            c.create_line(CX-ROAD_W/2, CY+off, CX+off, CY-ROAD_W/2,
                          fill="#2a3a00", width=1, dash=(4,6))
        # Centre median lines
        c.create_line(CX, 0,              CX, CY-ROAD_W/2-2, fill=C_LINE_MAIN, width=2)
        c.create_line(CX, CY+ROAD_W/2+2, CX, CANVAS_H,       fill=C_LINE_MAIN, width=2)
        c.create_line(0,             CY, CX-ROAD_W/2-2, CY,  fill=C_LINE_MAIN, width=2)
        c.create_line(CX+ROAD_W/2+2, CY, CANVAS_W,      CY,  fill=C_LINE_MAIN, width=2)
        # Dashed lane dividers
        dash = (12, 8)
        for sign in (-1, 1):
            x = CX + sign * LANE_W
            c.create_line(x, 0,              x, CY-ROAD_W/2-2, fill=C_LINE_DASH, width=1, dash=dash)
            c.create_line(x, CY+ROAD_W/2+2, x, CANVAS_H,       fill=C_LINE_DASH, width=1, dash=dash)
            y = CY + sign * LANE_W
            c.create_line(0,             y, CX-ROAD_W/2-2, y, fill=C_LINE_DASH, width=1, dash=dash)
            c.create_line(CX+ROAD_W/2+2, y, CANVAS_W,      y, fill=C_LINE_DASH, width=1, dash=dash)
        # Zebra crosswalks
        n = 5
        for i in range(n):
            ox = CX - ROAD_W/2 + i*(ROAD_W/n) + 4;  ow = ROAD_W/n - 8
            for y0 in [CY-ROAD_W/2-16, CY+ROAD_W/2+6]:
                c.create_rectangle(ox, y0, ox+ow, y0+10, fill=C_ZEBRA, outline="")
            oy = CY - ROAD_W/2 + i*(ROAD_W/n) + 4;  oh = ROAD_W/n - 8
            for x0 in [CX-ROAD_W/2-16, CX+ROAD_W/2+6]:
                c.create_rectangle(x0, oy, x0+10, oy+oh, fill=C_ZEBRA, outline="")
        # Direction arrows
        def arrow(x1, y1, x2, y2):
            c.create_line(x1,y1,x2,y2,fill="#37474f",width=2,arrow="last",arrowshape=(10,12,5))
        for lx in [-(LANE_W*0.5), -(LANE_W*1.5)]:
            arrow(CX+lx, 80,           CX+lx, CY-ROAD_W/2-20)  # N
            arrow(CX-lx, CANVAS_H-80, CX-lx, CY+ROAD_W/2+20)  # S
        for ly in [-(LANE_W*0.5), -(LANE_W*1.5)]:
            arrow(80,           CY-ly, CX-ROAD_W/2-20, CY-ly)   # W
            arrow(CANVAS_W-80, CY+ly, CX+ROAD_W/2+20, CY+ly)   # E

    # ─── Traffic lights ────────────────────────────────────────────────────
    def _draw_traffic_lights(self):
        self.light_bulbs: Dict[str, dict] = {}
        self.light_glow:  Dict[str, int]  = {}
        margin = ROAD_W / 2 + 8
        poles = {
            "North": (CX - ROAD_W/2 - 14, CY - margin),
            "South": (CX + ROAD_W/2 + 14, CY + margin),
            "West":  (CX - margin, CY + ROAD_W/2 + 14),
            "East":  (CX + margin, CY - ROAD_W/2 - 14),
        }
        c = self.canvas
        for lane, (bx, by) in poles.items():
            c.create_rectangle(bx-9, by-28, bx+9, by+28, fill="#1a1a2e", outline="#465a6e", width=1)
            r_id = c.create_oval(bx-6, by-24, bx+6, by-14, fill=C_RED,    outline="#111")
            y_id = c.create_oval(bx-6, by-6,  bx+6, by+4,  fill="#333",   outline="#111")
            g_id = c.create_oval(bx-6, by+14, bx+6, by+24, fill="#0a2a0a", outline="#111")
            glow = c.create_oval(bx-14, by-30, bx+14, by-12, fill="", outline="")
            self.light_bulbs[lane] = {"r": r_id, "y": y_id, "g": g_id}
            self.light_glow[lane]  = glow

    # ─── HUD ───────────────────────────────────────────────────────────────
    def _draw_hud(self):
        c = self.canvas
        # Top-left status box
        c.create_rectangle(10, 10, 220, 120, fill=C_HUD_BG, outline="#2a4060", width=2)
        c.create_text(115, 26, text="🚦 ADAPTIVE TRAFFIC SIM", fill="#64b5f6",
                      font=("Segoe UI", 10, "bold"))
        c.create_line(14, 34, 216, 34, fill="#1e3050")
        self._hud_state  = c.create_text(115, 52, text="Signal: Initializing...",
                                         fill=C_HUD_TXT, font=("Segoe UI", 9))
        self._hud_counts = c.create_text(115, 70, text="Total Vehicles: 0",
                                         fill="#90a4ae", font=("Segoe UI", 9))
        self._hud_reason = c.create_text(115, 88, text="",
                                         fill="#f39c12", font=("Segoe UI", 8, "italic"))
        self._hud_time   = c.create_text(115, 106, text="Frame: 0",
                                         fill="#546e7a", font=("Segoe UI", 8))

        # Per-lane count bubbles near each arm entry
        pos = {"North": (CX, 38), "South": (CX, CANVAS_H-38),
               "West": (38, CY),  "East": (CANVAS_W-38, CY)}
        self._lane_count_ids: Dict[str, int] = {}
        for lane, (lx, ly) in pos.items():
            c.create_rectangle(lx-42, ly-14, lx+42, ly+14,
                               fill="#0d1b2a", outline="#2a4060", width=1)
            self._lane_count_ids[lane] = c.create_text(
                lx, ly, text=f"{lane}: 0 🚗", fill="#eceff1",
                font=("Segoe UI", 8, "bold"))

    # ─── Public HUD update ─────────────────────────────────────────────────
    def update_hud_counts(self, counts_per_lane: Dict[str, int],
                          active_lane: str, reason: str, frame: int, total: int):
        c = self.canvas
        color_map = {"green": GLOW_GREEN, "yellow": GLOW_YELLOW, "red": "#ef9a9a"}
        for lane, cnt in counts_per_lane.items():
            color = color_map.get(self.signals.get(lane, "red"), "#eceff1")
            c.itemconfig(self._lane_count_ids[lane],
                         text=f"{lane}: {cnt} 🚗", fill=color)
        c.itemconfig(self._hud_counts, text=f"Total Vehicles: {total}")
        c.itemconfig(self._hud_reason, text=reason)
        c.itemconfig(self._hud_time,   text=f"Frame: {frame}")

    # ─── set_light ─────────────────────────────────────────────────────────
    def set_light(self, lane: str, state: str):
        state = state.upper()
        bulbs = self.light_bulbs[lane]
        glow  = self.light_glow[lane]
        c     = self.canvas
        c.itemconfig(bulbs["r"], fill="#330a0a")
        c.itemconfig(bulbs["y"], fill="#332a00")
        c.itemconfig(bulbs["g"], fill="#0a2a0a")
        c.itemconfig(glow, fill="", outline="")
        if state == "RED":
            c.itemconfig(bulbs["r"], fill=C_RED)
            c.itemconfig(glow, fill="#550000", outline=GLOW_RED, width=2)
            self.signals[lane] = "red"
        elif state == "YELLOW":
            c.itemconfig(bulbs["y"], fill=C_YELLOW)
            c.itemconfig(glow, fill="#555500", outline=GLOW_YELLOW, width=2)
            self.signals[lane] = "yellow"
        elif state == "GREEN":
            c.itemconfig(bulbs["g"], fill=C_GREEN)
            c.itemconfig(glow, fill="#005500", outline=GLOW_GREEN, width=2)
            self.signals[lane] = "green"
            c.itemconfig(self._hud_state, text=f"Signal: {lane} 🟢 GREEN", fill=GLOW_GREEN)

    # ─── Simulation loop ───────────────────────────────────────────────────
    def update_sim(self):
        if not self.running:
            return
        try:
            self._frame += 1
            # Auto-spawn — probability per lane per tick; physics gap-check handles max density
            for lane in self.DIRECTIONS:
                if random.random() < 0.05:
                    li = random.randint(0, N_LANES-1)
                    vt = random.choices(
                        ["car","bus","truck","ambulance"],
                        weights=[60,15,12,5,8])[0]
                    self._spawn_safely(lane, vt, li)

            total = 0
            for lane in self.DIRECTIONS:
                is_green = self.signals[lane] in ("green", "yellow")
                v_list   = self.vehicles[lane]
                total   += len(v_list)
                for i, v in enumerate(v_list):
                    front = None
                    for j in range(i-1, -1, -1):
                        if v_list[j].lane_idx == v.lane_idx:
                            front = v_list[j]; break
                    can_move = True
                    if front:
                        gap = abs(front.y-v.y) if lane in ("North","South") else abs(front.x-v.x)
                        if gap < 65: can_move = False
                    v.move(is_green, can_move)
                    for p in v.particles[:]:
                        p["age"] += 1
                        if p["age"] > 14:
                            self.canvas.delete(p["id"]); v.particles.remove(p)
                        else:
                            self.canvas.move(p["id"], random.uniform(-0.4,0.4), -0.4)
                on_screen = [v for v in v_list if not v.is_off_screen()]
                for v in v_list:
                    if v.is_off_screen(): v.cleanup()
                self.vehicles[lane] = on_screen

            if self._frame % 10 == 0:
                self.canvas.itemconfig(self._hud_counts, text=f"Total Vehicles: {total}")
                self.canvas.itemconfig(self._hud_time,   text=f"Frame: {self._frame}")

            if self.signal_callback:
                counts = {l: self._count_lane(l) for l in self.DIRECTIONS}
                self.signal_callback(counts)
        except Exception as e:
            logger.error(f"[Sim] {e}", exc_info=True)
        self.after(30, self.update_sim)

    # ─── API ───────────────────────────────────────────────────────────────
    def spawn_vehicle(self, lane: str, v_type: str, lane_idx: Optional[int] = None):
        if lane_idx is None: lane_idx = random.randint(0, N_LANES-1)
        self._spawn_safely(lane, v_type, lane_idx)
        logger.info(f"Manual spawn: {v_type} → {lane}[{lane_idx}]")

    def _spawn_safely(self, lane: str, v_type: str, lane_idx: int = 0):
        for v in self.vehicles[lane]:
            if v.lane_idx == lane_idx:
                if lane == "North":   d = abs(v.y - (-60))
                elif lane == "South": d = abs((CANVAS_H+60) - v.y)
                elif lane == "West":  d = abs(v.x - (-60))
                else:                 d = abs((CANVAS_W+60) - v.x)
                if d < 90: return
        self.vehicles[lane].append(VehicleEntity(self.canvas, lane, v_type, lane_idx))

    def _count_lane(self, lane: str) -> Dict[str, int]:
        c = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
        for v in self.vehicles[lane]:
            approaching = (
                (lane == "North" and v.y < CY) or
                (lane == "South" and v.y > CY) or
                (lane == "West"  and v.x < CX) or
                (lane == "East"  and v.x > CX)
            )
            if approaching:
                if v.v_type == "ambulance":
                    c["emergency"] = c.get("emergency", 0) + 1
                elif v.v_type in c:
                    c[v.v_type] += 1
                else:
                    c["car"] += 1
        return c

    def start(self):
        self.running = True
        self.update_sim()

    def stop(self):
        self.running = False
        for lane in self.DIRECTIONS:
            for v in self.vehicles[lane]: v.cleanup()
            self.vehicles[lane].clear()
