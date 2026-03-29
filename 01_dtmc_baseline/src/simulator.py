import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
import math

STATES = ["Source 0", "Source 1"]
TRANSITION_MATRIX = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])
INITIAL_PROBS = np.array([1.0, 0.0])

COLOR_BG = "#121212"
COLOR_CARD = "#1e1e1e"
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_SEC = "#b0b0b0"
COLOR_ACCENT_1 = "#00e5ff"
COLOR_ACCENT_2 = "#f50057"
COLOR_BALL = "#76ff03"
COLOR_PATH = "#333333"

class NeonMarkovSimulator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Neon DTMC Simulator")
        self.geometry("1200x750")
        self.configure(bg=COLOR_BG)
        
        # State Variables
        self.step_count = 0
        self.current_state_idx = 0 
        self.prob_vector = INITIAL_PROBS.copy()
        self.is_running = False
        self.animation_speed = 1.0 
        
        # UI Layout
        self.style_widgets()
        self.create_layout()
        self.update_idletasks() 
        self.draw_static_graph()
        self.update_ui_stats()

        # Bind resize event to redraw graph dynamically
        self.canvas.bind("<Configure>", lambda e: self.draw_static_graph())

    def style_widgets(self):
        """Configure ttk styles for dark theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Slider Style
        style.configure("TScale", background=COLOR_CARD, troughcolor="#333", bordercolor=COLOR_CARD)
        
        # Notebook Style
        style.configure("TNotebook", background=COLOR_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=COLOR_CARD, foreground=COLOR_TEXT_SEC, 
                        padding=[10, 5], font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "#2c2c2c")], 
                  foreground=[("selected", COLOR_ACCENT_1)])

    def create_layout(self):
        # Main Container
        main_container = tk.Frame(self, bg=COLOR_BG)
        main_container.pack(fill="both", expand=True, padx=20, pady=20)
        
        # --- Left Column: Canvas & Controls ---
        left_col = tk.Frame(main_container, bg=COLOR_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 20))
        
        # 1. Canvas Area (The Visualizer)
        self.canvas_frame = tk.Frame(left_col, bg=COLOR_CARD, bd=2, relief="flat")
        self.canvas_frame.pack(fill="both", expand=True, pady=(0, 20))
        
        self.canvas = tk.Canvas(self.canvas_frame, bg=COLOR_CARD, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        # 2. Controls Area
        controls = tk.Frame(left_col, bg=COLOR_CARD, padx=20, pady=15)
        controls.pack(fill="x")
        
        # Buttons
        btn_frame = tk.Frame(controls, bg=COLOR_CARD)
        btn_frame.pack(side="left")
        
        self.btn_step = tk.Button(btn_frame, text="STEP ▶", command=self.step,
                                 bg=COLOR_ACCENT_1, fg="black", font=("Segoe UI", 10, "bold"),
                                 relief="flat", activebackground="#00b8d4", padx=20, pady=5)
        self.btn_step.pack(side="left", padx=(0, 10))
        
        self.btn_auto = tk.Button(btn_frame, text="AUTO RUN", command=self.toggle_auto,
                                 bg="#333", fg="white", font=("Segoe UI", 10, "bold"),
                                 relief="flat", activebackground="#555", padx=20, pady=5)
        self.btn_auto.pack(side="left", padx=(0, 10))
        
        self.btn_reset = tk.Button(btn_frame, text="RESET", command=self.reset,
                                  bg=COLOR_CARD, fg="#ff5252", font=("Segoe UI", 10),
                                  relief="solid", bd=1, padx=20, pady=5)
        self.btn_reset.pack(side="left")

        # Speed Slider
        slider_frame = tk.Frame(controls, bg=COLOR_CARD)
        slider_frame.pack(side="right")
        
        tk.Label(slider_frame, text="SIMULATION SPEED", bg=COLOR_CARD, fg=COLOR_TEXT_SEC, 
                 font=("Segoe UI", 8, "bold")).pack(anchor="e")
        
        self.speed_scale = ttk.Scale(slider_frame, from_=0.1, to=3.0, orient="horizontal", 
                                    command=self.update_speed, length=150)
        self.speed_scale.set(1.0) 
        self.speed_scale.pack(pady=(5, 0))

        # --- Right Column: Dashboard ---
        right_col = tk.Frame(main_container, bg=COLOR_BG, width=350)
        right_col.pack(side="right", fill="y")
        
        # State Card
        self.create_card(right_col, "CURRENT STATE", self.create_state_view)
        # Distribution Card
        self.create_card(right_col, "PROBABILITY FLOW", self.create_dist_view)
        # Logs Card
        self.create_card(right_col, "SYSTEM LOGS", self.create_log_view, expand=True)

    def create_card(self, parent, title, widget_creator, expand=False):
        card = tk.Frame(parent, bg=COLOR_CARD, padx=15, pady=15)
        card.pack(fill="both", expand=expand, pady=(0, 15))
        
        tk.Label(card, text=title, bg=COLOR_CARD, fg=COLOR_ACCENT_1, 
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(fill="x", pady=(0, 10))
        widget_creator(card)

    def create_state_view(self, parent):
        self.lbl_state = tk.Label(parent, text=STATES[0], bg=COLOR_CARD, fg=COLOR_TEXT_MAIN, 
                                 font=("Segoe UI", 24, "bold"))
        self.lbl_state.pack(anchor="w")
        
        self.lbl_item = tk.Label(parent, text="Ready to start...", bg=COLOR_CARD, fg=COLOR_TEXT_SEC,
                                font=("Consolas", 12))
        self.lbl_item.pack(anchor="w", pady=(5,0))

    def create_dist_view(self, parent):
        self.bars = []
        colors = [COLOR_ACCENT_1, COLOR_ACCENT_2]
        
        for i, state in enumerate(STATES):
            frame = tk.Frame(parent, bg=COLOR_CARD)
            frame.pack(fill="x", pady=6)
            
            row = tk.Frame(frame, bg=COLOR_CARD)
            row.pack(fill="x")
            tk.Label(row, text=state, bg=COLOR_CARD, fg="#ddd", font=("Segoe UI", 10)).pack(side="left")
            pct = tk.Label(row, text="0.0%", bg=COLOR_CARD, fg=colors[i], font=("Consolas", 10, "bold"))
            pct.pack(side="right")
            
            # Progress Bar Background
            canvas = tk.Canvas(frame, height=6, bg="#333", highlightthickness=0)
            canvas.pack(fill="x", pady=(4, 0))
            rect = canvas.create_rectangle(0, 0, 0, 6, fill=colors[i], width=0)
            
            self.bars.append({"pct": pct, "canvas": canvas, "rect": rect, "max_w": 0})

    def create_log_view(self, parent):
        self.txt_log = tk.Text(parent, bg="#151515", fg="#ccc", font=("Consolas", 9), 
                              bd=0, highlightthickness=0, height=10)
        self.txt_log.pack(fill="both", expand=True)
        self.log("System initialized.")

    # Canvas Drawing 
    def draw_static_graph(self):
        """Draws nodes and static paths"""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        # Fallback if canvas has weird dimensions
        if w < 50: w = 800
        if h < 50: h = 500
        
        # Node Positions
        self.pos_s0 = (w * 0.3, h * 0.5)
        self.pos_s1 = (w * 0.7, h * 0.5)
        r = 40 # Node radius

        # 1. Draw Transitions 
        # S0 -> S1 
        self.draw_curved_arrow(self.pos_s0, self.pos_s1, r, -100, "0.4")
        # S1 -> S0 (Curved bottom)
        self.draw_curved_arrow(self.pos_s1, self.pos_s0, r, 100, "0.3")
        
        # Self Loops
        self.draw_self_loop(self.pos_s0, r, "left", "0.6")
        self.draw_self_loop(self.pos_s1, r, "right", "0.7")

        # 2. Draw Nodes
        self.draw_node(self.pos_s0, r, STATES[0], COLOR_ACCENT_1, self.current_state_idx == 0)
        self.draw_node(self.pos_s1, r, STATES[1], COLOR_ACCENT_2, self.current_state_idx == 1)

    def draw_node(self, pos, r, text, color, active):
        x, y = pos
        fill = color if active else "#222"
        outline = color if active else "#444"
        width = 3 if active else 1
        
        # Glow effect if active
        if active:
            for i in range(10, 0, -2):
                alpha = int(20 / i)
                self.canvas.create_oval(x-(r+i), y-(r+i), x+(r+i), y+(r+i), 
                                      outline=color, width=0, tags="glow")

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=fill, outline=outline, width=width, tags="node")
        self.canvas.create_text(x, y, text=text, fill="white" if active else "#888", 
                              font=("Segoe UI", 12, "bold"))

    def draw_curved_arrow(self, start, end, r, curve_h, label):
        """Draws a quadratic bezier-like curve"""
        x1, y1 = start
        x2, y2 = end
        
        # Control point for curve
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2 + curve_h
        
        # Adjust start/end to be on circle edge
        angle_start = math.atan2(cy - y1, cx - x1)
        angle_end = math.atan2(cy - y2, cx - x2)
        
        sx, sy = x1 + r*math.cos(angle_start), y1 + r*math.sin(angle_start)
        ex, ey = x2 + r*math.cos(angle_end), y2 + r*math.sin(angle_end)
        
        # Draw Curve
        self.canvas.create_line(sx, sy, cx, cy, ex, ey, smooth=True, width=2, 
                              arrow=tk.LAST, arrowshape=(16, 20, 6), fill=COLOR_PATH, tags="path")
        
        # Label 
        self.canvas.create_text(cx, cy - (10 if curve_h < 0 else -10), text=label, 
                              fill="#cccccc", font=("Consolas", 11, "bold"))

    def draw_self_loop(self, pos, r, side, label):
        x, y = pos
        offset = -1 if side == "left" else 1
        
        # Control points for a loop
        coords = [
            x, y - r,         
            x + (offset * r * 3), y - r * 2.5, 
            x + (offset * r * 3), y + r * 2.5, 
            x, y + r           
        ]
        
        self.canvas.create_line(coords, smooth=True, width=2, arrow=tk.LAST, 
                              arrowshape=(16, 20, 6), fill=COLOR_PATH, tags="path")
        
        lbl_x = x + (offset * r * 2.2)
        # Label
        self.canvas.create_text(lbl_x, y, text=label, fill="#cccccc", font=("Consolas", 11, "bold"))

    # --- Animation Logic ---
    def animate_transition(self, start_idx, end_idx, callback):
        """Animate the neon ball traveling between states"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 50: w, h = 800, 500
            
        p0 = (w * 0.3, h * 0.5)
        p1 = (w * 0.7, h * 0.5)
        
        steps = 40 
        delay = 0.015 / self.animation_speed 
        
        coords = []
        if start_idx == 0 and end_idx == 1:
            cx, cy = (p0[0] + p1[0])/2, (p0[1] + p1[1])/2 - 100
            coords = self.get_bezier_points(p0, (cx, cy), p1, steps)
        elif start_idx == 1 and end_idx == 0:
            cx, cy = (p0[0] + p1[0])/2, (p0[1] + p1[1])/2 + 100
            coords = self.get_bezier_points(p1, (cx, cy), p0, steps)
        elif start_idx == 0 and end_idx == 0:
            coords = self.get_loop_points(p0, 40, -1, steps)
        elif start_idx == 1 and end_idx == 1:
            coords = self.get_loop_points(p1, 40, 1, steps)

        ball_r = 8
        ball = self.canvas.create_oval(0, 0, 0, 0, fill=COLOR_BALL, outline="white", width=2)
        
        def move_step(step_idx):
            if step_idx >= len(coords):
                self.canvas.delete(ball)
                callback() 
                return
            
            x, y = coords[step_idx]
            self.canvas.coords(ball, x-ball_r, y-ball_r, x+ball_r, y+ball_r)
            self.update() 
            time.sleep(delay)
            move_step(step_idx + 1)
            
        threading.Thread(target=lambda: move_step(0), daemon=True).start()

    def get_bezier_points(self, p0, p1, p2, steps):
        points = []
        for t in np.linspace(0, 1, steps):
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            points.append((x, y))
        return points

    def get_loop_points(self, center, r, direction, steps):
        cx, cy = center
        points = []
        for t in np.linspace(0, 2*math.pi, steps):
            angle = -math.pi/2 + (t * direction)
            loop_cx = cx + (direction * r * 1.5)
            loop_cy = cy
            x = loop_cx + (r * 1.5 * math.cos(angle))
            y = loop_cy + (r * 2.0 * math.sin(angle))
            points.append((x, y))
        return points

    # --- Core Logic ---
    def update_speed(self, val):
        self.animation_speed = float(val)

    def log(self, msg):
        self.txt_log.insert("end", f"> {msg}\n")
        self.txt_log.see("end")

    def update_ui_stats(self):
        self.draw_static_graph()
        
        self.lbl_state.config(text=STATES[self.current_state_idx], 
                              fg=COLOR_ACCENT_1 if self.current_state_idx==0 else COLOR_ACCENT_2)
        
        for i, prob in enumerate(self.prob_vector):
            self.bars[i]["pct"].config(text=f"{prob*100:.1f}%")
            canvas_w = 300 
            bar_w = canvas_w * prob
            self.bars[i]["canvas"].coords(self.bars[i]["rect"], 0, 0, bar_w, 6)

    def step(self):
        if hasattr(self, 'animating') and self.animating: return 
        self.animating = True
        
        self.step_count += 1
        prev_idx = self.current_state_idx
        
        # 1. Math
        self.prob_vector = np.dot(self.prob_vector, TRANSITION_MATRIX)
        
        # 2. Sim
        current_probs = TRANSITION_MATRIX[self.current_state_idx]
        next_state_idx = np.random.choice([0, 1], p=current_probs)
        self.current_state_idx = next_state_idx
        
        # 3. Emissions
        emission_prob = 0.9 if next_state_idx == 0 else 0.95
        is_ok = np.random.random() < emission_prob
        item_str = "OK" if is_ok else "DEFECT"
        color = "#76ff03" if is_ok else "#ff3d00"
        
        self.lbl_item.config(text=f"Production Output: {item_str}", fg=color)
        self.log(f"t={self.step_count}: {STATES[prev_idx]} -> {STATES[next_state_idx]} ({item_str})")
        
        # 4. Animate then Update UI
        def on_complete():
            self.update_ui_stats()
            self.animating = False
            
        self.animate_transition(prev_idx, next_state_idx, on_complete)

    def reset(self):
        self.is_running = False
        self.btn_auto.config(text="AUTO RUN", bg="#333", fg="white")
        self.step_count = 0
        self.current_state_idx = 0
        self.prob_vector = INITIAL_PROBS.copy()
        self.txt_log.delete(1.0, "end")
        self.log("System Reset.")
        self.update_ui_stats()

    def toggle_auto(self):
        if self.is_running:
            self.is_running = False
            self.btn_auto.config(text="AUTO RUN", bg="#333", fg="white")
        else:
            self.is_running = True
            self.btn_auto.config(text="STOP", bg=COLOR_ACCENT_2, fg="white")
            threading.Thread(target=self.run_loop, daemon=True).start()

    def run_loop(self):
        while self.is_running:
            if not getattr(self, 'animating', False):
                self.after(0, self.step)
            time.sleep(1.0 / self.animation_speed)

if __name__ == "__main__":
    app = NeonMarkovSimulator()
    app.mainloop()

if __name__ == "__main__":
    app = NeonMarkovSimulator()
    app.mainloop()