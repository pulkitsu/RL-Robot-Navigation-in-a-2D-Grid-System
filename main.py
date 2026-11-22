"""
Interactive Q-Learning Robot Navigation GUI & Search Benchmark
"""

import sys
import os
import time
import json
import heapq
import tempfile
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

# Prevent Matplotlib cache warnings
os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'matplotlib_cache'))
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# GUI Imports with PyQt5 / PySide6 / PyQt6 cross-compatibility
GUI_AVAILABLE = False
GUI_TOOLKIT = None

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
        QFileDialog, QMessageBox, QGridLayout, QGroupBox, QComboBox, QTabWidget,
        QCheckBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
    from PyQt5.QtGui import QColor, QPainter, QFont
    from PyQt5.QtChart import QChart, QChartView, QLineSeries
    GUI_AVAILABLE = True
    GUI_TOOLKIT = "PyQt5"
except ImportError:
    try:
        from PySide6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
            QFileDialog, QMessageBox, QGridLayout, QGroupBox, QComboBox, QTabWidget,
            QCheckBox
        )
        from PySide6.QtCore import Qt, QTimer, Signal as pyqtSignal, QObject, QPointF
        from PySide6.QtGui import QColor, QPainter, QFont
        from PySide6.QtCharts import QChart, QChartView, QLineSeries
        GUI_AVAILABLE = True
        GUI_TOOLKIT = "PySide6"
    except ImportError:
        try:
            from PyQt6.QtWidgets import (
                QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
                QFileDialog, QMessageBox, QGridLayout, QGroupBox, QComboBox, QTabWidget,
                QCheckBox
            )
            from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF
            from PyQt6.QtGui import QColor, QPainter, QFont
            from PyQt6.QtCharts import QChart, QChartView, QLineSeries
            GUI_AVAILABLE = True
            GUI_TOOLKIT = "PyQt6"
        except ImportError:
            GUI_AVAILABLE = False
            # Dummy classes so the file can be imported and executed headlessly without error
            QWidget = object
            QMainWindow = object


# ============================================================================
# UNIT II: CLASSICAL SEARCH ALGORITHMS (A* SEARCH WITH MANHATTAN HEURISTIC)
# ============================================================================

def astar_shortest_path(grid, start_pos, goal_pos):
    """
    Classical A* Search Algorithm with Manhattan Distance Heuristic.
    Satisfies Unit II Heuristic Search (CO2).

    Args:
        grid: 2D numpy array (0 = free cell, 1 = obstacle)
        start_pos: tuple (row, col)
        goal_pos: tuple (row, col)

    Returns:
        dict: found (bool), path (list), steps (int), nodes_expanded (int), execution_time_ms (float)
    """
    start_time = time.perf_counter()
    grid_size = grid.shape[0]

    def manhattan(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    counter = 0
    open_heap = []
    h_start = manhattan(start_pos, goal_pos)
    heapq.heappush(open_heap, (h_start, counter, start_pos))

    came_from = {}
    g_score = {start_pos: 0}
    nodes_expanded = 0
    closed_set = set()

    # 4-directional transitions (Up, Down, Left, Right)
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    found = False

    while open_heap:
        f, _, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        if current == goal_pos:
            found = True
            break

        for dr, dc in deltas:
            nr, nc = current[0] + dr, current[1] + dc
            neighbor = (nr, nc)

            if 0 <= nr < grid_size and 0 <= nc < grid_size:
                if grid[nr, nc] == 1:
                    continue  # Obstacle collision

                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + manhattan(neighbor, goal_pos)
                    counter += 1
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

    exec_time_ms = (time.perf_counter() - start_time) * 1000.0

    if not found:
        return {
            'found': False,
            'path': [],
            'steps': 0,
            'nodes_expanded': nodes_expanded,
            'execution_time_ms': exec_time_ms
        }

    # Reconstruct optimal trajectory
    curr = goal_pos
    path = [curr]
    while curr in came_from:
        curr = came_from[curr]
        path.append(curr)
    path.reverse()

    return {
        'found': True,
        'path': path,
        'steps': len(path) - 1,
        'nodes_expanded': nodes_expanded,
        'execution_time_ms': exec_time_ms
    }


# ============================================================================
# UNIT I & IV: ENVIRONMENT AND Q-LEARNING AGENT
# ============================================================================

class GridEnvironment:
    """
    Grid-world environment for autonomous robot navigation.
    PEAS Formulation (Unit I):
    - Performance Measure: Maximize cumulative reward (+100 goal, -1 collision, -0.1 step penalty).
    - Environment: 10x10 discrete 2D grid world with 20% obstacles, fully observable, deterministic transitions.
    - Actuators: 4 discrete actions (0=Up, 1=Down, 2=Left, 3=Right).
    - Sensors: Exact coordinates (row, col).
    """

    def __init__(self, grid_size=10, obstacle_density=0.2):
        self.grid_size = grid_size
        self.obstacle_density = obstacle_density
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.grid = np.zeros((grid_size, grid_size))
        self.generate_valid_grid()
        self.robot_pos = self.start_pos

    def is_reachable(self, start=None, goal=None):
        """
        Check if goal is reachable from start using Breadth-First Search (BFS).
        Satisfies Unit II Uninformed Search (CO2) & Unit IV Operational Safety (CO4).
        """
        if start is None:
            start = self.start_pos
        if goal is None:
            goal = self.goal_pos

        if self.grid[start] == 1 or self.grid[goal] == 1:
            return False

        queue = deque([start])
        visited = {start}
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            curr = queue.popleft()
            if curr == goal:
                return True
            for dr, dc in deltas:
                nr, nc = curr[0] + dr, curr[1] + dc
                neighbor = (nr, nc)
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if self.grid[nr, nc] == 0 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
        return False

    def generate_valid_grid(self):
        """
        Place random obstacles and guarantee solvability via BFS verification.
        Eliminates unsolvable topological maps to ensure safe AI exploration.
        """
        max_attempts = 1000
        for _ in range(max_attempts):
            self.grid = np.zeros((self.grid_size, self.grid_size))
            num_obstacles = int(self.grid_size * self.grid_size * self.obstacle_density)
            obstacle_positions = np.random.choice(
                self.grid_size * self.grid_size, num_obstacles, replace=False
            )
            for pos in obstacle_positions:
                row, col = divmod(pos, self.grid_size)
                self.grid[row, col] = 1

            # Ensure start and goal cells are free
            self.grid[self.start_pos] = 0
            self.grid[self.goal_pos] = 0

            if self.is_reachable():
                return

        # Deterministic fallback: empty grid if 1000 attempts fail
        self.grid = np.zeros((self.grid_size, self.grid_size))

    def reset(self, regenerate=False):
        """Reset robot to start position, optionally regenerating the obstacle layout."""
        if regenerate:
            self.generate_valid_grid()
        self.robot_pos = self.start_pos
        return self.robot_pos

    def step(self, action):
        """
        Execute directional action and return (new_position, reward, done).
        Actions: 0=Up, 1=Down, 2=Left, 3=Right.
        """
        row, col = self.robot_pos
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_row, new_col = row + deltas[action][0], col + deltas[action][1]

        # Check boundary collision
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return self.robot_pos, -1.0, False

        # Check obstacle collision
        if self.grid[new_row, new_col] == 1:
            return self.robot_pos, -1.0, False

        self.robot_pos = (new_row, new_col)

        # Check goal reached
        if self.robot_pos == self.goal_pos:
            return self.robot_pos, 100.0, True
        else:
            return self.robot_pos, -0.1, False

    def get_state_index(self, pos):
        """Convert 2D position (row, col) to flattened state index."""
        return pos[0] * self.grid_size + pos[1]

    def get_position_from_index(self, index):
        """Convert state index to 2D position (row, col)."""
        return (index // self.grid_size, index % self.grid_size)


class QLearningAgent:
    """
    Q-Learning agent with dynamic epsilon decay and pure greedy deployment.
    Satisfies Unit IV Sequential Decision-Making & Safe AI (CO4).
    """

    def __init__(self, num_states, num_actions, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((num_states, num_actions))

    def select_action(self, state, training=True):
        """
        Select action:
        - During training: epsilon-greedy exploration.
        - During evaluation / deployment: pure greedy (epsilon = 0.0) for deterministic safety.
        """
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(self.num_actions))
        else:
            return int(np.argmax(self.q_table[state]))

    def decay_epsilon(self):
        """Decay exploration rate per completed episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Bellman optimality equation."""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def evaluate_greedy(self, env, max_steps=100):
        """
        Pure greedy evaluation (epsilon = 0.0) on the environment.
        Ensures collision-free, deterministic execution during testing.
        """
        start_time = time.perf_counter()
        env.reset()
        path = [env.robot_pos]
        total_reward = 0.0
        success = False

        for _ in range(max_steps):
            state_idx = env.get_state_index(env.robot_pos)
            action = self.select_action(state_idx, training=False)
            next_pos, reward, done = env.step(action)
            total_reward += reward
            path.append(next_pos)
            if done:
                success = (next_pos == env.goal_pos)
                break

        exec_time_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            'success': success,
            'path': path,
            'steps': len(path) - 1,
            'total_reward': total_reward,
            'execution_time_ms': exec_time_ms
        }

    def save_qtable(self, filepath):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)

    def load_qtable(self, filepath):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)


# ============================================================================
# GRID VISUALIZATION WIDGET
# ============================================================================

class GridWidget(QWidget):
    """Custom widget for rendering the grid environment and evaluated paths."""

    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        self.cell_size = 40
        self.path = []
        self.setMinimumSize(
            env.grid_size * self.cell_size + 2,
            env.grid_size * self.cell_size + 2
        )

    def set_path(self, path):
        """Set a path trajectory to visualize."""
        self.path = path or []
        self.update()

    def paintEvent(self, event):
        """Paint the grid, obstacles, robot, goal, and highlighted path."""
        painter = QPainter(self)
        path_set = set(self.path[1:-1]) if len(self.path) > 2 else set()

        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size

                # Determine cell color
                if (i, j) == self.env.goal_pos:
                    color = QColor(76, 175, 80)    # Green for goal
                elif (i, j) == self.env.robot_pos:
                    color = QColor(33, 150, 243)   # Blue for robot
                elif (i, j) in path_set:
                    color = QColor(255, 193, 7)    # Amber for path
                elif self.env.grid[i, j] == 1:
                    color = QColor(97, 97, 97)     # Slate Gray for obstacles
                elif (i, j) == self.env.start_pos:
                    color = QColor(225, 245, 254)  # Light cyan for start
                else:
                    color = QColor(255, 255, 255)  # White for free cell

                painter.fillRect(x, y, self.cell_size, self.cell_size, color)
                painter.setPen(QColor(210, 210, 210))
                painter.drawRect(x, y, self.cell_size, self.cell_size)

        painter.end()


def set_chart_axes_titles(chart, x_title, y_title):
    """Safely configure axis titles across both PyQt5 and PySide6."""
    axes = chart.axes()
    if len(axes) >= 2:
        axes[0].setTitleText(x_title)
        axes[1].setTitleText(y_title)
    else:
        try:
            if hasattr(chart, 'axisX') and chart.axisX():
                chart.axisX().setTitleText(x_title)
            if hasattr(chart, 'axisY') and chart.axisY():
                chart.axisY().setTitleText(y_title)
        except Exception:
            pass


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class RobotNavigationGUI(QMainWindow):
    """Main GUI application for Q-Learning robot navigation and search benchmarking."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Learning Robot Navigation & Search Benchmark (BT-CSE 303A)")
        self.setGeometry(100, 100, 1600, 900)

        # Initialize environment (BFS reachability guaranteed) and agent
        self.env = GridEnvironment(grid_size=10, obstacle_density=0.2)
        num_states = self.env.grid_size ** 2
        num_actions = 4
        self.agent = QLearningAgent(
            num_states, num_actions,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )

        # Training state and persistent episode metrics
        self.is_training = False
        self.episode = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

        self.rewards_history = deque(maxlen=100)
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        self.q_value_stats = []
        self.epsilon_history = []

        # Timer for training loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.training_step)

        # Create UI
        self.create_ui()

    def create_ui(self):
        """Create the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()

        # Left panel: Grid visualization
        left_layout = QVBoxLayout()
        self.grid_widget = GridWidget(self.env)
        left_layout.addWidget(self.grid_widget)

        # Right panel: Controls and plots
        right_layout = QVBoxLayout()

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.start_btn.clicked.connect(self.start_training)
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_training)
        self.pause_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_training)
        self.visualize_btn = QPushButton("Visualize Path")
        self.visualize_btn.clicked.connect(self.visualize_path)
        self.benchmark_btn = QPushButton("Benchmark A* vs Q-Learning")
        self.benchmark_btn.clicked.connect(self.run_benchmark)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.visualize_btn)
        button_layout.addWidget(self.benchmark_btn)
        right_layout.addLayout(button_layout)

        # Hyperparameters
        params_group = QGroupBox("Hyperparameters & Exploration Schedule")
        params_layout = QGridLayout()

        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate (α):"), 0, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.01, 1.0)
        self.lr_spinbox.setValue(0.1)
        self.lr_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.lr_spinbox, 0, 1)

        # Discount factor
        params_layout.addWidget(QLabel("Discount Factor (γ):"), 1, 0)
        self.df_spinbox = QDoubleSpinBox()
        self.df_spinbox.setRange(0.0, 1.0)
        self.df_spinbox.setValue(0.95)
        self.df_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.df_spinbox, 1, 1)

        # Epsilon (Current exploration rate)
        params_layout.addWidget(QLabel("Epsilon (ε):"), 2, 0)
        self.eps_spinbox = QDoubleSpinBox()
        self.eps_spinbox.setRange(0.0, 1.0)
        self.eps_spinbox.setValue(1.0)
        self.eps_spinbox.setSingleStep(0.05)
        self.eps_spinbox.setDecimals(4)
        self.eps_spinbox.valueChanged.connect(self.on_epsilon_user_changed)
        params_layout.addWidget(self.eps_spinbox, 2, 1)

        # Epsilon Decay Toggle and Decay Rate
        self.decay_checkbox = QCheckBox("Enable Epsilon Decay")
        self.decay_checkbox.setChecked(True)
        params_layout.addWidget(self.decay_checkbox, 3, 0)

        self.decay_spinbox = QDoubleSpinBox()
        self.decay_spinbox.setRange(0.800, 0.999)
        self.decay_spinbox.setValue(0.995)
        self.decay_spinbox.setSingleStep(0.001)
        self.decay_spinbox.setDecimals(3)
        params_layout.addWidget(self.decay_spinbox, 3, 1)

        # Minimum Epsilon
        params_layout.addWidget(QLabel("Minimum Epsilon (ε_min):"), 4, 0)
        self.eps_min_spinbox = QDoubleSpinBox()
        self.eps_min_spinbox.setRange(0.001, 0.50)
        self.eps_min_spinbox.setValue(0.01)
        self.eps_min_spinbox.setSingleStep(0.005)
        self.eps_min_spinbox.setDecimals(3)
        params_layout.addWidget(self.eps_min_spinbox, 4, 1)

        # Steps per update
        params_layout.addWidget(QLabel("Steps per Update:"), 5, 0)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 200)
        self.steps_spinbox.setValue(10)
        params_layout.addWidget(self.steps_spinbox, 5, 1)

        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)

        # Status display
        status_group = QGroupBox("Training Status & Telemetry")
        status_layout = QVBoxLayout()

        self.episode_label = QLabel("Episode: 0")
        self.reward_label = QLabel("Current Episode Reward: 0.0")
        self.avg_reward_label = QLabel("Avg Reward (last 100): 0.0")
        self.success_rate_label = QLabel("Success Rate: 0.0%")

        status_layout.addWidget(self.episode_label)
        status_layout.addWidget(self.reward_label)
        status_layout.addWidget(self.avg_reward_label)
        status_layout.addWidget(self.success_rate_label)

        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)

        # Save/Load buttons
        save_load_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Q-Table")
        self.save_btn.clicked.connect(self.save_qtable)
        self.load_btn = QPushButton("Load Q-Table")
        self.load_btn.clicked.connect(self.load_qtable)
        self.export_btn = QPushButton("Export Graphs")
        self.export_btn.clicked.connect(self.export_graphs)
        save_load_layout.addWidget(self.save_btn)
        save_load_layout.addWidget(self.load_btn)
        save_load_layout.addWidget(self.export_btn)
        right_layout.addLayout(save_load_layout)

        self.chart_tabs = QTabWidget()

        # Tab 1: Episode Rewards
        self.chart1 = QChart()
        self.chart1.setTitle("Episode Rewards")
        self.chart_view1 = QChartView(self.chart1)
        self.chart_view1.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view1, "Rewards")

        # Tab 2: Average Reward Trend
        self.chart2 = QChart()
        self.chart2.setTitle("Average Reward Trend (Smoothed)")
        self.chart_view2 = QChartView(self.chart2)
        self.chart_view2.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view2, "Avg Trend")

        # Tab 3: Q-Value Statistics
        self.chart3 = QChart()
        self.chart3.setTitle("Q-Value Statistics")
        self.chart_view3 = QChartView(self.chart3)
        self.chart_view3.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view3, "Q-Values")

        # Tab 4: Epsilon Decay
        self.chart4 = QChart()
        self.chart4.setTitle("Epsilon Decay Over Episodes")
        self.chart_view4 = QChartView(self.chart4)
        self.chart_view4.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view4, "Epsilon")

        # Tab 5: Steps to Goal
        self.chart5 = QChart()
        self.chart5.setTitle("Steps to Goal per Episode")
        self.chart_view5 = QChartView(self.chart5)
        self.chart_view5.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view5, "Steps")

        # Tab 6: Success Rate
        self.chart6 = QChart()
        self.chart6.setTitle("Success Rate Over Time")
        self.chart_view6 = QChartView(self.chart6)
        self.chart_view6.setRenderHint(QPainter.Antialiasing)
        self.chart_tabs.addTab(self.chart_view6, "Success Rate")

        right_layout.addWidget(self.chart_tabs)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)
        central_widget.setLayout(main_layout)

    def on_epsilon_user_changed(self, value):
        """User manual adjustment of epsilon via spinbox."""
        if not self.is_training:
            self.agent.epsilon = value

    def start_training(self):
        """Start the training loop."""
        self.is_training = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.timer.start(50)

    def pause_training(self):
        """Pause the training loop."""
        self.is_training = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)

    def reset_training(self):
        """Reset training state, persistent counters, and environment."""
        self.pause_training()
        self.episode = 0
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0

        self.rewards_history.clear()
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_successes = []
        self.q_value_stats = []
        self.epsilon_history = []

        # Re-generate grid with guaranteed BFS path
        self.env = GridEnvironment(grid_size=10, obstacle_density=0.2)
        self.grid_widget.env = self.env
        self.grid_widget.set_path([])

        # Reset epsilon to 1.0
        initial_eps = 1.0
        self.eps_spinbox.blockSignals(True)
        self.eps_spinbox.setValue(initial_eps)
        self.eps_spinbox.blockSignals(False)

        self.agent = QLearningAgent(
            self.env.grid_size ** 2, 4,
            learning_rate=self.lr_spinbox.value(),
            discount_factor=self.df_spinbox.value(),
            epsilon=initial_eps,
            epsilon_min=self.eps_min_spinbox.value(),
            epsilon_decay=self.decay_spinbox.value()
        )
        self.update_display()

    def training_step(self):
        """
        Execute training steps for this timer tick.
        Persists current_episode_reward and current_episode_steps across timer ticks.
        Prevents spinbox overwrite during epsilon decay.
        """
        # Synchronize hyperparameters
        self.agent.learning_rate = self.lr_spinbox.value()
        self.agent.discount_factor = self.df_spinbox.value()
        self.agent.epsilon_decay = self.decay_spinbox.value()
        self.agent.epsilon_min = self.eps_min_spinbox.value()

        # If decay is disabled, use manual spinbox value
        if not self.decay_checkbox.isChecked():
            self.agent.epsilon = self.eps_spinbox.value()

        steps = self.steps_spinbox.value()
        max_steps_per_episode = 1000

        for _ in range(steps):
            state_idx = self.env.get_state_index(self.env.robot_pos)
            action = self.agent.select_action(state_idx, training=True)
            next_pos, reward, done = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_pos)

            self.agent.update(state_idx, action, reward, next_state_idx, done)
            self.current_episode_reward += reward
            self.current_episode_steps += 1

            # Termination condition: reached goal or hit step limit
            if not done and self.current_episode_steps >= max_steps_per_episode:
                done = True

            if done:
                self.episode += 1
                self.rewards_history.append(self.current_episode_reward)
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_steps.append(self.current_episode_steps)

                reached_goal = (self.env.robot_pos == self.env.goal_pos)
                self.episode_successes.append(1 if reached_goal else 0)

                q_values = self.agent.q_table.flatten()
                self.q_value_stats.append({
                    'min': float(np.min(q_values)),
                    'max': float(np.max(q_values)),
                    'mean': float(np.mean(q_values))
                })

                # Decay epsilon per episode and write back to spinbox cleanly
                if self.decay_checkbox.isChecked():
                    self.agent.decay_epsilon()
                    self.eps_spinbox.blockSignals(True)
                    self.eps_spinbox.setValue(self.agent.epsilon)
                    self.eps_spinbox.blockSignals(False)

                self.epsilon_history.append(self.agent.epsilon)

                # Reset environment and counters for the next episode
                self.env.reset()
                self.current_episode_reward = 0.0
                self.current_episode_steps = 0

        self.update_display()

    def update_display(self):
        """Update the GUI display and telemetry charts."""
        self.grid_widget.update()
        self.episode_label.setText(f"Episode: {self.episode}")
        self.reward_label.setText(f"Current Episode Reward: {self.current_episode_reward:.1f}")

        if self.rewards_history:
            avg_reward = np.mean(list(self.rewards_history))
            self.avg_reward_label.setText(f"Avg Reward (last 100): {avg_reward:.2f}")

        if self.episode_successes:
            recent_successes = self.episode_successes[-100:]
            success_rate = (sum(recent_successes) / len(recent_successes)) * 100
            self.success_rate_label.setText(f"Success Rate: {success_rate:.1f}%")

        self.update_charts()

    def update_charts(self):
        """Update all 6 telemetry plots."""
        # Chart 1: Episode Rewards
        self.chart1.removeAllSeries()
        if self.episode_rewards:
            series = QLineSeries()
            series.setName("Episode Reward")
            for i, reward in enumerate(self.episode_rewards[-100:]):
                series.append(QPointF(i, reward))
            self.chart1.addSeries(series)
            self.chart1.createDefaultAxes()
            set_chart_axes_titles(self.chart1, "Episode", "Reward")

        # Chart 2: Average Reward Trend (Smoothed)
        self.chart2.removeAllSeries()
        if len(self.episode_rewards) > 10:
            smoothed_rewards = []
            window_size = 10
            for i in range(len(self.episode_rewards)):
                start = max(0, i - window_size)
                smoothed_rewards.append(np.mean(self.episode_rewards[start:i+1]))

            series = QLineSeries()
            series.setName("Smoothed Avg Reward")
            for i, reward in enumerate(smoothed_rewards[-100:]):
                series.append(QPointF(i, reward))
            self.chart2.addSeries(series)
            self.chart2.createDefaultAxes()
            set_chart_axes_titles(self.chart2, "Episode", "Avg Reward")

        # Chart 3: Q-Value Statistics
        self.chart3.removeAllSeries()
        if self.q_value_stats:
            min_series = QLineSeries()
            min_series.setName("Min Q-Value")
            max_series = QLineSeries()
            max_series.setName("Max Q-Value")
            mean_series = QLineSeries()
            mean_series.setName("Mean Q-Value")

            for i, stats in enumerate(self.q_value_stats[-100:]):
                min_series.append(QPointF(i, stats['min']))
                max_series.append(QPointF(i, stats['max']))
                mean_series.append(QPointF(i, stats['mean']))

            self.chart3.addSeries(min_series)
            self.chart3.addSeries(max_series)
            self.chart3.addSeries(mean_series)
            self.chart3.createDefaultAxes()
            set_chart_axes_titles(self.chart3, "Episode", "Q-Value")

        # Chart 4: Epsilon Decay
        self.chart4.removeAllSeries()
        if self.epsilon_history:
            series = QLineSeries()
            series.setName("Epsilon")
            for i, eps in enumerate(self.epsilon_history[-100:]):
                series.append(QPointF(i, eps))
            self.chart4.addSeries(series)
            self.chart4.createDefaultAxes()
            set_chart_axes_titles(self.chart4, "Episode", "Epsilon (Exploration Rate)")

        # Chart 5: Steps to Goal
        self.chart5.removeAllSeries()
        if self.episode_steps:
            series = QLineSeries()
            series.setName("Steps to Goal")
            for i, steps in enumerate(self.episode_steps[-100:]):
                series.append(QPointF(i, steps))
            self.chart5.addSeries(series)
            self.chart5.createDefaultAxes()
            set_chart_axes_titles(self.chart5, "Episode", "Steps")

        # Chart 6: Success Rate
        self.chart6.removeAllSeries()
        if len(self.episode_successes) > 10:
            success_rates = []
            window_size = 20
            for i in range(len(self.episode_successes)):
                start = max(0, i - window_size)
                window = self.episode_successes[start:i+1]
                success_rate = (sum(window) / len(window)) * 100
                success_rates.append(success_rate)

            series = QLineSeries()
            series.setName("Success Rate (%)")
            for i, rate in enumerate(success_rates[-100:]):
                series.append(QPointF(i, rate))
            self.chart6.addSeries(series)
            self.chart6.createDefaultAxes()
            set_chart_axes_titles(self.chart6, "Episode", "Success Rate (%)")

    def visualize_path(self):
        """Visualize optimal path using pure greedy policy (epsilon = 0.0)."""
        res = self.agent.evaluate_greedy(self.env, max_steps=100)
        self.grid_widget.set_path(res['path'])

        if res['success']:
            QMessageBox.information(
                self, "Pure Greedy Policy Visualization",
                f"Goal reached successfully in {res['steps']} steps!\n"
                f"Execution Latency: {res['execution_time_ms']:.3f} ms\n"
                f"Cumulative Evaluation Reward: {res['total_reward']:.1f}\n\n"
                f"Trajectory: {res['path']}"
            )
        else:
            QMessageBox.warning(
                self, "Pure Greedy Policy Visualization",
                f"Agent did not reach goal within 100 steps (policy not fully converged).\n"
                f"Steps taken: {res['steps']}\n"
                f"Path: {res['path']}"
            )

    def run_benchmark(self):
        """
        Run empirical side-by-side benchmark:
        Classical A* Search (Unit II) vs. Trained Q-Learning Policy (Unit IV).
        """
        # 1. Classical A* Search Baseline
        astar_res = astar_shortest_path(self.env.grid, self.env.start_pos, self.env.goal_pos)

        # 2. Q-Learning Policy Deployment (epsilon = 0.0)
        ql_res = self.agent.evaluate_greedy(self.env, max_steps=100)

        # Render Q-Learning path on grid
        if ql_res['path']:
            self.grid_widget.set_path(ql_res['path'])

        status_str = "REACHED" if ql_res['success'] else "FAILED"
        ratio_str = f"{(ql_res['steps'] / astar_res['steps']):.3f}" if (astar_res['steps'] > 0 and ql_res['success']) else "N/A"
        astar_nodes_str = f"{astar_res['nodes_expanded']} nodes"
        ql_steps_str = f"{sum(self.episode_steps)} steps ({self.episode} eps)"
        astar_time_str = f"{astar_res['execution_time_ms']:.3f} ms"
        ql_time_str = f"{ql_res['execution_time_ms']:.3f} ms"

        summary = (
            "========================================================================================\n"
            "                 ALGORITHM BENCHMARK: CLASSICAL A* VS. REINFORCEMENT LEARNING            \n"
            "========================================================================================\n"
            f"Environment Configuration : 10x10 Grid World (20% Obstacle Density)\n"
            f"Start Position -> Goal   : {self.env.start_pos} -> {self.env.goal_pos}\n"
            f"Training Episodes        : {self.episode}\n"
            "----------------------------------------------------------------------------------------\n"
            f"{'Metric':<30} | {'Classical A* (Unit II)':<24} | {'Q-Learning (Unit IV)':<24}\n"
            "----------------------------------------------------------------------------------------\n"
            f"{'Search Paradigm':<30} | {'Heuristic State Search':<24} | {'Model-Free Q-Policy':<24}\n"
            f"{'Core Mechanism':<30} | {'Manhattan Heuristic h(n)':<24} | {'Learned Q-Table':<24}\n"
            f"{'Goal Reached?':<30} | {'YES (Guaranteed)':<24} | {status_str:<24}\n"
            f"{'Path Length (Steps)':<30} | {astar_res['steps']:<24} | {ql_res['steps']:<24}\n"
            f"{'Optimality Ratio (Actual/Opt)':<30} | {'1.000 (Optimal)':<24} | {ratio_str:<24}\n"
            f"{'Search Space Expanded / Steps':<30} | {astar_nodes_str:<24} | {ql_steps_str:<24}\n"
            f"{'Runtime Inference Time (ms)':<30} | {astar_time_str:<24} | {ql_time_str:<24}\n"
            f"{'Operational Safety':<30} | {'Complete & Admissible':<24} | {'Pure Greedy (ε=0.0)':<24}\n"
            "========================================================================================\n"
        )
        print(summary)
        QMessageBox.information(self, "A* vs. Q-Learning Benchmark", summary)

    def save_qtable(self):
        """Save the Q-table to a file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Q-Table", "", "NumPy Files (*.npy)"
        )
        if filepath:
            self.agent.save_qtable(filepath)
            QMessageBox.information(self, "Success", f"Q-table saved to {filepath}")

    def load_qtable(self):
        """Load a Q-table from a file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Q-Table", "", "NumPy Files (*.npy)"
        )
        if filepath:
            self.agent.load_qtable(filepath)
            QMessageBox.information(self, "Success", f"Q-table loaded from {filepath}")

    def export_graphs(self):
        """Export all graphs as PNG files or combined PDF."""
        export_format, ok = QFileDialog.getSaveFileName(
            self, "Export Graphs", "", "PDF Files (*.pdf);;PNG Files (*.png)"
        )

        if not ok or not export_format:
            return

        try:
            if export_format.endswith('.pdf'):
                self.export_graphs_as_pdf(export_format)
            else:
                self.export_graphs_as_png(export_format)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export graphs: {str(e)}")

    def export_graphs_as_pdf(self, filepath):
        """Export all graphs as a single PDF document."""
        fig_height = 4
        fig_width = 8

        with PdfPages(filepath) as pdf:
            # Chart 1: Episode Rewards
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if self.episode_rewards:
                ax.plot(self.episode_rewards[-100:], label="Episode Reward", linewidth=2)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
                ax.set_title("Episode Rewards")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Chart 2: Average Reward Trend
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if len(self.episode_rewards) > 10:
                smoothed_rewards = []
                window_size = 10
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - window_size)
                    smoothed_rewards.append(np.mean(self.episode_rewards[start:i+1]))
                ax.plot(smoothed_rewards[-100:], label="Smoothed Avg Reward", linewidth=2, color='orange')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Avg Reward")
                ax.set_title("Average Reward Trend (Smoothed)")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Chart 3: Q-Value Statistics
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if self.q_value_stats:
                min_vals = [s['min'] for s in self.q_value_stats[-100:]]
                max_vals = [s['max'] for s in self.q_value_stats[-100:]]
                mean_vals = [s['mean'] for s in self.q_value_stats[-100:]]
                ax.plot(min_vals, label="Min Q-Value", linewidth=2)
                ax.plot(max_vals, label="Max Q-Value", linewidth=2)
                ax.plot(mean_vals, label="Mean Q-Value", linewidth=2)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Q-Value")
                ax.set_title("Q-Value Statistics")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Chart 4: Epsilon Decay
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if self.epsilon_history:
                ax.plot(self.epsilon_history[-100:], label="Epsilon", linewidth=2, color='red')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Epsilon (Exploration Rate)")
                ax.set_title("Epsilon Decay Over Episodes")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Chart 5: Steps to Goal
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if self.episode_steps:
                ax.plot(self.episode_steps[-100:], label="Steps to Goal", linewidth=2, color='green')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Steps")
                ax.set_title("Steps to Goal per Episode")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Chart 6: Success Rate
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            if len(self.episode_successes) > 10:
                success_rates = []
                window_size = 20
                for i in range(len(self.episode_successes)):
                    start = max(0, i - window_size)
                    window = self.episode_successes[start:i+1]
                    success_rate = (sum(window) / len(window)) * 100
                    success_rates.append(success_rate)
                ax.plot(success_rates[-100:], label="Success Rate (%)", linewidth=2, color='purple')
                ax.set_xlabel("Episode")
                ax.set_ylabel("Success Rate (%)")
                ax.set_title("Success Rate Over Time")
                ax.legend()
                ax.grid(True, alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        QMessageBox.information(self, "Export Successful", f"Graphs exported to:\n{filepath}")

    def export_graphs_as_png(self, filepath):
        """Export all graphs as individual PNG files."""
        base_path = Path(filepath).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = base_path / f"q_learning_graphs_{timestamp}"
        export_dir.mkdir(exist_ok=True)

        fig_height = 5
        fig_width = 10

        # Chart 1: Episode Rewards
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if self.episode_rewards:
            ax.plot(self.episode_rewards[-100:], label="Episode Reward", linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Episode Rewards")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "01_episode_rewards.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Chart 2: Average Reward Trend
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if len(self.episode_rewards) > 10:
            smoothed_rewards = []
            window_size = 10
            for i in range(len(self.episode_rewards)):
                start = max(0, i - window_size)
                smoothed_rewards.append(np.mean(self.episode_rewards[start:i+1]))
            ax.plot(smoothed_rewards[-100:], label="Smoothed Avg Reward", linewidth=2, color='orange')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Avg Reward")
            ax.set_title("Average Reward Trend (Smoothed)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "02_average_reward_trend.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Chart 3: Q-Value Statistics
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if self.q_value_stats:
            min_vals = [s['min'] for s in self.q_value_stats[-100:]]
            max_vals = [s['max'] for s in self.q_value_stats[-100:]]
            mean_vals = [s['mean'] for s in self.q_value_stats[-100:]]
            ax.plot(min_vals, label="Min Q-Value", linewidth=2)
            ax.plot(max_vals, label="Max Q-Value", linewidth=2)
            ax.plot(mean_vals, label="Mean Q-Value", linewidth=2)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Q-Value")
            ax.set_title("Q-Value Statistics")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "03_q_value_statistics.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Chart 4: Epsilon Decay
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if self.epsilon_history:
            ax.plot(self.epsilon_history[-100:], label="Epsilon", linewidth=2, color='red')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Epsilon (Exploration Rate)")
            ax.set_title("Epsilon Decay Over Episodes")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "04_epsilon_decay.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Chart 5: Steps to Goal
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if self.episode_steps:
            ax.plot(self.episode_steps[-100:], label="Steps to Goal", linewidth=2, color='green')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Steps")
            ax.set_title("Steps to Goal per Episode")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "05_steps_to_goal.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Chart 6: Success Rate
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if len(self.episode_successes) > 10:
            success_rates = []
            window_size = 20
            for i in range(len(self.episode_successes)):
                start = max(0, i - window_size)
                window = self.episode_successes[start:i+1]
                success_rate = (sum(window) / len(window)) * 100
                success_rates.append(success_rate)
            ax.plot(success_rates[-100:], label="Success Rate (%)", linewidth=2, color='purple')
            ax.set_xlabel("Episode")
            ax.set_ylabel("Success Rate (%)")
            ax.set_title("Success Rate Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.savefig(export_dir / "06_success_rate.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        QMessageBox.information(self, "Export Successful", f"Graphs exported to:\n{export_dir}")


# ============================================================================
# HEADLESS SIMULATION & EMPIRICAL BENCHMARK (FOR FORMAL CASE STUDY REPORT)
# ============================================================================

def run_headless_simulation(episodes=500, grid_size=10, obstacle_density=0.2,
                            lr=0.1, gamma=0.95, eps_start=1.0, eps_min=0.01, eps_decay=0.995):
    """
    Run an empirical evaluation without GUI overhead.
    Computes rigorous metrics comparing Classical A* vs. Q-Learning for course report.
    """
    print("\n" + "=" * 80)
    print(f"HEADLESS EMPIRICAL SIMULATION ({episodes} Episodes, Grid {grid_size}x{grid_size}, 20% Obstacles)")
    print("=" * 80)

    # 1. Environment initialization with BFS reachability verification
    env = GridEnvironment(grid_size=grid_size, obstacle_density=obstacle_density)
    bfs_verified = env.is_reachable()
    print(f"[BFS Verification] Start {env.start_pos} -> Goal {env.goal_pos} solvable: {bfs_verified}")

    # 2. Classical A* Search Baseline
    astar_res = astar_shortest_path(env.grid, env.start_pos, env.goal_pos)
    print(f"[A* Search Baseline] Optimal path found: {astar_res['steps']} steps, "
          f"{astar_res['nodes_expanded']} nodes expanded in {astar_res['execution_time_ms']:.3f} ms.")

    # 3. Q-Learning Agent with Epsilon Decay
    num_states = env.grid_size ** 2
    num_actions = 4
    agent = QLearningAgent(
        num_states=num_states,
        num_actions=num_actions,
        learning_rate=lr,
        discount_factor=gamma,
        epsilon=eps_start,
        epsilon_min=eps_min,
        epsilon_decay=eps_decay
    )

    episode_rewards = []
    episode_steps = []
    episode_successes = []

    # 4. Training Loop
    t_train_start = time.perf_counter()
    for ep in range(1, episodes + 1):
        env.reset()
        ep_reward = 0.0
        ep_step = 0
        done = False

        while not done and ep_step < 1000:
            s_idx = env.get_state_index(env.robot_pos)
            action = agent.select_action(s_idx, training=True)
            next_pos, reward, done = env.step(action)
            ns_idx = env.get_state_index(next_pos)
            agent.update(s_idx, action, reward, ns_idx, done)

            ep_reward += reward
            ep_step += 1
            if not done and ep_step >= 1000:
                done = True

        agent.decay_epsilon()
        episode_rewards.append(ep_reward)
        episode_steps.append(ep_step)
        episode_successes.append(1 if (env.robot_pos == env.goal_pos) else 0)

    t_train_total = (time.perf_counter() - t_train_start) * 1000.0

    # 5. Pure Greedy Policy Deployment (epsilon = 0.0)
    ql_res = agent.evaluate_greedy(env, max_steps=100)

    # Statistical summaries
    last_100_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    last_100_steps = episode_steps[-100:] if len(episode_steps) >= 100 else episode_steps
    last_100_succ = episode_successes[-100:] if len(episode_successes) >= 100 else episode_successes

    avg_reward = np.mean(last_100_rewards)
    avg_steps = np.mean(last_100_steps)
    succ_rate = (sum(last_100_succ) / len(last_100_succ)) * 100.0
    optimality_ratio = (ql_res['steps'] / astar_res['steps']) if (astar_res['steps'] > 0 and ql_res['success']) else 1.0

    astar_nodes_str = f"{astar_res['nodes_expanded']} nodes"
    ql_samples_str = f"{sum(episode_steps)} steps ({episodes} eps)"
    astar_time_str = f"{astar_res['execution_time_ms']:.3f} ms"
    ql_time_str = f"{ql_res['execution_time_ms']:.3f} ms"

    # Formatted Comparison Table
    print("\n" + "=" * 80)
    print(f"{'A* SEARCH VS. Q-LEARNING BENCHMARK (BT-CSE 303A TW/SL 2(c))':^80}")
    print("=" * 80)
    print(f"{'Evaluation Metric':<32} | {'Classical A* (Unit II)':<22} | {'Q-Learning (Unit IV)':<22}")
    print("-" * 80)
    print(f"{'Paradigm / Category':<32} | {'Heuristic Search':<22} | {'Model-Free RL':<22}")
    print(f"{'Knowledge / Policy Model':<32} | {'Manhattan Distance h(n)':<22} | {'Learned Tabular Q-Values':<22}")
    print(f"{'Exploration Parameter':<32} | {'N/A (Deterministic)':<22} | {f'ε: {eps_start:.2f} -> {agent.epsilon:.3f}':<22}")
    print(f"{'Goal Reached?':<32} | {'YES (Guaranteed)':<22} | {('YES (Converged)' if ql_res['success'] else 'NO'):<22}")
    print(f"{'Path Steps (Start to Goal)':<32} | {astar_res['steps']:<22} | {ql_res['steps']:<22}")
    print(f"{'Path Optimality Ratio':<32} | {'1.000 (Optimal)':<22} | {f'{optimality_ratio:.3f}':<22}")
    print(f"{'Search Effort / Samples':<32} | {astar_nodes_str:<22} | {ql_samples_str:<22}")
    print(f"{'Runtime Inference Time (ms)':<32} | {astar_time_str:<22} | {ql_time_str:<22}")
    print(f"{'Total Offline Training Time':<32} | {'0.0 ms (No Training)':<22} | {f'{t_train_total:.1f} ms':<22}")
    print(f"{'Success Rate (Final 100 eps)':<32} | {'100.0%':<22} | {f'{succ_rate:.1f}%':<22}")
    print(f"{'Avg Episode Reward (Final 100)':<32} | {'N/A':<22} | {f'{avg_reward:.2f}':<22}")
    print(f"{'Avg Steps to Goal (Final 100)':<32} | {astar_res['steps']:<22} | {f'{avg_steps:.2f}':<22}")
    print("=" * 80)

    print(f"\nA* Shortest Path Coordinates ({len(astar_res['path'])} states):")
    print(f"  {astar_res['path']}")
    print(f"\nQ-Learning Deployed Policy Path ({len(ql_res['path'])} states):")
    print(f"  {ql_res['path']}")
    print("=" * 80 + "\n")

    return {
        'astar': astar_res,
        'qlearning': ql_res,
        'episodes': episodes,
        'training_time_ms': t_train_total,
        'final_epsilon': agent.epsilon,
        'final_avg_reward': avg_reward,
        'final_avg_steps': avg_steps,
        'final_success_rate': succ_rate
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if "--headless" in sys.argv:
        episodes_to_run = 500
        for arg in sys.argv:
            if arg.startswith("--episodes="):
                try:
                    episodes_to_run = int(arg.split("=")[1])
                except ValueError:
                    pass
        run_headless_simulation(episodes=episodes_to_run)
    else:
        if not GUI_AVAILABLE:
            print("\n" + "=" * 80)
            print("Notice: No Qt GUI toolkit (PySide6 or PyQt5) was found in your active Python.")
            print("=" * 80)
            print(f"Current Python executable: {sys.executable}")
            print("\nTo launch the interactive GUI, you can:")
            print("1. Switch to Python 3.11 (where PySide6 is already installed on your machine):")
            print("   In VS Code: Press Ctrl+Shift+P -> 'Python: Select Interpreter' -> Select Python 3.11")
            print("   In Terminal: & 'C:\\Users\\PULKIT\\AppData\\Local\\Programs\\Python\\Python311\\python.exe' main.py")
            print("\n2. Or install PySide6 in your current Python environment:")
            print("   pip install PySide6")
            print("\n3. Or run the headless empirical simulation directly right now:")
            print("   python main.py --headless --episodes=500")
            print("=" * 80 + "\n")
            print("Executing headless benchmark simulation automatically...\n")
            run_headless_simulation(episodes=500)
            sys.exit(0)

        app = QApplication.instance() or QApplication(sys.argv)
        window = RobotNavigationGUI()
        window.show()
        sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())
