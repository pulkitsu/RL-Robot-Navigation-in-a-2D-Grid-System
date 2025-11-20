"""
RL Robot Navigation in a 2D Grid System
A PyQt5-based desktop application for visualizing and training a robot using Q-learning.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QGridLayout, QGroupBox, QComboBox, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPainter, QFont
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QBarSeries, QBarSet, QBarCategoryAxis
from PyQt5.QtCore import QPointF

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# ENVIRONMENT AND Q-LEARNING AGENT
# ============================================================================

class GridEnvironment:
    """Grid-world environment for robot navigation."""
    
    def __init__(self, grid_size=10, obstacle_density=0.2):
        """
        Initialize the grid environment.
        
        Args:
            grid_size: Size of the square grid
            obstacle_density: Proportion of cells that are obstacles
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size))
        
        # Place obstacles randomly
        num_obstacles = int(grid_size * grid_size * obstacle_density)
        obstacle_positions = np.random.choice(
            grid_size * grid_size, num_obstacles, replace=False
        )
        for pos in obstacle_positions:
            row, col = divmod(pos, grid_size)
            self.grid[row, col] = 1  # 1 = obstacle
        
        # Set start and goal positions (ensure they're not obstacles)
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)
        self.grid[self.start_pos] = 0
        self.grid[self.goal_pos] = 0
        
        self.robot_pos = self.start_pos
    
    def reset(self):
        """Reset robot to start position."""
        self.robot_pos = self.start_pos
        return self.robot_pos
    
    def step(self, action):
        """
        Execute action and return new state, reward, and done flag.
        
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        row, col = self.robot_pos
        
        # Action deltas
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_row, new_col = row + deltas[action][0], col + deltas[action][1]
        
        # Check bounds
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return self.robot_pos, -1, False
        
        # Check obstacle
        if self.grid[new_row, new_col] == 1:
            return self.robot_pos, -1, False
        
        self.robot_pos = (new_row, new_col)
        
        # Reward
        if self.robot_pos == self.goal_pos:
            return self.robot_pos, 100, True
        else:
            return self.robot_pos, -0.1, False
    
    def get_state_index(self, pos):
        """Convert position to state index."""
        return pos[0] * self.grid_size + pos[1]
    
    def get_position_from_index(self, index):
        """Convert state index to position."""
        return (index // self.grid_size, index % self.grid_size)


class QLearningAgent:
    """Q-Learning agent for robot navigation."""
    
    def __init__(self, num_states, num_actions, learning_rate=0.1, 
                 discount_factor=0.95, epsilon=0.1):
        """
        Initialize Q-Learning agent.
        
        Args:
            num_states: Number of states in the environment
            num_actions: Number of possible actions
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule."""
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
    
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
    """Custom widget for rendering the grid environment."""
    
    def __init__(self, env, parent=None):
        super().__init__(parent)
        self.env = env
        self.cell_size = 40
        self.setMinimumSize(
            env.grid_size * self.cell_size + 2,
            env.grid_size * self.cell_size + 2
        )
    
    def paintEvent(self, event):
        """Paint the grid, obstacles, robot, and goal."""
        painter = QPainter(self)
        
        # Draw grid
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                x = j * self.cell_size
                y = i * self.cell_size
                
                # Determine cell color
                if (i, j) == self.env.goal_pos:
                    color = QColor(76, 175, 80)  # Green for goal
                elif (i, j) == self.env.robot_pos:
                    color = QColor(33, 150, 243)  # Blue for robot
                elif self.env.grid[i, j] == 1:
                    color = QColor(100, 100, 100)  # Gray for obstacles
                else:
                    color = QColor(255, 255, 255)  # White for empty
                
                painter.fillRect(x, y, self.cell_size, self.cell_size, color)
                painter.drawRect(x, y, self.cell_size, self.cell_size)
        
        painter.end()


# ============================================================================
# MAIN GUI APPLICATION
# ============================================================================

class RobotNavigationGUI(QMainWindow):
    """Main GUI application for Q-Learning robot navigation."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-Learning Robot Navigation")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize environment and agent
        self.env = GridEnvironment(grid_size=10, obstacle_density=0.2)
        num_states = self.env.grid_size ** 2
        num_actions = 4
        self.agent = QLearningAgent(num_states, num_actions)
        
        # Training state
        self.is_training = False
        self.episode = 0
        self.rewards_history = deque(maxlen=100)
        self.episode_rewards = []
        self.episode_steps = []  # Track steps per episode
        self.episode_successes = []  # Track success/failure
        self.q_value_stats = []  # Track Q-value statistics
        self.epsilon_history = []  # Track epsilon decay
        
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
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.visualize_btn)
        right_layout.addLayout(button_layout)
        
        # Hyperparameters
        params_group = QGroupBox("Hyperparameters")
        params_layout = QGridLayout()
        
        # Learning rate
        params_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.01, 1.0)
        self.lr_spinbox.setValue(0.1)
        self.lr_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.lr_spinbox, 0, 1)
        
        # Discount factor
        params_layout.addWidget(QLabel("Discount Factor:"), 1, 0)
        self.df_spinbox = QDoubleSpinBox()
        self.df_spinbox.setRange(0.0, 1.0)
        self.df_spinbox.setValue(0.95)
        self.df_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.df_spinbox, 1, 1)
        
        # Epsilon
        params_layout.addWidget(QLabel("Epsilon (Exploration):"), 2, 0)
        self.eps_spinbox = QDoubleSpinBox()
        self.eps_spinbox.setRange(0.0, 1.0)
        self.eps_spinbox.setValue(0.1)
        self.eps_spinbox.setSingleStep(0.01)
        params_layout.addWidget(self.eps_spinbox, 2, 1)
        
        # Episodes per update
        params_layout.addWidget(QLabel("Steps per Update:"), 3, 0)
        self.steps_spinbox = QSpinBox()
        self.steps_spinbox.setRange(1, 100)
        self.steps_spinbox.setValue(10)
        params_layout.addWidget(self.steps_spinbox, 3, 1)
        
        params_group.setLayout(params_layout)
        right_layout.addWidget(params_group)
        
        # Status display
        status_group = QGroupBox("Training Status")
        status_layout = QVBoxLayout()
        
        self.episode_label = QLabel("Episode: 0")
        self.reward_label = QLabel("Current Reward: 0")
        self.avg_reward_label = QLabel("Avg Reward (last 100): 0")
        self.success_rate_label = QLabel("Success Rate: 0%")  # Add success rate display
        
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
        
        # Add layouts to main
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)
        central_widget.setLayout(main_layout)
    
    def start_training(self):
        """Start the training process."""
        self.is_training = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.timer.start(50)  # Update every 50ms
    
    def pause_training(self):
        """Pause the training process."""
        self.is_training = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def reset_training(self):
        """Reset the training."""
        self.pause_training()
        self.episode = 0
        self.rewards_history.clear()
        self.episode_rewards = []
        self.episode_steps = []  # Reset steps tracking
        self.episode_successes = []  # Reset success tracking
        self.q_value_stats = []  # Reset Q-value stats
        self.epsilon_history = []  # Reset epsilon history
        self.env = GridEnvironment(grid_size=10, obstacle_density=0.2)
        self.agent = QLearningAgent(
            self.env.grid_size ** 2, 4,
            learning_rate=self.lr_spinbox.value(),
            discount_factor=self.df_spinbox.value(),
            epsilon=self.eps_spinbox.value()
        )
        self.grid_widget.env = self.env
        self.update_display()
    
    def training_step(self):
        """Execute one training step."""
        # Update hyperparameters
        self.agent.learning_rate = self.lr_spinbox.value()
        self.agent.discount_factor = self.df_spinbox.value()
        self.agent.epsilon = self.eps_spinbox.value()
        
        # Run multiple steps per update
        steps = self.steps_spinbox.value()
        episode_reward = 0
        episode_step_count = 0  # Track steps in current episode
        
        for _ in range(steps):
            state_idx = self.env.get_state_index(self.env.robot_pos)
            action = self.agent.select_action(state_idx, training=True)
            next_pos, reward, done = self.env.step(action)
            next_state_idx = self.env.get_state_index(next_pos)
            
            self.agent.update(state_idx, action, reward, next_state_idx, done)
            episode_reward += reward
            episode_step_count += 1  # Increment step count
            
            if done:
                self.episode += 1
                self.rewards_history.append(episode_reward)
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(episode_step_count)  # Store steps
                self.episode_successes.append(1)  # Mark as success
                
                q_values = self.agent.q_table.flatten()
                self.q_value_stats.append({
                    'min': np.min(q_values),
                    'max': np.max(q_values),
                    'mean': np.mean(q_values)
                })
                
                self.epsilon_history.append(self.agent.epsilon)
                
                self.env.reset()
                episode_reward = 0
                episode_step_count = 0  # Reset step count
        
        self.update_display()
    
    def update_display(self):
        """Update the GUI display."""
        self.grid_widget.update()
        self.episode_label.setText(f"Episode: {self.episode}")
        
        if self.rewards_history:
            avg_reward = np.mean(list(self.rewards_history))
            self.avg_reward_label.setText(f"Avg Reward (last 100): {avg_reward:.2f}")
        
        if self.episode_successes:
            success_rate = (sum(self.episode_successes) / len(self.episode_successes)) * 100
            self.success_rate_label.setText(f"Success Rate: {success_rate:.1f}%")
        
        # Update all charts
        self.update_charts()
    
    def update_charts(self):
        """Update all reward plots."""
        self.chart1.removeAllSeries()
        if self.episode_rewards:
            series = QLineSeries()
            series.setName("Episode Reward")
            for i, reward in enumerate(self.episode_rewards[-100:]):
                series.append(QPointF(i, reward))
            self.chart1.addSeries(series)
            self.chart1.createDefaultAxes()
            self.chart1.axisX().setTitleText("Episode")
            self.chart1.axisY().setTitleText("Reward")
        
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
            self.chart2.axisX().setTitleText("Episode")
            self.chart2.axisY().setTitleText("Avg Reward")
        
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
            self.chart3.axisX().setTitleText("Episode")
            self.chart3.axisY().setTitleText("Q-Value")
        
        self.chart4.removeAllSeries()
        if self.epsilon_history:
            series = QLineSeries()
            series.setName("Epsilon")
            for i, eps in enumerate(self.epsilon_history[-100:]):
                series.append(QPointF(i, eps))
            self.chart4.addSeries(series)
            self.chart4.createDefaultAxes()
            self.chart4.axisX().setTitleText("Episode")
            self.chart4.axisY().setTitleText("Epsilon (Exploration Rate)")
        
        self.chart5.removeAllSeries()
        if self.episode_steps:
            series = QLineSeries()
            series.setName("Steps to Goal")
            for i, steps in enumerate(self.episode_steps[-100:]):
                series.append(QPointF(i, steps))
            self.chart5.addSeries(series)
            self.chart5.createDefaultAxes()
            self.chart5.axisX().setTitleText("Episode")
            self.chart5.axisY().setTitleText("Steps")
        
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
            self.chart6.axisX().setTitleText("Episode")
            self.chart6.axisY().setTitleText("Success Rate (%)")
    
    def visualize_path(self):
        """Visualize the optimal path using the trained Q-table."""
        self.env.reset()
        path = [self.env.robot_pos]
        max_steps = 100
        
        for _ in range(max_steps):
            state_idx = self.env.get_state_index(self.env.robot_pos)
            action = self.agent.select_action(state_idx, training=False)
            next_pos, _, done = self.env.step(action)
            path.append(next_pos)
            
            if done:
                break
        
        if self.env.robot_pos == self.env.goal_pos:
            QMessageBox.information(
                self, "Path Visualization",
                f"Reached goal in {len(path) - 1} steps!\nPath: {path}"
            )
        else:
            QMessageBox.warning(
                self, "Path Visualization",
                f"Did not reach goal after {max_steps} steps.\nPath: {path}"
            )
    
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
        """Export all graphs as PNG files or PDF."""
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
        """Export all graphs as a single PDF file."""
        fig_count = 6
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
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RobotNavigationGUI()
    window.show()
    sys.exit(app.exec_())
