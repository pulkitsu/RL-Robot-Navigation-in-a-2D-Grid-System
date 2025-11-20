# RL Robot Navigation in a 2D Grid-System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-5.15.9-green.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

A PyQt5-based desktop application for visualizing and training a robot using Q-learning reinforcement learning algorithm in a grid-world environment.

## 📸 Demo

*Main application window showing grid environment and training metrics and real-time training visualization*

https://github.com/user-attachments/assets/aaf340c3-5326-4e12-aa42-291f7163e925


## ✨ Features

- **Interactive Grid Environment**: 10x10 customizable grid with random obstacles
- **Real-time Training Visualization**: Watch the agent learn optimal paths dynamically
- **6 Analytics Charts**: 
  - Episode Rewards
  - Average Reward Trend (Smoothed)
  - Q-Value Statistics (Min/Max/Mean)
  - Epsilon Decay Over Time
  - Steps to Goal per Episode
  - Success Rate Over Time
- **Adjustable Hyperparameters**: Modify learning rate, discount factor, and epsilon on-the-fly
- **Model Persistence**: Save and load trained Q-tables
- **Export Capabilities**: Save all training metrics as PDF or individual PNG files


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. **Clone the repository:**
```bash
git clone https://github.com/pulkitsu/RL-Robot-Navigation-in-a-2D-Grid-System.git
cd RL-Robot-Navigation-in-a-2D-Grid-System
```
2. **Create a virtual environment (recommended):**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```


## 💻 Usage

### Basic Usage
Run the application:
```bash
python main.py
```
### Training the Agent
1. **Start Training**: Click the "Start Training" button to begin Q-learning
2. **Monitor Progress**: Watch the grid visualization and real-time charts
3. **Adjust Parameters**: Modify hyperparameters during training in the right panel
4. **Pause/Resume**: Use "Pause" button to temporarily stop training
5. **Visualize Learned Policy**: Click "Visualize Path" to see the optimal route


## ⚙️ Configuration

### Hyperparameters
| Parameter | Symbol | Range | Default | Description |
|-----------|--------|-------|---------|-------------|
| **Learning Rate** | α | 0.01 - 1.0 | 0.1 | Controls how much new information overrides old knowledge |
| **Discount Factor** | γ | 0.0 - 1.0 | 0.95 | Determines importance of future rewards vs immediate rewards |
| **Epsilon** | ε | 0.0 - 1.0 | 0.1 | Exploration rate (higher = more random exploration) |
| **Steps per Update** | - | 1 - 100 | 10 | Number of training steps before GUI refresh |

### Environment Configuration
You can modify the environment by changing these parameters in `main.py`:
```python
# In RobotNavigationGUI.__init__()
self.env = GridEnvironment(
    grid_size=10,          # Size of the grid (10x10)
    obstacle_density=0.2   # 20% of cells are obstacles
)
```

### Reward Structure

- **Reach Goal**: +100
- **Hit Obstacle/Wall**: -1
- **Normal Step**: -0.1 (encourages shorter paths)

## 📚 Documentation

### How Q-Learning Works

Q-Learning is a model-free reinforcement learning algorithm that learns the value of actions in different states. The agent updates its Q-table using:

```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Where:
- `s` = current state
- `a` = action taken
- `r` = reward received
- `s'` = next state
- `α` = learning rate
- `γ` = discount factor


## 📦 Project Structure

```
RL-Robot-Navigation-in-a-2D-Grid-System/
├── main.py              # Main application file (all code)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── screenshots/        # Demo images and GIFs
│   ├── main_window.png
│   └── training_demo.gif
├── saved_models/       # Directory for saved Q-tables
│   └── .gitkeep
└── exported_graphs/    # Directory for exported charts
    └── .gitkeep
```

## 🔬 Expected Results

After training for **500-1000 episodes** with default parameters:

- **Episode Rewards**: Stabilizes around 95-100
- **Success Rate**: Reaches 85-95%
- **Steps to Goal**: Reduces to ~18-20 steps (near-optimal)
- **Q-Values**: Mean Q-value converges to positive values

**Optimal Path Length**: 18 steps (for 10x10 grid from top-left to bottom-right)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Q-Learning algorithm based on Watkins & Dayan (1992)
- Built with PyQt5 for cross-platform GUI
- Visualization inspired by OpenAI Gym environments

## 📧 Contact

**Pulkit Sulekh** - https://www.linkedin.com/in/pulkitsulekh

Project Link: https://github.com/pulkitsu/RL-Robot-Navigation-in-a-2D-Grid-System

---

⭐ **Star this repo if you find it helpful!**
