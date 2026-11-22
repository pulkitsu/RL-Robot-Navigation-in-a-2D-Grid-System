# RL Robot Navigation in a 2D Grid System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![GUI](https://img.shields.io/badge/GUI-PySide6%20%7C%20PyQt5%20%7C%20PyQt6-green.svg)
![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)

A desktop application and benchmark suite for autonomous robot navigation on a discrete 2D grid. It compares classical search (**BFS** and **A\***) with **tabular Q-learning**, and lets you watch the agent learn in real time.

---

## 📸 Demo

*Main window: grid environment, real-time training and live telemetry charts.*

https://github.com/user-attachments/assets/135123f1-11c2-4a30-80bb-a8ace868110e

---

## 🏆 Results at a Glance

500-episode run on a 10×10 grid (start `(0, 0)`, goal `(9, 9)`, 20% obstacle density):

| Metric | A\* | Q-Learning |
| :--- | :---: | :---: |
| Deployed path length | 18 steps | 18 steps |
| Optimality ratio (L / L\*) | 1.000 | 1.000 |
| Final-100-episode success rate | 100% | 100% |
| Search effort / samples | 64 nodes expanded | 28,419 training steps |
| Offline training time | none | 465.4 ms |
| Online time per run | 0.117 ms | 0.069 ms |

Q-learning matches the A\* optimum after training and then acts by table lookup. A\* needs no training but searches from scratch each time. The timing figures are small, single-run measurements: A\* time covers a full search, Q-learning time covers following the stored table, so treat the ratio as indicative rather than a strict like-for-like comparison.

---

## ✨ Features

- **Interactive grid environment**: 10×10 grid with 20% random obstacles, fixed start and goal.
- **BFS solvability check**: every generated map is verified reachable; unsolvable maps are re-rolled automatically.
- **A\* benchmark**: Manhattan-distance heuristic (admissible and consistent) gives the optimal path length.
- **Tabular Q-learning**: temporal-difference updates with exponential ε-decay (1.0 → 0.01, factor 0.995 per episode).
- **Safe deployment**: path visualisation and benchmarks use pure greedy action selection (ε = 0).
- **Head-to-head comparison**: A\* vs Q-learning in the GUI or from the command line.
- **6 live telemetry charts**: episode rewards, smoothed reward (10-episode window), Q-value min/max/mean, ε decay, steps to goal, rolling success rate (20-episode window).
- **Adjustable hyperparameters** while training runs.
- **Persistence and export**: save/load Q-tables as `.npy`; export charts as PNGs or one PDF.
- **Toolkit flexibility**: works with PySide6, PyQt5 or PyQt6, with a headless CLI mode when no GUI toolkit is installed.

---

## 🎓 Syllabus Alignment

| Unit | CO | Concepts | Implementation in `main.py` |
| :--- | :---: | :--- | :--- |
| **I: Intelligent Agents & Problem Formulation** | CO1 | PEAS, environment classification (fully observable, deterministic, sequential, static, discrete) | `GridEnvironment`: 100 states, 4 actions, rewards +100 / −1.0 / −0.1 |
| **II: Problem Solving & Search** | CO2 | Uninformed search (BFS) and informed search (A\*) | `is_reachable()` and `astar_shortest_path()` |
| **IV: AI Planning, Robotics & Responsible AI** | CO4 | TD learning, policy auditability, safe deployment | `QLearningAgent`, `evaluate_greedy()`, `.npy` persistence |

---

## 🚀 Installation

**Requirements:** Python 3.8+ and pip.

```bash
# 1. Clone
git clone https://github.com/pulkitsu/RL-Robot-Navigation-in-a-2D-Grid-System.git
cd RL-Robot-Navigation-in-a-2D-Grid-System

# 2. Virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 3. Dependencies
pip install -r requirements.txt
```

If you install manually, PySide6 is the simplest option because it bundles Qt Charts:

```bash
pip install numpy matplotlib PySide6
```

With PyQt5 you must also install the charts add-on (`pip install numpy matplotlib PyQt5 PyQtChart`), since `main.py` imports `PyQt5.QtChart`.

---

## 💻 Usage

### Mode 1: Desktop GUI

```bash
python main.py
```

1. **Start Training** begins the Q-learning loop.
2. **Monitor** the robot on the grid and the six live charts.
3. **Adjust** learning rate, discount factor, ε and decay in the side panel.
4. **Pause / Resume** to inspect progress.
5. **Visualize Path** shows the learned greedy route (amber overlay).
6. **Benchmark** runs A\* vs Q-learning and shows path length, nodes visited and time.
7. **Export Graphs** saves all charts as PNGs or one PDF; **Save / Load Q-Table** stores weights as `.npy`.

### Mode 2: Headless benchmark (no GUI)

```bash
python main.py --headless --episodes=500
```

Prints the full comparison table, useful for collecting report data.

---

## ⚙️ Configuration

### Hyperparameters

| Parameter | Symbol | Range | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| Learning rate | α | 0.01 – 1.0 | 0.10 | How strongly new estimates override old Q-values |
| Discount factor | γ | 0.0 – 1.0 | 0.95 | Weight of future vs immediate reward |
| Initial epsilon | ε₀ | 0.0 – 1.0 | 1.00 | Starting exploration rate |
| Epsilon decay | λ | 0.800 – 0.999 | 0.995 | Per-episode factor: ε ← max(ε_min, ε · λ) |
| Minimum epsilon | ε_min | 0.001 – 0.50 | 0.01 | Exploration floor |
| Steps per update | – | 1 – 200 | 10 | Simulation steps per GUI refresh |

### Environment

```python
self.env = GridEnvironment(
    grid_size=10,          # 10x10 grid (100 states)
    obstacle_density=0.2   # 20% of cells targeted as obstacles
)
```

The generator places `int(100 × 0.2) = 20` obstacles and then clears the start and goal cells, so a layout can contain 19 or 20 obstacles.

### Rewards

| Event | Reward |
| :--- | :---: |
| Reach goal (9, 9), terminal | +100.0 |
| Hit obstacle or boundary (position unchanged) | −1.0 |
| Valid move | −0.1 |

With an 18-step optimal path, the best achievable return is 100 − 18 × 0.1 = 98.2.

---

## 📚 Algorithms

**BFS (solvability check).** A FIFO queue explores the grid level by level; if the goal is never dequeued, the map is rejected and regenerated.

**A\* (optimal baseline).** Nodes are ordered by `f(n) = g(n) + h(n)` with `h(n) = |r − r_goal| + |c − c_goal|`. Each step costs 1, so the heuristic never overestimates (admissible) and satisfies `h(n) ≤ c(n, n′) + h(n′)` (consistent), which guarantees an optimal path.

**Q-learning.**

```
Q(s, a) ← Q(s, a) + α [ r + γ · max_a′ Q(s′, a′) − Q(s, a) ]
```

Training uses ε-greedy action selection with exponential decay (ε ≈ 0.082 after 500 episodes). Deployment uses `a* = argmax_a Q(s, a)` with ε = 0.

### Learned 18-step path

```
(0,0) → (0,1) → (1,1) → (1,2) → (2,2) → (3,2) → (4,2) → (5,2) → (6,2) → (7,2)
      → (7,3) → (7,4) → (7,5) → (7,6) → (7,7) → (8,7) → (8,8) → (9,8) → (9,9)
```

`Test.npy` stores the converged 100×4 Q-table; following its greedy actions from `(0, 0)` reproduces this path.

---

## 🛡️ Safety and Limitations

- **Pre-flight BFS check** guarantees a solution exists before an agent is deployed.
- **ε = 0 at deployment** removes random exploratory moves from live operation.
- **Inspectable policy**: the Q-table is a plain 100×4 array that can be audited directly.
- **Limits**: tabular Q-learning scales as O(|S|·|A|) and the policy is tied to one fixed obstacle layout. Continuous or large state spaces need function approximation (e.g. DQN), and moving obstacles need replanning (e.g. D\* Lite). Reported results come from a single grid and run.

---

## 📦 Project Structure

```
RL-Robot-Navigation-in-a-2D-Grid-System/
├── main.py                           # BFS, A*, Q-learning, GUI and headless CLI
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── Case_Study_Report_BT_CSE_303A.pdf # Formal case study report (PDF)
├── Case_Study_Report_BT_CSE_303A.docx# Case study report (Word)
├── Case_Study_Report_BT_CSE_303A.md  # Case study report (Markdown)
├── Grid.png                          # 10x10 obstacle environment
├── Grid Goal Path.png                # 18-step greedy path overlay
├── 50 Episodes Graph.pdf             # Original vector export of training telemetry
├── Test.npy                          # Converged 100x4 Q-table
└── telemetry_charts/                 # PNGs of the six training charts
    ├── 01_episode_rewards.png
    ├── 02_smoothed_avg_reward.png
    ├── 03_q_value_statistics.png
    ├── 04_epsilon_decay.png
    ├── 05_steps_to_goal.png
    └── 06_success_rate.png
```

---

## 📄 License

Copyright © Pulkitsu 2026. All rights reserved.

Developed for academic evaluation. Not licensed for distribution, modification or commercial use without written permission from the author.

---

## 🙏 Acknowledgments

- Q-learning: Watkins & Dayan (1992).
- A\* search: Hart, Nilsson & Raphael (1968).
- Environment design inspired by the GridWorld examples in Sutton & Barto, *Reinforcement Learning: An Introduction*.
- GUI built with PySide6 / PyQt.

---

## 📧 Contact

**Pulkit Sulekh**: [LinkedIn](https://www.linkedin.com/in/pulkitsulekh)
Project: [github.com/pulkitsu/RL-Robot-Navigation-in-a-2D-Grid-System](https://github.com/pulkitsu/RL-Robot-Navigation-in-a-2D-Grid-System)
