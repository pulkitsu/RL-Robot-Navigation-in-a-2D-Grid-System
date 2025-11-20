# RL Robot Navigation in a 2D Grid-System

A comprehensive PyQt5-based desktop application for visualizing and training a robot using Q-learning in a grid-world environment. Features real-time training visualization, six interactive performance graphs, and advanced analytics.

https://github.com/user-attachments/assets/4e038303-797a-42c3-8598-3d65f3f3f6ee

## Features

### Core Functionality
- **Grid-Based Simulation**: 10x10 grid with obstacles, start position, and goal
- **Real-Time Visualization**: Watch the robot navigate and learn in real-time
- **Q-Learning Algorithm**: Implements standard Q-learning with epsilon-greedy exploration
- **Training Controls**: Start, Pause, and Reset buttons for full training control
- **Path Visualization**: Test the trained policy and visualize the optimal path

### Advanced Analytics (6 Interactive Graphs)
- **Episode Rewards**: Raw reward per episode showing training progress
  <img width="1749" height="962" alt="image" src="https://github.com/user-attachments/assets/e8c7a8e0-19e0-40b8-a3c8-47dce840ec1b" />

- **Average Reward Trend**: Smoothed 10-episode moving average for clearer learning trends
  <img width="1721" height="952" alt="image" src="https://github.com/user-attachments/assets/6b41c013-82af-423e-a132-70c9838a5bce" />

- **Q-Value Statistics**: Min, max, and mean Q-values to monitor algorithm convergence
  <img width="1713" height="953" alt="image" src="https://github.com/user-attachments/assets/fdbc158a-f739-4ae7-b1bc-cd8ccb218465" />

- **Epsilon Decay**: Exploration rate over time showing exploration-exploitation balance
  <img width="1748" height="963" alt="image" src="https://github.com/user-attachments/assets/4b13e5e5-c4d6-49ec-9087-117f6e4cc934" />

- **Steps to Goal**: Number of steps taken per episode measuring learning efficiency
  <img width="1692" height="957" alt="image" src="https://github.com/user-attachments/assets/fff3c07e-d7a2-4d63-a200-84f7882227db" />

- **Success Rate**: Rolling 20-episode success rate percentage
  <img width="1711" height="953" alt="image" src="https://github.com/user-attachments/assets/2f6ac6d7-08f9-4693-a82a-c6c6793f1220" />


### Hyperparameter Control
- **Learning Rate (α)**: Controls how quickly the agent learns (0.01-1.0)
- **Discount Factor (γ)**: Determines importance of future rewards (0.0-1.0)
- **Epsilon (ε)**: Controls exploration vs. exploitation (0.0-1.0)
- **Steps per Update**: Number of steps before updating the display

### Data Management
- **Save/Load Q-Tables**: Persist learned policies for later use
- **Export Graphs**: Export all 6 charts as individual PNG files or combined PDF
- **Timestamped Exports**: Organized file structure with automatic timestamps

## Requirements

- Python 3.10+
- PyQt5
- NumPy
- Matplotlib

## Installation

1. Install dependencies:
\`\`\`bash
pip install PyQt5 numpy matplotlib
\`\`\`

2. Run the application:
\`\`\`bash
python main.py
\`\`\`

## Usage

### Starting Training
1. Click **"Start Training"** to begin the Q-learning process
2. The robot will explore the grid and learn to reach the goal
3. Watch all 6 performance graphs update in real-time as the agent learns

### Monitoring Training Progress
- **Episode Rewards Tab**: View raw rewards per episode
- **Average Reward Trend Tab**: See smoothed learning curve (10-episode moving average)
- **Q-Value Statistics Tab**: Monitor convergence of Q-values (min, max, mean)
- **Epsilon Decay Tab**: Track exploration rate decrease over time
- **Steps to Goal Tab**: Measure learning efficiency (fewer steps = better learning)
- **Success Rate Tab**: View percentage of successful episodes (20-episode rolling average)

### Adjusting Hyperparameters
- **Learning Rate**: Controls convergence speed (higher = faster but less stable)
- **Discount Factor**: Determines long-term planning (higher = plan further ahead)
- **Epsilon**: Controls exploration (higher = more random exploration)
- **Steps per Update**: Increase for faster training, decrease for smoother visualization

### Visualizing the Learned Path
1. Train the agent for several episodes
2. Click **"Visualize Path"** to see the optimal path the agent learned
3. A message displays the number of steps taken to reach the goal

### Exporting Results
1. Click **"Export Graphs"** after training
2. Choose export format:
   - **PDF**: All 6 charts combined into a single document (ideal for reports)
   - **PNG**: Individual high-resolution images in a timestamped folder
3. Select destination folder and export

### Saving and Loading Q-Tables
- **Save Q-Table**: Export the learned Q-table to a `.npy` file for later use
- **Load Q-Table**: Import a previously saved Q-table to continue training or test

## How It Works

### Environment
- 10x10 grid with randomly placed obstacles
- Robot starts at (0, 0) and goal is at (9, 9)
- Actions: Up, Down, Left, Right
- Rewards: +100 for reaching goal, -1 for hitting obstacle/boundary, -0.1 per step

### Q-Learning Algorithm
The agent learns by updating Q-values using:
\`\`\`
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
\`\`\`

Where:
- α = learning rate
- r = reward
- γ = discount factor
- s' = next state

### Exploration Strategy
The agent uses epsilon-greedy exploration:
- With probability ε: choose a random action (explore)
- With probability 1-ε: choose the best known action (exploit)

### Performance Metrics
- **Average Reward**: Smoothed over 10 episodes to show learning trend
- **Q-Value Statistics**: Indicates convergence (stable values = converged)
- **Success Rate**: Percentage of episodes reaching the goal
- **Steps to Goal**: Efficiency metric (lower is better)

## Tips for Training

1. **Start with default hyperparameters** for stable learning
2. **Monitor the Average Reward Trend** to see if learning is progressing
3. **Check Success Rate** to verify the agent is finding the goal
4. **Increase epsilon** if the agent gets stuck in local patterns
5. **Decrease learning rate** for more stable convergence
6. **Increase discount factor** to make the agent plan further ahead
7. **Export graphs** periodically to track training progress over time
8. **Save Q-tables** before trying new hyperparameters

## File Structure

\`\`\`
.
├── main.py          # Complete application with all logic
└── README.md        # Application usage instructions
\`\`\`

## Output Files

When exporting graphs, the following files are created:

**PNG Export** (in timestamped folder):
- `1_episode_rewards.png`
- `2_average_reward_trend.png`
- `3_qvalue_statistics.png`
- `4_epsilon_decay.png`
- `5_steps_to_goal.png`
- `6_success_rate.png`

**PDF Export**:
- `q_learning_graphs_TIMESTAMP.pdf` (all 6 charts combined)

**Q-Table Export**:
- `q_table_TIMESTAMP.npy` (NumPy binary format)

## Troubleshooting

- **Agent not learning**: Try increasing epsilon or decreasing learning rate
- **Slow training**: Increase "Steps per Update" for faster training
- **Path not found**: The agent may need more training episodes
- **Graphs not updating**: Ensure "Steps per Update" is not too high
- **Export fails**: Check that you have write permissions in the selected folder

## Advanced Usage

### Analyzing Training Dynamics
1. Train the agent with different hyperparameters
2. Export graphs for each configuration
3. Compare the PDF exports to identify optimal settings
4. Use the Q-Value Statistics graph to detect convergence issues

### Continuing Training
1. Save your Q-table after initial training
2. Load the Q-table in a new session
3. Adjust hyperparameters and continue training
4. Export graphs to compare before/after improvements

