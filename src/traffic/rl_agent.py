import numpy as np
import random
import json
import os
from loguru import logger
from typing import Dict, List, Tuple

class TrafficRLAgent:
    """
    A simple Q-Learning agent that learns optimal green times per lane.
    State: Density level (Low, Medium, High, Critical)
    Action: Green time adjustment (-10s, -5s, 0s, +5s, +10s)
    Reward: Negative of (Waiting Vehicles + Lane Congestion)
    """
    
    ACTIONS = [-10, -5, 0, 5, 10]
    
    def __init__(self, 
                 lane_id: str, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.9, 
                 epsilon: float = 0.2):
        self.lane_id = lane_id
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # State: density_level (0-3)
        # Action index: 0-4
        self.q_table = defaultdict(lambda: np.zeros(len(self.ACTIONS)))
        self.history_file = f"outputs/rl_state_{lane_id}.json"
        
        # Load previous knowledge if exists
        self._load_q_table()

    def get_action(self, state: str) -> int:
        """Choose action using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Explore
            action_idx = random.randint(0, len(self.ACTIONS) - 1)
        else:
            # Exploit
            action_idx = np.argmax(self.q_table[state])
            
        return self.ACTIONS[action_idx]

    def update(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-Value based on reward experience."""
        action_idx = self.ACTIONS.index(action)
        
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-Learning Formula: Q(s,a) = Q(s,a) + alpha * [reward + gamma * maxQ(s',a') - Q(s,a)]
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q
        
        # Periodically save
        if random.random() < 0.05:
            self._save_q_table()

    def _save_q_table(self):
        """Persist q-table to disk."""
        try:
            # Convert default dict of numpy arrays to serializable dict
            serializable = {k: v.tolist() for k, v in self.q_table.items()}
            os.makedirs("outputs", exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(serializable, f)
        except Exception as e:
            logger.error(f"RL Save Error: {e}")

    def _load_q_table(self):
        """Load q-table from disk."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self.q_table[k] = np.array(v)
                logger.info(f"RL Agent {self.lane_id}: Loaded knowledge from disk.")
            except Exception as e:
                logger.error(f"RL Load Error: {e}")

from collections import defaultdict
