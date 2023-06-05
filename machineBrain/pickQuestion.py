import random
import torch
import math
from gameEnv import all_possible_characters
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_START = 1
EPS_END = 0.0001
EPS_DECAY = 0.0001

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DNN(nn.Module):
    def __init__(self, possible_actions):
        super().__init__()
        # DNN setup
        self.fc1 = nn.Linear(in_features=len(all_possible_characters), 
            out_features=48)
        self.fc2 = nn.Linear(in_features=48, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=len(possible_actions))

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = F.relu(self.fc3(t))
        t = F.relu(self.fc4(t))
        t = self.out(t)
        return t

class PickQuestionDNN():
    def __init__(self, possible_actions, asks_randomly):
        self.possible_actions = possible_actions
        self.current_step = 0
        self.total_questions_asked = 0
        self.next_state_net = None
        self.current_state_net = None
        self.asks_randomly = asks_randomly
        self.loadNN()

    def get_exploration_rate(self, time_step):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1 * time_step * EPS_DECAY)
    
    def get_current_q_val(self, states, actions):
        return self.current_state_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    def get_next_q_val(self, next_state, other_player_won):
        if torch.sum(next_state)==1:
            # Won! Utility is large
            return torch.tensor([100])
        elif self.total_questions_asked >= 20 or other_player_won:
            # Stuck in loop, lost, utility is low
            return torch.tensor([-100])

        return self.next_state_net(next_state).max(dim=1)[0].detach()
    
    def reset(self):
        self.total_questions_asked = 0

    def loadNN(self):
        # Get the previously stored policy net
        try:
            current_state_net = torch.load("./machineBrain/current_state_net.pt")
    
            if current_state_net:
                self.next_state_net = DNN(self.possible_actions).to(device)
                self.next_state_net.load_state_dict(current_state_net.state_dict())
                self.next_state_net.eval()
                self.current_state_net = current_state_net
        except:
            self.current_state_net = DNN(self.possible_actions).to(device)
            self.next_state_net = DNN(self.possible_actions).to(device)

    def pick_random_question_index(self):
        question_index = random.randrange(len(self.possible_actions))
        return question_index

    def pick_next_question(self, state):
        # State is one-hot-coded remaining characteres' indeces
        self.current_step += 1
        self.total_questions_asked += 1

        exploration_rate = self.get_exploration_rate(self.current_step)
        if self.asks_randomly or exploration_rate > random.random():
            # Do a random action. Action is the index number of the interview questions
            question_index = self.pick_random_question_index()
            return torch.tensor([question_index]).to(device), self.possible_actions[question_index]
        else:
            with torch.no_grad():
                # Give the action with the highest estimated q-value
                qvalues = self.current_state_net(state)
                question_index = torch.argmax(qvalues)
                return question_index, self.possible_actions[question_index]
