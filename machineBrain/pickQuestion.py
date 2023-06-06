import random
import torch
import math
from gameEnv import all_possible_characters, question_bank
import torch
import torch.nn as nn
import torch.nn.functional as F
from gameEnv import DEBUG_MODE, TRAINING
import torch.optim as optim

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
            out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
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
        self.optimizer = None
        self.loadNN()

    def get_exploration_rate(self, time_step):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-1 * time_step * EPS_DECAY)
    
    def get_current_q_val(self, states, actions):
        return self.current_state_net(states).gather(dim=1, index=actions)

    def get_next_q_val(self, next_states, game_statuses):
        # We will transform this clone
        next_qs = torch.clone(game_statuses)

        for i in range(len(next_qs)):
            game_status = next_qs[i][0]
            if game_status == 1:
                # Won! Utility is large
                next_qs[i] = torch.tensor([100])
            elif game_status == -1:
                # Stuck in loop, lost, utility is low or the other player won first
                next_qs[i] = torch.tensor([-100])
            elif game_status == 0:
                estimate = torch.tensor([self.next_state_net(next_states[i]).max().item()])
                next_qs[i] = estimate
        if DEBUG_MODE:
            print(f'Next q is {next_qs}')
        return next_qs
    
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

        self.optimizer = optim.SGD(params=self.current_state_net.parameters(), lr=0.001)

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

    def update_weights(self, current_states, actions, next_states, game_status):
        if self.asks_randomly:
            return

        self.optimizer.zero_grad()

        current_q_a_values = self.get_current_q_val(current_states, actions)
        next_q_values = self.get_next_q_val(next_states, game_status)

        rewards = torch.full_like(actions, -1)
        # num_eliminated = len(all_possible_characters) - torch.sum(next_state)
        # rewards = torch.tensor([num_eliminated])

        # Bellman equation
        target_q_values = rewards + 0.7 * next_q_values

        if DEBUG_MODE:
            print(f'Current qs are {self.current_state_net(current_states)}')
            print(f"Action is {actions}")
            print("Current Q Values are "+str(current_q_a_values))
            print("Target Q Value are "+str(target_q_values))
            current_policy_net = self.current_state_net(current_states)

        loss = F.mse_loss(current_q_a_values, target_q_values)
        loss.backward()
        self.optimizer.step() 

        if DEBUG_MODE:
            updated_policy_net = self.current_state_net(current_states)
            diff = updated_policy_net - current_policy_net
            print(f'Difference in the update {diff}')
            print(f"Biggest adjustment happened for action {question_bank[diff.argmax()]}")
            print(str(diff.argmax()) + " by "+ str(diff[0][diff.argmax().item()])+ "; avg is "+str(diff.mean()))


    def swap_neural_nets_for_stabilitiy(self, net=None):
        if net:
            self.next_state_net.load_state_dict(net)
        else:
            self.next_state_net.load_state_dict(
               self.current_state_net.state_dict()
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = .95*param_group['lr']
            