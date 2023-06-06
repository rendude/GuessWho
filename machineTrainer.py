import random
from collections import namedtuple
import torch
from machinePlayer import MachinePlayer
from oraclePlayer import OraclePlayer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch.nn.functional as F
import os
from gameEnv import all_possible_characters, question_bank, DEBUG_MODE, TRAINING, GAME_MODE, DEBUG_CHAR
from collections import defaultdict

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'game_status')
)

MEMORY_SIZE = 4000
BATCH_SIZE = 200
NUM_SIMULATED_GAMES = 2000
TARGET_UPDATE = 20
# Setting game_mode to true, prints out questions and answers

if DEBUG_MODE:
    BATCH_SIZE = 1
    NUM_SIMULATED_GAMES = 3

class MachineTrainer():
    def __init__(self):
        self.capacity = MEMORY_SIZE
        self.push_count = 0
        self.batch_size = BATCH_SIZE
        self.memory = []
        self.machine_player_1 = MachinePlayer()
        self.machine_player_2 = MachinePlayer()

    def sample_experience(self):
        sample = random.sample(self.memory, self.batch_size)
        if DEBUG_MODE and not TRAINING:
            sample = [self.memory[-1]]
        batch = Experience(*zip(*sample))
        t1 = torch.stack(batch.state)
        t2 = torch.stack(batch.action)
        t3 = torch.stack(batch.next_state)
        t4 = torch.stack(batch.game_status)
        return (t1, t2, t3, t4)

    def play_one_round(self):
        state = self.machine_player_1.state
        action, question_text = self.machine_player_1.ask_question()
        answer = self.machine_player_2.take_question(question_text)
        next_state = self.machine_player_1.take_answer(answer)

        if DEBUG_MODE:
            print(f'Player 1 asked {question_text} #{action}')
            print(f'Player 2 answered {answer}')
            print(f"Another eliminated {len(all_possible_characters) - torch.sum(next_state)} and they are")                        
            for i, diff in enumerate(next_state):
                if diff == 0:
                    print(f'{list(all_possible_characters.keys())[i]} was eliminated')
                if diff == 1:
                    print(f'{list(all_possible_characters.keys())[i]} remains')

        if self.machine_player_1.game_status == 0:
            # Player 2 gets to ask a question too in case they won
            action_2, question_text = self.machine_player_2.ask_question()
            answer = self.machine_player_1.take_question(question_text)
            self.machine_player_2.take_answer(answer)

        return Experience(state, torch.tensor([action]), next_state, torch.FloatTensor([self.machine_player_1.game_status]))
        
    def add_to_memory(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def can_provide_sample(self):
        return self.batch_size <= len(self.memory)

    def train(self):
        print("Starting training mode\n")
        num_ques_asked = []
        player_wins = 0
        player_win_rate = []
        remaining_when_lost = defaultdict(int)
        char_count = defaultdict(int)

        for episode in range(1, NUM_SIMULATED_GAMES+1):
            # print("Player 1")
            self.machine_player_1.reset()
            # print("Player 2")
            self.machine_player_2.reset()
            
            for timestep in count():
                self.add_to_memory(self.play_one_round())

                if self.can_provide_sample():
                    # learn a little bit every time
                    #print("I can start learning now")
                    states, actions, next_states, player_1_game_status = self.sample_experience()

                    # Backprop one step
                    self.machine_player_1.pick_question_brain.update_weights(states, actions, next_states, player_1_game_status)

                # If game ended
                if self.machine_player_1.game_status != 0 or self.machine_player_2.game_status != 0:
                    #print(f"Game over, player 1's {self.machine_player_1.game_status} and {torch.sum(self.machine_player_1.state)} remaining. took {self.machine_player_1.total_questions_asked} questions.")
                    if self.machine_player_1.game_status == 1:
                        player_wins += 1

                    remaining_when_lost[self.machine_player_2.given_char_name] += torch.sum(self.machine_player_1.state)
                    char_count[self.machine_player_2.given_char_name] += 1
                    questions_asked = self.machine_player_1.total_questions_asked
                    num_ques_asked.append(questions_asked)
                    player_win_rate.append((player_wins/episode)*100)
                    self.plot_questions_asked(num_ques_asked, 100)
                    self.plot_win_rate(player_win_rate, 100)
                    break
            
            if episode % TARGET_UPDATE == 0:
                self.machine_player_1.pick_question_brain.swap_neural_nets_for_stabilitiy()                
                self.machine_player_2.pick_question_brain.swap_neural_nets_for_stabilitiy(
                    self.machine_player_1.pick_question_brain.current_state_net.state_dict())
                if TRAINING:
                    torch.save(trainer.machine_player_1.pick_question_brain.current_state_net, "./machineBrain/trainedNet.pt")

    def plot_qvalues(self, state=None):
        x = self.machine_player_1.pick_question_brain.current_state_net(state)
        # Print the top 5 question candidates
        top_questions = torch.topk(x, 5)[1]

        for i in top_questions:
            print(self.machine_player_1.pick_question_brain.possible_actions[i] + " " + str(x[i].detach().item()))

    def get_moving_average(self, period, values):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= period:
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()


    def plot_questions_asked(self, values, moving_avg_period):
        plt.figure(2)
        plt.clf()
        plt.title("Training...")
        plt.xlabel('Episode')
        plt.ylabel('Avg Questions Asked')

        moving_avg = self.get_moving_average(moving_avg_period, values)
        plt.plot(moving_avg)
        plt.pause(0.01)
        #print("Episode", len(values), "\n", moving_avg_period, "average duration: ", moving_avg[-1])
        if is_ipython: display.clear_output(wait=True)

    def plot_win_rate(self, values, period: int):
        plt.figure(1)
        plt.clf()
        plt.title("Training...")
        plt.xlabel('Episode')
        plt.ylabel('Win rate')

        moving_avg = self.get_moving_average(period, values)
        plt.plot(moving_avg)
        plt.pause(0.01)
        #print("Episode", len(values), "\n", moving_avg_period, "average duration: ", moving_avg[-1])
        if is_ipython: display.clear_output(wait=True)

if TRAINING:
    try:
        os.remove('./machineBrain/trainedNet.pt')
        print("old net deleted")
    except:
        pass

# Sanity check the data set
for key, value in all_possible_characters.items():
    if len(value) != 27:
        print(key+" does not have the right number of attributes")


trainer = MachineTrainer()
trainer.train()