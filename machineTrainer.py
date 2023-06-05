import random
from collections import namedtuple
import torch
from machinePlayer import MachinePlayer
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import torch.optim as optim
import torch.nn.functional as F
import os
from gameEnv import all_possible_characters, question_bank

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

MEMORY_SIZE = 1000000
BATCH_SIZE = 1000
NUM_SIMULATED_GAMES = 6000
TARGET_UPDATE = 10
GAMMA = 0.7
LEARNING_RATE = 0.01
# Setting game_mode to true, prints out questions and answers
GAME_MODE = False
DEBUG_MODE = False
TRAINING = True
if DEBUG_MODE:
    BATCH_SIZE = 1
    NUM_SIMULATED_GAMES = 3

class MachineTrainer():
    def __init__(self):
        self.capacity = MEMORY_SIZE
        self.push_count = 0
        self.batch_size = BATCH_SIZE
        self.memory = []
        self.machine_player_1 = MachinePlayer(game_mode=GAME_MODE, debug_mode=DEBUG_MODE, 
            training_mode=TRAINING)
        if DEBUG_MODE:
            self.machine_player_1 = MachinePlayer(char_name="Claire", game_mode=GAME_MODE, debug_mode=DEBUG_MODE, 
                training_mode=TRAINING)
        self.machine_player_2 = MachinePlayer(game_mode=GAME_MODE, debug_mode=DEBUG_MODE, 
            training_mode=TRAINING)

    def sample_experience(self):
        sample = random.sample(self.memory, self.batch_size)
        if DEBUG_MODE and not TRAINING:
            sample = [self.memory[-1]]
        batch = Experience(*zip(*sample))
        t1 = torch.stack(batch.state)
        t2 = torch.cat(batch.action)
        t3 = torch.cat(batch.reward)
        t4 = torch.stack(batch.next_state)
        return (t1, t2, t3, t4)

    def get_experience_and_rewards(self):
        state = self.machine_player_1.state
        action, question_text = self.machine_player_1.ask_question()
        answer = self.machine_player_2.take_question(question_text)
        next_state, done = self.machine_player_1.take_answer(answer)

        reward = torch.tensor([-1])
        # num_eliminated = len(all_possible_characters) - torch.sum(next_state)
        # reward = torch.tensor([num_eliminated])

        # Player 2 gets to ask a question too in case they won
        action, question_text = self.machine_player_2.ask_question()
        answer = self.machine_player_1.take_question(question_text)
        self.machine_player_2.take_answer(answer)

        return Experience(state, action, next_state, reward)
        
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

        optimizer = optim.SGD(
            params=self.machine_player_1.pick_question_brain.current_state_net.parameters(), lr=LEARNING_RATE)

        for episode in range(NUM_SIMULATED_GAMES):
            # print("Player 1")
            self.machine_player_1.reset()
            # print("Player 2")
            self.machine_player_2.reset()
            
            for timestep in count():
                self.add_to_memory(self.get_experience_and_rewards())

                if self.can_provide_sample():
                    # learn a little bit every time
                    #print("I can start learning now")
                    states, actions, rewards, next_states = self.sample_experience()
                    if DEBUG_MODE:
                        print(f"Batch size is {BATCH_SIZE}")
                        print(f"machine 1 asked #{actions}: {question_bank[actions]}")
                        print(f"reward was {rewards}")
                        print(f"Total eliminated {len(all_possible_characters) - torch.sum(next_states)} and they are")                        

                        for i, next_state in enumerate(next_states[0]):
                            if next_state == 0:
                                print(f'{list(all_possible_characters.keys())[i]} was eliminated')

                        print(f"learn next states are "+ str(next_states))                        
                    optimizer.zero_grad()

                    current_q_a_values = self.machine_player_1.pick_question_brain.get_current_q_val(states, actions)
                    next_q_values = self.machine_player_1.pick_question_brain.get_next_q_val(next_states, self.machine_player_2.done)

                    # Bellman equation
                    target_q_values = rewards + GAMMA * next_q_values

                    if DEBUG_MODE:
                        print("Current Q Values are "+str(current_q_a_values))
                        print("Target Q Value are "+str(target_q_values.unsqueeze(1)))
                        current_policy_net = self.machine_player_1.pick_question_brain.current_state_net(states)

                    loss = F.mse_loss(current_q_a_values, target_q_values.unsqueeze(1))
                    loss.backward()
                    optimizer.step()

                    if DEBUG_MODE:
                        updated_policy_net = self.machine_player_1.pick_question_brain.current_state_net(states)
                        self.plot_qvalues(states[0])
                        if not TRAINING:
                            diff = updated_policy_net - current_policy_net
                            print(f"Biggest adjustment happened for action {question_bank[diff.argmax()]}")
                            print(str(diff.argmax()) + " by "+ str(diff[0][diff.argmax().item()])+ "; avg is "+str(diff.mean()))

                if self.
                if self.machine_player_1.done:
                    questions_asked = self.machine_player_1.pick_question_brain.total_questions_asked
                    if DEBUG_MODE:
                        print(f"Game over, player 1's state {self.machine_player_1.state}. took {questions_asked} questions.")
                    num_ques_asked.append(questions_asked)
                    self.plot(num_ques_asked, 100)
                    print("\n"*4)
                    break
            
            if episode % TARGET_UPDATE == 0:
                self.machine_player_1.pick_question_brain.next_state_net.load_state_dict(
                    self.machine_player_1.pick_question_brain.current_state_net.state_dict()
                )
                
                self.machine_player_2.pick_question_brain.next_state_net.load_state_dict(
                    self.machine_player_1.pick_question_brain.current_state_net.state_dict()
                )

                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1.01*param_group['lr']

            if TRAINING:
                avg = self.get_moving_average(100, num_ques_asked)[-1]
                print(avg)
                if avg < 4.8 and avg > 0:
                    break 

    def plot_qvalues(self, state=None):
        x = self.machine_player_1.pick_question_brain.current_state_net(state)
        # Print the top 5 question candidates
        top_questions = torch.topk(x, 5)[1]

        for i in top_questions:
            print(self.machine_player_1.pick_question_brain.possible_actions[i] + " " + str(x[i].detach().item()))

    def plot(self, values, moving_avg_period):
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

    def get_moving_average(self, period, values):
        values = torch.tensor(values, dtype=torch.float)
        if len(values) >= period:
            moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
            moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
            return moving_avg.numpy()
        else:
            moving_avg = torch.zeros(len(values))
            return moving_avg.numpy()

if TRAINING:
    try:
        os.remove('./machineBrain/trainedNet.pt')
        print("old net deleted")
    except:
        pass
trainer = MachineTrainer()
trainer.train()
if TRAINING:
    torch.save(trainer.machine_player_1.pick_question_brain.current_state_net, "./machineBrain/trainedNet.pt")