import random
import torch
import math
from gameEnv import all_possible_characters

# Takes an answer and eliminate characters, returning remaining characters as state
class TakeAnswerBrain():
    def __init__(self):
        self.remaining_characters = torch.ones(len(all_possible_characters))
        self.next_state = []

    def get_state(self):
        return self.remaining_characters.clone()

    def reset(self):
        self.remaining_characters = torch.ones(len(all_possible_characters))

    def eliminate_char(self, question_asked_index: int, answer_binary: int):
        for i, (char_name, char_attributes) in enumerate(all_possible_characters.items()):
            if question_asked_index < len(char_attributes) and char_attributes[question_asked_index] != answer_binary:
                self.remaining_characters[i] = 0
        #print("Remaining characters are "+str(self.remaining_characters))
    
    def take_answer(self, question_asked_index, answer_binary: int):
        done = False
        self.eliminate_char(question_asked_index, answer_binary)
        # Exit if there is one character left
        if torch.sum(self.remaining_characters) == 1:
            done = True
        return self.remaining_characters.clone(), done