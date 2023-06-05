from machineBrain.pickQuestion import PickQuestionDNN
from machineBrain.takeAnswer import TakeAnswerBrain
import string 

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from gameEnv import all_possible_characters, question_bank
import nltk
nltk.download('stopwords')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

class MachinePlayer():
    def __init__(self, char_name=None, game_mode=False, debug_mode=False, training_mode=False, asks_randomly=False):
        self.game_mode = game_mode
        self.debug_mode = debug_mode
        self.training_mode = training_mode
        self.given_char_name = self.assign_char(char_name)
        self.take_answer_brain = TakeAnswerBrain()
        self.pick_question_brain = PickQuestionDNN(question_bank, asks_randomly)
        self.question_asked_index = 0
        self.state = self.take_answer_brain.get_state()
        self.done = False

    def reset(self):
        if self.debug_mode:
            self.assign_char("Claire")
        else:
            self.assign_char()
        self.pick_question_brain.reset()
        self.take_answer_brain.reset()

    def assign_char(self, char_name=None):
        if char_name:
            self.given_char_name = char_name
        elif self.game_mode:
            # Don't pick an easy character
            self.given_char_name = random.choice(["Sam", "Bernard", "Richard", "Bill", "Tom"])
        else:
            self.given_char_name = random.choice(list(all_possible_characters.keys()))
        
        if self.debug_mode:
            print("Bot player was assigned "+self.given_char_name)
        return self.given_char_name

    def ask_question(self):
        q_index, question_text = self.pick_question_brain.pick_next_question(self.state)
        self.question_asked_index = q_index
        return q_index, question_text

    def take_question(self, human_question):
        def clean_text(text):
            # Clearn text first
            text = ' '.join([word for word in text.lower().split() if word not in stopwords.words('english') \
                             and word not in string.punctuation])
            return text
        
        # See https://towardsdatascience.com/calculating-string-similarity-in-python-276e18a7d33a
        def cosine_sim_vectors(vec1, vec2):
            vec1 = vec1.reshape(1, -1)
            vec2 = vec2.reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]
    
        # Using basic td-ls cosine to measure similarity for the demo
        # this can be swapped with any more advanced technique
        embeddings_text = question_bank + list(all_possible_characters.keys())
        
        # Add human question to the array of question banks so we vectorize in one go
        embeddings_text.append(human_question)

        ## Use text similarity score to approximate which question was asked
        embeddings = CountVectorizer().fit_transform(list(map(clean_text, embeddings_text))).toarray()

        # Store the similarity of each question to the question
        similarity_scores = [cosine_sim_vectors(embeddings[-1], question) for question in embeddings[:-1]]

        # Find the index of the most similar question.
        answer = "My answer is no"
        max_score = max(similarity_scores)
        if max_score > 0:
            index = similarity_scores.index(max_score)

            if index < len(question_bank):
                # Check if the character has that attribute
                if index < len(all_possible_characters[self.given_char_name]) and all_possible_characters[self.given_char_name][index] == 1:
                    answer = "My answer is yes"
            else: # Human player must have asked about a name
                # We took a short cut up top and appended char names to the question bank instead of separately
                char_index = index-len(question_bank)
                if all_possible_characters[char_index] == self.given_char_name:
                    answer = "Yes, but I get one more question to tie it because you went first."
        return answer

    def take_answer(self, answer: str):
        if answer.lower() in ["yes", "y", "yep", "sure", "my answer is yes", 1]:
            answer_binary = 1
        elif answer.lower() in ["no", "n", "nope", "my answer is no", 0]:
            answer_binary = 0
                    
        next_state, is_done = self.take_answer_brain.take_answer(self.question_asked_index, answer_binary)
        self.state = next_state

        if is_done:
            charIndex = (next_state == 1).nonzero()
            charName = list(all_possible_characters.keys())[charIndex]
            if self.game_mode:
                print("I know, your character is "+charName)
        elif self.pick_question_brain.total_questions_asked > 20:
            # The bot is stuck in a loop and lost
            is_done = True

        self.done = is_done
        return next_state, is_done
