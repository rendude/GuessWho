from machinePlayer import MachinePlayer
from gameEnv import question_bank, oracle_question_sequence, all_possible_characters, GAME_MODE


class OraclePlayer(MachinePlayer):
    def __init__(self):
        super().__init__()
        self.total_questions_asked = 0
        self.prev_ques_text = None
        self.answer_received = None
        self.game_status = 0

    def reset(self):
        super().reset()
        self.total_questions_asked = 0
        self.prev_ques_text = None
        self.answer_received = None

    def ask_question(self):
        if self.total_questions_asked == 0:
            question_text = "Do they have a big mouth?"
        else:
            question_text = oracle_question_sequence[self.prev_ques_text][self.answer_received]

        q_index = question_bank.index(question_text)
        self.total_questions_asked += 1
        self.prev_ques_text = question_text
        self.question_asked_index=q_index
        return q_index, question_text
    
    def take_answer(self, answer: str):
        if answer.lower() in ["yes", "y", "yep", "sure", "my answer is yes", 1]:
            answer_binary = 1
            self.answer_received = 'yes'
        elif answer.lower() in ["no", "n", "nope", "my answer is no", 0]:
            answer_binary = 0
            self.answer_received = "no"

        next_state, is_done = self.take_answer_brain.take_answer(self.question_asked_index, answer_binary)
        self.state = next_state

        if is_done:
            charIndex = (next_state == 1).nonzero()
            charName = list(all_possible_characters.keys())[charIndex]
            print("I know, your character is "+charName)
            self.game_status = 1
        elif self.pick_question_brain.total_questions_asked > 20:
            # The bot is stuck in a loop and lost
            self.game_status = -1

        return next_state

    def update_weights(self, current_states, actions, next_states, game_status):
        return