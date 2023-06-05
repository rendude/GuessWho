from humanPlayer import HumanPlayer
from machinePlayer import MachinePlayer

machine = MachinePlayer(game_mode=True)
human = HumanPlayer()

print(machine)

print("\n\nPlease pick a character but don't tell me who it is. Ready?\n")
start = input("1. Yes, bring it on. 2. No, take me to mommy:  ")
if start != 2:
    print("\n\nI'll let you go first\n")

    game_ongoing = True
    while game_ongoing:
        human_ques = human.ask_question()
        machine_answer = machine.take_question(human_ques)
        print(f"Machine: {machine_answer}\n\n")
        qindex, machine_question_text = machine.ask_question()
        print(f"Machine: {machine_question_text}")
        human_answer = input("Your choice, Yes, No: ")
        next_state, done = machine.take_answer(human_answer)
        if done:
            game_ongoing = False