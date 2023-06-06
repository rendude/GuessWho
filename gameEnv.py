GAME_MODE = False
DEBUG_MODE = True
TRAINING = True
DEBUG_CHAR = "George"


unsorted_characters = {
    "Alex": [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Alfred": [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    "Anita": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    "Anne": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    "Bernard": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    "Bill": [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    "Charles": [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Claire": [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    "David": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Eric": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "Frans": [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "George": [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    "Herman": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Joe": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Maria": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    "Max": [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Paul": [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Peter": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    "Philip": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    "Richard": [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Robert": [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    "Sam": [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Susan": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    "Tom": [1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

all_possible_characters = dict(sorted(unsorted_characters.items()))


# 27 real questions and 11 fake, 38 total
question_bank = [
    "Are they a man / male?",
    "Are they bald?",
    "Do they have hair?",
    "Do they have a mustache?",
    "Do they have a beard?",
    "Do they have a hat?",
    "Do they have glasses?",
    "Do they have rosy cheeks?",
    "Do they have black hair?",
    "Do they have red hair?",
    "Do they have blonde hair / Are they blond?",
    "Do they have brown hair?",
    "Do they have white hair?",
    "Do they have brown eyes?",
    "Do they have blue eyes?",
    "Do they have a big nose?",
    "Is their hair parted?",
    "Do they have curly hair?",
    "Is there stuff on their hair?",
    "Do they have long hair?",
    "Do they have a big mouth?",
    "Do they have red cheeks?",
    "Are they sad?",
    "Do they have facial hair?",
    "Do they have earrings?",
    "Are they a girl / female?",
    "Are they old?",
    "Who are they?",
    "What do they do?",
    "Do they have a weird smile?",
    "Do they look to the left?",
    "Do they look like a nice person?",
    "Are they attractive?",
    "Are they friendly-looking?",
    "Are they strong?",
    "Are they kind-looking?",
    "Do they seem rich?",
    "Do they look like a cat?"
]

oracle_question_sequence = {
    "Do they have a big mouth?": {
        'yes': "Do they have black hair?", 
        'no': "Do they have curly hair?",
    },

    "Do they have black hair?": {
        'yes': "Do they have a mustache?", 
        'no': "Is their hair parted?",
    },

    "Do they have a mustache?": {
    },

    "Is their hair parted?": {
        'yes': "Do they have white hair?", 
        'no': "Do they have a beard?",
    },

    "Do they have white hair?": {
        'yes': "Do they have a big nose?", 
        'no': "Do they have blue eyes?",
    },

    "Do they have a beard?": { 
        'no': "Do they have blonde hair / Are they blond?",
    },

    "Do they have curly hair?": { 
        'yes': "Do they have red hair?",
        'no': "Do they have long hair?"
    },

    "Do they have red hair?": { 
        'yes': "Are they bald?",
        'no': "Do they have earrings?"
    },

    "Do they have earrings?": { 
    },

    "Do they have blue eyes?": { 
    },

    "Do they have a big nose?": { 
    },

    "Do they have long hair?": { 
        'yes': "Do they have blonde hair / Are they blond?",
        'no': "Are they bald?"
    },

    "Do they have blonde hair / Are they blond?": { 
        'no': "Do they have blue eyes?"
    },

    "Are they bald?": { 
        'yes': "Do they have glasses?",
        'no': "Do they have a hat?",
    },

    "Do they have glasses?": {
        'yes': "Do they have blue eyes?",
        'no': "Do they have red cheeks?"
    },

    "Do they have a hat?": {
        'yes': "Are they sad?",
    },
}