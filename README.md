
CS221 Extra Credit Project

## Intro
Machine learns optimal strategy via self playing Guess Who game. Result is compared against a player that plays randomly and a player that uses an expert gamer's strategy.

## Setup

1. Make you sure you have python 3.7+. If not, `brew install python` or `brew update` and `brew upgrade python3`.

2. Make sure you have miniconda

3. Download Pytorch via `conda install pytorch torchvision -c pytorch`
  If you receive conda not found and you are using zsh, add `source ~/.bash_profile` to the
  top of your zsh file. Then type `source ~/.zshrc` in the terminal. That should make conda work.

4. `pip3 install -r reqs.txt`

5. `python3 play.py` to play the bot. Please don't misspell Yes, No answers because I don't have that scenario coded

6. `python3 machineTrainer.py` to see training in action