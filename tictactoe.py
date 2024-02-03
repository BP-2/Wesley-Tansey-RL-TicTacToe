#!/usr/bin/env python
# -*- coding: utf8; mode: python -*-
"""
Reference implementation of the Tic-Tac-Toe value function learning agent described in Chapter 1 of "Reinforcement Learning: An Introduction" by Sutton and Barto.

The agent contains a lookup table that maps states to values,
where initial values are 1 for a win, 0 for a draw or loss, and 0.5 otherwise.
At every move, the agent chooses either the maximum-value move (greedy) or,
with some probability epsilon, a random move (exploratory); by default epsilon=0.1.
The agent updates its value function (the lookup table) after every greedy move, following the equation:

    V(s) <- V(s) + alpha * [ V(s') - V(s) ]

This particular implementation addresses the question posed in Exercise 1.1:

    What would happen if the RL agent taught itself via self-play?

The result is that the agent learns only how to maximize its own potential payoff, without consideration
for whether it is playing to a win or a draw. Even more to the point, the agent learns a myopic strategy
where it basically has a single path that it wants to take to reach a winning state. If the path is blocked
by the opponent, the values will then usually all become 0.5 and the player is effectively moving randomly.

## License
- Created by Wesley Tansey, 1/21/2013
- Online: https://github.com/Naereen/Wesley-Tansey-RL-TicTacToe
- Code released under the [MIT license](http://mit-license.org).
"""

from __future__ import print_function  # Python 2/3 compatibility !
import random
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
import wandb

# States as integer : manual coding
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3

BOARD_FORMAT = """----------------------------
| {0} | {1} | {2} |
|--------------------------|
| {3} | {4} | {5} |
|--------------------------|
| {6} | {7} | {8} |
----------------------------"""
NAMES = [' ', 'X', 'O']


def printboard(state):
    """ Print the board from the internal state."""
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    """ An empty 3x3 state."""
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    """ Check if the state is gameover or not."""
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    """ Count who should play."""
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    """ Enumerate the different states from a state."""
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = idx // 3
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
    """ A RL agent abstraction."""

    def __init__(self, player, verbose=False, lossval=0, learning=True):
        """ Create a RL agent."""
        self.values = {}
        self.player = player
        self.verbose = verbose
        self.lossval = lossval
        self.learning = learning
        self.epsilon = 0.1
        self.alpha = 0.99
        self.prevstate = None
        self.prevscore = 0
        self.count = 0
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        """ Backup and reset self.prevstate and self.prevscore."""
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0

    def action(self, state):
        """ Play an action (epsilon-drunk policy between random and greedy)."""
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        """ Random policy !"""
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        """ Naive implementation of the greedy policy."""
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
                    if self.verbose:
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def backup(self, nextval):
        """ Backup the next value."""
        if self.prevstate is not None and self.learning:
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)

    def lookup(self, state):
        """ Lookup a state."""
        key = self.statetuple(state)
        if key not in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        """ Add a state."""
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)

    def winnerval(self, winner):
        """ Return the value of the winner (0, .5, 1, or self.lossval)."""
        if winner == self.player:
            #wandb.log({"result": "win", "player": "AI", "probability": 1})
            wandb.log({"training-result": "Win", "probability": 1})
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            #wandb.log({"training-result": "Draw", "probability": 1}, commit=False, media={"training-result": "text"})

            #wandb.log({"result": "Draw", "player": "AI", "probability": 1})
            return 0
        else:
            #wandb.log({"training-result": "Lose", "probability": 1}, commit=False, media={"training-result": "text"})
            #wandb.log({"result": "loss", "player": "AI", "probability": 1})
            return self.lossval

    def printvalues(self):
        """ Print the current internal values."""
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        """ Return a tuple of tuple for the current state."""
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        """ Print if verbose."""
        if self.verbose:
            print(s)


class Human(object):
    """ An interactive player. """
    def __init__(self, player):
        """ Create an interactive player."""
        self.player = player

    def action(self, state):
        """ Ask (with input(...)) the user to play."""
        printboard(state)
        action = str(input('Your move? '))
        return (int(action.split(',')[0]), int(action.split(',')[1]))

    def episode_over(self, winner):
        """ Check if you win."""
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))


def play(agent1, agent2):
    """ Play once."""
    state = emptystate()
    for i in range(9):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


def measure_performance_vs_random(agent1, agent2):
    """ A naive way to measure performance of two agents vs random."""
    epsilon1 = agent1.epsilon
    epsilon2 = agent2.epsilon
    agent1.epsilon = 0
    agent2.epsilon = 0
    agent1.learning = False
    agent2.learning = False
    r1 = Agent(1)
    r2 = Agent(2)
    r1.epsilon = 1
    r2.epsilon = 1
    probs = [0, 0, 0, 0, 0, 0]
    games = 100
    for i in range(games):
        winner = play(agent1, r2)
        if winner == PLAYER_X:
            probs[0] += 1.0 / games
            wandb.log({"result": "Win", "player": "USER", "probability": probs[0]})
        elif winner == PLAYER_O:
            probs[1] += 1.0 / games
            wandb.log({"result": "win", "player": "AI", "probability": probs[1]})
        else:
            probs[2] += 1.0 / games
    for i in range(games):
        winner = play(r1, agent2)
        if winner == PLAYER_O:
            probs[3] += 1.0 / games
        elif winner == PLAYER_X:
            probs[4] += 1.0 / games
        else:
            probs[5] += 1.0 / games
    agent1.epsilon = epsilon1
    agent2.epsilon = epsilon2
    agent1.learning = True
    agent2.learning = True
    return probs


def measure_performance_vs_each_other(agent1, agent2):
    """ A naive way to measure performance of two agents vs each other."""
    probs = [0, 0, 0]
    games = 100
    for i in range(games):
        winner = play(agent1, agent2)
        if winner == PLAYER_X:
            probs[0] += 1.0 / games
        elif winner == PLAYER_O:
            probs[1] += 1.0 / games
        else:
            probs[2] += 1.0 / games
    return probs


def main():
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)
    r1 = Agent(1, learning=False)
    r2 = Agent(2, learning=False)
    r1.epsilon = 1
    r2.epsilon = 1

    # Initialize Weights and Biases
    wandb.init(project="TIC-TAC-TOE", config={"epsilon": p1.epsilon, "alpha": p1.alpha, "lossval": p1.lossval})

    # Number of episodes
    num_episodes = 100

    for episode in range(num_episodes):
        # Self-play for training
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

        if episode % 10 == 0:
        # Measure performance against a random agent every 10 episodes
            perf_vs_random = measure_performance_vs_random(p1, r2)
            #wandb.log({"P1 vs Random": perf_vs_random, "episode": episode})

            # Measure performance against each other every 10 episodes
            perf_vs_each_other = measure_performance_vs_each_other(p1, p2)
            #wandb.log({"P1 vs P2": perf_vs_each_other, "episode": episode})

    # Visualize the learned values after training
    p1.printvalues()

    while True:
        p2.verbose = True
        p1 = Human(1)
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)


if __name__ == "__main__":
    main()