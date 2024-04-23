"""Games or Adversarial Search (Chapter 5)"""

import copy
import itertools
import random
from collections import namedtuple

import numpy as np

#from utils import vector_add
#My Sources used for this assignment
#https://www.youtube.com/watch?v=l-hh51ncgDI
#https://www.youtube.com/watch?v=trKjYdBASyQ
#Chapter 5 of our book 
#
#

GameState = namedtuple('GameState', 'to_move, utility, board, moves')

def gen_state(to_move='X', x_positions=[], o_positions=[], h=3, v=3):
    """Given whose turn it is to move, the positions of X's on the board, the
    positions of O's on the board, and, (optionally) number of rows, columns
    and how many consecutive X's or O's required to win, return the corresponding
    game state"""

    moves = set([(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]) - set(x_positions) - set(o_positions)
    moves = list(moves)
    board = {}
    for pos in x_positions:
        board[pos] = 'X'
    for pos in o_positions:
        board[pos] = 'O'
    return GameState(to_move=to_move, utility=0, board=board, moves=moves)


# ______________________________________________________________________________
# MinMax Search


def minmax(game, state):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    # Using the state we get a return of which players turn it is to move
    player = game.to_move(state)

    # Here we always have the player moving first so start with the Max_value(state)
    def max_value(state):
        # If this is the terminal state then return the utility/value of the state
        if game.terminal_test(state):
            return game.utility(state, player)
        maxValue = -np.inf
        for a in game.actions(state):
            maxValue = max(maxValue, min_value(game.result(state, a)))
        return maxValue

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        maxValue = np.inf
        for a in game.actions(state):
            maxValue = min(maxValue, max_value(game.result(state, a)))
        return maxValue

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a)))


def minmax_cutoff(game, state, depth):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the cutoff depth. At that level use evaluation func."""

    print ("Cutoff depth is:", depth)

    # Using the state we get a return of which players turn it is to move
    player = game.to_move(state)
    

    def max_value(state, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)
        
        
        maxValue = -np.inf 
        #for child in possible actions from state find the max value of the child
        #By calling min_value recursively
        for a in game.actions(state):
            maxValue = max(maxValue, min_value(game.result(state, a), depth - 1))
        return maxValue
    
    def min_value(state, depth):
        if depth == 0 or game.terminal_test(state):
            return game.evaluation_func(state)
        minValue = np.inf
        for a in game.actions(state):
            minValue = min(minValue, max_value(game.result(state, a), depth - 1))
        return minValue
    
    # Body of minmax_cutoff_decision:
    # Returns the action a that maximizes the value of children nodes
    # Final return to send the optimal action back up to the root node
    # this is the action that the algorithm has decided is the best move to make based on its traversal
    return max(game.actions(state), key=lambda a: min_value(game.result(state, a), depth))
     


# ______________________________________________________________________________


def expect_minmax(game, state):
    """
    [Figure 5.11]
    Return the best move for a player after dice are thrown. The game tree
	includes chance nodes along with min and max nodes.
	"""
    player = game.to_move(state)

    def max_value(state):
        v = -np.inf
        for a in game.actions(state):
            v = max(v, chance_node(state, a))
        return v

    def min_value(state):
        v = np.inf
        for a in game.actions(state):
            v = min(v, chance_node(state, a))
        return v

    def chance_node(state, action):
        res_state = game.result(state, action)
        if game.terminal_test(res_state):
            return game.utility(res_state, player)
        sum_chances = 0
        num_chances = len(game.chances(res_state))
        print("chance_node: to be completed by students")
        return 0 

    # Body of expect_minmax:
    return max(game.actions(state), key=lambda a: chance_node(state, a), default=None)


def alpha_beta_search(game, state):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)
    alpha = -np.inf
    beta = np.inf

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            alpha = max(alpha, v)
            #prune
            if alpha >= beta:
                return v
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            beta = min(beta, v)
            #Prune
            if beta <= alpha:
                return v
        return v

    # Body of alpha_beta_search:
    best_action = max(game.actions(state), key = lambda a: min_value(game.result(state, a), alpha, beta))

    return best_action


def alpha_beta_cutoff_search(game, state, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""
    print("alpha_beta_cutoff_search: may be used, if so, must be implemented by students")
    
    player = game.to_move(state)
    alpha = -np.inf
    beta = np.inf

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            alpha = max(alpha, v)
            #prune
            if alpha >= beta:
                return v
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            beta = min(beta, v)
            #Prune
            if beta <= alpha:
                return v
        return v

    # Body of alpha_beta_search:
    best_action = max(game.actions(state), key = lambda a: min_value(game.result(state, a), alpha, beta))

    return best_action
# ______________________________________________________________________________
# Players for Games


def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move


def random_player(game, state):
    """A player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None


def alpha_beta_player(game, state):
    return alpha_beta_search(game, state)

# If game.depth is -1 then minmax_player will be used i.e no depth limiting
# If game.depth is 0 then minmax_player will be used i.e depth limiting
# depth is stored in gdepth
def minmax_player(game,state):
    if( game.d == -1):
        return minmax(game, state)
    # Note changed this from minmax_cutoff( game , state ) to this to take into account the depth
    # could have done game.d inside of other function to get depth as well lol
    return minmax_cutoff(game, state, game.d)


def expect_minmax_player(game, state):
    return expect_minmax(game, state)


# ______________________________________________________________________________
# 


class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def evaluation_func(self, state):
        # Default implementation for Game which will get over written by tic tac toe class
        return 1

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


# Tic tac toe inherits from game 
class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to_move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'.
    depth = -1 means max search tree depth to be used."""

    def __init__(self, h=3, v=3, k=3, d=-1):
        self.h = h
        self.v = v
        self.k = k
        self.depth = d
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        # if state.utillity 0 means its a draw and hence all squares filled up
        # Any thing else means its a win or a draw
        return state.utility != 0 or len(state.moves) == 0
    

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return self.k if player == 'X' else -self.k
        else:
            return 0

    def evaluation_func(self, state):
        """computes value for a player on board after move.
            Likely it is better to conside the board's state from 
            the point of view of both 'X' and 'O' players and then subtract
            the corresponding values before returning."""

        
        """Compute the value for a player on the board after a move."""
        player = state.to_move
        opponent = 'O' if player == 'X' else 'X'
        
        # Check for winning moves
        if state.utility == self.k:
            return float('inf') if player == 'X' else float('-inf')
            
        elif state.utility == -self.k:
            return float('-inf') if player == 'X' else float('inf')
            
        
        # If neither X nor O wins and we are in the middle of the game

        # Algorithm to check if we have a line of two more more X's / O's with none of the other values blocking the line

        ValueX = 0
        ValueO = 0

        # Defining the 4 directions in which a line can be in 
        directions = [(0 , 1), (1, 0), (1, -1), (1, 1)]

        NumForWin = self.k

        for directionX , directionY in directions:
         
            # Go through all the cells in the board:

            for x in range(self.h):
                for y in range(self.v):
                    # Self.k_in_row counts how many times a certain 
                    # If a line of two or more X's is found with zero values of O add + 1 to score of X
                    # If there's a line of two or more 'X's with no 'O's blocking in the direction (dx, dy) from the cell (x, y), increment score_x
                    if self.k_in_row(state.board, (x, y), 'X', (directionX, directionY)) >= self.k and not self.k_in_row(state.board, (x, y), 'O', (directionX, directionY)):
                        ValueX += 10

                    # If there's a line of two or more 'O's with no 'X's blocking in the direction (dx, dy) from the cell (x, y), increment score_o
                    if self.k_in_row(state.board, (x, y), 'O', (directionX, directionY)) >= self.k and not self.k_in_row(state.board, (x, y), 'X', (directionX, directionY)):
                        ValueO += 10

        return ValueX - ValueO

		
    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player.
        This function checks if there are 'k' consecutive pieces of the same player in a row on the game board."""

        # delta_x and delta_y represent the direction of the line to check
        # For example, (1, 0) would represent a horizontal line, (0, 1) a vertical line, and (1, 1) a diagonal line
        (delta_x, delta_y) = delta_x_y

        # x and y = starting position on the board
        x, y = move

        # number of consequitive pieces of a player on board
        n = 0

        # Start at the given position and move in the direction specified by delta_x and delta_y
        # Check if the piece at the current position is the same as the player
        # If it is, increment n and move to the next position
        # Continue this until a piece of a different player or an empty space is encountered
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y

        # Reset the position back to the original move
        x, y = move

        # Now move in the opposite direction specified by delta_x and delta_y and do the same check
        # This is necessary because the line could extend in both directions from the original move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y

        # Because we counted the original move twice (once in each direction), decrement n by 1
        n -= 1

        # If the number of consecutive pieces of the same player in the line is greater than or equal to k, return True
        # Otherwise, return False
        return n >= self.k

    def chances(self, state):
        """Return a list of all possible states."""
        chances = []
        return chances
    
class Gomoku(TicTacToe):
    """Also known as Five in a row."""

    def __init__(self, h=15, v=16, k=5):
        TicTacToe.__init__(self, h, v, k)
