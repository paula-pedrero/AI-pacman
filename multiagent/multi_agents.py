# multi_agents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position() #Pacman position after taking the action
        new_food = successor_game_state.get_food() #food remaining on the board
        new_ghost_states = successor_game_state.get_ghost_states() #the ghosts positions and states
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states] #how many moves each ghost remains scared
        
        "*** YOUR CODE HERE ***"
        # Start with the score of the successor state
        score = successor_game_state.get_score()

        # Get the list of remaining food positions
        food_list = new_food.as_list()

        # Reward Pacman for being close to the nearest food
        if food_list:
            distances = [ 
                manhattan_distance(new_pos, food)
                for food in food_list
            ]

        closest_food = min(distances)
        score += 10.0 / (closest_food + 1)

        # Evaluate the distance to each ghost
        for ghost, scared_time in zip(new_ghost_states, new_scared_times):
            ghost_pos = ghost.get_position()
            distance = manhattan_distance(new_pos, ghost_pos)

            # If the ghost is scared, reward Pacman for approaching it
            if scared_time > 0:
                score += 20.0 / (distance + 1)

            # If the ghost is on Pacman's position, apply a large penalty
            elif distance == 0:
                score -= 1000

            # If a dangerous ghost is nearby, apply a penalty
            elif distance < 3:
                score -= 20.0 / distance

    # Return the final evaluation
        return score
        

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth) 

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #Recursive function that calculates the minimax value
        def minimax(state, agent_index, depth):
            # Stop if the game has ended or we reached depth zero
            if state.is_win() or state.is_lose() or depth==0:
                return self.evaluation_function(state)

            # Get all legal actions for the current agent
            actions=state.get_legal_actions(agent_index)

            # If there are no possible actions, evaluate the state
            if not actions:
                return self.evaluation_function(state)

            # Get the total number of agents (Pacman + ghosts)
            num_agents=state.get_num_agents()

            # Calculate which agent plays next
            next_agent = (agent_index + 1) % num_agents

            # Keep the same depth for the next agent
            next_depth=depth

            # If the next agent is Pacman, all ghosts have finished
            if next_agent==0:
                next_depth-=1
            # If it is Pacman's turn, use MAX
            if agent_index==0:
                # Start with the lowest possible value
                best_value = float("-inf")
                # Explore every possible action
                for action in actions:
                    # Generate the state after this action
                    successor = state.generate_successor(agent_index, action)
                    # Recursively calculate the value of that state
                    value = minimax(successor, next_agent, next_depth)
                    # Keep the highest value
                    best_value = max(best_value, value)
                #Return max value
                return best_value
            # Otherwise, it is a ghost's turn, so use MIN
            else:
                # Start with the highest possible value
                best_value = float("inf")
                # Explore every possible action
                for action in actions:
                    #Generate the state after thus action
                    successor = state.generate_successor(agent_index, action)
                    # Recursively calculate the value of that state
                    value = minimax(successor, next_agent, next_depth)
                    # Keep the lowest value
                    best_value = min(best_value, value)
                #return the minimum value
                return best_value
            # Get all possible actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        # If Pacman has no legal actions, return None
        if not legal_actions:
            return None
        # Start with no selected action
        best_action = None
        # Start with the lowest possible value
        best_value = float("-inf")
        # Get the total number of agents
        num_agents = game_state.get_num_agents()
        # After Pacman moves, the first ghost plays
        next_agent=1
        # Start searching with the configured depth
        next_depth = self.depth
        # Start searching with the configured depth
        for action in legal_actions:
            # Generate the state after Pacman takes this action
            successor = game_state.generate_successor(0,action)
            # Calculate the Minimax value of this state
            value = minimax(successor, next_agent, next_depth)
            #If this action has a higher value , save it
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
                
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        # Initialize alpha and beta
        alpha = float("-inf")
        beta = float("+inf")
        def alphaBetaPrunning(state, agent_index, depth, alpha, beta):
            # Stop if the game has ended or we reached depth zero
            if state.is_win() or state.is_lose() or depth==0:
                return self.evaluation_function(state)

            # Get all legal actions for the current agent
            actions=state.get_legal_actions(agent_index)

            # If there are no possible actions, evaluate the state
            if not actions:
                return self.evaluation_function(state)

            # Get the total number of agents (Pacman + ghosts)
            num_agents=state.get_num_agents()

            # Calculate which agent plays next
            next_agent = (agent_index + 1) % num_agents

            # Keep the same depth for the next agent
            next_depth=depth

            # If the next agent is Pacman, all ghosts have finished
            if next_agent==0:
                next_depth-=1
            # If it is Pacman's turn, use MAX
            if agent_index==0:
                # Start with the lowest possible value
                best_value = float("-inf")
                # Explore every possible action
                for action in actions:
                    # Generate the state after this action
                    successor = state.generate_successor(agent_index, action)
                    # Evaluate the successor recursively
                    value = alphaBetaPrunning(successor, next_agent, next_depth, alpha, beta)
                    # Keep the highest value
                    best_value = max(best_value, value)
                    #Update alpha value
                    alpha = max(best_value, alpha)
                    #Contidion to prune (beta cut)
                    if alpha>beta:
                        return best_value
                #Return max value
                return best_value
            # Otherwise, it is a ghost's turn, so use MIN
            else:
                # Start with the highest possible value
                best_value = float("inf")
                # Explore every possible action
                for action in actions:
                    #Generate the state after thus action
                    successor = state.generate_successor(agent_index, action)
                    # Evaluate the successor recursively
                    value = alphaBetaPrunning(successor, next_agent, next_depth, alpha, beta)
                    # Keep the lowest value
                    best_value = min(best_value, value)
                    #Update of beta
                    beta = min(best_value, beta)
                    #Condition to prune (alpha cut)
                    if beta<alpha:
                        return best_value
                #return the minimum value
                return best_value
                
        # Get all possible actions for Pacman
        legal_actions = game_state.get_legal_actions(0)
        # If Pacman has no legal actions, return None
        if not legal_actions:
            return None
        # Start with no selected action
        best_action = None
        # Start with the lowest possible value
        best_value = float("-inf")
        # Get the total number of agents
        num_agents = game_state.get_num_agents()
        # After Pacman moves, the first ghost plays
        next_agent=1
        # Start searching with the configured depth
        next_depth = self.depth
        # Start searching with the configured depth
        for action in legal_actions:
            # Generate the state after Pacman takes this action
            successor = game_state.generate_successor(0,action)
            # Calculate the value using alpha-beta pruning
            value = alphaBetaPrunning(successor, next_agent, next_depth, alpha, beta)
            #If this action has a higher value , save it
            if value > best_value:
                best_value = value
                best_action = action
            #Update alpha value
            alpha = max(best_value, alpha)
        return best_action
        util.raise_not_defined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()
    


# Abbreviation
better = better_evaluation_function
