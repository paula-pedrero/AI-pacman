# q_learning_agents.py
# ------------------
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


from game import *
from learning_agents import ReinforcementAgent
from feature_extractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - compute_value_from_q_values
        - compute_action_from_q_values
        - get_q_value
        - get_action
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.get_legal_actions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # Store the Q-values for each state-action pair.
        # Initially, all Q-values are 0.
        self.q_values = util.Counter()

    def get_q_value(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # Return the Q-value associated with this state and action.
        # If it doesn't exist, Counter returns 0 automatically.
        return self.q_values[(state, action)]


    def compute_value_from_q_values(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions available in this state.
        legal_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0
        if not legal_actions:
            return 0.0

        #Initialise the best value using first action
        best_value = self.get_q_value(state, legal_actions[0])
        #Check the remaining actions
        for action in legal_actions[1:]:
            #Get the Q-value of the current action
            q_value = self.get_q_value(state, action)

            #If this action has a higher value, update best_value
            if q_value > best_value:
                best_value = q_value
        #return the higher Q-value
        return best_value

    def compute_action_from_q_values(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get all legal actions available in this state.
        legal_actions = self.get_legal_actions(state)
        #If there are no legal actions, return None
        if not legal_actions:
            return None

        #Find the highest Q-value among all actions
        best_value = self.compute_value_from_q_values(state)

        #Create a list to store all actions with the highest value
        best_actions = []

        #Check every legal action

        for action in legal_actions:
            #Get the Q-value of the current action
            q_value = self.get_q_value(state, action)
            # If it has the highest value, add it to the list
            if q_value == best_value:
                best_actions.append(action)
        # Randomly choose one of the best actions.
        return random.choice(best_actions)



    def get_action(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flip_coin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.get_legal_actions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #If there are no legal actions, return None
        if not legal_actions:
            return None

        #Decide whether to explore or exploit
        if util.flip_coin(self.epsilon):
            #Exploration: choose a random legal action
            action = random.choice(legal_actions)
        else:
            #exploitation: choose the best known action
            action = self.compute_action_from_q_values(state)
        return action

        

    def update(self, state, action, next_state, reward):
        """
          The parent class calls this to observe a
          state = action => next_state and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #Get the current Q-value of the state-action pair
        current_q_value = self.get_q_value(state, action)

        #Get the maximum Q-value available in the next state
        next_value = self.compute_value_from_q_values(next_state)
        #Calculate the new Q-value using the Q-learning formula
        new_q_value = current_q_value + self.alpha * (reward +self.discount *next_value -current_q_value)

        #Save the updated Q-value
        self.q_values[(state, action)]= new_q_value


    def get_policy(self, state):
        return self.compute_action_from_q_values(state)

    def get_value(self, state):
        return self.compute_value_from_q_values(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, num_training=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        num_training - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['num_training'] = num_training
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def get_action(self, state):
        """
        Simply calls the get_action method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.get_action(self, state)
        self.do_action(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite get_q_value
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.feat_extractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()
        

    def update(self, state, action, next_state, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodes_so_far == self.num_training:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
