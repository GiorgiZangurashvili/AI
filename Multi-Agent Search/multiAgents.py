# multiAgents.py
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


from util import manhattanDistance
from game import Directions
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


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if currentGameState.isWin():
            return float('inf')

        if currentGameState.isLose():
            return -float('inf')

        min_distance_to_food = 1.0
        if newFood.asList():
            min_distance_to_food = min(util.manhattanDistance(newPos, food_pos) for food_pos in newFood.asList()) + 1.0

        min_distance_to_ghost = 1.0
        ghost_weight = 1.8
        if currentGameState.getGhostPositions():
            min_distance_to_ghost = min(util.manhattanDistance(newPos, ghost_pos) for ghost_pos in currentGameState.getGhostPositions()) + 1.0

        return successorGameState.getScore() + 1.0 / min_distance_to_food - ghost_weight / min_distance_to_ghost

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        minimax_val = self.minimax(gameState, self.depth, 0)
        return minimax_val[1]

    def minimax(self, gameState, depth, index):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        if index == 0:
            val = -float('inf')
            action = None
            for legal_action in gameState.getLegalActions(index):
                next_state = gameState.generateSuccessor(index, legal_action)
                minimax_val = self.minimax(next_state, depth, (index + 1) % gameState.getNumAgents())
                if minimax_val[0] > val:
                    val = minimax_val[0]
                    action = legal_action
            return val, action
        else:
            val = float('inf')
            action = None
            for legal_action in gameState.getLegalActions(index):
                next_state = gameState.generateSuccessor(index, legal_action)
                next_index = (index + 1) % gameState.getNumAgents()
                next_depth = depth
                if next_index == 0:
                    next_depth -= 1
                minimax_val = self.minimax(next_state, next_depth, next_index)
                if minimax_val[0] < val:
                    val = minimax_val[0]
                    action = legal_action
            return val, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')
        alpha_beta_val = self.alpha_beta(gameState, self.depth, 0, alpha, beta)
        return alpha_beta_val[1]

    def alpha_beta(self, gameState, depth, index, alpha, beta):
        if depth == 0 or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState), None
        if index == 0:
            val = -float('inf')
            action = None
            for legal_action in gameState.getLegalActions(index):
                next_state = gameState.generateSuccessor(index, legal_action)
                next_index = (index + 1) % gameState.getNumAgents()
                alpha_beta_val = self.alpha_beta(next_state, depth, next_index, alpha, beta)
                if alpha_beta_val[0] > val:
                    val = alpha_beta_val[0]
                    action = legal_action
                if val > beta:
                    return val, action
                alpha = max(alpha, val)
            return val, action
        else:
            val = float('inf')
            action = None
            for legal_action in gameState.getLegalActions(index):
                next_state = gameState.generateSuccessor(index, legal_action)
                next_index = (index + 1) % gameState.getNumAgents()
                next_depth = depth
                if next_index == 0:
                    next_depth -= 1
                alpha_beta_val = self.alpha_beta(next_state, next_depth, next_index, alpha, beta)
                if alpha_beta_val[0] < val:
                    val = alpha_beta_val[0]
                    action = legal_action
                if val < alpha:
                    return val, action
                beta = min(beta, val)
            return val, action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        expectimax_val = self.expectimax(gameState, self.depth, 0)
        return expectimax_val[1]

    def expectimax(self, gameState, depth, index):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None
        if index == 0:
            return self.max_val(gameState, depth, index)
        else:
            return self.exp_val(gameState, depth, index)

    def max_val(self, gameState, depth, index):
        val = -float('inf')
        action = None
        for legal_action in gameState.getLegalActions(index):
            next_state = gameState.generateSuccessor(index, legal_action)
            next_index = (index + 1) % gameState.getNumAgents()
            expectimax_val = self.expectimax(next_state, depth, next_index)
            if expectimax_val[0] > val:
                val = expectimax_val[0]
                action = legal_action
        return val, action

    def exp_val(self, gameState, depth, index):
        val = 0
        probability = 1.0 / len(gameState.getLegalActions(index))
        for legal_action in gameState.getLegalActions(index):
            next_state = gameState.generateSuccessor(index, legal_action)
            next_index = (index + 1) % gameState.getNumAgents()
            next_depth = depth
            if next_index == 0:
                next_depth -= 1
            expectimax_val = self.expectimax(next_state, next_depth, next_index)
            val += probability * expectimax_val[0]
        return val, None

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isLose():
        return -float('inf')

    if currentGameState.isWin():
        return float('inf')

    score = currentGameState.getScore()

    food_near_weight = 20.5
    ghost_near_weight = 27.5
    capsule_near_weight = 300.0

    pacman_pos = currentGameState.getPacmanPosition()

    min_dist_to_ghost = 1.0
    if currentGameState.getGhostPositions():
        min_dist_to_ghost = min([util.manhattanDistance(pacman_pos, ghost_pos) for ghost_pos in currentGameState.getGhostPositions()]) + 1
    min_dist_to_food = 1.0
    if currentGameState.getFood().asList():
        min_dist_to_food = min([util.manhattanDistance(pacman_pos, food_pos) for food_pos in currentGameState.getFood().asList()]) + 1
    min_dist_to_capsule = 1.0
    if currentGameState.getCapsules():
        min_dist_to_capsule = min([util.manhattanDistance(pacman_pos, capsule_pos) for capsule_pos in currentGameState.getCapsules()]) + 1

    score += food_near_weight / min_dist_to_food
    score -= ghost_near_weight / min_dist_to_ghost
    score += capsule_near_weight / min_dist_to_capsule

    return score

# Abbreviation
better = betterEvaluationFunction
