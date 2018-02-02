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
import random, util, math
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    # Bailey (Ahhyun Ahn) and I pair-programmed only on the first problem, Reflex Agent.
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def gridToList(self, grid):
        """ this returns a list of positions of stuff in stuffGrid (substitute stuff to either food/wall)"""
        gridList =[]
        for i in range(grid.width):
          for j in range(grid.height):
            if grid[i][j] == True:
              # print "grid[i][j]", grid[i][j]
              gridList.append((i,j))
        return gridList

    def getSuccessors(self, currentPos, wallList, grid):
        """ this returns a list of successors from current location """
        x, y = currentPos
        fourDirections = [(x-1, y, Directions.WEST), (x+1, y, Directions.EAST), \
                         (x, y+1, Directions.NORTH), (x, y-1, Directions.SOUTH)]
        return filter(lambda successorPos: self.isValidPos(successorPos[0], successorPos[1], wallList, grid), fourDirections)

    
    def isValidPos(self, x, y, wallList, grid):
        """ returns true if the position is valid in grid, and not a wall """
        if (x, y) not in wallList:
          return x > 0 and x < grid.width and y > 0 and y < grid.height

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
        newGhostPoses = [ghostState.getPosition() for ghostState in newGhostStates]

        numberOfFood = successorGameState.getNumFood()

        if numberOfFood == 0:
          return 1000

        closestFoodDistance = min([manhattanDistance(newPos, foodPos) for foodPos in self.gridToList(successorGameState.getFood())])

        stopPoint = 0
        if action == Directions.STOP:
          stopPoint = 20

        closestGhostDistance = min([manhattanDistance(newPos, ghostPosition) for ghostPosition in successorGameState.getGhostPositions()])

        if newPos in newGhostPoses:
          return -1000

        return - closestFoodDistance + 3*math.sqrt(closestGhostDistance) + 1.3*successorGameState.getScore()

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
        value = float("-inf")
        
        #self.index = 0 -> pacman 
        optimalList = [value, None]
        for action in gameState.getLegalActions(self.index):
          sucessorState = gameState.generateSuccessor(self.index, action)
          tempValue = self.value(sucessorState, self.depth, 1)
          if tempValue > value:
             value = tempValue
             optimalList = [value, action]
        # return optimal action     
        return optimalList[1]


    def value(self, state, depth, agentIndex):
      # check if we reached the terminal state or the bottom layer
      if state.isWin() or state.isLose() or (depth < 1):
        return self.evaluationFunction(state)

      # It's packman/maximizer
      if agentIndex == 0:
        #print "agentIndex: ", agentIndex
        return self.max_value(state, depth,agentIndex)

      # It's a ghost/minimizer
      else:
        return self.min_value(state, depth, agentIndex)

    def max_value(self, state, depth, agentIndex):
      value = float("-inf")

      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # increment agentIndex by 1 to go to the next layer/agent
        tempValue = self.value(successorState, depth, agentIndex+1)
        value = max(tempValue, value)
      return value

    def min_value(self, state, depth, agentIndex):
      value = float("inf")
      # numGhosts = numAgents-1 because we want to get the number of agents excluding pacman
      numGhosts = state.getNumAgents()-1

      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # Check if we have evaluated all the ghosts' sucessor states
        if agentIndex >= numGhosts:
          # We're done evaluating successor states for the minimizers 
          tempValue = self.value(successorState, depth-1, 0)
        else:
          # increment agentIndex by 1 to get to the next ghost
          tempValue = self.value(successorState, depth, agentIndex+1)

        value = min(tempValue, value)
      return value


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        
        #self.index = 0 -> pacman 
        optimalList = [value, None]
        for action in gameState.getLegalActions(self.index):
          sucessorState = gameState.generateSuccessor(self.index, action)
          tempValue = self.value(sucessorState, self.depth, 1, alpha, beta)

          if max(tempValue,value) > alpha:
            alpha = max(tempValue,value)

          if tempValue > value:
             value = tempValue
             optimalList = [value, action]
        # return optimal action 
        return optimalList[1]
    

    def value(self, state, depth, agentIndex, alpha, beta):
      # check if we reached the terminal state or the bottom layer
      if state.isWin() or state.isLose() or (depth < 1):
        return self.evaluationFunction(state)

      # It's packman/maximizer
      if agentIndex == 0:
        return self.max_value(state, depth, agentIndex, alpha, beta)

      # It's a ghost/minimizer
      else:
        return self.min_value(state, depth, agentIndex, alpha, beta)

    def max_value(self, state, depth, agentIndex, alpha, beta):
      value = float("-inf")

      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # increment agentIndex by 1 to go to the next layer/agent
        tempValue = self.value(successorState, depth, agentIndex+1, alpha, beta)
        
        if max(tempValue, value) > beta:
          value = max(tempValue, value)
          return value

        else:
          value = max(tempValue, value)
          #alpha -> maximizer's best option to root
          alpha = max(alpha, value)
        
      return value

    def min_value(self, state, depth, agentIndex, alpha, beta):
      value = float("inf")
      # numGhosts = numAgents-1 because we want to get the number of agents excluding pacman
      numGhosts = state.getNumAgents()-1

      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # numAgents-1 because we want to get the number of agents excluding pacman
        # Check if we have evaluated all the ghosts' sucessor states
        if agentIndex >= numGhosts:
          # We're done evaluating sucessor states for the minimizers 
          tempValue = self.value(successorState, depth-1, 0, alpha, beta)
        else:
          # increment agentIndex by 1 to get to the next ghost
          tempValue = self.value(successorState, depth, agentIndex+1, alpha, beta)

        if min(tempValue, value) < alpha:
          value = min(tempValue, value)
          return value

        else:
          value = min(tempValue, value)
          #beta -> minimizer's best option to root
          beta = min(beta, value)

      return value

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
        value = float("-inf")
        
        #self.index = 0 -> pacman 
        optimalList = [value, None]
        for action in gameState.getLegalActions(self.index):
          sucessorState = gameState.generateSuccessor(self.index, action)
          tempValue = self.value(sucessorState, self.depth, 1)
          if tempValue > value:
             value = tempValue
             optimalList = [value, action]
        # return optimal action     
        return optimalList[1]
        
    def value(self, state, depth, agentIndex):
      """
         This acts the same as minimax, except it calls the helper function exp_value
         in the place of where it used to call min_value. exp_value() returns the average
         of the values of the successor(children) states
      """
      # check if we reached the terminal state or the bottom layer
      if state.isWin() or state.isLose() or (depth < 1):
        return self.evaluationFunction(state)

      # It's packman/maximizer
      if agentIndex == 0:
        #print "agentIndex: ", agentIndex
        return self.max_value(state, depth,agentIndex)

      # It's a ghost
      else:
        return self.exp_value(state, depth, agentIndex)

    def max_value(self, state, depth, agentIndex):
      value = float("-inf")

      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # increment agentIndex by 1 to go to the next layer/agent
        tempValue = self.value(successorState, depth, agentIndex+1)
        value = max(tempValue, value)
      return value

    def exp_value(self, state, depth, agentIndex):
      value = 0
      # numGhosts = numAgents-1 because we want to get the number of agents excluding pacman
      numGhosts = state.getNumAgents()-1

      valueList = []
      # Store the values of all the successor states 
      for action in state.getLegalActions(agentIndex):
        successorState = state.generateSuccessor(agentIndex,action)
        # Check if we have evaluated all the ghosts' sucessor states
        if agentIndex >= numGhosts:
          # We're done evaluating successor states for the ghosts
          valueList += [self.value(successorState, depth-1, 0)]
        else:
          # increment agentIndex by 1 to get to the next ghost
          valueList += [self.value(successorState, depth, agentIndex+1)]

      # We take the average of the values of the successor states because the ghosts
      # are supposed to be acting uniformly at random
      value = sum(valueList)/len(valueList)

      return value


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: This evaluation function is not that different from my code
      for the reflex agent, except for a few things such as that it evaluates 
      states, rather than actions. 
      It rewards pacman for getting close to food, by taking the reciprocal of 
      the manhattan distance to the closest food into consideration, then subtracting
      the manhattan distance to the closest ghost. Lastly, pacman gets rewarded as 
      the current game score increases. It does NOT have any effect on this evaluation
      function when a ghost is "scared". 
    """
    score = 0

    if currentGameState.isWin():
      score += 10000000
    elif currentGameState.isLose():
      score -= 10000000

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = gridToList(currentGameState.getFood())

    closestFoodDistance = distanceToClosestFood(pacmanPos, foodList)   

    ghostPoses = currentGameState.getGhostPositions()
    closestGhostDistance = distanceToClosestGhost(pacmanPos, ghostPoses)

    score = (10/closestFoodDistance) - closestGhostDistance*10 + currentGameState.getScore()
    return score

def distanceToClosestFood(pacmanPos, foodList):
    foodDistance = 0
    manhattanToFood = [manhattanDistance(pacmanPos, foodPos) for foodPos in foodList]
    if len(manhattanToFood) != 0:
      foodDistance = min(manhattanToFood)
    else:
      foodDistance = 10
    return foodDistance

def distanceToClosestGhost(pacmanPos, ghostPoses):
  ghostDistance = 0

  for ghostPos in ghostPoses:
    manhattanToGhost = [manhattanDistance(pacmanPos, ghostPos)]
    if len(manhattanToGhost) != 0:
      ghostDistance = min(manhattanToGhost)
    else:
      ghostDistance = 0
  return ghostDistance


def gridToList(grid):
    """ this returns a list of positions of stuff in stuffGrid (substitute stuff to either food/wall)"""
    gridList =[]
    for i in range(grid.width):
      for j in range(grid.height):
        if grid[i][j] == True:
          gridList.append((i,j))
    return gridList


    
# Abbreviation
better = betterEvaluationFunction