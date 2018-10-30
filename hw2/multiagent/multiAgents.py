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
        from searchAgents import mazeDistance
        DistEvalFunc = mazeDistance
        DistEvalFunc = util.manhattanDistance

        curPos      = currentGameState.getPacmanPosition()   # pacman
        curCapsules = currentGameState.getCapsules()         # capsules positions
        curFood     = currentGameState.getFood()             # food grid
        curGhosts   = currentGameState.getGhostPositions()   # ghosts positions

        curScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

        newGhosts   = curGhosts                              # they are actually the same

        # weights
        avoid_ghost_weight = 100
        chase_ghost_weight = 10
        get_to_food_weight = 1
        get_to_capsules_weight = 5
        FinalScore = 0

        # AVOID GHOSTS
        # get the index of the ghost to avoid and collect the rest to chase_index
        avoid_ghost_score = None
        avoid_ghost_index = list()
        chase_ghost_index = list()
        ghost_to_avoid    = set()

        for i, times in enumerate(curScaredTimes):
            if times == 0: avoid_ghost_index.append(i)
            else: chase_ghost_index.append(i)

        newMinGhostDist = 99999

        for index in avoid_ghost_index:
            pos     = curGhosts[index]
            # newDist = DistEvalFunc(newPos, (int(pos[0]), int(pos[1])), currentGameState)
            newDist = DistEvalFunc(newPos, (int(pos[0]), int(pos[1])))
            if newDist <= 1: avoid_ghost_score = -1
            ghost_to_avoid.add((int(pos[0]), int(pos[1])))

        if avoid_ghost_score is None: avoid_ghost_score = 1

        FinalScore += avoid_ghost_score*avoid_ghost_weight
        
        # MOVE TOWARD NEAREST FOOD, BUT AVOID THE FOOD WITH THE GHOST ON IT
        curMinFoodDist = 99999
        newMinFoodDist = 99999
        
        for f in curFood.asList():
            if f not in ghost_to_avoid:
                # curMinFoodDist = min(curMinFoodDist, DistEvalFunc(curPos, f, currentGameState))
                curMinFoodDist = min(curMinFoodDist, DistEvalFunc(curPos, f))
                # newMinFoodDist = min(newMinFoodDist, DistEvalFunc(newPos, f, currentGameState))
                newMinFoodDist = min(newMinFoodDist, DistEvalFunc(newPos, f))

        if newMinFoodDist < curMinFoodDist: get_to_food_score = 1
        elif newMinFoodDist == curMinFoodDist: get_to_food_score = 0
        else: get_to_food_score = -1

        FinalScore += get_to_food_score*get_to_food_weight

        # CHASE THE NEAREST CHASEABLE GHOST
        chase_ghost_score = 0
        newMinChaseDist = 99999
        curMinChaseDist = 99999

        for index in chase_ghost_index:
            pos = curGhosts[index]
            # newMinChaseDist = min(newMinChaseDist, DistEvalFunc(newPos, (int(pos[0]), int(pos[1])), currentGameState))
            newMinChaseDist = min(newMinChaseDist, DistEvalFunc(newPos, (int(pos[0]), int(pos[1]))))
            # curMinChaseDist = min(curMinChaseDist, DistEvalFunc(curPos, (int(pos[0]), int(pos[1])), currentGameState))
            curMinChaseDist = min(curMinChaseDist, DistEvalFunc(curPos, (int(pos[0]), int(pos[1]))))

        if newMinChaseDist < curMinChaseDist: chase_ghost_score = 1
        elif newMinChaseDist == curMinChaseDist: chase_ghost_score = 0
        else: chase_ghost_score = -1

        FinalScore += chase_ghost_weight*chase_ghost_score

        # TARGET THE CAPSULES FIRST IF NONE OF THE GHOSTS ARE SCARED
        get_to_capsules_score = 0
        find_capsules = True

        for times in curScaredTimes:
            if times == 0: continue
            find_capsules = False
            break

        if find_capsules:

            curMinCapsuleDist = 99999
            newMinCapsuleDist = 99999

            for cap in curCapsules:
                # curMinCapsuleDist = min(curMinCapsuleDist, DistEvalFunc(curPos, cap, currentGameState))
                curMinCapsuleDist = min(curMinCapsuleDist, DistEvalFunc(curPos, cap))
                # newMinCapsuleDist = min(newMinCapsuleDist, DistEvalFunc(newPos, cap, currentGameState))
                newMinCapsuleDist = min(newMinCapsuleDist, DistEvalFunc(newPos, cap))

            if newMinCapsuleDist < curMinCapsuleDist: get_to_capsules_score = 1
            elif newMinCapsuleDist == curMinCapsuleDist: get_to_capsules_score = 0
            else: get_to_capsules_score = -1

        return FinalScore + get_to_capsules_weight*get_to_capsules_score
        return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0

        def MaxAgent(state, depth): # this is for pacman
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(PACMAN)
            BestAction = Directions.STOP # default is stop
            BestScore  = -99999
            ScoreActionPairs = list()
            for action in ACTIONS:
                # pacman takes its turn; ghosts' turn next
                # ghost index starts from 1
                score = MinAgent(state.generateSuccessor(PACMAN, action), 1, depth)
                ScoreActionPairs.append((score, action))
                BestScore = max(BestScore, score)

            # random pick one best action
            BestActions = [pair[1] for pair in ScoreActionPairs if pair[0] == BestScore]

            # remove 'STOP' if other option exists!!
            if len(BestActions) > 1:
                try: BestActions.remove(Directions.STOP)
                except: pass
            BestAction  = random.choice(BestActions)

            if depth == 0: return BestAction
            else: return BestScore

        def MinAgent(state, ghost, depth): # this is for ghosts
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(ghost)
            BestAction = Directions.STOP
            BestScore  = 99999
            for action in ACTIONS:
                if ghost == state.getNumAgents()-1: # the last ghost moves, next step will be taken by pacman
                    if depth == self.depth - 1: score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else: score = MaxAgent(state.generateSuccessor(ghost, action), depth+1)
                else: score = MinAgent(state.generateSuccessor(ghost, action), ghost+1, depth)
                BestScore = min(BestScore, score)
            return BestScore
        
        return MaxAgent(gameState, 0)
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        PACMAN = 0

        def MaxAgent(state, depth, alpha, beta): # this is for pacman
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(PACMAN)
            BestAction = Directions.STOP # default is stop
            BestScore  = -99999
            ScoreActionPairs = list()
            for action in ACTIONS:
                # pacman takes its turn; ghosts' turn next
                # ghost index starts from 1
                score = MinAgent(state.generateSuccessor(PACMAN, action), 1, depth, alpha, beta)
                ScoreActionPairs.append((score, action))
                BestScore = max(BestScore, score)

                # pruning
                if BestScore >= beta: return BestScore
                alpha = max(alpha, BestScore)

            # random pick one best action
            BestActions = [pair[1] for pair in ScoreActionPairs if pair[0] == BestScore]

            # remove 'STOP' if other option exists!!
            if len(BestActions) > 1:
                try: BestActions.remove(Directions.STOP)
                except: pass
            BestAction = random.choice(BestActions)

            if depth == 0: return BestAction
            else: return BestScore

        def MinAgent(state, ghost, depth, alpha, beta): # this is for ghosts
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(ghost)
            BestAction = Directions.STOP
            BestScore  = 99999
            for action in ACTIONS:
                if ghost == state.getNumAgents()-1: # the last ghost moves, next step will be taken by pacman
                    if depth == self.depth - 1: score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else: score = MaxAgent(state.generateSuccessor(ghost, action), depth+1, alpha, beta)
                else: score = MinAgent(state.generateSuccessor(ghost, action), ghost+1, depth, alpha, beta)
                BestScore = min(BestScore, score)
                # pruning
                if BestScore <= alpha: return BestScore
                beta = min(beta, BestScore)
            return BestScore
        
        return MaxAgent(gameState, 0, float('-inf'), float('inf'))
        util.raiseNotDefined()

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
        PACMAN = 0

        def MaxAgent(state, depth, alpha, beta): # this is for pacman
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(PACMAN)
            BestAction = Directions.STOP # default is stop
            BestScore  = -99999
            ScoreActionPairs = list()
            for action in ACTIONS:
                # pacman takes its turn; ghosts' turn next
                # ghost index starts from 1
                score = ChanceAgent(state.generateSuccessor(PACMAN, action), 1, depth, alpha, beta)
                ScoreActionPairs.append((score, action))
                BestScore = max(BestScore, score)

                # pruning
                if BestScore >= beta: return BestScore
                alpha = max(alpha, BestScore)

            # random pick one best action if such action exists
            if BestScore != -99999: BestActions = [pair[1] for pair in ScoreActionPairs if pair[0] == BestScore]
            else: BestActions = [BestAction]

            # remove 'STOP' if other option exists!!
            if len(BestActions) > 1:
                try: BestActions.remove(Directions.STOP)
                except: pass
            BestAction = random.choice(BestActions)

            if depth == 0: return BestAction
            else: return BestScore

        def ChanceAgent(state, ghost, depth, alpha, beta): # this is for ghosts
            if state.isWin() or state.isLose(): return self.evaluationFunction(state)
            ACTIONS    = state.getLegalActions(ghost)
            BestAction = Directions.STOP
            AveScore   = 0
            for count, action in enumerate(ACTIONS):
                if ghost == state.getNumAgents()-1: # the last ghost moves, next step will be taken by pacman
                    if depth == self.depth - 1: score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else: score = MaxAgent(state.generateSuccessor(ghost, action), depth+1, alpha, beta)
                else: score = ChanceAgent(state.generateSuccessor(ghost, action), ghost+1, depth, alpha, beta)
                beta = min(beta, score)
                AveScore += score
            return float(AveScore) / float(count+1)
        
        return MaxAgent(gameState, 0, float('-inf'), float('inf'))
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      
      1. EAT THE NEAREST FOOD
         
         In order to achieve this, not only the nearest distance to the food needs
         to be considered, the total food left on the map is also required to calculate
         the score for this term.

         The score will be eavulated as follow:

            (MaxFoodNum - curFoodNum) - MinFoodDist/MaxFoodDist
 
            where MaxFoodNum  = The estimated maximum number of foods that can appear on the map (number of grids)
                  curFoodNum  = The current number of food left on the map
                  MaxFoodDist = The estimated maximum distance between pacman and a food (number of grids)
                  MinFoodDist = The current distance between pacman and a nearest food

        The weight was set to 1
        The score will lie in [0, number of grids]

      2. AVOID THE GHOSTS

         Avoid the game state that any of the ghosts are 1 grid apart from pacman even if pacman will get a
         food or capsule (earn another 1 point for food or another 50 point for capsule) after going into this game state.

         The score will be evaluated as follow:

            AvoidGhostScore     if any of the ghosts are 1 grid apart from pacman
            -AvoidGhostScore    otherwise

            where it will be safer to set AvoidGhostScore > 50
            i.e. Set the wieght > 50

            The weight was set to 100 in the end

      3. TARGET THE NEARST CAPSULE IF NO GHOSTS ARE SCARED
      4. CHASE THE GHOSTS AFTER EATING A CAPSULE

         3 and 4 are combined
         The implementation is similar to 1.

         First:   ChaseGhostCost  = (number of scared ghost-1 + minimum distance to a scared ghost/MaxFoodDist)
         Second:  ChaseGhostCost /= Total number of ghost -> normalize to (0, 1)
         Finally: score = (MaxCapsuleNum - curCapsuleNum) - MinCapsuleDist/MaxFoodDist*A - ChaseGhostCost*B
                          where A and B are the weights

         To make chasing ghosts prior to finding capsules after a capsule is already eaten, simply set A<B given
         the constraint A,B>0, A+B=1, but it'll be safer to make A << b
         i.e. don't set A = 0.4, B = 0.6
         The combination (A, B) = (0.1, 0.9) works well in the end.

         To make finding capsules prior to finding foods, simply set the weight of this term high.
         After testing, setting the weight 50 works fine.

    """
    "*** YOUR CODE HERE ***"

    # avoid entering the losing state
    # this is essential
    if currentGameState.isLose(): return -99999
    
    from searchAgents import mazeDistance

    curPos      = currentGameState.getPacmanPosition()   # pacman
    curCapsules = currentGameState.getCapsules()         # capsules positions
    curFood     = currentGameState.getFood()             # food grid
    curGhosts   = currentGameState.getGhostPositions()   # ghosts positions

    curScaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    MAX_DIST_OF_MAP = curFood.width*curFood.height

    FinalScore            = 0
    FoodScoreWeight       = 1
    CombinedTermWeight    = 50
    AvoidGhostScoreWeight = 100

    # AVOID GHOSTS
    def check(pos, ghost):
        dist = util.manhattanDistance(pos, ghost)
        if dist <= 1: return False
        return True

    # get the index of the ghost to avoid and collect the rest to chase_index
    AvoidGhostScore   = None
    avoid_ghost_index = list()
    chase_ghost_index = list()
    ghost_to_avoid    = set()

    for i, times in enumerate(curScaredTimes):
        if times == 0: avoid_ghost_index.append(i)
        else: chase_ghost_index.append(i)

    for index in avoid_ghost_index:
        pos = curGhosts[index]
        if not check(curPos, pos): AvoidGhostScore = -1
        ghost_to_avoid.add((int(pos[0]), int(pos[1])))
    if AvoidGhostScore is None: AvoidGhostScore = 1

    # print 'AvoidGhostScore', AvoidGhostScore
    FinalScore += AvoidGhostScore*AvoidGhostScoreWeight

    # GET FOOD
    MinFoodDist = float('inf')
    MaxFoodNum  = MAX_DIST_OF_MAP
    for f in curFood.asList():
        if f not in ghost_to_avoid: MinFoodDist = min(MinFoodDist, mazeDistance(curPos, f, currentGameState))
        if MinFoodDist == 1: break # minima found, don't continue searching
    if MinFoodDist == float('inf'): MinFoodDist = 0 # to avoid score goes to -inf
    # THIS IS VERY IMPORTANT!!!!!!!
    FoodScore = (MaxFoodNum-curFood.count()) - float(MinFoodDist)/float(MAX_DIST_OF_MAP)

    FinalScore += FoodScoreWeight*FoodScore

    # FIND CAPSULES COMBINED WITH CHASE GHOST
    TotalGhostNum  = len(curGhosts)
    ScaredGhostNum = len(chase_ghost_index)
    MinCapsuleDist = float('inf')
    MinChaseDist   = float('inf')
    MaxCapsuleNum  = MAX_DIST_OF_MAP
    MaxScaredGhostNum = MAX_DIST_OF_MAP

    for cap in curCapsules:
        MinCapsuleDist = min(MinCapsuleDist, mazeDistance(curPos, cap, currentGameState))
        if MinCapsuleDist == 1: break
    if MinCapsuleDist == float('inf'): MinCapsuleDist = 0 # to avoid score goes to -inf after all capsules are eaten!

    for index in chase_ghost_index:
        pos = curGhosts[index]
        dist = mazeDistance(curPos, (int(pos[0]), int(pos[1])), currentGameState)
        MinChaseDist = min(MinChaseDist, dist)
        if MinChaseDist == 1: break
    if MinChaseDist == float('inf'): MinChaseDist = 0 # to avoid score goes to -inf

    ChaseGhostCost = (ScaredGhostNum-1 + float(MinChaseDist)/float(MAX_DIST_OF_MAP)) / float(TotalGhostNum)
    CapsuleScore = (MaxCapsuleNum-len(curCapsules)) - float(MinCapsuleDist)/float(MAX_DIST_OF_MAP)*0.1 - ChaseGhostCost*0.9

    # print 'combined score', CapsuleScore
    FinalScore += CombinedTermWeight*CapsuleScore

    return FinalScore
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

