import sys
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
from util import nearestPoint
import numpy as np
import operator

#### GLOBAL VARIABLES NEEDED FOR ASTAR ###

Walls = set()
NoWalls = set()


nearestEnemyLocation = None
latestFoodMissing = None

DEFENDING = []
validNextPositions = {}


####PARAMETERS########

EXPANSION = 6
DISCOUNT = 0.7
ITER=100
REWARD_STOP=-10.



###########################################


class PolicyValueAgent():

    def __init__(self, reward_matrix, discount=DISCOUNT, iterations=ITER):
        self.reward_matrix = reward_matrix
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() 

        for i in range(0, iterations):
            iteration_values = util.Counter()
            for state in reward_matrix.getStates():
                best_action = self.computeActionFromValues(state)
                if best_action:
                    iteration_values[state] = self.computeQValueFromValues(state, best_action)

            self.values = iteration_values.copy()

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
  
        policyvalue = 0
        for next_state, prob in self.reward_matrix.getTransitionStatesAndProbs(state, action):
            reward = self.reward_matrix.getReward(state, action, next_state)
            policyvalue += prob * (reward + self.discount * self.values[next_state])
        return policyvalue

    def computeActionFromValues(self, state):
        action_values = util.Counter()
        for action in self.reward_matrix.getPossibleActions(state):
            action_values[action] = self.computeQValueFromValues(state, action)
        return max(action_values.items(), key=operator.itemgetter(1))[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)



class Reward_matrix():
    def __init__(self, startState, states):
        self._startState = startState
        self._states = states

        legalMoves = util.Counter()
        for state in states:
            x, y = state
            if (x - 1, y) in states:
                legalMoves[(state, Directions.WEST)] = (x - 1, y)
            if (x + 1, y) in states:
                legalMoves[(state, Directions.EAST)] = (x + 1, y)
            if (x, y - 1) in states:
                legalMoves[(state, Directions.SOUTH)] = (x, y - 1)
            if (x, y + 1) in states:
                legalMoves[(state, Directions.NORTH)] = (x, y + 1)
        self._possibleActions = legalMoves
        self._rewards = util.Counter()

    def addReward(self, state, reward):
        self._rewards[state] += reward

    def addRewardWithNeighbours(self, state, reward):
        x, y = state
        self._rewards[state] += reward
        self._rewards[(x - 1, y)] += reward * 0.7
        self._rewards[(x + 1, y)] += reward * 0.7
        self._rewards[(x, y - 1)] += reward * 0.7
        self._rewards[(x, y + 1)] += reward * 0.7


    def getStates(self):
        return list(self._states)

    def getStartState(self):
        return self._startState

    def getPossibleActions(self, state):
        return [acts[1] for acts in self._possibleActions.keys() if acts[0] == state]

    def getTransitionStatesAndProbs(self, state, action):
        return [(self._possibleActions[(state, action)], 1), ]
        #all assumed to have 100% transition rate.

    def getReward(self, state, action, nextState):
        if action == Directions.STOP:
            return REWARD_STOP
        return min(self._rewards[state], self._rewards[nextState])

    def isTerminal(self, state):
        return False


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveAstar'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):

    def ConstructGrid(self, left, bottom, right, top):
        left = int(max(1, left))
        right = int(min(self.right, right))
        bottom = int(max(1, bottom))
        top = int(min(self.top, top))

        grids = set()
        
        for x in range(left, right + 1):
            for y in range(bottom, top + 1):
                grids.add((x, y))
        return grids.difference(self.walls)

    def getDistanceToHome(self, position):
        x, _ = position
        if (self.homeBoundary[0][0] - x) * self.side > 0:
            return 0
        distances = [self.distancer.getDistance(position, cell) for cell in self.homeBoundary]
        return min(distances)

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)


        self.walls = set(gameState.data.layout.walls.asList())
        self.top = max([y[1] for y in self.walls])
        self.right = max([x[0] for x in self.walls])

        self.side = 1 if gameState.isOnRedTeam(self.index) else -1

        # Determining home boundary
        self.homeBoundary_X = self.start[0] + ((self.right // 2 - 1) * self.side)
        cells = [(self.homeBoundary_X, y) for y in range(1, self.top)]
        self.homeBoundary = [item for item in cells if item not in self.walls]

        # Determining legal actions count for all cells
        legal_cells = self.ConstructGrid(1, 1, self.right, self.top)
        self._legalActions = util.Counter()
        for cell in legal_cells:
            x, y = cell
            if (x - 1, y) in legal_cells:
                self._legalActions[cell] += 1
            if (x + 1, y) in legal_cells:
                self._legalActions[cell] += 1
            if (x, y - 1) in legal_cells:
                self._legalActions[cell] += 1
            if (x, y + 1) in legal_cells:
                self._legalActions[cell] += 1


    def isAtHome(self, position):
        x, _ = position
        x_min = self.start[0]
        x_max = self.homeBoundary_X
        return x_min <= x <= x_max or x_max <= x <= x_min

    def assignRewards(self, grid, reward_matrix, rewardShape, current_position, target):
        distanceToTarget = self.distancer.getDistance(target, current_position)
        rewards = []

        for cell in grid:
            distance = self.distancer.getDistance(cell, target)
            if distance <= distanceToTarget:
                reward = rewardShape / max(float(distance), .5)
                random_Noise = reward / 10.
                reward += random.uniform(-random_Noise, random_Noise)
                reward_matrix.addReward(cell, reward)
                rewards.append((current_position, cell, distance, reward))
        return rewards

    def chooseAction(self, gameState):
        food_reward = 0.5
        dead_end_reward = -0.2
        ghost_reward = -1
        cap_reward = 3
        go_back_reward = 1
        nearby_definition = 6

        myState = gameState.getAgentState(self.index)
        current_position = myState.getPosition()
        distance_to_home = self.getDistanceToHome(current_position)
        x, y = current_position


        grid = self.ConstructGrid(x - nearby_definition, y - nearby_definition, x + nearby_definition, y + nearby_definition)
        grid = {cell for cell in grid if self.distancer.getDistance(current_position, cell) <= nearby_definition}
        reward_matrix = Reward_matrix(current_position, grid)

        # assigning rewards for food
        foods = self.getFood(gameState).asList()
        foodLeft = len(foods)
        if foodLeft >= 1:
            for foodPos in foods:
                self.assignRewards(grid, reward_matrix, rewardShape=food_reward, current_position=current_position,
                                   target=foodPos)

        # ghost analysis
        enemies = []
        enemyNearby = False

        for index in self.getOpponents(gameState):
            enemyState = gameState.getAgentState(index)
            enemyPos = enemyState.getPosition()

            if enemyPos:
                enemy_distance = self.distancer.getDistance(current_position, enemyPos)
                if enemy_distance <= nearby_definition:
                    enemyNearby = True
                    enemies.append((enemyState, enemyPos))

        if enemyNearby and not enemyState.isPacman:

            enemyMinDistance = nearby_definition
            enemyScaredTimer = min([item.scaredTimer for item, _ in enemies])
            for enemyState, enemyPos in enemies:
                if enemyState.scaredTimer >= 6:
                    continue
                enemy_distance = self.distancer.getDistance(current_position, enemyPos)
                reward = ghost_reward * foodLeft / 2. * (nearby_definition - enemy_distance + 1.)
                reward_matrix.addRewardWithNeighbours(enemyPos, reward)
                enemyMinDistance = min(enemyMinDistance, enemy_distance)

            for cell in grid:
                if self.isAtHome(cell) or enemyScaredTimer > 10:
                    continue

                # dead end (one way tunnel)
                cell_to_home_distance = self.getDistanceToHome(cell)
                enemy_dist = float(9 - enemyMinDistance) / 2.
                if cell_to_home_distance > distance_to_home:
                    reward = float(cell_to_home_distance - distance_to_home) * dead_end_reward * enemy_dist
                    reward_matrix.addReward(cell, reward)

                # penalising cells with fewer actions
                legalActions = self._legalActions[cell]
                foodInCell = cell in foods
                if legalActions == 1 and foodInCell == False:
                    reward = float(dead_end_reward * enemy_dist * 2)
                    reward_matrix.addRewardWithNeighbours(cell, reward)
                if legalActions == 2 and enemyMinDistance <= 3:
                    reward_matrix.addRewardWithNeighbours(cell, dead_end_reward)

                # capsules
                for capsule in self.getCapsules(gameState):
                     capreward = cap_reward * foodLeft  * (nearby_definition - enemy_distance + 1.)

                     self.assignRewards(grid, reward_matrix, rewardShape=capreward,
                                       current_position=current_position, target=capsule)

                # going home reward
                carrying_reward = myState.numCarrying
                carried_reward = go_back_reward * carrying_reward / 7
                self.assignGoHomeRewards(grid, reward_matrix, carried_reward, current_position)

        # time management
        timeLeft = gameState.data.timeleft // 5
        goingHome = (foodLeft <= 5) or (timeLeft < 30) or (timeLeft < (self.getDistanceToHome(current_position) + 10))
        if goingHome:
            self.assignGoHomeRewards(grid, reward_matrix, go_back_reward, current_position)

        if myState.numCarrying > 6 and self.getDistanceToHome(current_position) < 10:
            self.assignGoHomeRewards(grid, reward_matrix, go_back_reward, current_position)

        policy_calculator = PolicyValueAgent(reward_matrix, discount=0.7, iterations=100)

        best_action = policy_calculator.getAction(current_position)

        if goingHome:
            pass
        return best_action

    def final(self, gameState):
        CaptureAgent.final(self, gameState)

    def assignGoHomeRewards(self, grid, reward_matrix, rewardShape, current_position):
        for targetCell in self.homeBoundary:
            rewards = self.assignRewards(grid, reward_matrix, rewardShape=rewardShape, current_position=current_position, target=targetCell)
        return rewards


    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        return None

    def getWeights(self, gameState, action):
        return {'numInvaders': -10000, 'onDefense': 1000, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class ReflexCaptureAgentAstar(CaptureAgent):

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = util.Counter()
        self.discountFactor = 0.7
        self.ValidPos = {}
        self.PrevAction = None
        self.AttackHistory = []
        self.DefenceHistory = []
        self.offensiveEntry = None
        self.defensiveEntry = None

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        arr = np.zeros((gameState.data.layout.width - 2, gameState.data.layout.height - 2))
        noWallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                           not gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        WallsTemp = set([(index[0][0] + 1, index[0][1] + 1) for index in np.ndenumerate(arr) if
                         gameState.hasWall(index[0][0] + 1, index[0][1] + 1)])
        for x, y in noWallsTemp:
            availableMoves = []
            if (x + 1,
                y) not in WallsTemp and 0 < x + 1 < gameState.data.layout.width - 1 and 0 < y < gameState.data.layout.height - 1:
                availableMoves.append((x + 1, y))
            if (x,
                y + 1) not in WallsTemp and 0 < x < gameState.data.layout.width - 1 and 0 < y + 1 < gameState.data.layout.height - 1:
                availableMoves.append((x, y + 1))
            if (x - 1,
                y) not in WallsTemp and 0 < x - 1 < gameState.data.layout.width - 1 and 0 < y < gameState.data.layout.height - 1:
                availableMoves.append((x - 1, y))
            if (x,
                y - 1) not in WallsTemp and 0 < x < gameState.data.layout.width - 1 and 0 < y - 1 < gameState.data.layout.height - 1:
                availableMoves.append((x, y - 1))
            global validNextPositions
            key = str(x) + ',' + str(y)
            validNextPositions[key] = availableMoves
        self.ValidPos = validNextPositions
        global Walls
        Walls = WallsTemp
        global NoWalls
        NoWalls = noWallsTemp
        #########################
        # DEFENSIVE ENTRY POINT #
        #########################
        centralX = (gameState.data.layout.width / 2) - 1
        centralY = (gameState.data.layout.height / 2) - 2
        coordsUpper = []
        coordsLower = []
        coords = []
        for i in range(EXPANSION):
            coordsLower.append(
                [location for location in NoWalls if location[0] == (centralX - i) and location[1] <= centralY])
            coordsUpper.append(
                [location for location in NoWalls if location[0] == (centralX - i) and location[1] > centralY])
            coords.append([location for location in NoWalls if location[0] == (centralX - i)])
        self.defensiveEntry = min(coords, key=len)
        global latestFoodMissing
        latestFoodMissing = random.choice(self.getFoodYouAreDefending(gameState).asList())

        #########################
        # OFFENSIVE ENTRY POINT #
        #########################
        centralX = ((gameState.data.layout.width / 2) - 1) + 1
        centralY = (gameState.data.layout.height / 2) - 2
        coordsUpper = []
        coordsLower = []
        coords = []
        for i in range(EXPANSION):
            coordsLower.append(
                [location for location in NoWalls if location[0] == (centralX + i) and location[1] <= centralY])
            coordsUpper.append(
                [location for location in NoWalls if location[0] == (centralX + i) and location[1] > centralY])
            coords.append([location for location in NoWalls if location[0] == (centralX + i)])
        self.offensiveEntry = min(coords, key=len)


    def chooseAction(self, gameState):
        self.observationHistory.append(gameState)
        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.AproaxQvalue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        bestAction = None
        if foodLeft <= 0:
            bestDist = 511
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist

        else:
            bestAction = random.choice(bestActions)
        return bestAction

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def AproaxQvalue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def ValueFromQvalue(self, gameState):

        actions = gameState.getLegalActions(self.index)

        if actions:
            values = [self.AproaxQvalue(gameState, a) for a in actions]
            maxValue = max(values)
            return maxValue
        else:
            return 0

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}

    #####################
    # ASTAR LOGIC BEGIN #
    #####################

    def isGhost(self, gameState, index):
        pos = gameState.getAgentPosition(index)
        if pos is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (pos[0] < gameState.getWalls().width / 2))

    def isScared(self, gameState, index):
        isScared = bool(gameState.data.agentStates[index].scaredTimer)
        return isScared

    def isPacman(self, gameState, index):
        pos = gameState.getAgentPosition(index)
        if pos is None:
            return False
        return not (gameState.isOnRedTeam(index) ^ (pos[0] >= gameState.getWalls().width / 2))

    def aStarSearch(self, startPosition, gameState, goalPositions, avoidPositions=[], returnPosition=False):
        walls = gameState.getWalls()

        walls = walls.asList()

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVec = [Actions.directionToVector(action) for action in actions]
        actionVec = [tuple(int(number) for number in vector) for vector in actionVec]


        currentPos, currentPath, currentCost = startPosition, [], 0


        queue = util.PriorityQueueWithFunction(lambda x: x[2] +  
                                                         (100) * self.getMazeDistance(startPosition, x[0]) if
        												  x[0] in avoidPositions else 0 +  
                                    sum([self.getMazeDistance(x[0], Pos) for Pos in
                                         goalPositions]))

        visited = set([currentPos])

        while currentPos not in goalPositions:

            possiblePos = [((currentPos[0] + vector[0], currentPos[1] + vector[1]), action) for
                           vector, action in zip(actionVec, actions)]
            legalPositions = [(position, action) for position, action in possiblePos if position not in walls]

            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    queue.push((position, currentPath + [action], currentCost + 1))

            if len(queue.heap) == 0:
                return None
            else:
                currentPos, currentPath, currentCost = queue.pop()

        if returnPosition:
            return currentPath, currentPos
        else:
            return currentPath

    def getFoodDistance(self, gameState, action):  #
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        current_position = successor.getAgentState(self.index).getPosition()
        nearestDistance = 0
        if len(foodList) > 0:
            nearestDistance = min(
                [self.getMazeDistance(current_position, food) + abs(self.favoredY - food[1]) for food in foodList])
        return nearestDistance


class DefensiveAstar(ReflexCaptureAgentAstar):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        current_position = myState.getPosition()

        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(current_position, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -10000, 'onDefense': 1000, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    def Qvalueattacks(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def QValuesActions(self, state):
        b_val = -9999
        b_act = None
        for i in state.getLegalActions(self.index):
            value = self.Qvalueattacks(state, i)
            if value > b_val:
                b_act = [i]
                b_val = value
            elif value == b_val:
                b_act.append(i)
        if b_act == None:
            return Directions.STOP  
        return random.choice(b_act)  

    def nextLocatedFood(self, gamestate, ourPosition):
        foodList = self.getFoodYouAreDefending(gamestate).asList()
        dists = [(self.getMazeDistance(ourPosition, x), x) for x in foodList]
        if dists:
            return min(dists)[1]

    def chooseAction(self, state):
        hasFooodBeenEatenLastime = False
        foodMissing = []
        foodDefend = self.getFoodYouAreDefending(state).asList()
        if len(self.DefenceHistory):
            prev_state = self.DefenceHistory.pop()

            global nearestEnemyLocation
            if nearestEnemyLocation is None:
                nearestEnemyLocation = state.getInitialAgentPosition(self.getOpponents(state)[0])

            successor = self.getSuccessor(prev_state, self.PrevAction)
            current_position = successor.getAgentState(self.index).getPosition()
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            Ghosts = [agent for agent in enemies if
                      not agent.isPacman and agent.getPosition() is not None and not agent.scaredTimer > 0]
            if successor.getAgentState(self.getOpponents(state)[0]).isPacman or successor.getAgentState(
                    self.getOpponents(state)[1]).isPacman:
                global DEFENDING
                foodMissing = list(set(DEFENDING).difference(foodDefend))
                global latestFoodMissing
                hasFooodBeenEatenLastime = (len(foodDefend) != len(self.getFoodYouAreDefending(prev_state).asList()))
                if foodMissing:
                    latestFoodMissing = foodMissing[0]
                else:
                    foodMissing = [latestFoodMissing]

        DEFENDING = foodDefend

        self.DefenceHistory.append(state)

        global validNextPositions
        legalCoordinates = []
        keys = validNextPositions.keys()
        for key in keys:
            possibleMoves = len(validNextPositions[key])
            if possibleMoves == 1:
                x = int(key.split(',')[0])
                y = int(key.split(',')[1])
                legalCoordinates.append((x, y))

        nowalls = []
        for i in range(1, state.data.layout.height):
            if self.index % 2 == 0:
                if state.hasWall(int(state.data.layout.width / 2 - 1), i) == False:
                    nowalls.append((int(state.data.layout.width / 2 - 1), i))
            else:
                if state.hasWall(int(state.data.layout.width / 2), i) == False:
                    nowalls.append((int(state.data.layout.width / 2), i))


        walls = state.getWalls().asList()
        walls = list(set(walls))

        avoidPositions = []
        #######################
        # CHOOSE ACTION ASTAR #
        #######################
        food = self.getFood(state)
        enemyIndices = self.getOpponents(state)
        capsules = self.getCapsules(state)

        attackablePacmen = [state.getAgentPosition(i) for i in enemyIndices if
                            self.isPacman(state, i) and self.isGhost(state, self.index)]

        avoidPacmen = [state.getAgentPosition(i) for i in enemyIndices if
                       self.isPacman(state, i) and self.isScared(state, self.index)]

        anyEnemy = [state.getAgentState(i).isPacman for i in enemyIndices]

        nearestFood = self.nextLocatedFood(state, state.getAgentPosition(self.index))

        avoidGhost = [state.getAgentPosition(i) for i in enemyIndices if
                      self.isGhost(state, i) and self.isPacman(state, self.index)]

        if anyEnemy[0] or anyEnemy[1]:

            if attackablePacmen:
                goalPositions = set(attackablePacmen)
            else:
                goalPositions = set(foodMissing)

        else:
            goalPositions = set(foodMissing).union(set(self.defensiveEntry))

        if self.isScared(state, self.index) and (anyEnemy[0] or anyEnemy[1]):

            goalPositions = set(foodMissing)
            avoidPositions = set(avoidPacmen)
            newGoalPositions = []
            newAvoidPositions = []

            for positions in list(goalPositions):
                pos = str(positions[0]) + ',' + str(positions[1])
                if len(self.ValidPos[pos]) < 2:
                    newAvoidPositions.append(positions)
                else:
                    newGoalPositions.append(positions)
            if len(newGoalPositions) == 0:
                newGoalPositions.append(nearestFood)
            goalPositions = set(newGoalPositions)
            avoidPositions = set(avoidPositions).union(set(newAvoidPositions))

        if avoidGhost:
            avoidPositions = set(avoidPositions).union(set(avoidGhost))

        if goalPositions:

            currentPos = set([state.getAgentPosition(self.index)])
            if currentPos.issubset(goalPositions):
                goalPositions = goalPositions.difference(currentPos)

            astar_path = self.aStarSearch(state.getAgentPosition(self.index), state, goalPositions, avoidPositions)

        else:
            astar_path = None

        if astar_path:

            actionToDo = astar_path[0]
        else:
            actionToDo = self.QValuesActions(state)

        whatToDo = None
        legalActions = state.getLegalActions(self.index)
        if actionToDo in legalActions:
            whatToDo = actionToDo
        else:
            whatToDo = self.QValuesActions(state)

        self.PrevAction = whatToDo

        return whatToDo
