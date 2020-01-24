# myTeam.py
# ---------
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

import capture, random, time, util, game, math
from game import Directions
from captureAgents import CaptureAgent 
from util import nearestPoint
from util import Queue
from util import Stack
from util import Counter
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


# Global
tubes = []
defensiveTubes = []
walls = []

class Tree:
    def __init__(self, root):
        self.count = 1
        self.tree = root
        self.leaf = [root.value[0]]

    def insertNode(self, parent, child):
        id = self.count
        self.count += 1
        child.id = id
        parent.addChild(child)
        if parent.value[0] in self.leaf:
            self.leaf.remove(parent.value[0])
        parent.isLeaf = False
        self.leaf.append(child.value[0])

    def getParent(self, node):
        if node == self.tree:
            return None
        else:
            return self.tree.getParent(node)

    def backPropagation(self, root, node):
        (gameState, t1, n1) = node.value
        node.value = (gameState, t1 + root, n1 + 1)
        parent = self.getParent(node)
        if parent != None:
            self.backPropagation(root, parent)

    def chooseNode(self, node = None):
        if node == None:
            node = self.tree

        if node.isLeaf:
            return node
        else:
            nextNode = node.getChild()
            return self.chooseNode(nextNode)

class Node:
    def __init__(self, value, id=0):
        (gameState, tree, node) = value
        self.id = id
        self.children = []
        self.value = (gameState, float(tree), float(node))
        self.isLeaf = True

    def __str__(self):
        id = self.id
        (game, tree, node) = self.value
        return "Node in game"+ str(game) + ", id = " + str(id) + ", tree = " + str(tree) + ", node = " + str(node) + "."

    def addChild(self, child):
        self.children.append(child)

    def getChild(self):
        _, _, n1 = self.value
        result = None
        maxUCB = -999999
        for child in self.children:
            _, t2, n2 = child.value
            if n2 == 0:
                return child
            
            UCB = t2 + 1.96 * math.sqrt(math.log(n1) / n2)
            
            if maxUCB < UCB:
                maxUCB, result = UCB, child
        return result

    def getParent(self, node):
        for child in self.children:
            if child == node:
                return self
            else:
                nextNode = child.getParent(node)
                if nextNode != None:
                    return nextNode

class Guesser:

  def __init__(self, agent, gameState):
    self.beginState = gameState.getInitialAgentPosition(agent.index)
    self.agent = agent
    self.midP = gameState.data.layout.width/2
    self.enemies = self.agent.getOpponents(gameState)
    self.guess = {}
    self.Positions = []

    for p in gameState.getWalls().asList(False):
        self.Positions.append(p)
    
    for enemy in self.enemies:
        self.guess[enemy] = Counter()
        self.guess[enemy][gameState.getInitialAgentPosition(enemy)] = 1.0
        self.guess[enemy].normalize()

  def getPossiblePosition(self, enemy):
      return self.guess[enemy].argMax()

  def elapseTime(self):

    for enemy in self.enemies:
        count = Counter()
        for pos in self.Positions:
            count2 = Counter()
            allPossiblePos = [(pos[0]+i, pos[1]+j) for i in [-1,0,1] for j in [-1,0,1] if not (abs(i) == 1 and abs(j) == 1)]
            for pos2 in self.Positions:
                if pos2 in allPossiblePos:
                    count2[pos2] = 1.0
            count2.normalize()

            for newPos, prob in count2.items():
                count[newPos] = count[newPos] + self.guess[self.enemy][newPos] * prob

        count.normalize()
        self.guess[enemy] = count

  def observe(self, agent, gameState):

      myPos = gameState.getAgentPosition(agent.index)
      agentDistance = gameState.getAgentDistances()
      count = Counter()

      for enemy in self.enemies:
          for pos in self.Positions:
              actualDistance = util.manhattanDistance(myPos, pos)
              prob = gameState.getDistanceProb(actualDistance, agentDistance)

              if agent.red:
                  isPacman = pos[0] < self.midP
              else:
                  isPacman = pos[0] > self.midP

              if actualDistance <= 6 or isPacman != gameState.getAgentState(enemy).isPacman:
                  count[pos] = float(0)
              else:
                  count[pos] = self.guess[enemy][pos] * prob

          count.normalize()
          self.guess[enemy] = count
 
class uctCaptureAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    #global values
    global tubes 
    global walls  
    global Positions                                       
    global cleanRoad                                                    
    global defensiveTubes 

    CaptureAgent.registerInitialState(self, gameState)
    self.beginState = gameState.getAgentPosition(self.index)
    self.switchEntry = False
    self.nextEntry = None
    self.tubeStartPos = None
    self.safeCapsule = None
    self.nextFood = None
    self.nextTubeFood = None
    self.returnBoundaryPos = None
    self.enemyCapturedPos = None
    self.isStuck = False
    self.stuckCounter = 0
    self.enemyCapturedFood = 0
    self.enemyGuess = Guesser(self, gameState)
    self.invadersGuess = False
    
    walls = gameState.getWalls().asList()
    if len(tubes) == 0:
        Positions = []
        for pos in gameState.getWalls().asList(False):
            Positions.append(pos)
            
        tubes = getMapTubes(Positions)
        cleanRoad = list(set(Positions).difference(set(tubes)))

    width = gameState.data.layout.width

    redPositions = []
    bluePositions = []
    for pos in Positions:
        if pos[0] < width / 2:
            redPositions.append(pos)
        else:
            bluePositions.append(pos)

    if len(defensiveTubes) == 0:
        if self.red:
            defensiveTubes = getMapTubes(redPositions)
        else:
            defensiveTubes = getMapTubes(bluePositions)

  def chooseAction(self, gameState):
    actions = gameState.getLegalActions(self.index)
    rewards = []
    for a in actions:
        rewards.append(self.evaluate(gameState, a)) 

    Q = max(rewards)
    
    if self.isStuck:
        return self.MCT(gameState)
    
    options = [a for a, v in zip(actions, rewards) if v == Q]
    return random.choice(options)
  
  def getTimeLeft(self, gameState):
      return gameState.data.timeleft

  def MCT(self, gameState):
      root = Node((gameState, 0, 0))
      mct = Tree(root)
      startTime = time.time()
      while time.time() - startTime < 0.94:
          self.iteration(mct)

      nextState = mct.tree.getChild().value[0]
      (x1, y1) = gameState.getAgentPosition(self.index)
      (x2, y2) = nextState.getAgentPosition(self.index)
      if y1 + 1 == y2:
          return Directions.NORTH
      if x1 + 1 == x2:
          return Directions.EAST
      if y1 - 1 == y2:
          return Directions.SOUTH
      if x1 - 1 == x2:
          return Directions.WEST
      return Directions.STOP
  
  def evaluate(self, gameState, action):
    return self.getFeatures(gameState, action) * self.getWeights(gameState, action)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
        return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def isEmptyTube(self, gameState, successor):
    curPos = gameState.getAgentState(self.index).getPosition()
    sucPos = successor.getAgentState(self.index).getPosition()

    if curPos not in tubes and sucPos in tubes:

      self.tubeStartPos = curPos
      closed = []
      openStack = Stack()
      openStack.push((sucPos, 1))

      while not openStack.isEmpty():
        (x, y), size = openStack.pop()
        if self.getFood(gameState)[int(x)][int(y)]:
          return size

        if (x, y) not in closed:
          closed.append((x, y))
          lcoation = getSuccsorsPos(tubes,(x, y))
          for n in lcoation:
            if n not in closed:
              newSize = size + 1
              openStack.push((n, newSize))
    return 0

  def getTubeFood(self, gameState):
      curPos = gameState.getAgentState(self.index).getPosition()
      closed = []
      openQueue = Queue()
      openQueue.push(curPos)

      while not openQueue.isEmpty():
          (x, y) = openQueue.pop()
          if self.getFood(gameState)[int(x)][int(y)]:
              return (x, y)

          if (x, y) not in closed:
              closed.append((x, y))
              lcoation = getSuccsorsPos(tubes,(x, y))
              for i in lcoation:
                  if i not in closed:
                      openQueue.push(i)
      return None

  def getEntry(self,gameState):
        width = gameState.data.layout.width
        Positions = []
        redPositions = []
        bluePositions = []
        redEnter = []
        blueEnter = []
        for p in gameState.getWalls().asList(False):
            Positions.append(p)
        for p in Positions:
            if p[0] == width / 2 - 1:
                redPositions.append(p)
            if p[0] == width / 2:
                bluePositions.append(p)
                
        for r in redPositions:
            for b in bluePositions:
                if r[0] + 1 == b[0] and r[1] == b[1]:
                    redEnter.append(r)
                    blueEnter.append(b)
        if self.red:
            return redEnter
        else:
            return blueEnter
  
  def rollOut(self,gameState):
    counter = 20
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() is not None]
    ghostPos = [a.getPosition() for a in ghosts]
    currentState = gameState
    while counter != 0:
      counter -= 1
      actions = currentState.getLegalActions(self.index)
      nextAction = random.choice(actions)
      successor = self.getSuccessor(currentState,nextAction)
      nextPos = getNextPos(currentState.getAgentState(self.index).getPosition(),nextAction)
      if nextPos in ghostPos:
          return -9999
      currentState = successor
    return self.evaluate(currentState,'Stop')
  
  def expand(self, mct, node):
      actions = node.value[0].getLegalActions(self.index)
      actions.remove(Directions.STOP)
      for a in actions:
          successor = node.value[0].generateSuccessor(self.index, a)
          successorNode = Node((successor, 0, 0))
          mct.insertNode(node, successorNode)

  def iteration(self, mct):
      if mct.tree.children == []:
          self.expand(mct, mct.tree)
      else:
          leaf = mct.chooseNode()
          if leaf.value[2] == 0:
              r = self.rollOut(leaf.value[0])
              mct.backPropagation(r, leaf)
          elif leaf.value[2] == 1:
              self.expand(mct, leaf)
              newLeaf = random.choice(leaf.children)
              r = self.rollOut(newLeaf.value[0])
              mct.backPropagation(r, newLeaf)

class OffensiveReflexAgent(uctCaptureAgent):
  
  def getDistanceToHome(self, gameState):
      current = gameState.getAgentState(self.index).getPosition()
      width = gameState.data.layout.width
      Positions = []
      redPositions = []
      bluePositions = []
      for pos in gameState.getWalls().asList(False):
          Positions.append(pos)

      for pos in Positions:
          if pos[0] == width / 2 - 1:
              redPositions.append(pos)
          if pos[0] == width / 2:
              bluePositions.append(pos)
              
      if self.red:
          return min([self.getMazeDistance(current, a) for a in redPositions])
      else:
          return min([self.getMazeDistance(current, a) for a in bluePositions])

  def getFeatures(self, gameState, action):
    features = Counter()
    successor = self.getSuccessor(gameState, action)
    currentState = gameState.getAgentState(self.index).getPosition()
    nextState = successor.getAgentState(self.index).getPosition()
    nextPos = getNextPos(currentState,action) 
    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    safeCapsule = self.getCapsules(gameState)                         
    checkTube = self.isEmptyTube(gameState, successor)      
    distanceToHome = self.getDistanceToHome(successor)
    enemies = []
    invaders = []
    ghost = []
    passiveGhost = []
    activeGhost = [] 
    
    for e in self.getOpponents(gameState):
        enemies.append(gameState.getAgentState(e))

    for i in enemies:
        if i.isPacman and i.getPosition() is not None:
            invaders.append(i)

    for a in enemies:
        if not a.isPacman and a.getPosition() is not None and manhattanDistance(currentState,a.getPosition()) <= 5:
            ghost.append(a)

    for p in ghost:
        if p.scaredTimer > 1:
            passiveGhost.append(p)

    for c in ghost:
        if not(c in passiveGhost):
            activeGhost.append(c)
    
    currentFoodList = self.getFood(gameState).asList()    
    pathFood = []  
    tubeFood = [] 
    for food in currentFoodList:
        if food in tubes:
            tubeFood.append(food)
        else:
            pathFood.append(food)

    inEnemyArea = gameState.getAgentState(self.index).isPacman
    if inEnemyArea:
        self.switchEntry = False
    features['successorScore'] = self.getScore(successor)

    if len(ghost) == 0:
        self.safeCapsule = None
        self.nextFood = None
        self.nextTubeFood = None

    if nextPos in currentFoodList:
        self.enemyCapturedFood += 1

    if not inEnemyArea:
        self.enemyCapturedFood = 0

    if self.getTimeLeft(gameState)/4 < self.getDistanceToHome(gameState) + 3:
        features['distanceToHome'] = distanceToHome
        return features

    if len(activeGhost) == 0 and len(currentFoodList) != 0 and len(currentFoodList) >= 3:
      features['safeDistance'] = min([self.getMazeDistance(nextState, food) for food in currentFoodList])
      if nextState in self.getFood(gameState).asList():
          features['safeDistance'] = -1

    if len(currentFoodList) < 3:
      features['backToHome'] = distanceToHome

    if len(activeGhost) > 0 and len(currentFoodList) >= 3:
        dists = min([self.getMazeDistance(nextState, a.getPosition()) for a in activeGhost])
        features['distToGhost'] = 100 - dists
        ghostPos = [a.getPosition() for a in activeGhost]
        
        if nextPos in ghostPos:
            features['beEaten'] = 1
        if nextPos in [getSuccsorsPos(Positions,p) for p in ghostPos][0]:
            features['beEaten'] = 1
        
        if len(pathFood) > 0:
            features['pathFood'] = min([self.getMazeDistance(nextState, food) for food in pathFood])
            if nextState in pathFood:
              features['pathFood'] = -1
        elif len(pathFood) == 0:
            features['backToHome'] = distanceToHome

    if len(activeGhost) > 0 and len(currentFoodList) >= 3:
        if len(pathFood) > 0:
            safeFood = []
            for food in pathFood:
                dist = []
                for active in activeGhost:
                    dist.append(self.getMazeDistance(active.getPosition(), food))

                if self.getMazeDistance(currentState, food) < min(dist):
                    safeFood.append(food)

            if len(safeFood) != 0:
                dist = []
                for food in safeFood:
                    dist.append(self.getMazeDistance(currentState, food))

                for food in safeFood:
                    if self.getMazeDistance(currentState, food) == min(dist):
                        self.nextFood = food
                        break

    if len(activeGhost) > 0 and len(tubeFood) > 0 and len(passiveGhost) == 0 and len(currentFoodList) >= 3:
        safeFoodList = []
        for tf in tubeFood:
            tubeStartPos = getTubeStartPos(Positions, tubes, tf)
            distSum = self.getMazeDistance(currentState, tf) + self.getMazeDistance(tf, tubeStartPos)
            distGhost = []
            for active in activeGhost:
                distGhost.append(self.getMazeDistance(active.getPosition(), tubeStartPos))
            if distSum < min(distGhost):
                safeFoodList.append(tf)

        if len(safeFoodList) > 0:
            tfList = []
            for food in safeFoodList:
                tfList.append(self.getMazeDistance(currentState, food))
            
            for food in safeFoodList:
                if self.getMazeDistance(currentState, food) == tfList:
                    self.nextTubeFood = food
                    break

    if self.nextFood != None:
        features['goToSafeFood'] = self.getMazeDistance(nextState, self.nextFood)
        if nextState == self.nextFood:
            features['goToSafeFood'] = -0.0001
            self.nextFood = None

    if features['goToSafeFood'] == 0 and self.nextTubeFood != None:
        features['goToSafeFood'] = self.getMazeDistance(nextState, self.nextTubeFood)
        if nextState == self.nextTubeFood:
            features['goToSafeFood'] = 0
            self.nextTubeFood = None

    if len(activeGhost) > 0 and len(safeCapsule) != 0:
        for capsule in safeCapsule:
            capActive = []
            for active in activeGhost:
                capActive.append(self.getMazeDistance(capsule, active.getPosition()))
            if self.getMazeDistance(currentState, capsule) < min(capActive):
                self.safeCapsule = capsule

    if len(passiveGhost) > 0 and len(safeCapsule) != 0:
        for capsule in safeCapsule:
            curDist = self.getMazeDistance(currentState, capsule)
            passiveScareTime = passiveGhost[0].scaredTimer
            capPassive = []
            for pas in passiveGhost:
                capPassive.append(self.getMazeDistance(capsule, pas.getPosition()))
            if curDist >= passiveScareTime and curDist < min(capPassive):
                self.safeCapsule = capsule

    if currentState in tubes:
        for capsule in safeCapsule:
            if capsule in bfsSearchTube(currentState,tubes):
                self.safeCapsule = capsule

    if self.safeCapsule != None:
        features['distanceToCapsule'] = self.getMazeDistance(nextState, self.safeCapsule)
        if nextState == self.safeCapsule:
            features['distanceToCapsule'] = 0
            self.safeCapsule = None

    if len(activeGhost) == 0 and nextState in safeCapsule:
        features['leaveCapsule'] = 1                        #0.1

    if action == Directions.STOP: 
        features['stop'] = 1

    if successor.getAgentState(self.index).isPacman and not (currentState in tubes) and successor.getAgentState(self.index).getPosition() in tubes and checkTube == 0:
        features['emptyFoodTube'] = -1

    if len(activeGhost) > 0:
         dist = []
         for active in activeGhost:
             dist.append(self.getMazeDistance(currentState, active.getPosition()))
         minDist = min(dist) - 1 
         if checkTube != 0 and checkTube * 2 >= minDist:
             features['uselessMove'] = -1

    if len(passiveGhost) > 0:
         dist = min([self.getMazeDistance(currentState, a.getPosition()) for a in passiveGhost])
         minDist = passiveGhost[0].scaredTimer - 1
         if checkTube != 0 and checkTube * 2 >= minDist:
             features['uselessMove'] = -1


    if currentState in tubes and len(activeGhost) > 0:
        foodPos = self.getTubeFood(gameState)
        if foodPos == None:
            features['escapeTube'] = self.getMazeDistance(getNextPos(currentState,action), self.tubeStartPos)
        else:
            lengthToEscape = self.getMazeDistance(nextState, foodPos) + self.getMazeDistance(foodPos, self.tubeStartPos)
            ghostToEntry = min([self.getMazeDistance(self.tubeStartPos, a.getPosition()) for a in activeGhost])
            if ghostToEntry - lengthToEscape <= 1 and len(passiveGhost) == 0:
                features['escapeTube'] = self.getMazeDistance(getNextPos(currentState,action), self.tubeStartPos)

    if currentState in tubes and len(passiveGhost) > 0:
        foodPos = self.getTubeFood(gameState)
        if foodPos == None:
            features['escapeTube'] = self.getMazeDistance(getNextPos(currentState,action), self.tubeStartPos)
        else:
            lengthToEscape = self.getMazeDistance(nextState, foodPos) + self.getMazeDistance(foodPos, self.tubeStartPos)
            if  passiveGhost[0].scaredTimer - lengthToEscape <= 1:
                features['escapeTube'] = self.getMazeDistance(getNextPos(currentState,action), self.tubeStartPos)

    if not inEnemyArea and len(activeGhost) > 0 and self.stuckCounter != -1:
        self.stuckCounter += 1

    if inEnemyArea or nextState == self.nextEntry:
        self.stuckCounter = 0
        self.nextEntry = None

    if self.stuckCounter > 10:
        self.stuckCounter = -1
        self.nextEntry = random.choice(self.getEntry(gameState))

    if self.nextEntry != None and features['goToSafeFood'] == 0:
        features['goForNextEntry'] = self.getMazeDistance(nextState,self.nextEntry)

    return features

  def getWeights(self, gameState, action):
    return {
        'successorScore': 1, 
        'distanceToHome': -100, 
        'safeDistance': -2, 
        'pathFood': -3,
        'distToGhost': -10, 
        'beEaten': -99999,
        'goToSafeFood': -11,
        'distanceToCapsule': -1200,
        'backToHome': -1,
        'leaveCapsule': -1,
        'stop': -50, 
        'emptyFoodTube': 100,
        'uselessMove': 100,
        'escapeTube': -1001,
        'goForNextEntry': -1001
    }

class DefensiveReflexAgent(uctCaptureAgent):

  def distanceToBoundary(self, gameState):
      curPos = gameState.getAgentState(self.index).getPosition()
      width = gameState.data.layout.width
      Positions = []
      redPositions = []
      bluePositions = []
      for p in gameState.getWalls().asList(False):
          Positions.append(p)
      for p in Positions:
          if p[0] == width / 2 - 1:
              redPositions.append(p)
          if p[0] == width / 2:
              bluePositions.append(p)

      if self.red:
          return min([self.getMazeDistance(curPos, r) for r in redPositions])
      else:
          return min([self.getMazeDistance(curPos, b) for b in bluePositions])
  
  def isBlockedTube(self, cuurentInvaders, currentPostion, curCapsule):
    if len(cuurentInvaders) == 1:
      invadersPos = cuurentInvaders[0].getPosition()
      if invadersPos in tubes:
        tubeStartPos = getTubeStartPos(Positions, tubes, invadersPos)
        if self.getMazeDistance(tubeStartPos,currentPostion) <= self.getMazeDistance(tubeStartPos,invadersPos) and curCapsule not in bfsSearchTube(invadersPos,tubes):
           return True
    return False

  def isLostFood(self):
        preState = self.getPreviousObservation()
        currentState = self.getCurrentObservation()
        myCurrFood = self.getFoodYouAreDefending(currentState).asList()
        myLastFood = self.getFoodYouAreDefending(preState).asList()
        if len(myCurrFood) < len(myLastFood):
            for f in myLastFood:
                if f not in myCurrFood:
                    return f
        return None

  def getFeatures(self, gameState, action):
    features = Counter()
    successor = self.getSuccessor(gameState, action)
    curPos = gameState.getAgentState(self.index).getPosition() 
    currentState = gameState.getAgentState(self.index)
    sucState = successor.getAgentState(self.index)
    sucPos = sucState.getPosition()       
    curCapsule = self.getCapsulesYouAreDefending(gameState) 
    distToBoundary = self.distanceToBoundary(successor) 

    features['defensiveState'] = 100
    if sucState.isPacman: 
        features['defensiveState'] = 0

    if self.returnBoundaryPos == None:
        features['returnBoundaryPos'] = self.distanceToBoundary(successor)

    if self.distanceToBoundary(successor) <= 2:
        self.returnBoundaryPos = 0
    
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    curEnemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    curInvaders = [a for a in curEnemies if a.isPacman and a.getPosition() != None]

    if self.invadersGuess:
        self.enemyGuess.observe(self, gameState)
        enemyPos = self.enemyGuess.getPossiblePosition(curInvaders[0])
        features['goTubeEntry'] = self.getMazeDistance(enemyPos, sucPos)
        self.enemyGuess.elapseTime()

    if self.isBlockedTube(curInvaders, curPos, curCapsule) and currentState.scaredTimer == 0:
        features['goTubeEntry'] = self.getMazeDistance(getTubeStartPos(Positions,tubes,curInvaders[0].getPosition()),sucPos)
        return features

    if curPos in defensiveTubes and len(curInvaders) == 0:
        features['outOfTube'] = self.getMazeDistance(self.beginState, sucPos)

    features['numOfInvaders'] = len(invaders)

    if len(curInvaders) == 0 and not successor.getAgentState(self.index).isPacman and currentState.scaredTimer == 0:
        if  curPos not in defensiveTubes and successor.getAgentState(self.index).getPosition() in defensiveTubes:
            features['uselessMove'] = -1

    if len(invaders) > 0 and currentState.scaredTimer == 0:
        dists = [self.getMazeDistance(sucPos, inv.getPosition()) for inv in invaders]
        features['distanceToInvader'] = min(dists)
        features['distToBoundary'] = self.distanceToBoundary(successor)

    if len(invaders) > 0 and currentState.scaredTimer != 0:
        dists = min([self.getMazeDistance(sucPos, inv.getPosition()) for inv in invaders])
        features['tracking'] = math.pow(dists-2, 2)
        if curPos not in defensiveTubes and successor.getAgentState(self.index).getPosition() in defensiveTubes:
            features['uselessMove'] = -1
    '''
    if len(invaders) > 0 and len(curCapsule) != 0:
        dist2 = [self.getMazeDistance(c, sucPos) for c in curCapsule]
        features['returnCapsule'] = min(dist2)
    '''
    if action == Directions.STOP: 
        features['stop'] = 100
    else:
        features['stop'] = -100

    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == reverse: 
        features['reverse'] = 1

    if self.getPreviousObservation() != None:
      if len(invaders) == 0 and self.isLostFood() != None:
          self.enemyCapturedPos = self.isLostFood()

      if self.enemyCapturedPos != None and len(invaders) == 0:
          features['goToLostFood'] = self.getMazeDistance(sucPos,self.enemyCapturedPos)

      if sucPos == self.enemyCapturedPos or len(invaders) > 0:
          self.enemyCapturedPos = None

    return features

  def getWeights(self, gameState, action):
    return {
        'numOfInvaders': -100, 
        'defensiveState': 10, 
        'distanceToInvader': -10, 
        'distToBoundary':-3,
        'returnCapsule': -3, 
        'uselessMove': 200,
        'tracking': -100, 
        'goTubeEntry': -10, 
        'outOfTube': -0.1,
        'stop': -100, 
        'reverse': -2,
        'returnBoundaryPos': -2,
        'goToLostFood': -1
    }

def bfsSearchTube(pos, tubes):
    if not (pos in tubes):
        return None

    closed = []
    openQueue = Queue()
    openQueue.push(pos)
    
    while not openQueue.isEmpty():
        current = openQueue.pop()
        if not (current in closed):
            closed.append(current)
            succssors = getSuccsorsPos(tubes, current)
            for node in succssors:
                if not(node in closed):
                    openQueue.push(node)
    return closed

def getNextPos(pos, move):
    x, y = pos
    if move == Directions.NORTH:
        return (x, y + 1)
    if move == Directions.EAST:
        return (x + 1, y)
    if move == Directions.SOUTH:
        return (x, y - 1)
    if move == Directions.WEST:
        return (x - 1, y)
    return pos

def countSuccsorsPos(Positions, pos):
    count = 0
    x,y = pos
    if (x + 1, y) in Positions:
        count += 1
    if (x - 1, y) in Positions:
        count += 1
    if (x, y + 1) in Positions:
        count += 1
    if (x, y - 1) in Positions:
        count += 1
    return count

def getSuccsorsPos(Positions, pos):
    result = []
    x,y = pos
    if (x + 1, y) in Positions:
        result.append((x + 1, y))
    if (x - 1, y) in Positions:
        result.append((x - 1, y))
    if (x, y + 1) in Positions:
        result.append((x, y + 1))
    if (x, y - 1) in Positions:
        result.append((x, y - 1))
    return result

def getTubeStartPos(Positions, tubes, pos):
    if not(pos in tubes):
        return None

    search = bfsSearchTube(pos, tubes)
    for pos2 in search:
        x, y = pos2
        if (x + 1, y) in Positions and not((x + 1, y) in tubes):
            return (x + 1, y)
        elif (x - 1, y) in Positions and not((x - 1, y) in tubes):
            return (x - 1, y)
        elif (x, y + 1) in Positions and not((x, y + 1) in tubes):
            return (x, y + 1)
        elif (x, y - 1) in Positions and not((x, y - 1) in tubes):
            return (x, y - 1)

def manhattanDistance(pos1, pos2):
    (x1, y1), (x2, y2)  = pos1, pos2
    return abs(x2 - x1) + abs(y2 - y1) 

def findTubes(Positions, tubes):
    result = tubes
    for pos in Positions:
        diff = countSuccsorsPos(Positions, pos) - countSuccsorsPos(tubes, pos)  
        if diff == 1 and not(pos in tubes):
            result.append(pos)
    return result

def getMapTubes(Positions):
    tubes = []
    while len(tubes) != len(findTubes(Positions, tubes)):
        tubes = findTubes(Positions, tubes) #get all tubes 
    return tubes
