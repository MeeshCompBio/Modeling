# Reinforced learning (RL)

## General tutorial links
[Reinforcement Learning: A Tutorial](http://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf)

## Overview

### Gradient Descent (GD)
* Used while training a machine learning model (optimization algorithm) to iteratively minimize function to its local minimum (trying to find the least deviation between preciction and outcome)
* Test GD is by plotting the cost function as it runs
    * Num iterations on the x-axes and cost-function on y-axes.  
* Multiple types fo gradient decent
    * Batch GD (vanilla)
        * Calcualtes error for each example, model gets updates after all examples have been evaluated
        * computationally efficient but entire dataset needs to be in memory
        * prodices a stable error gradient/convergence, does not always produce the best result
    * Stochastic GD
        * Update parameters for each training example (one by one). The frequency of update allow for detailed rate of improvement
        * Computationally expensive, frequency of updates can also result in noise
    * Mini batch GD
        * Combination of Batch and Stochastic
            * Split training set in small batches and performs update after each batch
            * Very popular when using neural networks  

### Back Propogation
[GitHub Example](https://github.com/mattm/simple-neural-network)
* goal is to optimize weights so neural network can learn how map arbitrary inputs to outputs
* Updates each of the network weights so actual output is closer target output
* Give insights into how changing weights and biases changes network behavior
* Based around four fundamental equations
* Allows way to comput error and gradient fo the cost function

### Intro to RL
* Dynamic programming was been used to solve problems of optimization and control
* Supervised learning is a general method for trainign parameterized function aproximator
    * SL required known answers to questions and we don't always know them  
* RL combines dynamic programming and supervised learning yeilding a machine learning system
    * In RL a computer is given a goal to achieve the learns how to achieve the goal by trial and error actions within the environment

### Parts of the RL problem
* In the standard reinforcement learning model an agent interacts with its environment
    * Interaction takes for in agent sensing the environment and based off sensory input, choosing an action
    * The action will then in turn change the environment and that is then communicated to the agent with scalar reinforcement

#### Environment
* Every RL system learns a mappring from sitations/actions within a dynamic environment
* Environment must be observable (at least partially) by the RL system
* Observations can be readings, descriptions etc
* Actions can be low or high level (such as winning or losing)
* If the RL system can perfectly observe all infor in the environment that might affect action choice, the RL chooses based on trur states fo the environment

#### Reinforcement function
* The exact function of future reinforcements the agent wants to maximize
    * IE, there is mapping from state/action paris that leads to reinforcement
    * Then the RL agent will recieve reinforcement in the form of a scalar value
    * RL Agent will then learn to perform actions that will maximize reinforcements based off initial state and going to final one

* Three classes for reinforment fucntions (although more complex ones can be made)
    * Pure delayed reward and avoidance problems
    * Reinfocements are all zero except at the terminal state
    * The sign of scalar reinforment at terminal state indicates if it is a goal state (reward) or a state that should be avoided (penalty)
    * An example would be backgammon where state is current state of the board, actions are legal moves, reinforment function is zero at every turn until there is a win or loss

* Minimum time to goal
    * agent to perform actions that generate the least amount of actions or trajectory to goal state
    * Car on the hill example
        * Car stuck between two steep hills
        * The goal of agent (driver) is to drive up right side hill to top
        * State is cars position and velocity
        * Actions or forward, reverse or no thrust
        * Dynamics included that car can't just directly thrust up hill (must use momentum of going back and forth)
        * The reinforcement function is -1 for all state transitions except the transition of the goal state in which 0 is returns
        * Agent wants to maximize reinforcement so it wants to minimize the time it takes to reach goal state

* Games
    * Above examples want to maximize the reinforcement fucntion but we can also minimize it
    * IE reinforcement is is a function of limited resources and agent must learn to conserve them while trying to achive goal
    * Alternaticely function can be used the context of games environment.
    * In a game scenarion, the RL system the learn to generate the optimal behavior be finding the maximin, minimax or saddlepoint of the reinforcement funciton
    * Beacause the actions are chosen independantly and executed simultanerously, the RL agent learns to choose actions that would best outcome for a given playing in a "worst case" scenario

#### Value function
* A policy determins which action should be performed in each state; a policy is mapping from states to actions
* The value of a state is defined as the sum of the reinforcements recieved when starting in that state and following a fixed policy to a terminal state
* The optimal policy would be mapping from states to actions that maximizes the sum of the reinforcements when starting at an arbitrary state and performing actions to reach terminal state
* The value of a state is dependant on it policy
* The value function is a mapping from state to stae values and can be approximated using any type of function approiximator
* for examples, if you have 4 x 4 grid and you want to get to either top left or bottom right corner, and you can only go up, down, left or right. The optimal value function would "show" the smallest number of movement required to get to either cell depending on where you start

### Approcimating the value function
* RL is a hard problem becuase the system might perform actions and might not know if it was good or bad
* A way to combat this is through dynamic programming
    * If the system does something bad immeadiatly, then it does not do that again
    * If all the actions in the certain situation leads to bad results then the situation should be avoided to begin with  
* The Essence of Dynamic Programming
    * Initially the approcimation of the optimal value fuction is poor. AKA state to state mappign is not valid
* Value iteration
    * Done using gradient decent on the mean squared bellman resifual
    