# Reinforced learning (RL)

## General tutorial links
[Reinforcement Learning: A Tutorial](http://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf)  
[Introduction to deep reinforcement learning](https://medium.com/@jonathan_hui/rl-introduction-to-deep-reinforcement-learning-35c25e04c199)
[Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)  
[Deep Q-network](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)  
[Deep Reinforcement Learning Series](https://medium.com/@jonathan_hui/rl-deep-reinforcement-learning-series-833319a95530)  
[Intro to reinforcement learning](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)  
[RL video tutorials](https://www.youtube.com/watch?v=q2ZOEFAaaI0&feature=youtu.be)  

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
* The RL loop outputs a sequence of state, action, and reward

### Parts of the RL problem
* In the standard reinforcement learning model an agent interacts with its environment
    * Interaction takes for in agent sensing the environment and based off sensory input, choosing an action
    * An agent will learn from the environment by interacting with is and recieving rewards for performing actions
    * The action will then in turn change the environment and that is then communicated to the agent with scalar reinforcement
    * An action is the same as a control (written as a or u)
    * State can be written as s or x

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
* A policy rells us how to act from a particular state (we want to find the one that makes the most rewarding decisions)
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


## Intro to reinforcement learning  
* RL is just a computational approach of learning form action
* RL process can be modeled as a loop that works like this
    * Agent recieves state S0 from the environment
    * Based of the S0 state, the agent then takes an action S0
    * The environment then takes an action A0
    * Environment then gives a reward to the agent
    * THe RL then keeps looping in a sequence of state, action, reward

### Central idea of reward hypothesis
* All goals can be described by the maximization of the expected cumulative reward
* Cumulative rewards at each step can be sumed as Gt = sum(k0 to T) Rt + k +1
    * In reality, you can't add rewards like that, rewards earlier in game are more likely to happen since they are more predicatble
* Consider a cat and mouse game, where there is a grid and mouse must find cheese but not get caught by the cat
* Any cheese near the cat will be discounted since we are more likely to get caught by the cat
* We define a discount rate gamma to discount the rewards (must be between 0 and 1)
* The large the gamma, the smaller the discount, meaingin the agent cares more about the long term reward
* Smaller the gamma, the bigger the discount, this means the agent cares more abotu the short term reward
* Value for gamma, can be discounted for each step depending on the scenario
    * Gt = sum(k0 to T) gamma to the power of k * Rt + k +1 (where gamma 0:1)
    * simple is each reward will be discounted by gamma to the exponent of the time step

### Episodic or continuous tasks
* A taks is an instance of a RL problem
* An episodic task has a start and end point (terminal state)
    * a list of states, actions, rewards and new states
* Continuous tasks have no terminal states (continues forever)
    * Agent must learn how to choose the best actions and simultanerously interact with the environment (like stock trading)
    * The agent will continue to run until we decided to stop it

### Monte Carlo vs TD learning methods
* Monte Carlo is collectin the rewards at the end of the episode and then calculating the maxiumum expected future reward
* Temporal difference learning estimated the rewards at each step
* Monte Carlo
    * When an episode ends, the agent looks that the total cumulative reward to see how well it performed.
    * Rewards are only recieved at the end of the game
    * We will then start a new game with the added knowledge, hoping the agent makes better decisions with each iteration

* TD learning
    * Will not wait until the end of the rpisode to update the maximum future reward estimation
    * It will update its value estimation V for the non-terminal states St occuring at that experience
    * This is called TD(0) or one step TD (update the value function after any induvidual step)
    * Wiil only waint until the next time step to update the value estimates
    * At any time t +1, they immeadiately form a TD target using the observed reward Rt+1 and the current value estimate (V(St+1))
    * TD target is an estiamtion, you update the precious estimate by updating it towards a one-step target

### Exploration/ Exploitation trade off
* Exploration is finding more information about the environment
* Exploitation is exploting known information to maximize the reward
* The goal of the RL agent is to maximize the expected cumulative reward (but we can fall into trap)
* You must define rules to aid in exploration so agent does not fall into loop and only does conservative things when there is a large benefit for taking risks for instance

### Three approaches for RL learning
#### Value based
* goal is to optimize the value function V(s)
* The value function is a function that rells us the maximum expected fufutre reward the agent will get at each state
* Value at each state is the total amount of the reward an agent can expect to accumulat over the future depending on the state you start at
* The agend will use the value function to select which state to choose at each step (agent will take the stat will the biggest value)

#### Policy based
* We want to directly optimized the policy function without using a value function
* The policy is what defines the agend behavior at a given time
* When you learn a policy function, this allows you to map each state to the best corresponding action
* Deterministic policy means that a policy at a given state will always return the same action
* Stochastic policy means to output a distribution probability over acitons

#### Model based
* We create a model of the behavior of the environment
* Each environment will need a different model representation

