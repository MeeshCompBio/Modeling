# Reinforced learning (RL)

## General tutorial links
[Reinforcement Learning: A Tutorial](http://www.cs.toronto.edu/~zemel/documents/411/rltutorial.pdf)  
[Introduction to deep reinforcement learning](https://medium.com/@jonathan_hui/rl-introduction-to-deep-reinforcement-learning-35c25e04c199)
[Simple Reinforcement Learning with Tensorflow](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)  
[Deep Q-network](https://medium.com/@jonathan_hui/rl-dqn-deep-q-network-e207751f7ae4)  
[Deep Reinforcement Learning Series](https://medium.com/@jonathan_hui/rl-deep-reinforcement-learning-series-833319a95530)  
[Intro to reinforcement learning](https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419)  
[RL video tutorials](https://www.youtube.com/watch?v=q2ZOEFAaaI0&feature=youtu.be)  
[Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C)](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)  
[Conveloution in machine learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)

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

## RL with Q-learning
* A value based RL alogirthm
* Creates a table where we caluculate the maximum expected future reward for each action at each state
* For this example, grid based try to find princess, each tile has four possible actions
* Computationally grid is transformed into a table
    * row will be state and columns will be actions (for this example)
* Q stands for quality of the action (think of the table as a cheat sheet)
* Each Q table scroe will be the maximum expected future reward that we would get if we take the action at that state with the best policy given

### Q-learning algorithm
* The action-value function or Q-function takes two inputs (state and action), it will then reutrn the expected future reward of action at that state
* Before exploration, Q-table will give arbitrary fixed value (mostly 0), as environment is explored, Q table will iteravely update and give us better and better approximantions using the Bellman equation
* Q-learning algorithm process
    * Initialize Q-table
    * 2) Choose an action
    * Perform action
    * Measure reward
    * Update Q
    * return to 2)  
* Q-table is but with m cols (number of actions) and n rows (number of states), values are initialized at 0
* Steps will be repeated until max number of episodes are met (user specfied)
* Choosing and action a at the current state s is based off the current Q-val esitmates
* What it everything is 0 first, what happens next?
* Using epsilon greedy strategy
    * We set an exploration rate (epsilon) to 1 in the beginning. This will be the rate of steps that we will do randomly
    * The rate must be at its highest value becuase we don't know anything abotu values in Q-table
    * We then generate a random number, if it is bigger than epsilon, then we will "exploit" (select best action at each step), or else we will explore
    * Idea is to have big epsilon at the beginning so we do more exploration then reduce it over time as the agent becomes more confident at estimating Q-values  
* Take the action a and observe the outcomes state s' and reward r, now update the function Q(s,a)
* We then take the action a that we chose in step 3, and then performing this action returns us a new state s' and a reward r
* Then update Q(s,a) using the Bellman equation

## Deep Q-learning
* Creating and updateing a Q-table can be ineffective in large environments
* Examples below correspond to teaching an agent to play doom (big environment with millions of different states)
* DQ nerual new takes a stack of four frames as an input and pass though the network then output a vector of Q-values for each action possible at the given state
* Take the largest Q-value from the vector to find the best action
* Preprocessing can be important, reducing complexity of states can greatly reduce computational time
* This example will be using stacked frames so there is an idea of motion, one frame will not tell us this information
* Frames are processed by three convoloution layes which allow you to exploit spatial relationshiips in images
* Each laywith will used [ELU](https://arxiv.org/pdf/1511.07289.pdf) as an activation function and one output layer that produces the Q-calue estimation for each action
* We need to watch out for previous experiences, to do this we create a replay buffer
* A replay buffer stores experience tuples while interactin with the environment and then we sample a small batch of tuple to feed the network
* This prevents the network from only learning about what it has immeadiately done
* If we train the network in sequential order, we eisk the after being influenced by correlation
* By sampling the replay buffer at random, we can break this correlation
    * This precents action values from osicllatinf or divergin catastrophically  
* We want to update our neural net weights to reduce the error  
* The error or TD error, is calculated by taking the difference between our Q_target(maximum possible value from the next state) and Q_value (our current prediction of the Q-value)  
* We do this by sampling the environment where we perform action and store the observed experience in tuples in a replay memory
* Select a small batch of tuple randomly and learn from it using a gradient decent update step  

## Improvements in Deep Q Learning
* A lot of improvements have been made in deep Q learning since 2004
    * Fixed Q-targets
    * double DQNs
    * dueling DQNs (DDQN)
    * Prioritized experience replay (PER)  

### Fixed Q-targets
* In deep Q learning, when we want to calculate the TD error (aks the loss), we caluculate the difference between the TD target (Q_target) and the current Q value (estimation of Q)
* The catch is that we don't have anr idead of the real TD target, it needs to be estimated
* Using the bellman equation, TD target is the reward of taking that action at a state plus the discounted highest Q calue for the next state
* The issue with this is that the parameters (weights) for estimating the target and Q value are the same
* This means that for every step of training, the q-value AND target value shifts (like chasing a moving target)
* To combat this, we use fixed Q-targets which what introduced by DeepMind
    * Using a seperate network with e fixed parameter(w) for estimating the TD target
    * At every Tau step, we copy the parameters from our DQN network to update the target network
    * This results in more stable learning because the target fuction stays fixed for a while  
    * Q-target implementation  
        * First create two seperate networks (DQN)
        * during the training, calculate the TD target using the target network
        * Update the target network with the DQNetwork every tau step (tau is a hyperparameter that we define)  

###  Double DQNs
* Also known as double learning, havles the problem of the overestimation of Q-values
* When normally calculating the TD target. How are we sure that the best action for the nex state is the action with the highest Q-value?
* We know tha tthe accuracy of q-values depends on what action we tried and neighboring state we explored
* Because of this, at the start of training, we don't have enough informaiton about the best action to take. Therfore, taking the max q value (which is noisey) as the best actino to take can lead to false positives.
* If non-optimal action are regularly given at a higher Q-value then the optimal best action, the learning will be complicated. 
* To fix this, we end up using two networks to decouple the action selection from the target Q value generation
    * Use out DQN network to select that is the best action to take for the next state (the action with the highest Q value)
    * use our target network to calculate the target Q value of taking the action at the next state
The Double DQN helps us reduce the overestimation of q values and as a consequence, helps us train faster and have more stable learning.  

### Dueling DQN (aka DDQN)
* Remeber that Q-calues correspond to how good it is to be at that stat and taking an action at that state Q(s,a)
* This decomposes to V(s) the calue of being at the stat and A(s,a) the advantage of taking that action at that stae (how much better it is to take this action bersus all other possible actions at that state)  
    * With DDQN we will separate the estimator of these two elements using two new streams
    * One that estimates the state value (s)
    * One that estimates the advantage for each action A(s,a)
    * The two streams will be compbined using a special aggregation lawyer to get the estimate Q(s,a)  
* We need to calculate these estimate seperately before combining because, our DDQN can learn which states are valuable without having to learn the effect of each action at each state (since it is calculating V(s))  
* A normal DQN needs to calculate the balur of each action at that state, but can be affected if all actions at a state are bad
* By decoupling, we are able to use this method for state where the actions to not affect the environment in a relavent way
    * In most states, the choice of action has no effect on what happens  
* We can force our advantage funciton estimator to have 0 advantage at the chosed action to no fall in into the isse of identifiability  
* To do that, we subtract the average advantage of all actions possible of the state
    * This helps us accelerate training
    * Becuase we can calculate the value of a state without calculating the Q(s,a) for each action be decoupling the streams  
* The only thing we need to modify for implementation is to add streams to DQN architecture  

### Proritized Experience Replay
* PER idea is that some experiences may be more important than others for our traning but might occure less frequently
* Becuase we sample batch randomly, the important experiences occur rarely and have no chance of being selected
* We use PER to try and change sampling distribution by using a criterion to define the priority of each tuple experience
* We want to take in a priority experience where there is a big difference between our prediction and the TD target since it means that we have a lot to learn about it  
    * We use the absolut value of the magnitude of our TD error and put that prioity in the experience of each replay buffer
    * Can't just do greed prioritization because it will lead to always sampleing same experiences
    * Need to use stochastic prioritization which generated the probability of being chosen for a replay
    * As a consequence, furing each timsetp, we will get a batch of samples with this probability distribution and train our network on it
    * The way we sample the experiences must match the underlying distribution they came from
    * But, becuase we priority sampling, purely random sampling is abandonded and we introduce bias toward high-priority samples
    * If we update our weights normally, we take a risk of overfitting
    * To help alleviate this, we will update our weights with only a small protion of experiences that we consider really interesting
    * To correct this bias, we use importance sampling weights (IS) that will adjust the updating by reducing the weights of the oger seen samples
    * Weights corresponding to hihg-prority samples will have little adjustment since they will be seen many times, low priority samples will have full update
    * The role of b (importance of smampling weight), is annealed up to 1 over the duration of the training, because the weights are more important in the end of learning when our q values begin to converge  
* To impliment, we can't sorts all the experience replay buffers according to proiorities since it will not be computationally efficient
* An unsorted sumtree will be used instead (binary tree with max of two children at each node)
* This resuts in the deepest nodes (leaves), contain the priority calues and a data array that points to leave containing the experiences (O(log n))  
* We then create a memory object that will contain our sumtree and data
* To sample a minibatch of size k, the range [0, to totoal priority] will be divided into k ranges (value is uniformly sampled from reach range)
* The transitions (experiences) that correspond to each of these sampled values are retrieved from the sumtree  

## Introduction to policy gradients
* In policy-based methods, instead of learning a value function that tells us what is the expected sum of rewards given a state and an action, we learn directly the policy function that maps state to action (select actions wtihout using a value function)
* This means that we directly try to optimize our policy function without worrying about the value function. We will directly parmeterize (select an action without a value funciton)
* Two types of policy based methods deterministic or stochastic
* Deterministic
    * A deterministic policy is policy that maps state to actions, You give it a state and the function returns an action to take
    * Deterministic policies are used in deterministic envronments (where the actions taken determine the outcome). Aka there is no uncertainty like moving a chess pawn from A2 to A3  
* Stochastic
    * Output a probability distribution over actions
    * Instead of being sure of taking an action a, there is prbability we will take a different one
    * This policy is used when the environment is uncertain (Partailly Observable Markov Decision Process)
    * Stochastic us used more then deterministic  

#### Three main advantages in using Policy Gradients  
* policy-based methods have better convergence properties
    * Other value based methods can have a big oscillation while training, This is because of the choice of action may change dramatically for an arbitrarily small change in the estimated action values.
    * With policy gradients, we just follow the gradient to find the best parameters, (This results in a smooth update of our policy at each step)
    * Since we use a gradient to find the best parameters, we see a smooth update for our policy at each step.
    * Because a gradient is used to find the best parameters, we are guarenteed to converge on local maxiumu (worst case), or global maxiumum (best case)  
* Policy gradients are more effective in high dimensional action spaces or when using continous actions
    * Deep Q-learning assigns a score for each possible actoin, but what is there is an infinite number of actions
    * For Policy based methods, you can adjust the parameters directly  
* Policy gradients can lean stchastic policies
    * There are two advantages though
    * We don't need to implement and exploration/exploitation trade off
    * We also don't need to worry about perceptual aliasing (when two states seem to be the same but need different actions)
    * Some disadvantages, they can converge on the local maxima rather than the global optimum
    * The can also take longer to train, since they converge slower (step by step)  

### Policy Search
* Our policy pie that a parameter theta. This pie outputs a probibilaty distribution of actions
* How do we know that the policy is good?
* We must find the best parameters (theta) to mazimize a score function J(theta)
* There are two steps to this
    * Measure the quality of a pie (policy) with a poilcy score function J(theta)
    * Use policy gradient acent to find the best parameter (theta) that improve our pie  

#### Fist Step: the policy score fucntion
* To mesure how good our policy is, we use the objective function (Policy Score Function) that calculates the expected reward of policy
* Three methods exist for optimizing policies. The choice depends only on the environment and the objectives we have
* Episodic environment
    * 