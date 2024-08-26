**Resilience-based explainable reinforcement learning in chemical process safety**

For future applications of artificial intelligence, namely reinforcement learning (RL), we develop a resilience-based explainable RL agent to make decisions about the activation of mitigation systems.
The applied reinforcement learning algorithm is Deep Q-learning and the reward function is resilience. We investigate two explainable reinforcement learning methods, which are the decision tree, as a policy-explaining method, and the Shapley value
as a state-explaining method.

The policy can be visualized in the agent's state space using a decision tree for better understanding. We compare the agent's decision boundary with the runaway boundaries defined by runaway criteria, namely the divergence criterion and modified dynamic condition.
Figure 1 shows the decision boundaries of the decision tree and the two critical curves. The class in which no action is taken, which means that the agent does not intervene, is shown in blue, and the classes where the agent intervenes are the synthetic
and original actions. are shown in red and brown, respectively. The actions taken by the agent are the original actions and the actions sampled by SMOTEENN are the synthetic actions, and the policy of the RL agent is split in the decision tree.
The critical curve of the divergence criterion is coloured orange, and the critical curve of the modified dynamic condition criteria is coloured green. With the specific settings of the intervention system, the runaway limit defined by the divergence criterion
describes quite well the policy of the RL agent. The modified dynamic condition is more strict from this context, however, it follows that in this specific system the modified dynamic condition can be used as an early warning detection system, while the RL agent
can be used to make decisions in the intervention policy.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1976bc8f-212c-4f7d-a356-6f58ae0b738b" alt="Figure_dt (3)">
  
  Figure 1: The performed actions by the agent at specific concentration-temperature states compared to the critical curves of runaway criteria
</div>

Shapley value explains the contribution of the state variables on the behavior of the agent over time. We define the state contributions to a decision using Shapley values, where we explain the difference between the two output of the neurons,
so the difference between the Q-values in a given state for performing the first (no intervention) and second action (intervention). In other words, we calculate how much more it is worth for us to intervene than not, based on the future cumulative rewards of the actions.
Figure 2 presents the reactor operation in the concentration-temperature phase space on the left and presents the difference of the Q-values over time with the corresponding Shapley values. The reactor temperature increases and the monomer concentration decreases over time.
Until 400 minutes, the monomer concentration has a positive effect on performing the intervention, while based on the temperature values, intervention is not necessary. After 400 minutes based on the temperature values, we should intervene,
but the decreasing monomer concentration can compensate for this for up to 800 minutes.

<div align="center">
<div style="display: flex; justify-content: space-around; align-items: center;">

  <figure style="margin: 15px; text-align: center;">
    <img src="https://github.com/user-attachments/assets/fe98f303-eddd-42d7-a4eb-8382ac1f7ff2" alt="Figure 1" style="width: 400px;">
  </figure>

  <figure style="margin: 15px; text-align: center;">
    <img src="https://github.com/user-attachments/assets/fe085fd3-2976-4ba9-9041-272134f72b34" alt="Figure 2" style="width: 400px;">
  </figure>

  Figure 2: The monomer concentration-temperature phase space is shown in the left figure with the initial temperature and concentration, and the model output is described on the right figure.
  The right figure shows how the temperature and monomer concentration affect the model output, if they are in blue, it decreases the output value, and if they are in red, it increases the output value.

</div>
</div>

The results show that the decisions of the artificial agent in a resilience-based mitigation system can be explained and can be presented in a transparent way.
