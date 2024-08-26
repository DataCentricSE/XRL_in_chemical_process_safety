## Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from resilience import resilience
from model_env import init_params, call_ode
from RL_utils import list_flattener
from agent import DQNAgent
from sklearn.preprocessing import MinMaxScaler
import shap
import torch


## Define Agent
n_episodes = 1
best_score = -np.inf
load_checkpoint = True
n_neur = 128
ln = 1
iteration = 5

s_a = []
s_a2 = []

t_trajectory = []
T_trajectory = []
M_trajectory = []
tinj_tr = []

agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001, input_dims=2, n_actions=2, mem_size=2000, eps_min=0,
                 batch_size=32, ln=ln, n_neuron=n_neur, replace=100, eps_dec=1 / (n_episodes * 0.95), chkpt_dir='',
                 algo='DQNAgent_' + str(ln) + '_' + str(n_neur) + '_' + str(iteration), target_tau=1e-4)

scores = []
win_pct_list = []
T_l2 = []


## Define variables for MinMaxScaler
T_m = []
dT1_m = []
M_m = []
T0 = 330
M0 = 8.6981 / 2
I0, D0, V0, Tj0 = init_params()
x0 = [V0, M0, I0, D0, T0, Tj0]
t, V, M, I, D, T, Tj = call_ode((0, 60000), 0, x0)
dT1 = np.diff(T) / np.diff(t)
T_m.append(T)
dT1_m.append(dT1)
M_m.append(M[:10000])

T0 = 324
M0 = 8.6981 / 2
I0, D0, V0, Tj0 = init_params()
x0 = [V0, M0, I0, D0, T0, Tj0]
t, V, M, I, D, T, Tj = call_ode((0, 60000), 0, x0)
dT1 = np.diff(T) / np.diff(t)
T_m.append(T)
dT1_m.append(dT1)
M_m.append(M)

T_m = list_flattener(T_m)
T_m = np.array(T_m)
dT1_m = list_flattener(dT1_m)
dT1_m = np.array(dT1_m)
M_m = list_flattener(M_m)
M_m = np.array(M_m)

T_sc = MinMaxScaler(feature_range=(0, 1))
dT1_sc = MinMaxScaler(feature_range=(-1, 1))
M_sc = MinMaxScaler(feature_range=(0, 1))

T_sc.fit(T_m.reshape(-1, 1))
dT1_sc.fit(dT1_m.reshape(-1, 1))
M_sc.fit(M_m.reshape(-1, 1))


## Test Agent
for episode in range(0, n_episodes):

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = 0

    # T0 = uniform(324, 330)
    # M0 = uniform(4, 5)

    T0 = 327  #
    M0 = 4.3  # 4.4 or 4.3

    I0, D0, V0, Tj0 = init_params()
    x0 = [V0, M0, I0, D0, T0, Tj0]
    t_l, V_l, M_l, I_l, D_l, T_l, Tj_l, Fh_l, tinj_l = [[0]], [[V0]], [[M0]], [[I0]], [[D0]], [[T0]], [[Tj0]], [], []

    t, V, M, I, D, T, Tj = call_ode((0, 1000 * 60), 0, x0)  # define Rb R_baseline

    args = t, T, M, V, 0, 0, T0
    Q_res, Rb = resilience(args)

    Fh = 0
    inj = 60  # how many seconds to inject
    ts, te = 0, 1

    a1 = 0  # 0 if there has not yet been an injection, an auxiliary variable
    s_a_m = []  # we collect the states and actions here

    while True:
        tspan = (ts * 60, (te * 60) + 1)
        t, V, M, I, D, T, Tj = call_ode(tspan, Fh, x0)

        t_l.append(t[1:])
        M_l.append(M[1:])
        T_l.append(T[1:])
        V_l.append(V[1:])

        x0 = [V[-1], M[-1], I[-1], D[-1], T[-1], Tj[-1]]
        dT1 = (np.diff(T_l[-1]) / np.diff(t_l[-1]))

        if a1 == 0:  # watch at 1 minute

            # agent decides
            T_act_sc = T_sc.transform(T_l[-1].reshape(-1, 1))[-1]
            dT1_act_sc = dT1_sc.transform(dT1.reshape(-1, 1))[-1]
            M_act_sc = M_sc.transform(M_l[-1].reshape(-1, 1))[-1]
            state = (T_act_sc[0], M_act_sc[0])

            action = agent.choose_action(state)

            # action = 0
            if action == 0:  # if the agent does not intervene (action=0), no Fh is added
                Fh = 0
            else:  # if the agent intervenes, add Fh and increase the auxiliary variable to 1
                Fh = 14
                a1 += 1

                tinj_l.append(te)  # we collect these variables for plottingClick to apply
                Fh_l.append(Fh)
                tinj_tr.append(tinj_l)

            ts += 1
            te += 1

            s_a_m.append([state, action])

        else:  # here a1=1, so the agent has already intervened
            print('Beavatkozott: ' + str(tinj_l) + 'epizód: ' + str(episode) + 'Rb: ' + str(Rb) + 'Max_T: ' + str(
                max(list_flattener(T_l))))
            Fh = 0
            ts += 1
            te = 1000

        if t_l[-1][-1] >= 999.9:
            break

    if action == 0:
        print(
            'Nem avatkozott be, epizód: ' + str(episode) + 'Rb: ' + str(Rb) + 'Max_T: ' + str(max(list_flattener(T_l))))
    t_l = list_flattener(t_l)
    t_l = np.array(t_l)
    M_l = list_flattener(M_l)
    M_l = np.array(M_l)
    T_l = list_flattener(T_l)
    T_l = np.array(T_l)
    V_l = list_flattener(V_l)
    V_l = np.array(V_l)

    t_trajectory.append(t_l)
    T_trajectory.append(T_l)
    M_trajectory.append(M_l)

    args = t_l, T_l, M_l, V_l, tinj_l, inj, T0
    Q_res, R = resilience(args)

    for i in range(0, len(s_a_m)):
        if i == len(s_a_m) - 1:
            if len(s_a_m) >= 1000:
                s_a_m[i].append((T_sc.transform(T_l.reshape(-1, 1))[-1][0],
                                 M_sc.transform(M_l.reshape(-1, 1))[-1][0]))
                s_a_m[i].append(R)
            else:
                s_a_m[i].append((T_sc.transform(T_l.reshape(-1, 1))[(len(s_a_m) + 1) * 60][0],
                                 M_sc.transform(M_l.reshape(-1, 1))[(len(s_a_m) + 1) * 60][0]))
                s_a_m[i].append(R)
        else:
            s_a_m[i].append(s_a_m[i + 1][0])
            s_a_m[i].append(R)

    for i in range(0, len(s_a_m)):
        state = s_a_m[i][0]
        action = s_a_m[i][1]

        s_a2.append((state[0], state[1], action))
        s_a_d = pd.DataFrame(s_a2, columns=['T', 'c', 'action'])

        reward = s_a_m[i][3]
        if Rb > 0.99 and action == 1:  # if the resilience is good, but the agent still intervenes
            reward = -5 * (1 / tinj_l[0])
        if Rb > 0.99 and action == 0:
            reward = 1

        new_state = s_a_m[i][2]

    if not load_checkpoint:
        agent.store_transition(state, action, reward, new_state)
        agent.learn()

    agent.decrement_epsilon()
    scores.append(reward)

    mean_scores = np.mean(scores[-50:])
    win_pct_list.append(mean_scores)
    if episode % 50 == 0:
        if episode % 50 == 0:
            print('episode', episode, 'reward-mean %.4f' % mean_scores, 'epsilon %.2f' % agent.epsilon)

    if mean_scores > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = mean_scores

scoresd = pd.Series(scores)


## SHAP
state_t = s_a_d.iloc[:, 0:2].values
action_t = s_a_d.iloc[:, -1].values

st = []
for j in state_t:
    state_tensor = torch.tensor([j], dtype=torch.float).to(agent.q_eval.device)
    q2 = agent.q_eval.forward(state_tensor)[0][1].detach().numpy()
    q1 = agent.q_eval.forward(state_tensor)[0][0].detach().numpy()
    st.append(q2 - q1)

def model_for_agent(s):
    st = []
    for j in s:
        state_tensor = torch.tensor([j], dtype=torch.float).to(agent.q_eval.device)
        q2 = agent.q_eval.forward(state_tensor)[0][1].detach().numpy()
        q1 = agent.q_eval.forward(state_tensor)[0][0].detach().numpy()
        st.append(q2-q1)
    return np.array(st)

explainer = shap.KernelExplainer(model_for_agent, state_t)
shap_values = explainer.shap_values(state_t)


## Plot the shapley values
state_T = T_sc.inverse_transform(state_t[:,0].reshape((-1,1)))
state_M = M_sc.inverse_transform(state_t[:,1].reshape((-1,1)))

plt.figure(42)
plt.plot(state_T-273, state_M)
if M0 == 4.4:
    plt.plot(54, 4.4, marker=".", color="tab:blue", markersize=10, label="$T_{0}=54 °C$, $M_{0}=4.4$ $mol/L$")
elif M0 == 4.3:
    plt.plot(54, 4.3, marker=".", color="tab:blue", markersize=10, label="$T_{0}=54 °C$, $M_{0}=4.3$ $mol/L$")
plt.xlabel('Temperature (°C)')
plt.ylabel('Monomer concentration (mol/L)')
plt.legend()
plt.show()


## shapley f(x)
plt.figure(24)
x = np.arange(0,len(shap_values))
y1 = st-shap_values[:,0]
y2 = st-shap_values[:,1]
y3 = st
plt.plot(y3, color='black', alpha=0.5)
plt.fill_between(x, y1, y3, where=(y1>y3), color='cornflowerblue', alpha=0.8)
plt.fill_between(x, y1, y3, where=(y1<y3), color='red', alpha=0.5)
plt.fill_between(x, y2, y3, where=(y2>y3), color='cornflowerblue', alpha=0.8)
plt.fill_between(x, y2, y3, where=(y2<y3), color='red', alpha=0.5)
plt.xlabel('Time (min)')
plt.ylabel('Output value')
if M0 == 4.3:
    plt.text(170, -4, "Temperature", ha='center', va='center',color='black', alpha=0.8)
    plt.text(820, -8, "Temperature", ha='center', va='center', color='black', alpha=0.8)
elif M0 == 4.4:
    plt.text(150, -1.6, "Monomer\nconcentration", ha='center', va='center',color='black', alpha=0.8)
    plt.text(670, -0.1, "Monomer\nconcentration", ha='center', va='center',color='black', alpha=0.8)
    plt.text(150, -0.4, "Temperature", ha='center', va='center', color='black', alpha=0.8)
    plt.text(670, -1.5, "Temperature", ha='center', va='center', color='black', alpha=0.8)
plt.legend(['$Q(s,a_{2})-Q(s,a_{1})$'], loc='upper right')


