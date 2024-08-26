## Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from resilience import resilience
from model_env import init_params, call_ode
from RL_utils import list_flattener
from agent import DQNAgent
from random import uniform
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from matplotlib.colors import ListedColormap
import seaborn as sns


## Define Agent
n_episodes = 100
scoresd_to_excel = pd.DataFrame([], index=range(0,n_episodes), columns=range(0,1))
best_score = -np.inf
load_checkpoint = True
n_neur = 128
ln = 1
iteration = 5

s_a = []

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

    T0 = uniform(324, 330)
    M0 = uniform(4, 5)

    I0, D0, V0, Tj0 = init_params()
    x0 = [V0, M0, I0, D0, T0, Tj0]
    t_l, V_l, M_l, I_l, D_l, T_l, Tj_l, Fh_l, tinj_l = [[0]], [[V0]], [[M0]], [[I0]], [[D0]], [[T0]], [[Tj0]], [], []

    t, V, M, I, D, T, Tj = call_ode((0, 1000 * 60), 0, x0)  # determination of baseline resilience
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

            # action = 0 if ts < 10 else 1
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

        T_inv = T_sc.inverse_transform(state[0].reshape(-1, 1))[-1] - 273
        M_inv = M_sc.inverse_transform(state[1].reshape(-1, 1))[-1]
        s_a.append((T_inv[0], M_inv[0], action))
        s_a_d = pd.DataFrame(s_a, columns=['T', 'c', 'action'])

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
scoresd_to_excel.loc[:, iteration] = scoresd.values


## Drop data for decision tree
index_label = s_a_d[s_a_d['T'] < 53].index.tolist()
s_a_d_filtered = s_a_d.drop(index_label)


## Split train and test
feature_cols = ['T', 'c']
X = s_a_d_filtered[feature_cols]
y = s_a_d_filtered.action
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

a2 = 0  # 0 - over sampling, 1 - under sampling, 2 - no change

if a2 == 0:
    # over sampling
    X_train, y_train = SMOTEENN().fit_resample(X_train, y_train)
elif a2 == 1:
    # under sampling
    rus = RandomUnderSampler(random_state=0)
    X_train, y_train = rus.fit_resample(X_train, y_train)
elif a2 == 2:
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)


## Values of alpha
if a2 == 0 or a2 == 1:
    clf = tree.DecisionTreeClassifier(random_state=0)
elif a2 == 2:
    clf = tree.DecisionTreeClassifier(random_state=0, class_weight="balanced")

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

plt.figure(11)
plt.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
plt.xlabel("Effective alpha")
plt.ylabel("Total impurity of leaves")
plt.title("Total Impurity vs effective alpha for training set")
plt.show(block=True)


## Train with effective alpha
clfs = []
for ccp_alpha in ccp_alphas:
    if a2 == 0 or a2 == 1:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    elif a2 == 2:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, class_weight='balanced')
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]

plt.figure(12)
plt.subplot(211)
plt.plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Number of nodes")
plt.title("Number of nodes vs alpha")
plt.subplot(212)
plt.plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Depth of tree")
plt.title("Depth vs alpha")
plt.tight_layout()


## Maximizing the accuracy with alpha
train_sc = [balanced_accuracy_score(y_train, clf.predict(X_train)) for clf in clfs]
test_sc = [balanced_accuracy_score(y_test, clf.predict(X_test)) for clf in clfs]

plt.figure(13)
plt.plot(ccp_alphas, train_sc, marker="o", label="train", drawstyle="steps-post")
plt.plot(ccp_alphas, test_sc, marker="o", label="test", drawstyle="steps-post")
plt.xlabel("Alpha")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


## Decision tree
if a2 == 0 or a2 == 1:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.001)
elif a2 == 2:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.03, class_weight='balanced')

clf.fit(X_train, y_train)

y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

acc_train = balanced_accuracy_score(y_train, y_pred_train)
acc_test = balanced_accuracy_score(y_test, y_pred_test)


## Plot a decision tree
# plt.figure(figsize=(12, 12))
# tree.plot_tree(clf, feature_names=feature_cols, class_names=['0', '1'])


## Visualising
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set.iloc[:, 0].min() - 1, stop=X_set.iloc[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set.iloc[:, 1].min() - 1, stop=X_set.iloc[:, 1].max() + 1, step=0.01))
plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.55, cmap=ListedColormap(colors=('cornflowerblue', 'lightcoral')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

data_v = X_set
data_v['action'] = y_set
data_f = s_a_d_filtered[s_a_d_filtered["action"]==1]
data_f.replace(to_replace=1, value=2, inplace=True)
frames = [data_v, data_f]
result = pd.concat(frames)

sns.scatterplot(data=result, x="T", y="c", hue="action", palette=(["mediumblue","tab:red","tab:brown"]), linewidth=0.3)
plt.xlabel("Temperature (°C)")
plt.ylabel("Monomer concentration (mol/L)")
plt.show()

# Modified dynamic condition (MDC)
dHr = -69919.56  # J/mol
rhocp = 1507.248  # J/L K
Ep = 29572.898  # J/mol
R = 8.314  # J/mol/K
Tj0 = 288  # K
beta = -dHr/rhocp  # KL/mol
TR = np.arange(324,348,0.1)
c = []
for T in TR:
    c.append((1/beta)/((Ep/(R*T**2))-(1/(T-Tj0))))

plt.plot(TR-273,c, label='mdc', linewidth=3, color='green')
plt.xlabel('Temperature (°C)')
plt.ylabel('Concentration (mol/L')
plt.show()

# Divergence criterion
dHr = -69919.56  # J/mol
rhocp = 1507.248  # J/L K

Ep = 29572.898  # J/mol
Ed = 123853.658  # J/mol
Et = 7008.702  # J/mol

R = 8.314  # J/mol/K
Tw = 288  # K
UA = 293.076  # W/K
V = 3000

kp0 = 1.06e7  # l/mol/s
kd0 = 5.95e13  # 1/s
kt0 = 1.25e9  # l/mol/s

beta = -dHr/rhocp  # KL/mol
alpha = UA/(V*rhocp)

TR = np.arange(324,348,0.1)
f = 0.6
I = 0.294
cm = []

for T in TR:
    kd = kd0 * np.exp(-Ed / (R * T))  # 1/s
    kt = kt0 * np.exp(-Et / (R * T))  # l/mol/s
    a1 = (2 * f * I * kd) / kt
    lambda0 = np.sqrt(a1)  # mol/l
    cm.append((alpha+kp0*np.exp(-Ep/(R*T))*lambda0)/(beta*kp0*lambda0*np.exp(-Ep/(R*T))*(Ep/(R*T**2))))

plt.plot(TR-273,cm, label='Div', linewidth=3, color='orange')
plt.xlabel('Temperature (°C)')
plt.ylabel('Concentration (mol/L)')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ['0 (no action)', '1 (synthetic action)', '1 (original action)', 'Modified Dynamic Condition',
                     'Divergence criterion'], loc='upper right')

