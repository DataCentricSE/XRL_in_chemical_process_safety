## Import packages
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from model_env import get_reac_params, init_params


def resilience(args):
    ## Absorption performance

    t_l, T_l, M_l, V_l, tinj_l, inj, T0 = args
    UA, _, _ = get_reac_params()
    Tj0 = init_params()[-1]
    Tmax = 393  # MAT
    Q = []  # heat duty
    for i in range(0, len(t_l)):
        Q.append(UA * (T_l[i] - Tj0))

    if max(T_l) == T_l[0]:  # if max(T) = first element of T, then Resilience=1
        Q_res = np.ones(len(t_l) - 6)
        R = 1
    else:  # if T increases
        ta_0 = np.where(t_l == min(t_l))[0][0]  # beginning of absorption phase, t = 0

        if tinj_l != 0 and tinj_l != [] and max(T_l) < Tmax:  # if injection happens, and max(T) < MAT
            ta_end = np.where(t_l >= (tinj_l[0]) + (inj / 60))[0][0]  # end of absorption phase, until injection

        else:
            # 1. if reactor runaway happens, 2. if no injection and no runaway happen,
            # 3. if no injection happen, but a runaway would happen
            ta_end = np.where(T_l == max(T_l))[0][0]  # end of absorption phase, until max(T)

        ta = t_l[ta_0:ta_end - 1]  # time range of absorption phase
        Ta = T_l[ta_0:ta_end - 1]  # temperature range of absorption phase

        # Determination of dynamic MTTF_vec (mean time to failure) and lambda_0 (baseline failure rate)
        m_abs = (np.diff(Ta) / np.diff(ta))  # slope of temperature trajectory

        if np.any(m_abs <= 0) and np.any(m_abs > 0):  # drop negative values
            m_abs[np.where(m_abs <= 0)] = m_abs[np.where(m_abs <= 0)[0][-1] - 1]
        if np.all(m_abs < 0):
            m_abs = -m_abs  # if every m_abs is a tiny negative number

        MTTF_vec = (Tmax - T_l[ta_0:ta_end - 2]) / m_abs  # MTTF
        if np.any(MTTF_vec <= 0):  # if T > Tmax so MTTF <= 0, it is replaced by the previous non-zero value
            MTTF_vec[np.where(MTTF_vec <= 0)] = MTTF_vec[np.where(MTTF_vec <= 0)[0][0] - 1]

        lambda_0 = 1 / MTTF_vec  # baseline failure rate

        # Determination of Q_abs
        Q_a = np.array(Q[ta_0:ta_end - 2])  # heat duty in absorption phase
        Qmax = UA * (Tmax - Tj0)  # maximum heat duty
        d1 = Q_a / Qmax
        lambda_abs = lambda_0 * np.exp(d1)  # failure rate

        i_abs = integrate.cumtrapz(lambda_abs, ta[0:len(d1)])  # integration in absorption phase
        Q_abs = np.exp(-i_abs)  # performance of absorption phase


        ## Recovery performance

        if ta_end == np.where(t_l == t_l[-1])[0][0]:  # if there is no recovery phase
            Q_rec = []  # performance of recovery phase

        elif max(T_l) > Tmax:  # if the reactor has run away and exploded, the performance is zero in recovery phase
            tr_end = np.where(t_l == t_l[-1])[0][0]  # end of recovery phase
            rr = (T_l[ta_end:tr_end])
            Q_rec = np.zeros(len(rr))  # performance of recovery phase

        else:
            # but if there is a recovery phase
            tr_end = np.where(t_l == t_l[-1])[0][0]  # end of recovery phase
            m_rec = (np.diff(T_l[ta_end:tr_end]) / np.diff(t_l[ta_end:tr_end]))  # slope of temperature trajectory
            if np.any(m_rec == 0):  # if the derivative is zero, it is replaced by the following non-zero value
                m_rec[np.where(m_rec == 0)[0][0]] = m_rec[np.where(m_rec == 0)[0][0] + 1]

            # Determination of dynamic MTTR_vec (mean time to repair) and mu_rec (recovery rate)
            MTTR_vec = (T_l[-1] - T_l[ta_end:tr_end - 1]) / m_rec
            if np.any(MTTR_vec == 0):  # if MTTR is zero, it is replaced by the previous non-zero value
                MTTR_vec[np.where(MTTR_vec == 0)] = MTTR_vec[np.where(MTTR_vec > 0)[0][0] - 1]

            mu_rec = 1 / MTTR_vec  # recovery rate

            # Determination of Q_rec
            t_r = t_l[ta_end:tr_end - 1]  # time range of recovery phase
            i_rec = integrate.cumtrapz(mu_rec, t_r)  # integration in recovery phase
            Q_rec = Q_abs[-1] + (1 - Q_abs[-1]) * (1 - np.exp(-i_rec))  # performance of recovery phase


        ## Resilience assessment from Q_abs and Q_rec

        Q_res = np.concatenate((Q_abs, Q_rec))  # putting together Q_abs and Q_rec

        if np.all(Q_res) > 0.99999:  # if all Q_res is close to 1
            Q_res[np.where(Q_res > 1)] = 1

        if min(Q_res) < 0.95:
            t_res = np.where(Q_res < 0.95)[0][-2]  # end of phases, if performance decreases a lot
        else:
            t_res = np.where(Q_res <= 1)[0][-1]  # end of phases, if performance does not decrease a lot

        i_Q = integrate.trapz(Q_res[:t_res], t_l[:t_res])  # integration of Q_res
        R = i_Q / (t_l[:t_res][-1] - ta_0)  # resilience assessment

    return Q_res, R


## Plot reactor temperature and performance

def plot_res(t_l, T_l, Q_res, tinj_l, R):
    plt.figure(5, figsize=(13, 5))
    plt.subplot(121)
    plt.plot(t_l[0:len(Q_res)], T_l[0:len(Q_res)] - 273, label='$t_{}$ = {:.0f} min'.format('{inj}', tinj_l[0]))
    plt.xlabel('Time (min)')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    plt.subplot(122)
    plt.plot(t_l[0:len(Q_res)], Q_res, label='R = {:.3f}'.format(R))
    plt.plot(t_l[0:len(Q_res)], np.ones(len(Q_res)), 'k--')
    plt.xlabel('Time (min)')
    plt.ylabel('Performance, Q(t)')
    plt.legend()
