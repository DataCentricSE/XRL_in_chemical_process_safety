## Import packages
import numpy as np
import scipy


## Model vars and equations

# Initial parameters
def init_params():
    # M0 = 8.6981 / 2  # mol/l
    I0 = 0.5888 / 2  # mol/l
    D0 = 0  # mol/l
    V0 = 3000  # l
    Tj0 = 288  # K

    return I0, D0, V0, Tj0


# Parameters of styrene polymerization
def get_params():
    f = 0.6

    kd0 = 5.95e13  # 1/s
    kp0 = 1.06e7  # l/mol/s
    kt0 = 1.25e9  # l/mol/s

    Ed = 123853.658  # J/mol
    Ep = 29572.898  # J/mol
    Et = 7008.702  # J/mol

    R = 8.314  # J/mol/K
    dHr = -69919.56  # J/mol
    rhocp = 1507.248  # J/l/K
    rhocpj = 4045.7048  # J/l/K

    rhocph = 866 * 1.717  # J/l/K
    Thin = 288  # K

    return f, kd0, kp0, kt0, Ed, Ep, Et, R, dHr, rhocp, rhocpj, rhocph, Thin


def get_reac_params():
    UA = 293.076  # W/K
    Vj = 3312.4  # l
    Fj = 0.131  # l/s

    return UA, Vj, Fj


# Model equations of styrene polymerization
def intsys(t, x, args):
    Tj0 = init_params()[-1]
    f, kd0, kp0, kt0, Ed, Ep, Et, R, dHr, rhocp, rhocpj, rhocph, Thin = get_params()
    UA, Vj, Fj = get_reac_params()

    V, M, I, D, T, Tj = x
    kd = kd0 * np.exp(-Ed / (R * T))  # 1/s
    kp = kp0 * np.exp(-Ep / (R * T))  # l/mol/s
    kt = kt0 * np.exp(-Et / (R * T))  # l/mol/s

    Fh = args

    a1 = (2 * f * I * kd) / kt
    if a1 >= 0:
        lambda0 = np.sqrt(a1)  # mol/l
    else:
        lambda0 = 0

    dVdt = Fh  # l/s
    dMdt = -kp * M * lambda0  # mol/l/s
    dIdt = -kd * I  # mol/l/s
    dDdt = kt * lambda0 ** 2  # mol/l/s
    dTdt = (UA) / (rhocp * (V)) * (Tj - T) + ((-dHr) / (rhocp)) * kp * M * lambda0 +\
           Fh * rhocph * (Thin - T) / ((V) * rhocp)  # K/s
    dTjdt = (Fj / Vj) * (Tj0 - Tj) - (UA) / (rhocpj * Vj) * (Tj - T)  # K/s

    return dVdt, dMdt, dIdt, dDdt, dTdt, dTjdt


def call_ode(tspan, Fh, x0):
    t_eval = np.arange(tspan[0], tspan[1])
    args = tuple([Fh])
    ode = scipy.integrate.solve_ivp(intsys, tspan, t_eval=t_eval, y0=x0, method='BDF', args=args)

    t = ode.t / 60
    V = ode.y[0]
    M = ode.y[1]
    I = ode.y[2]
    D = ode.y[3]
    T = ode.y[4]
    Tj = ode.y[5]

    return t, V, M, I, D, T, Tj
