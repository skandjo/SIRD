import numpy as np
import matplotlib.pyplot as plt

def euler_sird(beta, gamma, mu, S0, I0, R0, D0, dt, days):
    # Initialisation des variables
    steps = int(days / dt)
    S, I, R, D = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)
    
    # Conditions initiales
    S[0], I[0], R[0], D[0] = S0, I0, R0, D0

    # Boucle de simulation
    for t in range(steps - 1):
        S[t + 1] = S[t] - dt * beta * S[t] * I[t]
        I[t + 1] = I[t] + dt * (beta * S[t] * I[t] - gamma * I[t] - mu * I[t])
        R[t + 1] = R[t] + dt * gamma * I[t]
        D[t + 1] = D[t] + dt * mu * I[t]

    return S, I, R, D

# Paramètres du modèle
beta = 0.5  # Taux de transmission
gamma = 0.15  # Taux de récupération
mu = 0.015  # Taux de mortalité
S0, I0, R0, D0 = 0.99, 0.01, 0.0, 0.0  # Conditions initiales
dt = 0.01  # Pas de temps
days = 90  # Nombre de jours simulés
# Simulation
S, I, R, D = euler_sird(beta, gamma, mu, S0, I0, R0, D0, dt, days)

# Temps
time = np.linspace(0, days, int(days / dt))

# Tracer les courbes
plt.figure(figsize=(10, 6))
plt.plot(time, S, label='Susceptibles (S)')
plt.plot(time, I, label='Infectés (I)')
plt.plot(time, R, label='Rétablis (R)')
plt.plot(time, D, label='Décédés (D)')
plt.xlabel('Temps (jours)')
plt.ylabel('Proportion de la population')
plt.title('Dynamique SIRD')
plt.legend()
plt.grid()
plt.show()
