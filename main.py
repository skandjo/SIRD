import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from tqdm import tqdm

###############
### ETAPE 1 ###
###############
# Discrétisation pour S

# Formulation discrète des dérivées
# S'(X[i]) = (S(X[i+1]) - S(X[i])) / step
# S'(X[i]) = -beta * S(X[i]) * I(X[i])
# (S(X[i+1]) - S(X[i])) / step = -beta * S(X[i]) * I(X[i])
# S(X[i+1]) = S(X[i]) - step * beta * S(X[i]) * I(X[i])
# I(X[i+1]) = I(X[i]) + step * (beta * S(X[i]) * I(X[i]) - gamma * I(X[i]) - mu * I(X[i]))
# R(X[i+1]) = R(X[i]) + step * gamma * I(X[i])
# D(X[i+1]) = D(X[i]) + step * mu * I(X[i])

# Fonction pour la méthode d'Euler
def euler_sird(params, S0, I0, R0, D0, step, days):
    beta, gamma, mu = params
    T = np.arange(0, days, step)  # Ajout de dt pour inclure le dernier point
    S, I, R, D = np.zeros(len(T)), np.zeros(len(T)), np.zeros(len(T)), np.zeros(len(T))

    # Conditions initiales
    S[0], I[0], R[0], D[0] = S0, I0, R0, D0

    # Boucle de simulation pour S, I, R, D
    for index in range(len(T) - 1):
        S[index + 1] = S[index] - step * beta * S[index] * I[index]
        I[index + 1] = I[index] + step * (beta * S[index] * I[index] - gamma * I[index] - mu * I[index])
        R[index + 1] = R[index] + step * gamma * I[index]
        D[index + 1] = D[index] + step * mu * I[index]
    
    return T, S, I, R, D


###############
### ETAPE 2 ###
###############
# Paramètres du modèle
params = (0.5, 0.15, 0.015)
S0, I0, R0, D0 = 0.99, 0.01, 0.0, 0.0  # Conditions initiales
step = 0.01  # Pas de temps
days = 90  # Nombre de jours simulés

# Fonction de tracé
def graph(params, S0, I0, R0, D0, dt, days):
    T, S, I, R, D = euler_sird(params, S0, I0, R0, D0, dt, days)
    plt.figure(figsize=(10, 6))
    plt.plot(T, S, label="S(t): Susceptibles")
    plt.plot(T, I, label="I(t): Infectés")
    plt.plot(T, R, label="R(t): Récupérés")
    plt.plot(T, D, label="D(t): Décès")
    plt.title("Évolution du modèle SIRD")
    plt.xlabel("Temps (jours)")
    plt.ylabel("Proportions")
    plt.legend()
    plt.grid()
    plt.show()


graph(params, S0, I0, R0, D0, step, days)


###############
### ETAPE 3 ###
###############
# Chargement des données du fichier CSV
data_frame = pd.read_csv('sird_dataset.csv')
time_data = data_frame['Jour']
S_data = data_frame['Susceptibles']
I_data = data_frame['Infectés']
R_data = data_frame['Rétablis']
D_data = data_frame['Décès']

# Fonction de coût (MSE)
def cost_function(params, S_data, I_data, R_data, D_data, step, days):
    _, S, I, R, D = euler_sird(params, S_data[0], I_data[0], R_data[0], D_data[0], step, days)
    S, I, R, D = S[::100], I[::100], R[::100], D[::100]
    mse = np.mean((S - S_data)**2 + (I - I_data)**2 + (R - R_data)**2 + (D - D_data)**2)
    return mse

# Grid Search pour optimiser les paramètres
beta_values = np.linspace(0.25, 0.5, 10)
gamma_values = np.linspace(0.08, 0.15, 10)
mu_values = np.linspace(0.005, 0.015, 10)

# Création des combinaisons de paramètres
param_combinations = product(beta_values, gamma_values, mu_values)
best_params = None
best_cost = float('inf')

# Recherche sur la grille avec barre de progression
for beta, gamma, mu in tqdm(param_combinations, total=len(beta_values) * len(gamma_values) * len(mu_values), desc="Recherche Grid"):
    cost = cost_function((beta, gamma, mu), S_data, I_data, R_data, D_data, 0.01, 90)
    if cost < best_cost:
        best_cost = cost
        best_params = [beta, gamma, mu]

print(f"Meilleurs paramètres : {best_params}")

# Simulation avec les meilleurs paramètres
T, S_opt, I_opt, R_opt, D_opt = euler_sird(best_params, S0, I0, R0, D0, 0.01, 90)
S_opt, I_opt, R_opt, D_opt = S_opt[::100], I_opt[::100], R_opt[::100], D_opt[::100]

# Tracé des résultats optimaux
plt.figure(figsize=(10, 6))
plt.plot(time_data, S_opt, label='S Modèle')
plt.plot(time_data, I_opt, label='I Modèle')
plt.plot(time_data, R_opt, label='R Modèle')
plt.plot(time_data, D_opt, label='D Modèle')
plt.plot(time_data, S_data, label='S Empirique', linestyle='--', color='blue')
plt.plot(time_data, I_data, label='I Empirique', linestyle='--', color='red')
plt.plot(time_data, R_data, label='R Empirique', linestyle='--', color='green')
plt.plot(time_data, D_data, label='D Empirique', linestyle='--', color='black')
plt.legend()
plt.grid()
plt.show()


###############
### ETAPE 4 ###
###############
# Scénario sans intervention
T, S_no, I_no, R_no, D_no = euler_sird(best_params, S0, I0, R0, D0, 0.01, 90)

# Scénario avec intervention (réduction de beta)
beta_reduced = best_params[0] / 2
best_params[0] = beta_reduced

T, S_inter, I_inter, R_inter, D_inter = euler_sird(best_params, S0, I0, R0, D0, 0.01, 90)

# Comparaison des deux scénarios
plt.figure(figsize=(10, 6))
plt.plot(T, I_no, label='Infectés sans intervention')
plt.plot(T, I_inter, label='Infectés avec intervention')
plt.xlabel('Temps (jours)')
plt.ylabel('Proportion de la population infectée')
plt.title('Impact de la réduction de β sur l’épidémie')
plt.legend()
plt.grid()
plt.show()
