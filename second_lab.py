import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ----------- Функція для моделі SIR -----------------
def sir_model(beta, gamma, S0, I0, R0, N, t_max=160, dt=0.1):
    """
    Простенька реалізація моделі SIR з методом Рунге-Кутти 4-го порядку (RK4)

    beta  - коефіцієнт зараження (ймовірність передачі)
    gamma - коефіцієнт одужання (1/середня тривалість хвороби)
    S0, I0, R0 - початкові значення (здорові, інфіковані, одужалі)
    N - загальна кількість населення
    t_max - тривалість моделювання (у днях)
    dt - крок інтегрування
    """

    # створюємо часовий масив
    t = np.arange(0, t_max, dt)

    # ініціалізуємо порожні масиви для результатів
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))

    # задаємо початкові умови
    S[0] = S0
    I[0] = I0
    R[0] = R0

    # Основний цикл інтегрування
    for i in range(1, len(t)):
        # поточні значення
        s, i_, r = S[i - 1], I[i - 1], R[i - 1]

        # визначаємо диференціальні рівняння
        def dSdt(s, i_): return -beta * s * i_ / N

        def dIdt(s, i_, r): return beta * s * i_ / N - gamma * i_

        def dRdt(i_): return gamma * i_

        # метод Рунге-Кутти 4-го порядку (точніший за Ейлера)
        k1_S = dSdt(s, i_)
        k1_I = dIdt(s, i_, r)
        k1_R = dRdt(i_)

        k2_S = dSdt(s + 0.5 * k1_S * dt, i_ + 0.5 * k1_I * dt)
        k2_I = dIdt(s + 0.5 * k1_S * dt, i_ + 0.5 * k1_I * dt, r + 0.5 * k1_R * dt)
        k2_R = dRdt(i_ + 0.5 * k1_I * dt)

        k3_S = dSdt(s + 0.5 * k2_S * dt, i_ + 0.5 * k2_I * dt)
        k3_I = dIdt(s + 0.5 * k2_S * dt, i_ + 0.5 * k2_I * dt, r + 0.5 * k2_R * dt)
        k3_R = dRdt(i_ + 0.5 * k2_I * dt)

        k4_S = dSdt(s + k3_S * dt, i_ + k3_I * dt)
        k4_I = dIdt(s + k3_S * dt, i_ + k3_I * dt, r + k3_R * dt)
        k4_R = dRdt(i_ + k3_I * dt)

        # обчислюємо нові значення
        S[i] = s + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        I[i] = i_ + (dt / 6) * (k1_I + 2 * k2_I + 2 * k3_I + k4_I)
        R[i] = r + (dt / 6) * (k1_R + 2 * k2_R + 2 * k3_R + k4_R)

    return t, S, I, R


# ----------- Запуск експериментів -----------------

N = 1000  # населення
I0 = 1  # початкова кількість інфікованих
R0 = 0  # початкова кількість одужалих
S0 = N - I0 - R0

# набір параметрів для експерименту (варіюємо β)
betas = [0.1, 0.2, 0.3, 0.5, 0.8]
gamma = 0.1  # фіксоване γ

plt.figure(figsize=(10, 6))
for beta in betas:
    t, S, I, R = sir_model(beta, gamma, S0, I0, R0, N)
    plt.plot(t, I, label=f'β={beta}')

plt.title("Залежність кількості інфікованих I(t) від часу при різних β")
plt.xlabel("Час (дні)")
plt.ylabel("Кількість інфікованих")
plt.legend()
plt.grid(True)
plt.show()

# ----------- Інший експеримент -----------------
# міняємо γ, а β фіксоване
gammas = [0.05, 0.1, 0.2, 0.3]
beta = 0.3

plt.figure(figsize=(10, 6))
for gamma in gammas:
    t, S, I, R = sir_model(beta, gamma, S0, I0, R0, N)
    plt.plot(t, I, label=f'γ={gamma}')

plt.title("Вплив коефіцієнта одужання γ на динаміку інфікованих")
plt.xlabel("Час (дні)")
plt.ylabel("Кількість інфікованих")
plt.legend()
plt.grid(True)
plt.show()

# ----------- Побудова таблиці підсумків -----------------
results = []
for beta in betas:
    t, S, I, R = sir_model(beta, gamma, S0, I0, R0, N)
    Imax = np.max(I)
    t_peak = t[np.argmax(I)]
    R_final = R[-1]
    R0_val = beta * S0 / gamma
    results.append([beta, gamma, round(R0_val, 2), round(Imax, 2), round(t_peak, 2), round(R_final, 2)])

df = pd.DataFrame(results, columns=["β", "γ", "R₀", "Iмакс", "Час піку (дні)", "R на кінці"])
print(df)

# ----------- Діаграма -----------------
plt.figure(figsize=(8, 5))
plt.bar([str(b) for b in betas], [np.max(sir_model(b, gamma, S0, I0, R0, N)[2]) for b in betas])
plt.title("Пікові значення інфікованих Imax для різних β")
plt.xlabel("β (ймовірність зараження)")
plt.ylabel("Максимум I")
plt.grid(True, axis='y')
plt.show()
