import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import pandas as pd


class AnalyticalMM4Model:
    """
    Аналітична модель системи M/M/4 з необмеженою чергою
    """

    def __init__(self, X, mu):
        """
        Ініціалізація параметрів моделі

        Args:
            X: Середній інтервал між надходженням партій (хв)
            mu: Інтенсивність обслуговування однієї дільниці (партій/хв)
        """
        self.X = X
        self.mu = mu
        self.lambd = 1.0 / X  # Інтенсивність надходження
        self.n = 4  # Кількість дільниць

    def calculate_metrics(self):
        """Розрахунок основних показників ефективності системи"""

        # Коефіцієнт завантаження
        rho = self.lambd / (self.n * self.mu)

        # Перевірка умови стаціонарності
        if rho >= 1:
            print("УВАГА: Система нестабільна (ρ ≥ 1)")
            return None

        # Ймовірність того, що система порожня (P0)
        sum_part = 0
        for k in range(self.n):
            sum_part += (self.lambd / self.mu) ** k / factorial(k)

        p0 = 1.0 / (sum_part + ((self.lambd / self.mu) ** self.n /
                                (factorial(self.n) * (1 - rho))))

        # Ймовірність того, що всі дільниці зайняті
        p_all_busy = ((self.lambd / self.mu) ** self.n /
                      (factorial(self.n) * (1 - rho))) * p0

        # Середня кількість партій у черзі
        Lq = p_all_busy * rho / (1 - rho)

        # Середня кількість партій у системі
        Ls = Lq + (self.lambd / self.mu)

        # Середній час очікування в черзі
        Wq = Lq / self.lambd

        # Середній час перебування в системі
        Ws = Wq + (1.0 / self.mu)

        # Розподіл ймовірностей станів
        states_prob = self.calculate_state_probabilities(p0)

        return {
            'rho': rho,
            'p0': p0,
            'p_all_busy': p_all_busy,
            'Lq': Lq,  # Середня довжина черги
            'Ls': Ls,  # Середня кількість у системі
            'Wq': Wq,  # Середній час в черзі
            'Ws': Ws,  # Середній час в системі
            'states_prob': states_prob
        }

    def calculate_state_probabilities(self, p0):
        """Розрахунок ймовірностей кожного стану системи"""
        states_prob = [p0]  # P0

        for k in range(1, self.n + 1):
            pk = ((self.lambd / self.mu) ** k / factorial(k)) * p0
            states_prob.append(pk)

        # Для станів k > n
        k = self.n + 1
        while True:
            pk = states_prob[self.n] * ((self.lambd / (self.n * self.mu)) ** (k - self.n))
            if pk < 1e-6:  # Критерій зупинки
                break
            states_prob.append(pk)
            k += 1

        return states_prob

    def find_max_X_for_queue(self, target_queue=20, mu_estimate=1 / 45):
        """
        Пошук максимального X, при якому середня черга не перевищує задане значення

        Args:
            target_queue: Максимальна допустима середня довжина черги
            mu_estimate: Оцінка інтенсивності обслуговування
        """
        print("Пошук оптимального X аналітичним методом...")
        print(f"Ціль: Lq ≤ {target_queue} партій")

        # Метод бісекції
        low = 5.0
        high = 120.0
        tolerance = 0.1

        iteration = 0
        Lq = 0
        while high - low > tolerance and iteration < 20:
            mid = (low + high) / 2
            self.X = mid
            self.lambd = 1.0 / mid

            results = self.calculate_metrics()

            if results is None:
                # Система нестабільна - збільшуємо X
                low = mid
            else:
                Lq = results['Lq']

                if Lq <= target_queue:
                    high = mid  # Можна зменшити інтервал
                else:
                    low = mid  # Потрібно збільшити інтервал

            iteration += 1
            print(f"Ітерація {iteration}: X = {mid:.2f}, Lq = {Lq:.2f}")

        optimal_X = (low + high) / 2
        return optimal_X


def estimate_processing_time():
    """
    Оцінка середнього часу обробки партії для розрахунку mu
    """
    print("\nОцінка часу обробки партії:")

    # Детерміновані складові
    pre_processing = 5  # хв (для 2 з 4 деталей)
    assembly = 32  # хв (4 деталі по 8 хв)
    regulation = 8  # хв (середній час)

    # Ймовірнісні складові
    base_time = pre_processing + assembly + regulation

    # Усереднення з урахуванням ймовірностей
    # P(брак) = 0.01 → час = 0
    # P(заміна) = 0.02 → час = base_time + 3
    # P(норма) = 0.97 → час = base_time

    avg_time = (0.01 * 0 + 0.02 * (base_time + 3) + 0.97 * base_time)

    print(f"  Попередня обробка: {pre_processing} хв")
    print(f"  Складання: {assembly} хв")
    print(f"  Регулювання: {regulation} хв")
    print(f"  Базова обробка: {base_time} хв")
    print(f"  Усереднений час: {avg_time:.2f} хв")

    mu = 1.0 / avg_time  # Інтенсивність обслуговування
    print(f"  Інтенсивність обслуговування μ: {mu:.4f} партій/хв")

    return mu


def plot_analytical_results(X_values, results_list):
    """Побудова графіків для аналітичної моделі"""

    # Створення DataFrame для зручності
    df = pd.DataFrame(results_list)

    # 1. Графік залежності довжини черги від X
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(df['X'], df['Lq'], 'bo-', linewidth=2, markersize=6)
    plt.axhline(y=20, color='red', linestyle='--', label='Ліміт черги (20)')
    plt.xlabel('Інтервал надходження партій X (хв)')
    plt.ylabel('Середня довжина черги Lq')
    plt.title('Залежність середньої довжини черги від інтервалу X')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Графік завантаження системи
    plt.subplot(2, 2, 2)
    plt.plot(df['X'], df['ρ'], 'ro-', linewidth=2, markersize=6)
    plt.axhline(y=1, color='red', linestyle='--', label='Границя стабільності')
    plt.xlabel('Інтервал надходження партій X (хв)')
    plt.ylabel('Коефіцієнт завантаження ρ')
    plt.title('Завантаження системи')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. Графік часу очікування
    plt.subplot(2, 2, 3)
    plt.plot(df['X'], df['Wq'], 'go-', linewidth=2, markersize=6)
    plt.xlabel('Інтервал надходження партій X (хв)')
    plt.ylabel('Середній час очікування Wq (хв)')
    plt.title('Середній час очікування в черзі')
    plt.grid(True, alpha=0.3)

    # 4. Графік ймовірності простою
    plt.subplot(2, 2, 4)
    plt.plot(df['X'], df['P0'], 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Інтервал надходження партій X (хв)')
    plt.ylabel('Ймовірність простою P0')
    plt.title('Ймовірність простою системи')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 5. Розподіл ймовірностей станів для оптимального X
    optimal_idx = np.argmin(np.abs(df['Lq'] - 20))
    optimal_X = df.iloc[optimal_idx]['X']

    model_optimal = AnalyticalMM4Model(X=optimal_X, mu=1 / 45)
    optimal_results = model_optimal.calculate_metrics()

    if optimal_results:
        states_prob = optimal_results['states_prob']
        states = range(len(states_prob))

        plt.figure(figsize=(12, 6))
        plt.bar(states[:15], states_prob[:15], color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Кількість партій у системі')
        plt.ylabel('Ймовірність')
        plt.title(f'Розподіл ймовірностей станів системи (X = {optimal_X:.1f} хв)')
        plt.grid(True, alpha=0.3)
        plt.xticks(states[:15])
        plt.show()


def plot_comparative_analysis():
    """Порівняльний аналіз впливу X на показники системи"""

    X_range = np.linspace(10, 50, 20)
    metrics_data = []

    mu = 1 / 45  # Фіксована інтенсивність обслуговування

    for X in X_range:
        model = AnalyticalMM4Model(X=X, mu=mu)
        results = model.calculate_metrics()

        if results:
            metrics_data.append({
                'X': X,
                'Lq': results['Lq'],
                'Wq': results['Wq'],
                'rho': results['rho'],
                'P0': results['p0'],
                'Ls': results['Ls']
            })

    df_comparative = pd.DataFrame(metrics_data)

    # Створення комбінованого графіка
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Перша вісь - довжина черги
    color = 'blue'
    ax1.set_xlabel('Інтервал надходження партій X (хв)')
    ax1.set_ylabel('Середня довжина черги Lq', color=color)
    line1 = ax1.plot(df_comparative['X'], df_comparative['Lq'], color=color, linewidth=2, label='Lq')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Ліміт черги')

    # Друга вісь - коефіцієнт завантаження
    ax2 = ax1.twinx()
    color = 'green'
    ax2.set_ylabel('Коефіцієнт завантаження ρ', color=color)
    line2 = ax2.plot(df_comparative['X'], df_comparative['rho'], color=color, linewidth=2, label='ρ')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Границя стабільності')

    # Комбінована легенда
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')

    plt.title('Вплив інтервалу X на довжину черги та завантаження системи')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    """Основна функція для аналітичного дослідження"""
    print("=" * 60)
    print("АНАЛІТИЧНА МОДЕЛЬ СИСТЕМИ M/M/4")
    print("=" * 60)

    # Оцінка інтенсивності обслуговування
    mu = estimate_processing_time()

    # Аналіз для різних значень X
    X_values = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    results_list = []

    print("\nАналіз для різних значень X:")
    for X in X_values:
        model = AnalyticalMM4Model(X=X, mu=mu)
        results = model.calculate_metrics()

        if results:
            results_list.append({
                'X': X,
                'λ': 1 / X,
                'ρ': results['rho'],
                'P0': results['p0'],
                'Lq': results['Lq'],
                'Ls': results['Ls'],
                'Wq': results['Wq'],
                'Ws': results['Ws']
            })
            print(f"X = {X}: Lq = {results['Lq']:.2f}, ρ = {results['rho']:.3f}")

    # Вивід таблиці результатів
    df_results = pd.DataFrame(results_list)
    print("\nТАБЛИЦЯ РЕЗУЛЬТАТІВ АНАЛІТИЧНОЇ МОДЕЛІ:")
    print(df_results.round(4))

    # Побудова графіків
    plot_analytical_results(X_values, results_list)
    plot_comparative_analysis()

    # Пошук оптимального X
    print("\nПошук оптимального значення X...")
    model_opt = AnalyticalMM4Model(X=20, mu=mu)
    optimal_X = model_opt.find_max_X_for_queue(target_queue=20, mu_estimate=mu)

    print(f"\nРЕЗУЛЬТАТ:")
    print(f"Оптимальний інтервал X = {optimal_X:.2f} хв")
    print(f"Інтенсивність λ = {1 / optimal_X:.4f} партій/хв")

    # Перевірка для оптимального X
    model_final = AnalyticalMM4Model(X=optimal_X, mu=mu)
    final_results = model_final.calculate_metrics()

    if final_results:
        print(f"\nПоказники для оптимального X:")
        print(f"  Середня довжина черги: {final_results['Lq']:.2f} партій")
        print(f"  Коефіцієнт завантаження: {final_results['rho']:.3f}")
        print(f"  Середній час очікування: {final_results['Wq']:.2f} хв")


if __name__ == "__main__":
    main()