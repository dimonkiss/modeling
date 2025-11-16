import simpy
import random
import numpy as np
import math
from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd

# Параметри моделі
X = 10  # Початкове значення (хвилини)
SIM_TIME = 480  # Тривалість зміни (хвилин)
NUM_STATIONS = 4  # Кількість дільниць

# Змінні для збору статистики
queue_lengths = []
max_queue_length = 0


class Workshop:
    def __init__(self, env, num_stations):
        self.env = env
        self.stations = simpy.Resource(env, num_stations)
        self.queue_length = 0
        self.queue_history = []
        self.processed_batches = 0

    def process_batch(self, batch_id):
        """Процес обробки партії деталей"""
        start_time = self.env.now

        # Очікування вільної дільниці
        with self.stations.request() as request:
            self.queue_length += 1
            self.queue_history.append((self.env.now, self.queue_length))
            yield request
            self.queue_length -= 1
            self.queue_history.append((self.env.now, self.queue_length))

            # Попередня обробка (для 2 з 4 деталей)
            yield self.env.timeout(5)

            # Складання (4 деталі по 8 хв)
            yield self.env.timeout(32)

            # Регулювання
            regulation_time = self.get_regulation_time()
            if regulation_time > 0:  # Якщо не брак
                yield self.env.timeout(regulation_time)
                self.processed_batches += 1

        # print(f"Партія {batch_id} оброблена за {self.env.now - start_time:.2f} хв")

    def get_regulation_time(self):
        """Визначення часу регулювання з урахуванням ймовірнісних подій"""
        base_time = self.lognormal_time(8, 0.5)  # Логнормальний розподіл

        # Перевірка на брак
        rand_val = random.random()
        if rand_val < 0.01:
            return 0  # Виріб бракований
        elif rand_val < 0.03:  # 0.01 + 0.02
            return base_time + 3  # Заміна деталі
        else:
            return base_time

    def lognormal_time(self, mean, sigma):
        """Генерація логнормально розподіленого часу"""
        # Перетворення параметрів
        mu = math.log(mean ** 2 / math.sqrt(sigma ** 2 + mean ** 2))
        sigma_real = math.sqrt(math.log(1 + (sigma ** 2 / mean ** 2)))

        return random.lognormvariate(mu, sigma_real)


def batch_generator(env, workshop, X):
    """Генератор партій деталей"""
    batch_id = 0
    while True:
        # Експоненційний інтервал між партіями
        yield env.timeout(random.expovariate(1.0 / X))
        batch_id += 1
        env.process(workshop.process_batch(batch_id))


def monitor_queue(env, workshop, queue_data):
    """Моніторинг довжини черги"""
    while True:
        queue_data.append(workshop.queue_length)
        yield env.timeout(1)  # Перевірка кожну хвилину


def run_simulation(X_value):
    """Запуск однієї симуляції"""
    queue_data = []

    env = simpy.Environment()
    workshop = Workshop(env, NUM_STATIONS)

    # Запуск процесів
    env.process(batch_generator(env, workshop, X_value))
    env.process(monitor_queue(env, workshop, queue_data))

    # Запуск симуляції
    env.run(until=SIM_TIME)

    # Розрахунок статистики
    avg_queue = mean(queue_data) if queue_data else 0
    max_queue = max(queue_data) if queue_data else 0

    return avg_queue, max_queue, workshop.processed_batches


def run_multiple_simulations(X_values, num_runs=100):
    """Запуск множинних симуляцій для різних значень X"""
    results = {}

    for X in X_values:
        print(f"Моделювання для X = {X}...")

        avg_queues = []
        max_queues = []
        processed_batches = []

        for run in range(num_runs):
            avg_q, max_q, processed = run_simulation(X)
            avg_queues.append(avg_q)
            max_queues.append(max_q)
            processed_batches.append(processed)

        results[X] = {
            'avg_queues': avg_queues,
            'max_queues': max_queues,
            'processed_batches': processed_batches
        }

    return results


def create_results_table(results):
    """Створення таблиці результатів"""
    table_data = []

    for X, data in results.items():
        table_data.append({
            'X (хв)': X,
            'λ (1/хв)': round(1 / X, 4),
            'Середня черга': f"{np.mean(data['avg_queues']):.2f} ± {np.std(data['avg_queues']):.2f}",
            'Макс. черга': f"{np.mean(data['max_queues']):.2f} ± {np.std(data['max_queues']):.2f}",
            'Мін. макс. черга': np.min(data['max_queues']),
            'Макс. макс. черга': np.max(data['max_queues']),
            'Оброблено партій': f"{np.mean(data['processed_batches']):.1f} ± {np.std(data['processed_batches']):.1f}"
        })

    df = pd.DataFrame(table_data)
    return df


def plot_results(results):
    """Побудова графіків результатів"""

    # 1. Гістограми розподілу максимальної черги
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    X_values = list(results.keys())

    for i, X in enumerate(X_values):
        if i < len(axes):
            max_queues = results[X]['max_queues']
            axes[i].hist(max_queues, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            axes[i].axvline(20, color='red', linestyle='--', linewidth=2, label='Ліміт 20')
            axes[i].set_xlabel('Максимальна довжина черги')
            axes[i].set_ylabel('Частота')
            axes[i].set_title(f'Розподіл макс. черги (X={X})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    # Приховати зайві субплоди
    for i in range(len(X_values), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

    # 2. Діаграми середніх значень з похибками
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Середня довжина черги
    avg_means = [np.mean(results[X]['avg_queues']) for X in X_values]
    avg_stds = [np.std(results[X]['avg_queues']) for X in X_values]

    axes[0, 0].errorbar(X_values, avg_means, yerr=avg_stds, fmt='o-', capsize=5, color='blue')
    axes[0, 0].set_xlabel('Інтервал X (хв)')
    axes[0, 0].set_ylabel('Середня довжина черги')
    axes[0, 0].set_title('Середня довжина черги з похибками')
    axes[0, 0].grid(True)

    # Максимальна довжина черги
    max_means = [np.mean(results[X]['max_queues']) for X in X_values]
    max_stds = [np.std(results[X]['max_queues']) for X in X_values]

    axes[0, 1].errorbar(X_values, max_means, yerr=max_stds, fmt='o-', capsize=5, color='red')
    axes[0, 1].axhline(20, color='black', linestyle=':', linewidth=2, label='Ліміт 20')
    axes[0, 1].set_xlabel('Інтервал X (хв)')
    axes[0, 1].set_ylabel('Максимальна довжина черги')
    axes[0, 1].set_title('Максимальна довжина черги з похибками')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Кількість оброблених партій
    proc_means = [np.mean(results[X]['processed_batches']) for X in X_values]
    proc_stds = [np.std(results[X]['processed_batches']) for X in X_values]

    axes[1, 0].errorbar(X_values, proc_means, yerr=proc_stds, fmt='o-', capsize=5, color='green')
    axes[1, 0].set_xlabel('Інтервал X (хв)')
    axes[1, 0].set_ylabel('Кількість партій')
    axes[1, 0].set_title('Оброблено партій за зміну')
    axes[1, 0].grid(True)

    # Ймовірність перевищення ліміту черги
    prob_over_20 = [np.mean(np.array(results[X]['max_queues']) > 20) for X in X_values]

    axes[1, 1].plot(X_values, prob_over_20, 'o-', color='orange', linewidth=2)
    axes[1, 1].set_xlabel('Інтервал X (хв)')
    axes[1, 1].set_ylabel('Ймовірність')
    axes[1, 1].set_title('Ймовірність перевищення черги > 20')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # 3. Boxplot для максимальних черг
    plt.figure(figsize=(12, 6))

    max_queues_data = [results[X]['max_queues'] for X in X_values]
    box = plt.boxplot(max_queues_data, labels=X_values, patch_artist=True)

    # Зафарбовування boxplot
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lavender']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.axhline(20, color='red', linestyle='--', linewidth=2, label='Ліміт 20')
    plt.xlabel('Інтервал X (хв)')
    plt.ylabel('Максимальна довжина черги')
    plt.title('Розподіл максимальної довжини черги (boxplot)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def find_optimal_X_simulation():
    """Пошук оптимального X імітаційним методом"""
    print("Пошук оптимального X імітаційним методом...")

    low = 10.0
    high = 40.0
    target_queue = 20

    for iteration in range(6):
        mid = (low + high) / 2

        # 50 прогонів для перевірки
        max_queues = []
        for _ in range(50):
            _, max_q, _ = run_simulation(mid)
            max_queues.append(max_q)

        avg_max_queue = np.mean(max_queues)
        prob_over_20 = np.mean(np.array(max_queues) > 20)

        if avg_max_queue <= target_queue and prob_over_20 < 0.1:
            high = mid  # Можна зменшити інтервал
        else:
            low = mid  # Потрібно збільшити інтервал

        print(f"Ітерація {iteration + 1}: X = {mid:.2f}, Макс. черга = {avg_max_queue:.2f}, "
              f"P(>20) = {prob_over_20:.3f}")

    optimal_X = (low + high) / 2

    # Фінальна перевірка
    final_queues = []
    for _ in range(100):
        _, max_q, _ = run_simulation(optimal_X)
        final_queues.append(max_q)

    final_avg = np.mean(final_queues)
    final_prob = np.mean(np.array(final_queues) > 20)

    print(f"\nОПТИМАЛЬНЕ ЗНАЧЕННЯ: X = {optimal_X:.2f} хв")
    print(f"Середня макс. черга: {final_avg:.2f}")
    print(f"Ймовірність перевищення: {final_prob:.3f}")

    return optimal_X


# Запуск дослідження
if __name__ == "__main__":
    print("=" * 60)
    print("ІМІТАЦІЙНА МОДЕЛЬ ЦЕХУ - АНАЛІЗ ЗА 100 ПРОГОНІВ")
    print("=" * 60)

    # Тестування для різних значень X
    X_values = [10, 15, 20, 25, 30]

    # Запуск 100 прогонів для кожного X
    results = run_multiple_simulations(X_values, num_runs=100)

    # Створення таблиці результатів
    print("\nТАБЛИЦЯ РЕЗУЛЬТАТІВ (100 прогонів для кожного X):")
    df = create_results_table(results)
    print(df.to_string(index=False))

    # Побудова графіків
    print("\nПОБУДОВА ГРАФІКІВ...")
    plot_results(results)

    # Пошук оптимального X
    print("\nПОШУК ОПТИМАЛЬНОГО ЗНАЧЕННЯ X...")
    optimal_X = find_optimal_X_simulation()

    print("\n" + "=" * 60)
    print("ВИСНОВКИ:")
    print("=" * 60)
    print(f"1. Оптимальний інтервал надходження партій: {optimal_X:.2f} хв")
    print(f"2. При X = {optimal_X:.2f} хв система забезпечує виконання умови")
    print("   (черга не перевищує 20 партій протягом зміни)")
    print("3. Результати базуються на 100 прогонах моделі для кожного X")
    print("4. Графіки демонструють розподіл та стабільність показників")