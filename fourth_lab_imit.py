import simpy
import random
import numpy as np
import math
from statistics import mean

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
            yield self.env.timeout(regulation_time)

        print(f"Партія {batch_id} оброблена за {self.env.now - start_time:.2f} хв")

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


def monitor_queue(env, workshop):
    """Моніторинг довжини черги"""
    while True:
        queue_lengths.append(workshop.queue_length)
        yield env.timeout(1)  # Перевірка кожну хвилину


def run_simulation(X_value):
    """Запуск однієї симуляції"""
    global queue_lengths, max_queue_length
    queue_lengths = []

    env = simpy.Environment()
    workshop = Workshop(env, NUM_STATIONS)

    # Запуск процесів
    env.process(batch_generator(env, workshop, X_value))
    env.process(monitor_queue(env, workshop))

    # Запуск симуляції
    env.run(until=SIM_TIME)

    # Розрахунок статистики
    avg_queue = mean(queue_lengths)
    max_queue = max(queue_lengths)

    print(f"X = {X_value}: Середня черга = {avg_queue:.2f}, Максимальна черга = {max_queue}")

    return avg_queue, max_queue


def find_optimal_X():
    """Пошук оптимального значення X методом бісекції"""
    low = 1.0
    high = 60.0
    target_queue = 20

    while high - low > 0.1:
        mid = (low + high) / 2
        avg_queue, max_queue = run_simulation(mid)

        if max_queue <= target_queue:
            high = mid  # Можна зменшити інтервал
        else:
            low = mid  # Потрібно збільшити інтервал

        print(f"Пошук: X = {mid}, Черга = {max_queue}")

    optimal_X = (low + high) / 2
    print(f"\nОптимальне значення X: {optimal_X:.2f} хвилин")
    return optimal_X


# Запуск дослідження
if __name__ == "__main__":
    # Тестування для конкретного значення X
    test_X = 15
    avg_q, max_q = run_simulation(test_X)
    print(f"\nРезультат для X = {test_X}:")
    print(f"Середня довжина черги: {avg_q:.2f}")
    print(f"Максимальна довжина черги: {max_q}")

    # Пошук оптимального значення (закоментуйте, якщо потрібно тільки тестування)
    # optimal_X = find_optimal_X()