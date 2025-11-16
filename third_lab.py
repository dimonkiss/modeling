import numpy as np
import matplotlib.pyplot as plt

# -------------------
# Параметри моделі
# -------------------
SHIFT_DURATION = 8 * 60  # хвилини
MEAN_ARRIVAL = 10  # середній інтервал надходження партії, хв
PARTY_SIZE = 4
PREP_TIME = 5  # попередня обробка, хв
ASSEMBLY_TIME = 8  # складання, хв
REG_MEAN = 8  # середнє регулювання, хв
REPLACE_TIME = 3  # заміна деталі, хв
DEFECT_WHOLE = 0.01
DEFECT_PART = 0.02

# Кількість прогонів
N_RUNS = 100


# -------------------
# Функції моделі
# -------------------
def lognormal_time(mean):
    sigma = 0.25  # параметр розсіювання
    mu = np.log(mean) - 0.5 * sigma ** 2
    return np.random.lognormal(mu, sigma)


def simulate_one_run(log_events=False):
    t = 0
    next_arrival = np.random.exponential(MEAN_ARRIVAL)
    prep_queue = []
    assembly_queue = []
    regulate_queue = []
    finished = 0
    resource_prep_free = True
    resource_assembly_free = True
    resource_regulate_free = True

    # Статистика
    time_in_system = []

    # Події
    prep_end_time = 0
    assembly_end_time = 0
    regulate_end_time = 0

    # Для графіків: розміри черг у часі
    queue_history = {'time': [], 'prep': [], 'assembly': [], 'regulate': []}

    while t < SHIFT_DURATION or prep_queue or assembly_queue or regulate_queue:
        next_event = min([next_arrival if t < SHIFT_DURATION else float('inf'),
                          prep_end_time if not resource_prep_free else float('inf'),
                          assembly_end_time if not resource_assembly_free else float('inf'),
                          regulate_end_time if not resource_regulate_free else float('inf')])
        t = next_event

        # Збереження розмірів черг
        queue_history['time'].append(t)
        queue_history['prep'].append(len(prep_queue))
        queue_history['assembly'].append(len(assembly_queue))
        queue_history['regulate'].append(len(regulate_queue))

        # Надходження нової партії
        if t == next_arrival and t < SHIFT_DURATION:
            if log_events: print(f"[{t:.1f} хв] Надходження партії")
            for _ in range(PARTY_SIZE):
                need_prep = np.random.rand() < 0.5
                if need_prep:
                    prep_queue.append({'arrival': t})
                else:
                    assembly_queue.append({'arrival': t})
            next_arrival = t + np.random.exponential(MEAN_ARRIVAL)

        # Завершення попередньої обробки
        if t == prep_end_time and not resource_prep_free:
            resource_prep_free = True
            part = prep_queue.pop(0)
            assembly_queue.append({'arrival': part['arrival']})
            prep_end_time = 0
            if log_events: print(f"[{t:.1f} хв] Завершено попередню обробку деталі")

        # Початок обробки попередньої
        if resource_prep_free and prep_queue:
            part = prep_queue[0]
            resource_prep_free = False
            prep_end_time = t + PREP_TIME
            if log_events: print(f"[{t:.1f} хв] Початок попередньої обробки деталі")

        # Завершення складання
        if t == assembly_end_time and not resource_assembly_free:
            resource_assembly_free = True
            part = assembly_queue.pop(0)
            regulate_queue.append({'arrival': part['arrival']})
            assembly_end_time = 0
            if log_events: print(f"[{t:.1f} хв] Завершено складання деталі")

        # Початок складання
        if resource_assembly_free and assembly_queue:
            part = assembly_queue[0]
            resource_assembly_free = False
            assembly_end_time = t + ASSEMBLY_TIME
            if log_events: print(f"[{t:.1f} хв] Початок складання деталі")

        # Завершення регулювання
        if t == regulate_end_time and not resource_regulate_free:
            resource_regulate_free = True
            part = regulate_queue.pop(0)
            rnd = np.random.rand()
            if rnd < DEFECT_WHOLE:
                if log_events: print(f"[{t:.1f} хв] Виріб бракований і вилучений")
            elif rnd < DEFECT_WHOLE + DEFECT_PART:
                regulate_queue.append({'arrival': t})
                regulate_end_time = t + REPLACE_TIME
                if log_events: print(f"[{t:.1f} хв] Замінено деталь, регулювання продовжується")
            else:
                finished += 1
                time_in_system.append(t - part['arrival'])
            regulate_end_time = 0

        # Початок регулювання
        if resource_regulate_free and regulate_queue:
            part = regulate_queue[0]
            resource_regulate_free = False
            regulate_end_time = t + lognormal_time(REG_MEAN)
            if log_events: print(f"[{t:.1f} хв] Початок регулювання деталі")

    avg_time = np.mean(time_in_system) if time_in_system else 0
    return finished, avg_time, queue_history


# -------------------
# Один прогін моделі
# -------------------
finished, avg_time, queue_hist = simulate_one_run(log_events=True)
print(f"\nОдиночний прогін: завершено виробів = {finished}, середній час перебування = {avg_time:.2f} хв")

# Динаміка черг для одного прогона
plt.figure(figsize=(12, 5))
plt.plot(queue_hist['time'], queue_hist['prep'], label='Попередня обробка: кількість деталей у черзі', color='blue')
plt.plot(queue_hist['time'], queue_hist['assembly'], label='Складання: кількість деталей у черзі', color='green')
plt.plot(queue_hist['time'], queue_hist['regulate'], label='Регулювання: кількість деталей у черзі', color='red')
plt.title("Динаміка розмірів черг на кожному етапі за один прогін")
plt.xlabel("Час, хв")
plt.ylabel("Розмір черги")
plt.grid(True)
plt.legend()
plt.show()

# -------------------
# 100 прогонів моделі
# -------------------
finished_list = []
time_list = []

for _ in range(N_RUNS):
    f, t_avg, _ = simulate_one_run()
    finished_list.append(f)
    time_list.append(t_avg)

# Середні показники
print(f"\nСередні показники за {N_RUNS} прогонів:")
print(f"Середня кількість виробів: {np.mean(finished_list):.2f}")
print(f"Середній час перебування виробу: {np.mean(time_list):.2f} хв")

# Гістограми за 100 прогонів
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(finished_list, bins=15, color='skyblue', edgecolor='black')
plt.title("Гістограма: Кількість завершених виробів за 100 прогонів")
plt.xlabel("Кількість виробів")
plt.ylabel("Частота прогонів")
plt.grid(axis='y')
plt.legend(["Кількість виробів, завершених протягом зміни"])

plt.subplot(1, 2, 2)
plt.hist(time_list, bins=15, color='salmon', edgecolor='black')
plt.title("Гістограма: Середній час перебування виробу за 100 прогонів")
plt.xlabel("Середній час перебування (хв)")
plt.ylabel("Частота прогонів")
plt.grid(axis='y')
plt.legend(["Середній час перебування виробу в системі"])

plt.tight_layout()
plt.show()
