# ===============================
# Распределение задач KAN13–KAN23 между A, B, C с помощью PuLP (ILP)
# ===============================

import pulp

# -------------------------------
# 1. ВХОДНЫЕ ДАННЫЕ
# -------------------------------

# Список задач (фазы 3)
tasks = [
    "KAN13", "KAN14", "KAN15", "KAN16", "KAN17",
    "KAN18", "KAN19", "KAN20", "KAN21", "KAN22", "KAN23"
]

# Участники проекта
people = ["A", "B", "C"]

# Оценка задач в Story Points
story_points = {
    "KAN13": 3,   
    "KAN14": 8,   
    "KAN15": 3,   
    "KAN16": 5,   
    "KAN17": 2,   
    "KAN18": 2,   
    "KAN19": 3,   
    "KAN20": 3,   
    "KAN21": 5,   
    "KAN22": 2,  
    "KAN23": 3   
}

# Предпочтения: насколько человеку интересно делать задачу (1–5)
preferences = {
    "A": {
        "KAN13": 5, "KAN14": 4, "KAN15": 4, "KAN16": 2, "KAN17": 2,
        "KAN18": 3, "KAN19": 3, "KAN20": 3, "KAN21": 4, "KAN22": 5, "KAN23": 5
    },
    "B": {
        "KAN13": 2, "KAN14": 3, "KAN15": 2, "KAN16": 5, "KAN17": 5,
        "KAN18": 4, "KAN19": 4, "KAN20": 5, "KAN21": 3, "KAN22": 4, "KAN23": 3
    },
    "C": {
        "KAN13": 3, "KAN14": 3, "KAN15": 5, "KAN16": 3, "KAN17": 3,
        "KAN18": 3, "KAN19": 3, "KAN20": 3, "KAN21": 5, "KAN22": 4, "KAN23": 4
    }
}

# Ограничения по загрузке в Story Points (ёмкость на фазу / спринт)
capacity = {
    "A": 16,
    "B": 13,
    "C": 10
}

# -------------------------------
# 2. СОЗДАНИЕ МОДЕЛИ
# -------------------------------


model = pulp.LpProblem("TaskAssignmentPhase3_SP", pulp.LpMaximize)

# -------------------------------
# 3. ПЕРЕМЕННЫЕ МОДЕЛИ
# -------------------------------


x = pulp.LpVariable.dicts(
    "x",                                      # базовое имя переменных
    [(t, p) for t in tasks for p in people],  # все пары (задача, человек)
    lowBound=0,                               # нижняя граница
    upBound=1,                                # верхняя граница
    cat="Binary"                              # тип — бинарная
)

# -------------------------------
# 4. ЦЕЛЕВАЯ ФУНКЦИЯ
# -------------------------------

model += pulp.lpSum(
    preferences[p][t] * x[(t, p)]  # вклад пары (t,p): интерес * факт назначения
    for t in tasks
    for p in people
), "TotalPreference"

# -------------------------------
# 5. ОГРАНИЧЕНИЯ
# -------------------------------

for t in tasks:
    model += pulp.lpSum(
        x[(t, p)] for p in people
    ) == 1, f"OnePersonPerTask_{t}"


for p in people:
    model += pulp.lpSum(
        story_points[t] * x[(t, p)] for t in tasks
    ) <= capacity[p], f"Capacity_{p}"

# -------------------------------
# 6. РЕШЕНИЕ ЗАДАЧИ
# -------------------------------

model.solve(pulp.PULP_CBC_CMD(msg=False))

# -------------------------------
# 7. ВЫВОД РЕЗУЛЬТАТОВ
# -------------------------------

print("Статус решения:", pulp.LpStatus[model.status])
print()

print("Распределение задач:")
for t in tasks:
    for p in people:
        # Если x[(t,p)] близко к 1 — значит задача t назначена человеку p
        if pulp.value(x[(t, p)]) > 0.5:
            print(f"  {t} -> {p}")

print("\nЗагрузка по людям (Story Points):")
for p in people:
    load_p = sum(story_points[t] * pulp.value(x[(t, p)]) for t in tasks)
    print(f"  {p}: {load_p} SP")

print("\nСуммарное значение предпочтений (целевая функция):",
      pulp.value(model.objective))
