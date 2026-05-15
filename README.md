# ML Project — Классификация коммерческой успешности фильмов

**Студент:** Швыркина Алина Юрьевна

**Группа:** БИВ234

---

# Описание задачи

В проекте решается задача многоклассовой классификации коммерческой успешности фильмов.

Целевая переменная отсутствует в исходном датасете, поэтому формируется самостоятельно на основе показателя:

```text
ROI = revenue / budget
```

После этого фильмы разбиваются на 3 класса:

* `low`
* `medium`
* `high`

Такой подход позволяет оценивать не абсолютную прибыль фильма, а относительную окупаемость.

Основная метрика качества:

```text
macro F1
```

Дополнительно используется:

```text
accuracy
```

`macro F1` выбрана как основная метрика, потому что задача является многоклассовой и важно учитывать качество предсказаний для всех классов.

---

# Источник данных

Используется датасет:

**IMDb Movies User Friendly Dataset**

Источник:
[https://www.kaggle.com/datasets/jacopoferretti/idmb-movies-user-friendly?resource=download](https://www.kaggle.com/datasets/jacopoferretti/idmb-movies-user-friendly?resource=download)

В проекте используется файл:

```text
MOVIES.csv
```

---

# Описание датасета

После загрузки данных было получено:

* 44 985 строк;
* 23 признака.

В датасете присутствуют:

### Числовые признаки

* `budget`
* `revenue`
* `runtime`
* `popularity`
* `vote_average`
* `vote_count`

### Категориальные признаки

* `adult`
* `original_language`
* `has_homepage`

### Текстовые и составные признаки

* `overview`
* `tagline`
* `genre`
* `companies`
* `countries`
* `languages`

Датасет содержит как финансовые, так и текстовые признаки, что делает его подходящим для ML-задачи классификации.

---

# Самостоятельный парсинг данных

Для выполнения требования по самостоятельному сбору данных в проект добавлен отдельный скрипт:

```text
src/data/parse_tmdb_movies.py
```

Скрипт выполняет:

* запросы к TMDb API;
* обход страниц `discover/movie`;
* получение подробной информации по фильмам;
* сохранение результата в CSV.

Пример запуска:

```bash
export TMDB_API_KEY=<your_api_key>

python -m src.data.parse_tmdb_movies \
  --pages 3 \
  --output data/processed/parsed_tmdb_movies_sample.csv
```

Полученные данные могут использоваться как дополнительный источник фильмов и признаков.

---

# Обработка данных

В notebook выполнены:

* анализ структуры данных;
* проверка пропусков;
* проверка дубликатов;
* преобразование `release_date` в datetime;
* создание признаков по дате релиза;
* создание признаков по длинам текстов;
* создание признаков по количеству жанров, компаний, стран и языков;
* логарифмические преобразования;
* обработка выбросов через clipping по верхнему 99-му перцентилю;
* удаление признаков с leakage.

Также выполнены:

* train / validation / test split;
* кодирование категориальных признаков;
* масштабирование числовых признаков внутри pipeline моделей.

---

# Визуализации

В notebook построены:

* распределения числовых признаков;
* heatmap корреляций;
* распределения классов;
* графики по финансовым признакам;
* визуализации признаков после feature engineering.

Визуализации используются для первичного анализа данных и проверки распределений признаков.

---

# Baseline

В качестве baseline используется:

```python
DummyClassifier(strategy="most_frequent")
```

Baseline-метрики на validation:

| model           | valid_accuracy | valid_macro_f1 |
| --------------- | -------------: | -------------: |
| DummyClassifier |         0.3331 |         0.1666 |

Baseline показывает качество модели без использования полезных признаков и используется как нижняя граница качества.

---

# Эксперименты и модели

В проекте обучены baseline и 5 полноценных ML-моделей:

1. `LogisticRegression`
2. `KNeighborsClassifier`
3. `DecisionTreeClassifier`
4. `RandomForestClassifier`
5. `HistGradientBoostingClassifier`

---

# Результаты моделей

Validation-метрики:

| model                | train_accuracy | valid_accuracy | train_macro_f1 | valid_macro_f1 |
| -------------------- | -------------: | -------------: | -------------: | -------------: |
| RandomForest         |         0.9912 |         0.6736 |         0.9912 |         0.6687 |
| LogisticRegression   |         0.6354 |         0.6347 |         0.6326 |         0.6324 |
| HistGradientBoosting |         0.9482 |         0.6358 |         0.9482 |         0.6300 |
| DecisionTree         |         0.7783 |         0.6064 |         0.7766 |         0.5988 |
| KNeighbors           |         0.6358 |         0.5697 |         0.6298 |         0.5625 |
| DummyClassifier      |         0.3333 |         0.3331 |         0.1667 |         0.1666 |

---

# Подбор гиперпараметров

Для `RandomForestClassifier` выполнен перебор гиперпараметров через `GridSearchCV`.

Использовались параметры:

```python
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 10],
    "model__min_samples_leaf": [1, 2],
}
```

Лучшие параметры:

```text
model__max_depth = None
model__min_samples_leaf = 2
model__n_estimators = 300
```

Лучший результат cross-validation:

```text
macro F1 = 0.6305
```

---

# Почему выбрана RandomForest

`RandomForestClassifier` показал лучший результат среди протестированных моделей:

```text
valid_macro_f1 = 0.6687
```

Сравнение:

| model                | valid_macro_f1 |
| -------------------- | -------------: |
| RandomForest         |         0.6687 |
| LogisticRegression   |         0.6324 |
| HistGradientBoosting |         0.6300 |
| DecisionTree         |         0.5988 |
| KNeighbors           |         0.5625 |

RandomForest выбран как лучшая модель проекта, потому что:

* показал наибольший `macro F1`;
* лучше обрабатывает нелинейные зависимости;
* устойчивее одиночного дерева решений;
* показывает более стабильные результаты на validation.

---

# Финальный результат

Оценка лучшей модели на test:

| model        | test_accuracy | test_macro_f1 |
| ------------ | ------------: | ------------: |
| RandomForest |        0.6431 |        0.6401 |

---

# Выводы

В ходе проекта:

* проведён анализ и очистка данных;
* выполнен feature engineering;
* реализован baseline;
* обучены 5 полноценных ML-моделей;
* проведено сравнение моделей;
* выполнен подбор гиперпараметров;
* реализован самостоятельный парсинг данных через TMDb API.

Основные выводы:

* baseline значительно слабее обучаемых моделей;
* engineered features улучшают качество классификации;
* наиболее сложным оказался класс `medium`;
* RandomForest показал лучшее качество среди протестированных моделей.

---

# Воспроизводимость проекта

## Fixed seed

Во всех экспериментах используется:

```python
RANDOM_STATE = 42
```

Это обеспечивает воспроизводимость разбиения данных и результатов моделей.

---

# Docker

В проект добавлены:

* `Dockerfile`
* `docker-compose.yml`

Запуск:

```bash
docker compose up --build
```

После запуска notebook доступен на:

```text
http://localhost:8888
```

---

# Линтеры и качество кода

В проекте используются:

* `ruff`
* `black`

Настройки находятся в:

```text
pyproject.toml
```

Проверка:

```bash
ruff check src tests
black --check src tests
```

---

# Версии библиотек

Версии библиотек зафиксированы в:

```text
requirements.txt
```

---

# Структура проекта

```text
.
├── data
├── notebooks
├── src
├── tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# Запуск проекта

## 1. Создать окружение

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 2. Установить зависимости

```bash
pip install -r requirements.txt
```

---

## 3. Положить датасет

Скачать `MOVIES.csv` и положить:

```text
data/raw/MOVIES.csv
```

---

## 4. Запустить notebook

```bash
jupyter notebook notebooks/cp1.ipynb
```

В notebook выполнить:

```text
Kernel → Restart & Run All
```
