# credit-fraud-prediction
# Классификация мошеннических транзакций

Проект по бинарной классификации мошеннических транзакций с использованием scikit-learn и CatBoost.

## Описание

Модель предсказывает вероятность мошенничества для каждой транзакции. Основная метрика оценки — ROC AUC.

В проекте сравнивались следующие модели:

- RidgeClassifier
- RandomForestClassifier
- HistGradientBoostingClassifier
- CatBoostClassifier

## Данные

Данные загружаются из CSV-файлов:

- `train.csv` — обучающая выборка
- `test.csv` — тестовая выборка

Целевая переменная — `is_fraud`:
- `1` — мошенническая транзакция
- `0` — обычная транзакция

## Методология

1. Предобработка данных выполняется с помощью `Pipeline`.
2. Модели оцениваются с помощью стратифицированной кросс-валидации.
3. Основная метрика — ROC AUC.
4. Для редкого положительного класса дополнительно анализируются precision, recall и F1-score.
5. Порог классификации подбирается с учётом допустимого количества предупреждений (alert budget).

## Результаты

На тестовой выборке:

| Модель | ROC AUC |
|---|---:|
| RidgeClassifier | 0.927 |
| RandomForestClassifier | 0.908 |
| HistGradientBoostingClassifier | 0.940 |
| CatBoostClassifier | 0.940 |

Финальная модель — `HistGradientBoostingClassifier` с подобранными гиперпараметрами и `class_weight="balanced"`.

## Использование

```python
import pandas as pd

test_data = pd.read_csv("test.csv")

test_prob = best_gbc.predict_proba(test_data)[:, 1]

submission = pd.DataFrame({
    "id": test_data["id"],
    "is_fraud": test_prob
})

submission.to_csv("submission.csv", index=False)
```

## Результат сабмита

Score: **0.94495**

## Зависимости

- Python 3
- pandas
- numpy
- scikit-learn
- catboost
