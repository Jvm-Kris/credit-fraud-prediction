# credit-fraud-prediction
# Классификация мошеннических транзакций

Проект по бинарной классификации мошеннических транзакций для соревнования Kaggle **Playground Series — Season 6 Episode 11**.

## Описание

Модель предсказывает вероятность мошенничества для каждой транзакции. Оценка сабмита производится по ROC AUC между предсказанной вероятностью и истинным значением `is_fraud`.

## Данные

Данные не включены в репозиторий из-за большого размера.

Скачайте их через `kagglehub`:

```python
import kagglehub

path = kagglehub.competition_download("playground-series-s6e11")
print("Path to competition files:", path)
```

После скачивания будут доступны файлы:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

Целевая переменная — `is_fraud`:

- `1` — мошенническая транзакция
- `0` — обычная транзакция

## Модели

В проекте сравнивались:

- RidgeClassifier
- RandomForestClassifier
- HistGradientBoostingClassifier
- CatBoostClassifier

## Результаты

| Модель | ROC AUC |
|---|---:|
| RidgeClassifier | 0.927 |
| RandomForestClassifier | 0.908 |
| HistGradientBoostingClassifier | 0.940 |
| CatBoostClassifier | 0.940 |

Финальная модель — `HistGradientBoostingClassifier` с `class_weight="balanced"`.

## submission.csv

Файл submission.csv содержит вероятности мошенничества:

```csv
id,is_fraud
699960,0.3
699961,0.9
699962,0.2
```

## Результат

Public score: **0.94495**

## Зависимости

- Python 3
- pandas
- numpy
- scikit-learn
- catboost
- kagglehub
