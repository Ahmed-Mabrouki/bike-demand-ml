"""Reproducible, chronological benchmark on UCI daily bike rentals."""
from pathlib import Path
from urllib.request import urlopen
import hashlib
import io
import json
import platform
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip'
# Calendar features are available in advance. Weather is intentionally excluded:
# observed daily weather is not a weather forecast available the evening before.
FEATURES = ['mnth', 'weekday', 'holiday', 'workingday', 'trend',
            'month_sin', 'month_cos', 'lag_1', 'lag_7', 'mean_7']


def download():
    path = ROOT / 'data/raw/day.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(DATA_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in ['day.csv', 'Readme.txt']:
            (path.parent / name).write_bytes(archive.read(name))
    return path


def validate(df):
    required = {'dteday', 'cnt', 'casual', 'registered', 'mnth', 'weekday',
                'holiday', 'workingday'}
    if not required.issubset(df):
        raise ValueError(f'Missing columns: {required - set(df)}')
    if df[list(required)].isna().any().any():
        raise ValueError('Missing values in required columns')
    if df.dteday.duplicated().any() or not df.dteday.is_monotonic_increasing:
        raise ValueError('Dates must be unique and sorted')
    if not df.dteday.diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise ValueError('Daily calendar has gaps; lags would be invalid')
    if (df.cnt < 0).any() or not df.cnt.eq(df.casual + df.registered).all():
        raise ValueError('Invalid rental counts')
    for col, maximum in [('mnth', 12), ('weekday', 6), ('holiday', 1), ('workingday', 1)]:
        if not df[col].between(1 if col == 'mnth' else 0, maximum).all():
            raise ValueError(f'Invalid {col}')


def make_features(df):
    df = df.copy()
    df['trend'] = (df.dteday - pd.Timestamp('2011-01-01')).dt.days
    df['month_sin'] = np.sin(2 * np.pi * df.mnth / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.mnth / 12)
    df['lag_1'] = df.cnt.shift(1)
    df['lag_7'] = df.cnt.shift(7)
    df['mean_7'] = df.cnt.shift(1).rolling(7).mean()
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def split_data(df):
    train = df[df.dteday < '2012-07-01'].copy()
    valid = df[(df.dteday >= '2012-07-01') & (df.dteday < '2012-10-01')].copy()
    test = df[df.dteday >= '2012-10-01'].copy()
    if any(part.empty for part in [train, valid, test]):
        raise ValueError('Empty time split')
    return train, valid, test


def scores(y, prediction):
    return {'mae': float(mean_absolute_error(y, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction)))}


def main():
    path = download()
    raw = pd.read_csv(path, parse_dates=['dteday'])
    validate(raw)
    df = make_features(raw)
    train, valid, test = split_data(df)
    candidates = {
        'Gradient boosting': HistGradientBoostingRegressor(
            max_iter=180, max_leaf_nodes=10, l2_regularization=10,
            learning_rate=0.05, early_stopping=False, random_state=42),
        'Random forest': RandomForestRegressor(
            n_estimators=250, min_samples_leaf=4, max_features=0.8,
            random_state=42, n_jobs=2),
    }
    validation = {'Semaine precedente': scores(valid.cnt, valid.lag_7)}
    for name, model in candidates.items():
        model.fit(train[FEATURES], train.cnt)
        validation[name] = scores(valid.cnt, np.maximum(0, model.predict(valid[FEATURES])))
    # Choose on validation only, including the naive baseline as a valid winner.
    selected = min(validation, key=lambda key: validation[key]['mae'])
    # Explain the strongest learned model on validation, never on the test set.
    learned_name = min(candidates, key=lambda key: validation[key]['mae'])
    importance = permutation_importance(candidates[learned_name], valid[FEATURES],
        valid.cnt, scoring='neg_mean_absolute_error', n_repeats=10, random_state=42)
    explanation = sorted([{'feature': name, 'mae_increase': float(value),
                           'std': float(std)} for name, value, std in
                          zip(FEATURES, importance.importances_mean, importance.importances_std)],
                         key=lambda item: item['mae_increase'], reverse=True)
    if selected == 'Semaine precedente':
        predictions = test.lag_7.to_numpy()
    else:
        model = candidates[selected]
        history = pd.concat([train, valid])
        model.fit(history[FEATURES], history.cnt)
        predictions = np.maximum(0, model.predict(test[FEATURES]))
    result = test[['dteday', 'cnt', 'lag_7']].copy()
    result['prediction'] = predictions
    result['absolute_error'] = np.abs(result.cnt - result.prediction)
    test_metrics = scores(test.cnt, predictions)
    baseline = scores(test.cnt, test.lag_7)
    gain = 100 * (1 - test_metrics['mae'] / baseline['mae'])
    report = {
        'source': DATA_URL, 'sha256_day_csv': hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_rows': len(raw), 'usable_rows': len(df), 'features': FEATURES,
        'versions': {'python': platform.python_version(), 'pandas': pd.__version__,
                     'numpy': np.__version__, 'scikit_learn': sklearn.__version__},
        'splits': {name: {'rows': len(part), 'start': str(part.dteday.min().date()),
                         'end': str(part.dteday.max().date())}
                   for name, part in [('train', train), ('validation', valid), ('test', test)]},
        'validation': validation, 'selected': selected, 'test': test_metrics,
        'test_baseline': baseline, 'mae_improvement_percent': gain,
        'importance_model': learned_name, 'validation_importance': explanation,
        'protocol': 'Rolling one-day-ahead evaluation; actual previous-day totals available; fixed fitted model throughout test. No observed future weather.',
    }
    folder = ROOT / 'reports'
    folder.mkdir(exist_ok=True)
    (folder / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    result.to_csv(folder / 'predictions.csv', index=False)
    df.to_csv(folder / 'features.csv', index=False)
    with sqlite3.connect(folder / 'analytics.sqlite') as db:
        raw.assign(dteday=raw.dteday.dt.strftime('%Y-%m-%d')).to_sql('daily_rentals', db, if_exists='replace', index=False)
        result.assign(dteday=result.dteday.dt.strftime('%Y-%m-%d')).to_sql('test_predictions', db, if_exists='replace', index=False)
    from src.reporting import render
    render(report, result, raw, folder)
    print(json.dumps({'selected': selected, 'test': test_metrics, 'baseline': baseline,
                      'gain_percent': gain}, indent=2))


if __name__ == '__main__':
    main()
