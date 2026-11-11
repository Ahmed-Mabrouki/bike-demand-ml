"""Guard against evaluation errors that would invalidate portfolio results."""
import unittest
import numpy as np
import pandas as pd
from src.pipeline import FEATURES, make_features, split_data, validate


class TemporalIntegrityTests(unittest.TestCase):
    def fixture(self):
        dates = pd.date_range('2011-01-01', '2012-12-31')
        return pd.DataFrame({'dteday': dates, 'cnt': np.arange(len(dates)) + 10,
            'casual': np.arange(len(dates)), 'registered': 10, 'mnth': dates.month,
            'weekday': (dates.dayofweek + 1) % 7, 'holiday': 0, 'workingday': 1})

    def test_future_targets_cannot_change_past_features(self):
        raw = self.fixture()
        cutoff = pd.Timestamp('2012-08-01')
        before = make_features(raw)
        raw.loc[raw.dteday >= cutoff, 'cnt'] += 100000
        after = make_features(raw)
        pd.testing.assert_frame_equal(before.loc[before.dteday <= cutoff, FEATURES],
                                      after.loc[after.dteday <= cutoff, FEATURES])

    def test_lags_use_actual_previous_dates(self):
        features = make_features(self.fixture())
        self.assertEqual(features.iloc[0].lag_1, 16)
        self.assertEqual(features.iloc[0].lag_7, 10)
        self.assertEqual(features.iloc[0].mean_7, 13)

    def test_splits_are_ordered_disjoint_and_cover_data(self):
        features = make_features(self.fixture())
        train, valid, test = split_data(features)
        self.assertLess(train.dteday.max(), valid.dteday.min())
        self.assertLess(valid.dteday.max(), test.dteday.min())
        self.assertEqual(sum(map(len, (train, valid, test))), len(features))
        self.assertFalse({'cnt', 'casual', 'registered', 'instant'} & set(FEATURES))

    def test_calendar_gaps_and_bad_counts_rejected(self):
        raw = self.fixture()
        validate(raw)
        with self.assertRaises(ValueError):
            validate(raw.drop(index=5))
        raw.loc[5, 'cnt'] = -1
        with self.assertRaises(ValueError):
            validate(raw)


if __name__ == '__main__':
    unittest.main()
