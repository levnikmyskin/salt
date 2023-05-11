from unittest import TestCase
from active_learning.load_data import filter_files, ALFileInfo, ALPolicy
from active_learning import (
    ActiveLearningConfig,
    ActiveLearningDatasetGenerator,
    InitialSeedPolicy,
)
from datetime import datetime
import numpy as np


class TestActiveLearning(TestCase):
    def test_filter_files(self):
        date_created = datetime(420, 4, 20)
        date_before = datetime(320, 4, 20)
        date_searching = datetime(420, 3, 20)
        checkpoint_search = {1000, 5000, 20000}
        fs = [
            ALFileInfo(
                ALPolicy.RELEVANCE_SAMPLING,
                1000,
                int(date_created.timestamp()),
                100_000,
                "",
            ),
            ALFileInfo(
                ALPolicy.RELEVANCE_SAMPLING,
                5000,
                int(date_created.timestamp()),
                100_000,
                "",
            ),
            ALFileInfo(
                ALPolicy.RELEVANCE_SAMPLING,
                5000,
                int(date_before.timestamp()),
                100_000,
                "",
            ),
            ALFileInfo(
                ALPolicy.RELEVANCE_SAMPLING,
                10000,
                int(date_created.timestamp()),
                100_000,
                "",
            ),
            ALFileInfo(
                ALPolicy.RELEVANCE_SAMPLING,
                20000,
                int(date_before.timestamp()),
                100_000,
                "",
            ),
        ]

        filters = [
            lambda f: f.checkpoint in checkpoint_search,
            lambda f: f.timestamp > int(date_searching.timestamp()),
        ]
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 2)

        checkpoint_search = {10_000, 1000}
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 2)

        checkpoint_search = {20_000}
        filtered = list(filter_files(fs, filters))
        self.assertEqual(len(filtered), 0)

    def test_active_learning_generation(self):
        pool_size = 5_000
        data_size = 2000
        batch_size = 100
        validation_size = 500
        x = np.random.default_rng().random((pool_size, 100), dtype=float)
        y = np.random.default_rng().integers(0, 2, size=pool_size)
        config = ActiveLearningConfig(
            ALPolicy.RELEVANCE_SAMPLING,
            x,
            y,
            batch_size,
            InitialSeedPolicy.DETERMINISTIC,
            {data_size},
            random_initial_pos=2,
            save=False,
            validation_size=validation_size,
        )
        gen = ActiveLearningDatasetGenerator(config)
        tr, val = gen.generate_dataset(data_size, 5, batch_size, pool_size)
        self.assertLessEqual(len(val), validation_size)
        self.assertEqual(len(tr) + len(val), data_size)
        self.assertSetEqual(set(tr) & set(val), set())
        config.validation_size = 0
        gen = ActiveLearningDatasetGenerator(config)
        tr, val = gen.generate_dataset(data_size, 5, batch_size, pool_size)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(tr), data_size)
