"""
hybrid_model.py — Гибридная модель прогнозирования футбольных матчей.

Архитектура:
    [33 признака] -> LightGBM (poisson regression) -> lambda_home, lambda_away
                                                         |
                                                         v
                                              Dixon-Coles V2 -> Матрица P(gh, ga)
                                                         |
                                                         v
                                              Isotonic Calibration (val set)
                                                         |
                                                         v
                                              ВСЕ рынки: 1X2, ТБ, ИТБ, ОЗ, AH0, точные счёта

Ключевые принципы:
- Target = goals (НЕ xG!) — Poisson работает на count data
- 3-way time split: train(70%) / val(15%) / test(15%)
- rho оптимизируется на val (без leakage в test)
- Isotonic calibration на val — исправляет overconfidence Poisson
- Регуляризация вместо клиппинга (max_depth=5, L1/L2, early_stopping)
- Никакого клиппинга lambda — только численная защита от 0

Author: Hybrid Model V2
"""

import asyncpg
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, brier_score_loss
import joblib
import os
import logging
from typing import Dict, Optional, List
from .features_v2 import FeatureEngineerV2
from .dixon_coles_v2 import DixonColesV2

logger = logging.getLogger(__name__)

# Рынки для калибровки: (ключ в predict(), описание для логов)
CALIBRATION_MARKETS = [
    'prob_home', 'prob_draw', 'prob_away',
    'over_2_5_prob', 'over_1_5_prob',
    'btts_yes_prob',
    'home_ah0_prob', 'away_ah0_prob',
]


class HybridModel:

    def __init__(self, model_path: str = "ml/hybrid_model.pkl"):
        self.model_path = model_path
        self.model_lambda_home = None    # LightGBM -> lambda для голов хозяев
        self.model_lambda_away = None    # LightGBM -> lambda для голов гостей
        self.feature_columns = None
        self.is_trained = False
        self.dixon_coles = DixonColesV2(rho=0.05)
        self.metrics = {}
        # Isotonic calibration models (один IsotonicRegression на рынок)
        self.calibrators: Dict[str, IsotonicRegression] = {}

    def get_lgb_params(self) -> Dict:
        """
        Параметры LightGBM для Poisson regression.

        Регуляризация вместо клиппинга:
        - max_depth=5: жёсткий лимит глубины (косвенный контроль lambda)
        - num_leaves=15: мало листов -> гладкие предсказания, нет экстремумов
        - min_child_samples=30: минимум данных в каждом листе
        - lambda_l1/l2=0.1: L1/L2 штрафуют за слишком сложную модель
        - early_stopping=50: остановка при переобучении
        """
        return {
            'objective': 'poisson',
            'metric': 'poisson',
            'boosting_type': 'gbdt',
            'num_leaves': 15,
            'max_depth': 5,
            'learning_rate': 0.03,
            'min_child_samples': 30,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'n_estimators': 500,
            'verbosity': -1,
            'n_jobs': -1,
            'seed': 42,
        }

    # ==========================
    # TRAIN
    # ==========================
    async def train(self, pool: asyncpg.Pool) -> Dict[str, float]:
        """
        Полный цикл обучения гибридной модели.

        Шаги:
        1. Загрузка данных из PostgreSQL
        2. Построение матрицы признаков (33 признака, EWMA)
        3. Time-based 3-way split: train(70%) / val(15%) / test(15%)
        4. Обучение 2-х LightGBM моделей (lambda_home, lambda_away)
        5. Оптимизация rho (Dixon-Coles) на val
        6. Isotonic calibration на val (исправляет overconfidence)
        7. Оценка на test (чистая, без leakage)
        8. Сохранение модели
        """
        logger.info("=== НАЧАЛО ОБУЧЕНИЯ ГИБРИДНОЙ МОДЕЛИ ===")

        fe = FeatureEngineerV2(pool)

        # 1. Загрузка данных
        df = await fe.get_training_data()

        if len(df) == 0:
            return {"error": "No data in database"}

        if len(df) < 200:
            return {"error": f"Not enough data: {len(df)} matches (minimum 200)"}

        # 2. Time-based 3-way split СНАЧАЛА (до feature engineering!)
        # train (70%) -> feature engineering + LightGBM учится
        # val (15%)   -> rho + calibration (без leakage в test!)
        # test (15%)  -> финальная оценка (чистая)
        #
        # КРИТИЧЕСКИ: global_means и league encoding считаются ТОЛЬКО на train!
        split_train = int(len(df) * 0.70)
        split_val = int(len(df) * 0.85)

        train_raw = df.iloc[:split_train]
        val_raw = df.iloc[split_train:split_val]
        test_raw = df.iloc[split_val:]

        logger.info(f"Raw split: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

        # 2b. Global stats ТОЛЬКО на train (без leakage!)
        fe._compute_global_stats(train_raw)

        # 3. Feature engineering на train
        feature_df = await fe.build_feature_matrix(train_raw)

        if len(feature_df) < 100:
            return {"error": "Not enough feature rows"}

        self.feature_columns = fe.get_feature_columns()
        feature_df = feature_df.fillna(0)

        train_df = feature_df

        # 3b. Feature engineering на val (с train global stats — уже зафиксированы)
        val_df = await fe.build_feature_matrix(val_raw)
        val_df = val_df.fillna(0)

        # 3c. Feature engineering на test (с train global stats)
        test_df = await fe.build_feature_matrix(test_raw)
        test_df = test_df.fillna(0)

        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)} матчей")

        X_train = train_df[self.feature_columns].values
        X_val = val_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values

        # Targets (реальные голы, целочисленные)
        y_train_home = train_df['target_goals_home'].values
        y_val_home = val_df['target_goals_home'].values
        y_test_home = test_df['target_goals_home'].values
        y_train_away = train_df['target_goals_away'].values
        y_val_away = val_df['target_goals_away'].values
        y_test_away = test_df['target_goals_away'].values

        # Sample weights (time decay)
        weights_train = train_df['time_weight'].values

        # 4. Обучение моделей (early_stopping по val — НЕ по test!)
        logger.info("Обучение model_lambda_home (Poisson)...")
        self.model_lambda_home = self._train_model(
            X_train, y_train_home, weights_train,
            X_val, y_val_home
        )

        logger.info("Обучение model_lambda_away (Poisson)...")
        self.model_lambda_away = self._train_model(
            X_train, y_train_away, weights_train,
            X_val, y_val_away
        )

        # 5. Оптимизация rho (Dixon-Coles) на VAL (без leakage в test!)
        # Используем time_weights: свежие матчи важнее для rho
        val_time_weights = val_df['time_weight'].values.tolist()
        logger.info("Оптимизация rho (Dixon-Coles) на validation (с time decay)...")
        self._optimize_rho(X_val, y_val_home, y_val_away, val_time_weights)

        # 6. Isotonic calibration на VAL (исправляет overconfidence Poisson)
        logger.info("Isotonic calibration на validation...")
        self._calibrate(X_val, y_val_home, y_val_away)

        # 7. Оценка моделей на TEST (чистая, без leakage)
        metrics = self._evaluate_models(X_test, y_test_home, y_test_away)
        logger.info(f"Метрики: {metrics}")

        # 8. Сохранение
        self.is_trained = True
        self.metrics = metrics
        self.save()

        logger.info("=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===")
        return metrics

    def _train_model(self, X_train, y_train, weights_train,
                     X_val, y_val):
        """Обучить одну LightGBM модель с Poisson objective."""
        params = self.get_lgb_params()

        model = lgb.LGBMRegressor(**params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        model.fit(
            X_train, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )

        # Важность признаков
        importance = dict(zip(
            self.feature_columns,
            model.feature_importances_
        ))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        logger.info("Важность признаков (top-10):")
        for name, imp in sorted_imp[:10]:
            logger.info(f"  {name}: {imp}")

        return model

    # ==========================
    # OPTIMIZE RHO
    # ==========================
    def _optimize_rho(self, X, y_home, y_away, time_weights=None):
        """
        Оптимизировать rho на val-выборке.
        Используем предсказанные lambda, реальные голы и time_weights.

        Args:
            X: feature matrix val-выборки
            y_home: реальные голы хозяев
            y_away: реальные голы гостей
            time_weights: веса по времени (свежие матчи важнее)
        """
        pred_home = self.model_lambda_home.predict(X)
        pred_away = self.model_lambda_away.predict(X)

        actual_scores = []
        lambda_h_list = []
        lambda_a_list = []

        for i in range(len(y_home)):
            lh = max(1e-6, pred_home[i])
            la = max(1e-6, pred_away[i])
            gh = int(y_home[i])
            ga = int(y_away[i])

            actual_scores.append((gh, ga))
            lambda_h_list.append(lh)
            lambda_a_list.append(la)

        if len(actual_scores) >= 50:
            self.dixon_coles.optimize_rho(
                actual_scores, lambda_h_list, lambda_a_list,
                time_weights=time_weights
            )
        else:
            logger.warning(f"Слишком мало данных для оптимизации rho: {len(actual_scores)}")

    # ==========================
    # CALIBRATION (Isotonic Regression)
    # ==========================
    def _get_raw_predictions(self, X: np.ndarray) -> List[Dict[str, float]]:
        """Получить сырые (некалиброванные) предсказания Dixon-Coles для набора матчей."""
        preds = []
        pred_home = self.model_lambda_home.predict(X)
        pred_away = self.model_lambda_away.predict(X)

        for i in range(len(X)):
            lh = max(1e-6, pred_home[i])
            la = max(1e-6, pred_away[i])

            score_probs = self.dixon_coles.predict_score_probability(lh, la)
            markets = self.dixon_coles.calculate_all_markets(score_probs, lh, la)
            preds.append(markets)

        return preds

    def _calibrate(self, X_val, y_val_home, y_val_away):
        """
        Isotonic calibration на val-выборке.

        Poisson + Dixon-Coles почти всегда overconfident:
        модель говорит P(home)=60%, а реально home выигрывает 52%.
        Isotonic regression подгоняет сырые вероятности к реальным частотам.

        Монотонность: если P_raw=0.3 калибруется в 0.25,
        то P_raw=0.4 калибруется в >=0.25 (нет инверсий).
        """
        raw_preds = self._get_raw_predictions(X_val)

        # Реальные исходы для каждого рынка
        actual_home = (y_val_home > y_val_away).astype(float)
        actual_draw = (y_val_home == y_val_away).astype(float)
        actual_away = (y_val_home < y_val_away).astype(float)
        actual_over25 = ((y_val_home + y_val_away) > 2.5).astype(float)
        actual_over15 = ((y_val_home + y_val_away) > 1.5).astype(float)
        actual_btts = ((y_val_home > 0) & (y_val_away > 0)).astype(float)
        actual_home_ah0 = (actual_home + 0.5 * actual_draw)
        actual_away_ah0 = (actual_away + 0.5 * actual_draw)

        # Реальные исходы для ITB рынков
        actual_home_itb05 = (y_val_home >= 1).astype(float)  # хозяева забьют 1+
        actual_home_itb15 = (y_val_home >= 2).astype(float)  # хозяева забьют 2+
        actual_away_itb05 = (y_val_away >= 1).astype(float)  # гости забьют 1+
        actual_away_itb15 = (y_val_away >= 2).astype(float)  # гости забьют 2+

        # Маппинг: ключ рынка -> (сырая вероятность из predict, реальный исход)
        # Каждый рынок имеет СВОЙ калибратор!
        market_outcomes = {
            # 1X2
            'prob_home': (lambda m: m['prob_home'], actual_home),
            'prob_draw': (lambda m: m['prob_draw'], actual_draw),
            'prob_away': (lambda m: m['prob_away'], actual_away),
            # Тоталы
            'over_2_5_prob': (lambda m: m['over_2_5_prob'], actual_over25),
            'over_1_5_prob': (lambda m: m['over_1_5_prob'], actual_over15),
            # Обе забьют
            'btts_yes_prob': (lambda m: m['btts_yes_prob'], actual_btts),
            # Индивидуальные тоталы (СВОИ калибраторы, НЕ prob_home/prob_away!)
            'home_itb_0_5_prob': (lambda m: m['home_itb_0_5_prob'], actual_home_itb05),
            'home_itb_1_5_prob': (lambda m: m['home_itb_1_5_prob'], actual_home_itb15),
            'away_itb_0_5_prob': (lambda m: m['away_itb_0_5_prob'], actual_away_itb05),
            'away_itb_1_5_prob': (lambda m: m['away_itb_1_5_prob'], actual_away_itb15),
            # Азиатская фора
            'home_ah0_prob': (lambda m: m['home_ah0_prob'], actual_home_ah0),
            'away_ah0_prob': (lambda m: m['away_ah0_prob'], actual_away_ah0),
        }

        self.calibrators = {}

        for market_key, (prob_extractor, actual_outcome) in market_outcomes.items():
            raw_probs = np.array([prob_extractor(m) for m in raw_preds])

            # Isotonic regression: monotone, non-parametric
            # y_min=0.01, y_max=0.99 — не даём вероятности выйти за [1%, 99%]
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
            ir.fit(raw_probs, actual_outcome)
            self.calibrators[market_key] = ir

            # Логируем сдвиг калибровки (средний до/после)
            raw_mean = float(np.mean(raw_probs))
            calibrated_mean = float(np.mean(ir.transform(raw_probs)))
            shift = calibrated_mean - raw_mean
            logger.info(f"  Calibration {market_key}: "
                        f"raw_mean={raw_mean:.4f} -> calibrated={calibrated_mean:.4f} "
                        f"(shift={shift:+.4f})")

    def _apply_calibration(self, prob: float, market_key: str) -> float:
        """Применить isotonic calibration к одной вероятности."""
        if market_key not in self.calibrators:
            return prob  # нет калибратора -> возвращаем как есть

        calibrated = self.calibrators[market_key].transform(np.array([prob]))[0]
        return float(calibrated)

    # ==========================
    # EVALUATE
    # ==========================
    def _evaluate_models(self, X_test, y_test_home, y_test_away) -> Dict[str, float]:
        """
        Оценить качество моделей на тестовой выборке (чистая, без leakage).
        Включает Brier Score для оценки калибровки.
        """
        metrics = {}

        pred_home = self.model_lambda_home.predict(X_test)
        pred_away = self.model_lambda_away.predict(X_test)

        # MAE для lambda
        mae_home = mean_absolute_error(y_test_home, pred_home)
        metrics['mae_lambda_home'] = round(mae_home, 4)
        mae_away = mean_absolute_error(y_test_away, pred_away)
        metrics['mae_lambda_away'] = round(mae_away, 4)

        # Средние lambda vs реальные голы
        metrics['mean_pred_lambda_home'] = round(float(np.mean(pred_home)), 3)
        metrics['mean_actual_goals_home'] = round(float(np.mean(y_test_home)), 3)
        metrics['mean_pred_lambda_away'] = round(float(np.mean(pred_away)), 3)
        metrics['mean_actual_goals_away'] = round(float(np.mean(y_test_away)), 3)

        # rho
        metrics['rho'] = round(self.dixon_coles.rho, 4)

        # Brier Score (до и после калибровки) для ключевых рынков
        raw_preds = self._get_raw_predictions(X_test)

        actual_home = (y_test_home > y_test_away).astype(float)
        actual_over25 = ((y_test_home + y_test_away) > 2.5).astype(float)
        actual_btts = ((y_test_home > 0) & (y_test_away > 0)).astype(float)

        # Brier score до калибровки
        raw_home = np.array([m['prob_home'] for m in raw_preds])
        raw_o25 = np.array([m['over_2_5_prob'] for m in raw_preds])
        raw_btts = np.array([m['btts_yes_prob'] for m in raw_preds])

        bs_home_raw = brier_score_loss(actual_home, raw_home)
        bs_o25_raw = brier_score_loss(actual_over25, raw_o25)
        bs_btts_raw = brier_score_loss(actual_btts, raw_btts)

        # Brier score после калибровки
        cal_home = np.array([self._apply_calibration(p, 'prob_home') for p in raw_home])
        cal_o25 = np.array([self._apply_calibration(p, 'over_2_5_prob') for p in raw_o25])
        cal_btts = np.array([self._apply_calibration(p, 'btts_yes_prob') for p in raw_btts])

        bs_home_cal = brier_score_loss(actual_home, cal_home)
        bs_o25_cal = brier_score_loss(actual_over25, cal_o25)
        bs_btts_cal = brier_score_loss(actual_btts, cal_btts)

        metrics['brier_home_raw'] = round(bs_home_raw, 4)
        metrics['brier_home_calibrated'] = round(bs_home_cal, 4)
        metrics['brier_over25_raw'] = round(bs_o25_raw, 4)
        metrics['brier_over25_calibrated'] = round(bs_o25_cal, 4)
        metrics['brier_btts_raw'] = round(bs_btts_raw, 4)
        metrics['brier_btts_calibrated'] = round(bs_btts_cal, 4)

        # Улучшение калибровки (отрицательное = лучше)
        metrics['brier_home_improvement'] = round(bs_home_cal - bs_home_raw, 4)
        metrics['brier_over25_improvement'] = round(bs_o25_cal - bs_o25_raw, 4)
        metrics['brier_btts_improvement'] = round(bs_btts_cal - bs_btts_raw, 4)

        logger.info(f"Brier Score (test): home raw={bs_home_raw:.4f} cal={bs_home_cal:.4f} | "
                     f"over25 raw={bs_o25_raw:.4f} cal={bs_o25_cal:.4f} | "
                     f"btts raw={bs_btts_raw:.4f} cal={bs_btts_cal:.4f}")

        return metrics

    # ==========================
    # PREDICT
    # ==========================
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Полный прогноз для матча.

        Поток:
        1. Features -> LightGBM -> lambda_home, lambda_away
        2. lambda -> Dixon-Coles -> Матрица P(gh, ga)
        3. Матрица -> ВСЕ рынки (сырые вероятности)
        4. Isotonic calibration -> скорректированные вероятности
        5. fair odds = 1 / calibrated_prob

        Args:
            features: Dict с 33 признаками (от FeatureEngineerV2.get_prediction_features)

        Returns:
            Dict с калиброванными вероятностями и справедливыми коэффициентами
        """
        if not self.is_trained:
            self.load()

        if not self.is_trained:
            raise Exception("Гибридная модель не обучена!")

        # 1. Формируем X matrix
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])

        # 2. Предсказываем lambda (ожидаемые голы)
        lambda_home = self.model_lambda_home.predict(X)[0]
        lambda_away = self.model_lambda_away.predict(X)[0]

        # Численная защита от 0 (Poisson: lambda > 0, log(0) = -inf)
        lambda_home = max(1e-6, lambda_home)
        lambda_away = max(1e-6, lambda_away)

        logger.info(f"Predict: lambda_home={lambda_home:.3f}, lambda_away={lambda_away:.3f}, "
                     f"rho={self.dixon_coles.rho:.4f}")

        # 3. Dixon-Coles: матрица вероятностей счёта
        score_probs = self.dixon_coles.predict_score_probability(
            lambda_home, lambda_away)

        # 4. ВСЕ рынки из матрицы (сырые)
        markets = self.dixon_coles.calculate_all_markets(
            score_probs, lambda_home, lambda_away)

        # 5. Isotonic calibration для основных рынков
        prob_home_cal = self._apply_calibration(markets['prob_home'], 'prob_home')
        prob_draw_cal = self._apply_calibration(markets['prob_draw'], 'prob_draw')
        prob_away_cal = self._apply_calibration(markets['prob_away'], 'prob_away')
        over25_cal = self._apply_calibration(markets['over_2_5_prob'], 'over_2_5_prob')
        over15_cal = self._apply_calibration(markets['over_1_5_prob'], 'over_1_5_prob')
        btts_cal = self._apply_calibration(markets['btts_yes_prob'], 'btts_yes_prob')
        home_ah0_cal = self._apply_calibration(markets['home_ah0_prob'], 'home_ah0_prob')
        away_ah0_cal = self._apply_calibration(markets['away_ah0_prob'], 'away_ah0_prob')

        # 5b. Isotonic calibration для ITB рынков (СВОИ калибраторы!)
        home_itb05_cal = self._apply_calibration(markets['home_itb_0_5_prob'], 'home_itb_0_5_prob')
        home_itb15_cal = self._apply_calibration(markets['home_itb_1_5_prob'], 'home_itb_1_5_prob')
        away_itb05_cal = self._apply_calibration(markets['away_itb_0_5_prob'], 'away_itb_0_5_prob')
        away_itb15_cal = self._apply_calibration(markets['away_itb_1_5_prob'], 'away_itb_1_5_prob')

        # 6. Fair odds из КАЛИБРОВАННЫХ вероятностей
        def safe_odd(p):
            return round(1.0 / p, 2) if p > 0.001 else 99.99

        return {
            # 1X2 (калиброванные)
            'prob_home': round(prob_home_cal, 4),
            'prob_draw': round(prob_draw_cal, 4),
            'prob_away': round(prob_away_cal, 4),
            'fair_odd_home': safe_odd(prob_home_cal),
            'fair_odd_draw': safe_odd(prob_draw_cal),
            'fair_odd_away': safe_odd(prob_away_cal),

            # Тоталы (калиброванные)
            'over25_prob': round(over25_cal * 100, 1),
            'over25_odd': safe_odd(over25_cal),
            'over15_prob': round(over15_cal * 100, 1),
            'over15_odd': safe_odd(over15_cal),

            # Обе забьют (калиброванные)
            'btts_prob': round(btts_cal * 100, 1),
            'btts_odd': safe_odd(btts_cal),

            # Индивидуальные тоталы Home (СВОИ калибраторы!)
            'home_itb05_prob': round(home_itb05_cal * 100, 1),
            'home_itb05_odd': safe_odd(home_itb05_cal),
            'home_itb15_prob': round(home_itb15_cal * 100, 1),
            'home_itb15_odd': safe_odd(home_itb15_cal),

            # Индивидуальные тоталы Away (СВОИ калибраторы!)
            'away_itb05_prob': round(away_itb05_cal * 100, 1),
            'away_itb05_odd': safe_odd(away_itb05_cal),
            'away_itb15_prob': round(away_itb15_cal * 100, 1),
            'away_itb15_odd': safe_odd(away_itb15_cal),

            # Азиатская фора 0 (калиброванные)
            'home_ah0_prob': round(home_ah0_cal, 4),
            'fair_odd_home_ah0': safe_odd(home_ah0_cal),
            'away_ah0_prob': round(away_ah0_cal, 4),
            'fair_odd_away_ah0': safe_odd(away_ah0_cal),

            # Expected goals (lambda — без калибровки, это регрессия)
            'expected_goals_home': round(lambda_home, 3),
            'expected_goals_away': round(lambda_away, 3),
            'xg_home': round(lambda_home, 3),
            'xg_away': round(lambda_away, 3),

            # Точные счёта (без калибровки — из матрицы, не бинарный рынок)
            'exact_scores': {
                k: round(v * 100, 1) for k, v in markets['exact_scores'].items()
            },

            # Мета-информация
            'model_status': 'hybrid_v2',
            'rho': round(markets['rho'], 4),
        }

    # ==========================
    # SAVE / LOAD
    # ==========================
    def save(self):
        """Сохранить модель + калибраторы в .pkl файл."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        joblib.dump({
            'model_lambda_home': self.model_lambda_home,
            'model_lambda_away': self.model_lambda_away,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'rho': self.dixon_coles.rho,
            'rho_optimized': self.dixon_coles._rho_optimized,
            'metrics': self.metrics,
            'calibrators': self.calibrators,
        }, self.model_path)

        logger.info(f"Гибридная модель сохранена: {self.model_path} "
                     f"(калибраторов: {len(self.calibrators)})")

    def load(self) -> bool:
        """Загрузить модель + калибраторы из .pkl файла."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Файл модели не найден: {self.model_path}")
            return False

        try:
            data = joblib.load(self.model_path)

            self.model_lambda_home = data['model_lambda_home']
            self.model_lambda_away = data['model_lambda_away']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
            self.metrics = data.get('metrics', {})

            # Восстанавливаем rho
            rho = data.get('rho', 0.05)
            self.dixon_coles = DixonColesV2(rho=rho)
            self.dixon_coles._rho_optimized = data.get('rho_optimized', False)

            # Восстанавливаем калибраторы
            self.calibrators = data.get('calibrators', {})

            n_cal = len(self.calibrators)
            logger.info(f"Гибридная модель загружена: {self.model_path} "
                         f"(rho={rho:.4f}, калибраторов: {n_cal})")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
