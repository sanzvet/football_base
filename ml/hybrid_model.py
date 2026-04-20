"""
hybrid_model.py — Гибридная модель прогнозирования футбольных матчей.

Архитектура:
    [35 признаков] -> LightGBM (poisson regression) -> lambda_home, lambda_away
                                                         |
                                                         v
                                              Lambda Scaling Calibration (val2)
                                                         |
                                                         v
                                              Dixon-Coles V2 -> Матрица P(gh, ga)
                                                         |
                                                         v
                                              Isotonic Calibration (val2 set)
                                                         |
                                                         v
                                              Joint Normalization (1X2 sum = 1.0)
                                                         |
                                                         v
                                              ВСЕ рынки: 1X2, ТБ, ИТБ, ОЗ, AH0, точные счёта

Ключевые принципы:
- Target = goals (НЕ xG!) — Poisson работает на count data
- 4-way time split: train(70%) / val1(7.5%) / val2(7.5%) / test(15%)
- val1: rho + early_stopping (модель подбирается)
- val2: lambda scaling + isotonic calibration (калибровка)
- Нет leakage: val1 и val2 НЕ пересекаются
- Joint normalization: P(H) + P(D) + P(A) = 1.0 после калибровки
- Lambda scaling: mean_actual / mean_pred (простое и эффективное)
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
from typing import Dict, Optional, List, Tuple
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
        # ОШИБКА №10 FIX: Lambda scaling calibration
        self.lambda_scale_home = 1.0  # mean_actual_home / mean_pred_home
        self.lambda_scale_away = 1.0  # mean_actual_away / mean_pred_away

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
        2. Построение матрицы признаков (35 признаков, EWMA)
        3. Time-based 4-way split: train(70%) / val1(7.5%) / val2(7.5%) / test(15%)
        4. Обучение 2-х LightGBM моделей (lambda_home, lambda_away)
        5. Оптимизация rho (Dixon-Coles) на val1
        6. Lambda scaling calibration на val2
        7. Isotonic calibration на val2 (без leakage в test!)
        8. Оценка на test (чистая, без leakage)
        9. Сохранение модели
        """
        logger.info("=== НАЧАЛО ОБУЧЕНИЯ ГИБРИДНОЙ МОДЕЛИ ===")

        fe = FeatureEngineerV2(pool)

        # 1. Загрузка данных
        df = await fe.get_training_data()

        if len(df) == 0:
            return {"error": "No data in database"}

        if len(df) < 200:
            return {"error": f"Not enough data: {len(df)} matches (minimum 200)"}

        # ============================================================
        # ОШИБКА №8 FIX: 4-way split вместо 3-way
        # ============================================================
        # Было: train(70%) / val(15%) / test(15%)
        #   val использовался ДВАЖДЫ: rho + calibration → leakage!
        #
        # Стало: train(70%) / val1(7.5%) / val2(7.5%) / test(15%)
        #   val1 → rho + early_stopping (подбор модели)
        #   val2 → lambda scaling + calibration (калибровка)
        #   val1 и val2 НЕ пересекаются → нет leakage
        #
        # При 1000 матчей: train=700, val1=75, val2=75, test=150
        # При 200 матчей:  train=140, val1=30, val2=30, test=30
        # ============================================================
        split_train = int(len(df) * 0.70)
        split_val = int(len(df) * 0.85)
        val_mid = split_train + (split_val - split_train) // 2

        train_raw = df.iloc[:split_train]
        val1_raw = df.iloc[split_train:val_mid]
        val2_raw = df.iloc[val_mid:split_val]
        test_raw = df.iloc[split_val:]

        logger.info(
            f"Raw split: train={len(train_raw)}, val1={len(val1_raw)}, "
            f"val2={len(val2_raw)}, test={len(test_raw)}"
        )

        # 2b. Global stats ТОЛЬКО на train (без leakage!)
        fe._compute_global_stats(train_raw)

        # 2c. Rolling league encoding — строим на train+val1+val2+test вместе
        #     (rolling expanding mean с shift(1) исключает текущий матч)
        #     Это правильно: val/test видят только ПРЕДЫДУЩИЕ матчи
        fe._compute_rolling_league_encoding(
            pd.concat([train_raw, val1_raw, val2_raw, test_raw], ignore_index=True)
        )

        # 3. Feature engineering на train
        feature_df = await fe.build_feature_matrix(train_raw)

        if len(feature_df) < 100:
            return {"error": "Not enough feature rows"}

        self.feature_columns = fe.get_feature_columns()

        train_df = feature_df

        # 3b. Feature engineering на val1 (для rho + early_stopping)
        val1_df = await fe.build_feature_matrix(val1_raw)

        # 3c. Feature engineering на val2 (для calibration)
        val2_df = await fe.build_feature_matrix(val2_raw)

        # 3d. Feature engineering на test (для оценки)
        test_df = await fe.build_feature_matrix(test_raw)

        logger.info(
            f"Train: {len(train_df)}, Val1: {len(val1_df)}, "
            f"Val2: {len(val2_df)}, Test: {len(test_df)} матчей"
        )

        X_train = train_df[self.feature_columns].values
        X_val1 = val1_df[self.feature_columns].values
        X_val2 = val2_df[self.feature_columns].values
        X_test = test_df[self.feature_columns].values

        # Targets (реальные голы, целочисленные)
        y_train_home = train_df['target_goals_home'].values
        y_val1_home = val1_df['target_goals_home'].values
        y_val2_home = val2_df['target_goals_home'].values
        y_test_home = test_df['target_goals_home'].values
        y_train_away = train_df['target_goals_away'].values
        y_val1_away = val1_df['target_goals_away'].values
        y_val2_away = val2_df['target_goals_away'].values
        y_test_away = test_df['target_goals_away'].values

        # Sample weights (time decay)
        weights_train = train_df['time_weight'].values

        # 4. Обучение моделей (early_stopping по val1 — НЕ по test!)
        logger.info("Обучение model_lambda_home (Poisson)...")
        self.model_lambda_home = self._train_model(
            X_train, y_train_home, weights_train,
            X_val1, y_val1_home
        )

        logger.info("Обучение model_lambda_away (Poisson)...")
        self.model_lambda_away = self._train_model(
            X_train, y_train_away, weights_train,
            X_val1, y_val1_away
        )

        # ============================================================
        # 5. Оптимизация rho на VAL1 (без leakage в val2 и test!)
        # ============================================================
        val1_time_weights = val1_df['time_weight'].values.tolist()
        val1_league_labels = (
            val1_raw['league'].values.tolist()
            if 'league' in val1_raw.columns else None
        )
        logger.info(
            "Оптимизация rho (Dixon-Coles) на val1 "
            "(time decay + per-league)..."
        )
        self._optimize_rho(
            X_val1, y_val1_home, y_val1_away,
            val1_time_weights, val1_league_labels
        )

        # ============================================================
        # 6. Lambda scaling calibration на VAL2 (ОШИБКА №10 FIX)
        # ============================================================
        logger.info("Lambda scaling calibration на val2...")
        self._calibrate_lambda(X_val2, y_val2_home, y_val2_away)

        # ============================================================
        # 7. Isotonic calibration на VAL2 (ОШИБКА №8 FIX: не на val1!)
        # ============================================================
        logger.info("Isotonic calibration на val2...")
        self._calibrate(X_val2, y_val2_home, y_val2_away)

        # 8. Оценка моделей на TEST (чистая, без leakage)
        metrics = self._evaluate_models(X_test, y_test_home, y_test_away)
        logger.info(f"Метрики: {metrics}")

        # 9. Сохранение
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
    def _optimize_rho(self, X, y_home, y_away, time_weights=None, league_labels=None):
        """
        Оптимизировать rho на val-выборке.
        Используем предсказанные lambda, реальные голы, time_weights и league_labels.

        Args:
            X: feature matrix val-выборки
            y_home: реальные голы хозяев
            y_away: реальные голы гостей
            time_weights: веса по времени (свежие матчи важнее)
            league_labels: названия лиг (для per-league rho)
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
                time_weights=time_weights,
                league_labels=league_labels
            )
        else:
            logger.warning(f"Слишком мало данных для оптимизации rho: {len(actual_scores)}")

    # ==========================
    # CALIBRATION (Isotonic Regression)
    # ==========================
    def _get_raw_predictions(self, X: np.ndarray,
                             apply_lambda_scale: bool = False) -> List[Dict[str, float]]:
        """
        Получить сырые (некалиброванные) предсказания Dixon-Coles.

        Args:
            X: feature matrix
            apply_lambda_scale: если True — применить lambda scaling (ОШИБКА №10 FIX)
                Для _calibrate() и _evaluate_models() — True (хотим видеть эффект scaling)
                Для других случаев — False (чистый LightGBM output)
        """
        preds = []
        pred_home = self.model_lambda_home.predict(X)
        pred_away = self.model_lambda_away.predict(X)

        for i in range(len(X)):
            lh = max(1e-6, pred_home[i])
            la = max(1e-6, pred_away[i])

            # ОШИБКА №10 FIX: Применяем lambda scaling
            if apply_lambda_scale:
                lh *= self.lambda_scale_home
                la *= self.lambda_scale_away

            score_probs, tail_prob = self.dixon_coles.predict_score_probability(lh, la)
            markets = self.dixon_coles.calculate_all_markets(score_probs, lh, la, tail_prob=tail_prob)
            preds.append(markets)

        return preds

    # ==========================
    # LAMBDA SCALING CALIBRATION
    # ==========================
    def _calibrate_lambda(self, X: np.ndarray, y_home: np.ndarray,
                          y_away: np.ndarray):
        """
        Калибровка lambda через простое scaling.

        ОШИБКА №10 FIX:
        LightGBM Poisson почти всегда систематически смещён:
          - lambda_home = 1.3, а реальный mean = 1.5 → scale = 1.15
          - lambda_away = 1.1, а реальный mean = 1.0 → scale = 0.91

        Если не калибровать lambda → ВСЕ рынки наследуют ошибку!
        Dixon-Coles, isotonic — всё опирается на lambda.

        Формула:
          scale = mean(actual_goals) / mean(pred_lambda)

        Клэмпинг: [0.85, 1.15] — не даём слишком сильно сдвигать.
        Если scale за пределами → логируем warning (модель может быть переобучена).
        """
        pred_home = self.model_lambda_home.predict(X)
        pred_away = self.model_lambda_away.predict(X)

        mean_pred_h = float(np.mean(pred_home))
        mean_actual_h = float(np.mean(y_home))
        mean_pred_a = float(np.mean(pred_away))
        mean_actual_a = float(np.mean(y_away))

        raw_scale_h = mean_actual_h / max(mean_pred_h, 1e-6)
        raw_scale_a = mean_actual_a / max(mean_pred_a, 1e-6)

        # Клэмпинг: не даём экстремальных scaling
        SCALE_MIN, SCALE_MAX = 0.85, 1.15
        self.lambda_scale_home = float(np.clip(raw_scale_h, SCALE_MIN, SCALE_MAX))
        self.lambda_scale_away = float(np.clip(raw_scale_a, SCALE_MIN, SCALE_MAX))

        # Logging
        logger.info(
            f"Lambda scaling (val2):"
            f"\n  home: pred_mean={mean_pred_h:.3f} actual_mean={mean_actual_h:.3f} "
            f"raw_scale={raw_scale_h:.4f} clamped={self.lambda_scale_home:.4f}"
            f"\n  away: pred_mean={mean_pred_a:.3f} actual_mean={mean_actual_a:.3f} "
            f"raw_scale={raw_scale_a:.4f} clamped={self.lambda_scale_away:.4f}"
        )

        if abs(raw_scale_h - 1.0) > 0.15:
            logger.warning(
                f"Lambda home scaling {raw_scale_h:.3f} за пределами [{SCALE_MIN}, {SCALE_MAX}]. "
                f"Возможно переобучение или мало данных."
            )
        if abs(raw_scale_a - 1.0) > 0.15:
            logger.warning(
                f"Lambda away scaling {raw_scale_a:.3f} за пределами [{SCALE_MIN}, {SCALE_MAX}]. "
                f"Возможно переобучение или мало данных."
            )

    # ==========================
    # ISOTONIC CALIBRATION
    # ==========================
    def _calibrate(self, X_val, y_val_home, y_val_away):
        """
        Isotonic calibration на val2-выборке.

        Poisson + Dixon-Coles почти всегда overconfident:
        модель говорит P(home)=60%, а реально home выигрывает 52%.
        Isotonic regression подгоняет сырые вероятности к реальным частотам.

        ОШИБКА №8 FIX: Используем val2 (НЕ val1!)
        val1 уже использован для rho → двойное использование = leakage.

        ОШИБКА №9 FIX: 1X2 калибруется раздельно, но после —
        нормализация: P(H) + P(D) + P(A) = 1.0

        Монотонность: если P_raw=0.3 калибруется в 0.25,
        то P_raw=0.4 калибруется в >=0.25 (нет инверсий).
        """
        # ОШИБКА №10 FIX: raw predictions С lambda scaling
        raw_preds = self._get_raw_predictions(X_val, apply_lambda_scale=True)

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

    @staticmethod
    def _normalize_1x2(p_home: float, p_draw: float,
                      p_away: float) -> Tuple[float, float, float]:
        """
        ОШИБКА №9 FIX: Joint normalization 1X2.

        После раздельной калибровки P(home), P(draw), P(away):
          P(home) + P(draw) + P(away) ≠ 1.0
          → коэффициенты не соответствуют fair odds
          → arbitrage!

        Решение: s = p_h + p_d + p_a; p_h /= s; p_d /= s; p_a /= s
        Это сохраняет пропорции но гарантирует sum = 1.0.
        """
        s = p_home + p_draw + p_away
        if s > 0 and abs(s - 1.0) > 1e-6:
            return p_home / s, p_draw / s, p_away / s
        return p_home, p_draw, p_away

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

        # MAE для lambda (БЕЗ scaling — чистое качество LightGBM)
        mae_home = mean_absolute_error(y_test_home, pred_home)
        metrics['mae_lambda_home'] = round(mae_home, 4)
        mae_away = mean_absolute_error(y_test_away, pred_away)
        metrics['mae_lambda_away'] = round(mae_away, 4)

        # MAE для lambda ПОСЛЕ scaling (ОШИБКА №10 FIX)
        scaled_home = pred_home * self.lambda_scale_home
        scaled_away = pred_away * self.lambda_scale_away
        mae_home_scaled = mean_absolute_error(y_test_home, scaled_home)
        mae_away_scaled = mean_absolute_error(y_test_away, scaled_away)
        metrics['mae_lambda_home_scaled'] = round(mae_home_scaled, 4)
        metrics['mae_lambda_away_scaled'] = round(mae_away_scaled, 4)
        metrics['lambda_scale_home'] = round(self.lambda_scale_home, 4)
        metrics['lambda_scale_away'] = round(self.lambda_scale_away, 4)

        # Средние lambda vs реальные голы
        metrics['mean_pred_lambda_home'] = round(float(np.mean(pred_home)), 3)
        metrics['mean_actual_goals_home'] = round(float(np.mean(y_test_home)), 3)
        metrics['mean_pred_lambda_away'] = round(float(np.mean(pred_away)), 3)
        metrics['mean_actual_goals_away'] = round(float(np.mean(y_test_away)), 3)

        # rho
        metrics['rho'] = round(self.dixon_coles.rho, 4)

        # Brier Score (до и после калибровки) для ключевых рынков
        # ОШИБКА №10 FIX: raw predictions С lambda scaling
        raw_preds = self._get_raw_predictions(X_test, apply_lambda_scale=True)

        actual_home = (y_test_home > y_test_away).astype(float)
        actual_draw = (y_test_home == y_test_away).astype(float)
        actual_away = (y_test_home < y_test_away).astype(float)
        actual_over25 = ((y_test_home + y_test_away) > 2.5).astype(float)
        actual_btts = ((y_test_home > 0) & (y_test_away > 0)).astype(float)

        # Brier score до калибровки
        raw_home = np.array([m['prob_home'] for m in raw_preds])
        raw_draw = np.array([m['prob_draw'] for m in raw_preds])
        raw_away = np.array([m['prob_away'] for m in raw_preds])
        raw_o25 = np.array([m['over_2_5_prob'] for m in raw_preds])
        raw_btts = np.array([m['btts_yes_prob'] for m in raw_preds])

        bs_home_raw = brier_score_loss(actual_home, raw_home)
        bs_o25_raw = brier_score_loss(actual_over25, raw_o25)
        bs_btts_raw = brier_score_loss(actual_btts, raw_btts)

        # Brier score после калибровки (С joint normalization! ОШИБКА №9 FIX)
        cal_home = np.array([self._apply_calibration(p, 'prob_home') for p in raw_home])
        cal_draw = np.array([self._apply_calibration(p, 'prob_draw') for p in raw_draw])
        cal_away = np.array([self._apply_calibration(p, 'prob_away') for p in raw_away])
        cal_o25 = np.array([self._apply_calibration(p, 'over_2_5_prob') for p in raw_o25])
        cal_btts = np.array([self._apply_calibration(p, 'btts_yes_prob') for p in raw_btts])

        # ОШИБКА №9 FIX: normalize 1X2 before Brier score
        for i in range(len(cal_home)):
            cal_home[i], cal_draw[i], cal_away[i] = self._normalize_1x2(
                cal_home[i], cal_draw[i], cal_away[i]
            )

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
            features: Dict с 35 признаками (от FeatureEngineerV2.get_prediction_features)

        Returns:
            Dict с калиброванными вероятностями и справедливыми коэффициентами
        """
        if not self.is_trained:
            self.load()

        if not self.is_trained:
            raise Exception("Гибридная модель не обучена!")

        # 1. Формируем X matrix
        # ОШИБКА №5 FIX: default 0 заменён на None — если фича отсутствует,
        # это ошибка в get_prediction_features, а не тихое превращение в 0
        X = np.array([[features.get(col) for col in self.feature_columns]])
        if np.any([v is None for v in X[0]]):
            missing = [col for col in self.feature_columns if col not in features]
            logger.warning(f"Prediction: missing features: {missing}")
            # Fallback: заполняем None нулями (последняя линия защиты)
            X = np.array([[features.get(col, 0) for col in self.feature_columns]])

        # 2. Предсказываем lambda (ожидаемые голы)
        lambda_home = self.model_lambda_home.predict(X)[0]
        lambda_away = self.model_lambda_away.predict(X)[0]

        # ОШИБКА №10 FIX: Lambda scaling calibration
        lambda_home *= self.lambda_scale_home
        lambda_away *= self.lambda_scale_away

        # Численная защита от 0 (Poisson: lambda > 0, log(0) = -inf)
        lambda_home = max(1e-6, lambda_home)
        lambda_away = max(1e-6, lambda_away)

        logger.info(f"Predict: lambda_home={lambda_home:.3f} (scaled), lambda_away={lambda_away:.3f} (scaled), "
                     f"rho={self.dixon_coles.rho:.4f}")

        # 3. Dixon-Coles: матрица вероятностей счёта
        # Используем per-league rho если доступен
        league = features.get('_league', None)
        score_probs, tail_prob = self.dixon_coles.predict_score_probability(
            lambda_home, lambda_away)

        # 4. ВСЕ рынки из матрицы + tail (сырые)
        markets = self.dixon_coles.calculate_all_markets(
            score_probs, lambda_home, lambda_away, league=league,
            tail_prob=tail_prob)

        # 5. Isotonic calibration для основных рынков
        prob_home_cal = self._apply_calibration(markets['prob_home'], 'prob_home')
        prob_draw_cal = self._apply_calibration(markets['prob_draw'], 'prob_draw')
        prob_away_cal = self._apply_calibration(markets['prob_away'], 'prob_away')

        # ОШИБКА №9 FIX: Joint normalization — P(H) + P(D) + P(A) = 1.0
        prob_home_cal, prob_draw_cal, prob_away_cal = self._normalize_1x2(
            prob_home_cal, prob_draw_cal, prob_away_cal
        )

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

            # Expected goals (lambda — ПОСЛЕ scaling!)
            'expected_goals_home': round(lambda_home, 3),
            'expected_goals_away': round(lambda_away, 3),
            'xg_home': round(lambda_home, 3),
            'xg_away': round(lambda_away, 3),

            # Lambda scaling factors (диагностика)
            'lambda_scale_home': round(self.lambda_scale_home, 4),
            'lambda_scale_away': round(self.lambda_scale_away, 4),

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
            'rho_per_league': self.dixon_coles.rho_per_league,
            'metrics': self.metrics,
            'calibrators': self.calibrators,
            # ОШИБКА №10 FIX: save lambda scaling
            'lambda_scale_home': self.lambda_scale_home,
            'lambda_scale_away': self.lambda_scale_away,
        }, self.model_path)

        logger.info(f"Гибридная модель сохранена: {self.model_path} "
                     f"(калибраторов: {len(self.calibrators)}, "
                     f"per-league rho: {len(self.dixon_coles.rho_per_league)})")

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

            # Восстанавливаем rho + per-league rho
            rho = data.get('rho', 0.05)
            self.dixon_coles = DixonColesV2(rho=rho)
            self.dixon_coles._rho_optimized = data.get('rho_optimized', False)
            self.dixon_coles.rho_per_league = data.get('rho_per_league', {})

            # Восстанавливаем калибраторы
            self.calibrators = data.get('calibrators', {})

            # ОШИБКА №10 FIX: restore lambda scaling
            self.lambda_scale_home = data.get('lambda_scale_home', 1.0)
            self.lambda_scale_away = data.get('lambda_scale_away', 1.0)

            n_cal = len(self.calibrators)
            n_league_rho = len(self.dixon_coles.rho_per_league)
            logger.info(f"Гибридная модель загружена: {self.model_path} "
                         f"(rho={rho:.4f}, per-league={n_league_rho}, "
                         f"калибраторов: {n_cal})")
            return True

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
