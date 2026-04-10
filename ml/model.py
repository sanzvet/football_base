import asyncio
import asyncpg
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .features import FeatureEngineer
from .dixon_coles import DixonColes
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class FootballModel:
    """
    ML-модель для прогноза футбольных матчей.
    LightGBM + Time Decay Weighting + Dixon-Coles Correction
    """
    
    def __init__(self, model_path: str = "ml/ml_model.pkl"):
        self.model_path = model_path
        self.model_home = None  # Модель для победы хозяев
        self.model_draw = None  # Модель для ничьей
        self.model_away = None  # Модель для победы гостей
        self.model_goals_h = None  # Модель для голов хозяев
        self.model_goals_a = None  # Модель для голов гостей
        self.feature_columns = None
        self.is_trained = False
        self.dixon_coles = DixonColes(rho=0.05)  # Коррекция Диксона-Колза
    
    def get_lgb_params(self) -> Dict:
        """
        Параметры LightGBM.
        """
        return {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,  # В конструкторе (не в fit!)
            'n_jobs': -1,
            'seed': 42,
            'n_estimators': 500,
            'early_stopping_rounds': 50
        }
    
    async def train(self, pool: asyncpg.Pool) -> Dict[str, float]:
        """
        Обучить модель на данных из БД.
        """
        logger.info("🚀 Начинаем обучение модели...")
        
        # 1. Получаем данные
        fe = FeatureEngineer(pool)
        df = await fe.get_training_data()
        
        if len(df) == 0:
            logger.error("❌ Нет данных для обучения!")
            return {"error": "No data"}
        
        # 2. Строим матрицу признаков
        feature_df = await fe.build_feature_matrix(df)
        
        if len(feature_df) < 100:
            logger.error(f"❌ Слишком мало данных: {len(feature_df)} матчей")
            return {"error": "Not enough data"}
        
        # 3. Получаем список признаков
        self.feature_columns = fe.get_feature_columns()
        
        # 4. Заполняем пропуски
        feature_df = feature_df.fillna(0)
        
        # 5. Разделяем на признаки и цели
        X = feature_df[self.feature_columns].values
        y_home = feature_df['result_home'].values
        y_draw = feature_df['result_draw'].values
        y_away = feature_df['result_away'].values
        y_goals_h = feature_df['goals_h'].values
        y_goals_a = feature_df['goals_a'].values
        
        # 6. Time Decay веса
        sample_weights = feature_df['time_weight'].values
        
        logger.info(f"📊 Данные: {X.shape[0]} матчей, {X.shape[1]} признаков")
        
        # 7. Time Series Split (чтобы не заглядывать в будущее)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 8. Обучаем модели
        logger.info("🏠 Обучаем модель для победы хозяев...")
        self.model_home = self._train_model(X, y_home, sample_weights, tscv)
        
        logger.info("🤝 Обучаем модель для ничьей...")
        self.model_draw = self._train_model(X, y_draw, sample_weights, tscv)
        
        logger.info("✈️ Обучаем модель для победы гостей...")
        self.model_away = self._train_model(X, y_away, sample_weights, tscv)
        
        logger.info("⚽ Обучаем модель для голов хозяев...")
        self.model_goals_h = self._train_model_regression(X, y_goals_h, sample_weights, tscv)
        
        logger.info("⚽ Обучаем модель для голов гостей...")
        self.model_goals_a = self._train_model_regression(X, y_goals_a, sample_weights, tscv)
        
        self.is_trained = True
        
        # 9. Сохраняем модель
        self.save()
        
        # 10. Оценка качества
        metrics = self._evaluate_models(X, y_home, y_draw, y_away)
        
        logger.info("✅ Обучение завершено!")
        logger.info(f"📈 Метрики: {metrics}")
        
        return metrics
    
    def _train_model(self, X: np.ndarray, y: np.ndarray, 
                     weights: np.ndarray, cv) -> lgb.LGBMClassifier:
        """
        Обучить бинарную модель с кросс-валидацией.
        """
        model = lgb.LGBMClassifier(**self.get_lgb_params())
        
        # Простая валидация (последние 20% как тест)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train, w_val = weights[:split_idx], weights[split_idx:]
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val]
            # verbosity удалён - он уже задан в конструкторе!
        )
        
        return model
    
    def _train_model_regression(self, X: np.ndarray, y: np.ndarray,
                                 weights: np.ndarray, cv) -> lgb.LGBMRegressor:
        """
        Обучить регрессионную модель (для голов).
        """
        params = self.get_lgb_params()
        params['objective'] = 'regression'
        params['metric'] = 'rmse'
        
        model = lgb.LGBMRegressor(**params)
        
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        w_train, w_val = weights[:split_idx], weights[split_idx:]
        
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val]
            # verbosity удалён - он уже задан в конструкторе!
        )
        
        return model
    
    def _evaluate_models(self, X: np.ndarray, y_home: np.ndarray, 
                         y_draw: np.ndarray, y_away: np.ndarray) -> Dict[str, float]:
        """
        Оценить качество моделей.
        """
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        
        metrics = {}
        
        if self.model_home:
            pred_home = self.model_home.predict_proba(X_test)[:, 1]
            metrics['home_log_loss'] = log_loss(y_home[split_idx:], pred_home)
            metrics['home_brier'] = brier_score_loss(y_home[split_idx:], pred_home)
        
        if self.model_draw:
            pred_draw = self.model_draw.predict_proba(X_test)[:, 1]
            metrics['draw_log_loss'] = log_loss(y_draw[split_idx:], pred_draw)
            metrics['draw_brier'] = brier_score_loss(y_draw[split_idx:], pred_draw)
        
        if self.model_away:
            pred_away = self.model_away.predict_proba(X_test)[:, 1]
            metrics['away_log_loss'] = log_loss(y_away[split_idx:], pred_away)
            metrics['away_brier'] = brier_score_loss(y_away[split_idx:], pred_away)
        
        return metrics
    
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Сделать прогноз для конкретного матча.
        Возвращает: 1X2, xG, Тотал 2.5, ОЗ, справедливые коэффициенты.
        """
        if not self.is_trained:
            self.load()
        
        if not self.is_trained:
            raise Exception("Модель не обучена!")
        
        # Преобразуем признаки в вектор
        X = np.array([[features.get(col, 0) for col in self.feature_columns]])
        
        # Предсказания вероятностей 1X2
        prob_home = self.model_home.predict_proba(X)[0, 1] if self.model_home else 0.33
        prob_draw = self.model_draw.predict_proba(X)[0, 1] if self.model_draw else 0.33
        prob_away = self.model_away.predict_proba(X)[0, 1] if self.model_away else 0.33

        # Справедливые коэффициенты для форы 0
        fair_odd_home_ah0 = (1 - prob_draw) / prob_home if prob_home > 0 else 99.99
        fair_odd_away_ah0 = (1 - prob_draw) / prob_away if prob_away > 0 else 99.99

        # Также можно вычислить вероятности «не проиграть»
        home_ah0_win_or_push = prob_home + prob_draw
        away_ah0_win_or_push = prob_away + prob_draw
        
        # Предсказания голов (xG)
        xg_home = max(0, self.model_goals_h.predict(X)[0] if self.model_goals_h else 1.5)
        xg_away = max(0, self.model_goals_a.predict(X)[0] if self.model_goals_a else 1.2)
        
        # Нормализация вероятностей 1X2 (чтобы сумма = 1)
        total = prob_home + prob_draw + prob_away
        if total > 0:
            prob_home /= total
            prob_draw /= total
            prob_away /= total
        
        # ========== DIXON-COLES: Расчёт рынков ==========
        dc_markets = self.dixon_coles.calculate_market_probabilities(xg_home, xg_away)
        
        # Справедливые коэффициенты 1X2 (из LightGBM)
        fair_odd_home = 1 / prob_home if prob_home > 0 else 99.99
        fair_odd_draw = 1 / prob_draw if prob_draw > 0 else 99.99
        fair_odd_away = 1 / prob_away if prob_away > 0 else 99.99
        
        # Справедливые коэффициенты рынков (из Dixon-Coles)
        dc_odds = self.dixon_coles.get_fair_odds(dc_markets)
        
        # ========== Индивидуальные голы (из распределения Пуассона) ==========
        # > 0.5 = хотя бы 1 гол = 1 - P(0)
        home_over_05 = (1 - poisson.pmf(0, xg_home)) * 100
        away_over_05 = (1 - poisson.pmf(0, xg_away)) * 100

        # > 1.5 = хотя бы 2 гола = 1 - P(0) - P(1)
        home_over_15 = (1 - poisson.pmf(0, xg_home) - poisson.pmf(1, xg_home)) * 100
        away_over_15 = (1 - poisson.pmf(0, xg_away) - poisson.pmf(1, xg_away)) * 100
        
        # ========== Точные счёта (Топ-5) ==========
        score_probs = self.dixon_coles.predict_score_probability(xg_home, xg_away)
        exact_scores = {
            f"{gh}:{ga}": prob * 100 
            for (gh, ga), prob in sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        }
        
        return {
            # === Исход 1X2 (LightGBM) ===
            'prob_home': prob_home,
            'prob_draw': prob_draw,
            'prob_away': prob_away,
            'fair_odd_home': fair_odd_home,
            'fair_odd_draw': fair_odd_draw,
            'fair_odd_away': fair_odd_away,

            'fair_odd_home_ah0': fair_odd_home_ah0,
            'fair_odd_away_ah0': fair_odd_away_ah0,
            'home_ah0_prob': home_ah0_win_or_push,
            'away_ah0_prob': away_ah0_win_or_push,
            
            # === Ожидаемые голы ===
            'xg_home': xg_home,
            'xg_away': xg_away,
            
            # === Индивидуальные голы ===
            'home_over_05': home_over_05,
            'home_over_15': home_over_15,
            'away_over_05': away_over_05,
            'away_over_15': away_over_15,
            
            # === Рынки ставок (Dixon-Coles) ===
            'over25_prob': dc_markets['over_2_5'] * 100,
            'over25_odd': dc_odds['over_2_5_odd'],
            'btts_prob': dc_markets['btts_yes'] * 100,
            'btts_odd': dc_odds['btts_yes_odd'],
            
            # === Точные счёта (Топ-5) ===
            'exact_scores': exact_scores
        }
    
    def save(self):
        """
        Сохранить модель на диск.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        data = {
            'model_home': self.model_home,
            'model_draw': self.model_draw,
            'model_away': self.model_away,
            'model_goals_h': self.model_goals_h,
            'model_goals_a': self.model_goals_a,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(data, self.model_path)
        logger.info(f"💾 Модель сохранена: {self.model_path}")
    
    def load(self) -> bool:
        """
        Загрузить модель с диска.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"⚠️ Модель не найдена: {self.model_path}")
            return False
        
        try:
            data = joblib.load(self.model_path)
            self.model_home = data['model_home']
            self.model_draw = data['model_draw']
            self.model_away = data['model_away']
            self.model_goals_h = data['model_goals_h']
            self.model_goals_a = data['model_goals_a']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
            logger.info(f"✅ Модель загружена: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Получить важность признаков.
        """
        if not self.model_home or not self.feature_columns:
            return pd.DataFrame()
        
        importance = self.model_home.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        
        return df