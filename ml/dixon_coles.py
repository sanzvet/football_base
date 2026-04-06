import numpy as np
import math
from typing import Dict, Tuple, List
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)


class DixonColes:
    """
    Коррекция Диксона-Колза для калибровки вероятностей точного счёта.
    
    Учитывает, что:
    - Низкие счета (0:0, 1:0, 0:1, 1:1) происходят чаще, чем предсказывает Пуассон
    - Корреляция между голами хозяев и гостей
    """
    
    def __init__(self, rho: float = 0.05):
        """
        rho: параметр корреляции (обычно 0.03-0.07)
        """
        self.rho = rho
    
    def tau(self, goals_h: int, goals_a: int, 
            lambda_h: float, lambda_a: float) -> float:
        """
        Функция коррекции τ (tau) из статьи Диксона-Колза.
        
        Корректирует вероятность счёта на основе:
        - Низкие голы (0 или 1) получают больший вес
        - Учитывает корреляцию между командами
        """
        if goals_h == 0 and goals_a == 0:
            return 1 - lambda_h * lambda_a * self.rho
        elif goals_h == 0 and goals_a == 1:
            return 1 + lambda_h * self.rho
        elif goals_h == 1 and goals_a == 0:
            return 1 + lambda_a * self.rho
        elif goals_h == 1 and goals_a == 1:
            return 1 - self.rho
        else:
            return 1.0
    
    def predict_score_probability(self, lambda_h: float, lambda_a: float,
                                   max_goals: int = 10) -> Dict[Tuple[int, int], float]:
        """
        Предсказать вероятности всех возможных счётов.
        
        Args:
            lambda_h: ожидаемые голы хозяев (из модели)
            lambda_a: ожидаемые голы гостей (из модели)
            max_goals: максимальное количество голов для расчёта
            
        Returns:
            Dict[(goals_h, goals_a), probability]
        """
        score_probs = {}
        
        for g_h in range(max_goals + 1):
            for g_a in range(max_goals + 1):
                # Базовая вероятность Пуассона
                prob = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)
                
                # Коррекция Диксона-Колза
                prob *= self.tau(g_h, g_a, lambda_h, lambda_a)
                
                score_probs[(g_h, g_a)] = prob
        
        # Нормализация (чтобы сумма = 1)
        total = sum(score_probs.values())
        if total > 0:
            score_probs = {k: v / total for k, v in score_probs.items()}
        
        return score_probs
    
    def calculate_market_probabilities(self, lambda_h: float, lambda_a: float) -> Dict[str, float]:
        """
        Рассчитать вероятности для популярных рынков ставок.
        
        Returns:
            Dict с вероятностями:
            - over_2_5: Тотал больше 2.5
            - under_2_5: Тотал меньше 2.5
            - btts_yes: Обе забьют - Да
            - btts_no: Обе забьют - Нет
            - exact_1_0, exact_2_1, etc.: Точный счёт
        """
        score_probs = self.predict_score_probability(lambda_h, lambda_a)
        
        markets = {}
        
        # Тотал больше 2.5
        markets['over_2_5'] = sum(
            prob for (g_h, g_a), prob in score_probs.items() 
            if g_h + g_a > 2.5
        )
        markets['under_2_5'] = 1 - markets['over_2_5']
        
        # Обе забьют (BTTS)
        markets['btts_yes'] = sum(
            prob for (g_h, g_a), prob in score_probs.items() 
            if g_h > 0 and g_a > 0
        )
        markets['btts_no'] = 1 - markets['btts_yes']
        
        # Популярные точные счёта
        for g_h in range(4):
            for g_a in range(4):
                key = f"exact_{g_h}_{g_a}"
                markets[key] = score_probs.get((g_h, g_a), 0)
        
        # Исход 1X2 (из точных счётов)
        markets['result_home'] = sum(
            prob for (g_h, g_a), prob in score_probs.items() 
            if g_h > g_a
        )
        markets['result_draw'] = sum(
            prob for (g_h, g_a), prob in score_probs.items() 
            if g_h == g_a
        )
        markets['result_away'] = sum(
            prob for (g_h, g_a), prob in score_probs.items() 
            if g_h < g_a
        )
        
        return markets
    
    def get_fair_odds(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Рассчитать справедливые коэффициенты из вероятностей.
        """
        odds = {}
        for market, prob in probabilities.items():
            if prob > 0:
                odds[f"{market}_odd"] = 1 / prob
            else:
                odds[f"{market}_odd"] = 99.99  # Бесконечный кэф для 0%
        return odds