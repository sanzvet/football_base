"""
dixon_coles_v2.py — Улучшенная модель Dixon-Coles с оптимизацией rho.

Улучшения по сравнению с V1:
- rho (корреляция) подбирается через scipy.optimize из данных
- Метод optimize_rho() находит оптимальную корреляцию по логарифмическому правдоподобию
- Все рынки рассчитаны из единой матрицы счёта:
    1X2, ТБ 1.5, ТБ 2.5, ОЗ, ИТБ 0.5/1.5 Home/Away, AH0
- Калибровка: нормализация матрицы (сумма = 1.0)
- Диагностика: логирование lambda и rho для каждого прогноза

Author: Hybrid Model V2
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


class DixonColesV2:
    """
    Улучшенная модель Диксона-Колза.

    Основная идея: распределение голов по Пуассону с коррекцией
    для низких счётов (0-0, 1-0, 0-1, 1-1), где корреляция между
    голами хозяев и гостей отлична от нуля.

    tau() — функция коррекции:
    - rho > 0: больше 0-0 и 1-1, меньше 0-1 и 1-0 (defensive/attacking correlation)
    - rho < 0: больше 0-1 и 1-0, меньше 0-0 и 1-1
    """

    def __init__(self, rho: float = 0.05):
        """
        Args:
            rho: начальное значение корреляции (0.03-0.15 типично).
                 Будет оптимизировано через optimize_rho().
        """
        self.rho = rho
        self._rho_optimized = False

    def tau(self, goals_h: int, goals_a: int,
            lambda_h: float, lambda_a: float) -> float:
        """
        Функция коррекции tau из статьи Dixon-Coles (1997).

        Корректирует вероятность счёта для низких голов:
        - (0,0): 1 - lambda_h * lambda_a * rho
        - (0,1): 1 + lambda_h * rho
        - (1,0): 1 + lambda_a * rho
        - (1,1): 1 - rho
        - остальные: 1.0 (без коррекции)

        Важно: tau ограничен снизу max(1e-6, tau), потому что:
        - При больших lambda и rho > 0: tau(0,0) = 1 - lambda_h*lambda_a*rho < 0
        - Отрицательный tau → отрицательная вероятность → сломанная матрица
        - Используем 1e-6 (а не 0.01!) чтобы не завышать вероятности в ×10
        """
        if goals_h == 0 and goals_a == 0:
            tau_val = 1 - lambda_h * lambda_a * self.rho
        elif goals_h == 0 and goals_a == 1:
            tau_val = 1 + lambda_h * self.rho
        elif goals_h == 1 and goals_a == 0:
            tau_val = 1 + lambda_a * self.rho
        elif goals_h == 1 and goals_a == 1:
            tau_val = 1 - self.rho
        else:
            return 1.0

        # Tau не может быть отрицательным — это сломает вероятности
        # 1e-6 = минимальный guard: не искажает реальные вероятности,
        # в отличие от 0.01 (который завышал бы в ×10)
        return max(1e-6, tau_val)

    def predict_score_probability(self, lambda_h: float, lambda_a: float,
                                  max_goals: int = 7) -> Dict[Tuple[int, int], float]:
        """
        Матрица вероятностей всех возможных счётов.

        Использует бивариантное распределение Пуассона с коррекцией
        Диксона-Колза для низких счётов.

        Args:
            lambda_h: ожидаемые голы хозяев (предсказано ML)
            lambda_a: ожидаемые голы гостей (предсказано ML)
            max_goals: максимальное количество голов (8 достаточно)

        Returns:
            Dict[(goals_h, goals_a), probability] — нормализованная матрица
        """
        # Численная защита от 0/отрицательных (Poisson требует λ > 0)
        # Никакого клиппинга сверху — модель должна быть регуляризована на этапе обучения
        lambda_h = max(1e-6, lambda_h)
        lambda_a = max(1e-6, lambda_a)

        score_probs = {}

        for g_h in range(max_goals + 1):
            for g_a in range(max_goals + 1):
                # Базовая вероятность Пуассона (независимые распределения)
                prob = poisson.pmf(g_h, lambda_h) * poisson.pmf(g_a, lambda_a)

                # Коррекция Диксона-Колза (только для низких счётов)
                prob *= self.tau(g_h, g_a, lambda_h, lambda_a)

                score_probs[(g_h, g_a)] = prob

        # Нормализация (сумма всех вероятностей = 1.0)
        total = sum(score_probs.values())
        if total > 0:
            score_probs = {k: v / total for k, v in score_probs.items()}

        return score_probs

    def optimize_rho(self, actual_scores: List[Tuple[int, int]],
                     lambda_h_list: List[float],
                     lambda_a_list: List[float],
                     time_weights: Optional[List[float]] = None) -> float:
        """
        Оптимизировать rho (корреляцию) через Maximum Likelihood Estimation.

        Ищет rho, который максимизирует взвешенное логарифмическое правдоподобие
        наблюденных счётов при заданных lambda_h и lambda_a.

        Args:
            actual_scores: список реальных счётов [(goals_h, goals_a), ...]
            lambda_h_list: список ожидаемых голов хозяев
            lambda_a_list: список ожидаемых голов гостей
            time_weights: веса матчей по времени (опционально).
                Свежие матчи имеют больший вес, старые — меньший.
                Если None — все матчи имеют одинаковый вес (1.0).
                Формула: exp(-days_from_end / 365)

        Returns:
            Оптимальное значение rho
        """
        n = len(actual_scores)
        use_time_decay = time_weights is not None
        if time_weights is None:
            time_weights = [1.0] * n

        # Нормализация весов (сумма = n)
        w_sum = sum(time_weights)
        if w_sum > 0:
            norm_weights = [w * n / w_sum for w in time_weights]
        else:
            norm_weights = [1.0] * n

        def neg_log_likelihood(rho_val):
            """Взвешенное отрицательное лог-правдоподобие (минимизируем)."""
            self.rho = rho_val
            total_ll = 0.0

            for i, (gh, ga) in enumerate(actual_scores):
                lh = max(1e-6, lambda_h_list[i])
                la = max(1e-6, lambda_a_list[i])

                prob = poisson.pmf(gh, lh) * poisson.pmf(ga, la)
                prob *= self.tau(gh, ga, lh, la)

                if prob > 1e-10:
                    total_ll += norm_weights[i] * np.log(prob)
                else:
                    total_ll += norm_weights[i] * -23  # log(1e-10) ~ -23

            return -total_ll

        # L2 регуляризация: штрафуем за экстремальные значения rho
        # Предотвращает прилипание к границам [-0.2, 0.2] на шуме
        def neg_log_likelihood_with_penalty(rho_val):
            base_nll = neg_log_likelihood(rho_val)
            penalty = 10.0 * rho_val ** 2  # L2: lambda * rho^2
            return base_nll + penalty

        # Оптимизация в диапазоне rho [-0.2, 0.2]
        result = minimize_scalar(
            neg_log_likelihood_with_penalty,
            bounds=(-0.20, 0.20),
            method='bounded'
        )

        self.rho = result.x
        self._rho_optimized = True

        logger.info(f"Dixon-Coles rho оптимизирован: rho={self.rho:.4f} "
                     f"(neg_log_likelihood={result.fun:.2f}, "
                     f"matches={n}, time_decay={'ON' if use_time_decay else 'OFF'})")

        return self.rho

    def calculate_all_markets(self, score_probs: Dict[Tuple[int, int], float],
                              lambda_h: float, lambda_a: float) -> Dict[str, float]:
        """
        Рассчитать ВСЕ рынки ставок из единой матрицы вероятностей счёта.

        Включает:
        - 1X2 (исход матча)
        - ТБ 1.5, ТБ 2.5 (общие тоталы)
        - ОЗ (обе забьют)
        - ИТБ 0.5/1.5 Home и Away (индивидуальные тоталы)
        - AH0 (азиатская фора 0) Home и Away
        - Точные счёта (top-5)

        Args:
            score_probs: матрица вероятностей от predict_score_probability()
            lambda_h: ожидаемые голы хозяев
            lambda_a: ожидаемые голы гостей

        Returns:
            Dict с вероятностями и справедливыми коэффициентами для всех рынков
        """
        markets = {}

        # ==============================
        # 1X2 (Исход матча)
        # ==============================
        prob_home = sum(p for (gh, ga), p in score_probs.items() if gh > ga)
        prob_draw = sum(p for (gh, ga), p in score_probs.items() if gh == ga)
        prob_away = sum(p for (gh, ga), p in score_probs.items() if gh < ga)

        markets['prob_home'] = prob_home
        markets['prob_draw'] = prob_draw
        markets['prob_away'] = prob_away

        # ==============================
        # Общие тоталы
        # ==============================
        # ТБ 2.5: сумма голов > 2.5
        over_25 = sum(p for (gh, ga), p in score_probs.items()
                      if gh + ga > 2.5)
        markets['over_2_5_prob'] = over_25

        # ТБ 1.5: сумма голов > 1.5
        over_15 = sum(p for (gh, ga), p in score_probs.items()
                      if gh + ga > 1.5)
        markets['over_1_5_prob'] = over_15

        # ==============================
        # Обе забьют (BTTS)
        # ==============================
        btts_yes = sum(p for (gh, ga), p in score_probs.items()
                       if gh > 0 and ga > 0)
        btts_no = 1.0 - btts_yes
        markets['btts_yes_prob'] = btts_yes
        markets['btts_no_prob'] = btts_no

        # ==============================
        # Индивидуальные тоталы (Home)
        # ==============================
        # ИТБ 0.5 Home: хозяева забьют хотя бы 1 гол
        # P(gh >= 1) = 1 - P(gh = 0) = сумма вероятностей где gh > 0
        home_itb_05 = sum(p for (gh, ga), p in score_probs.items()
                          if gh > 0)
        markets['home_itb_0_5_prob'] = home_itb_05

        # ИТБ 1.5 Home: хозяева забьют хотя бы 2 гола
        # P(gh >= 2) = 1 - P(gh = 0) - P(gh = 1)
        home_itb_15 = sum(p for (gh, ga), p in score_probs.items()
                          if gh >= 2)
        markets['home_itb_1_5_prob'] = home_itb_15

        # ==============================
        # Индивидуальные тоталы (Away)
        # ==============================
        # ИТБ 0.5 Away: гости забьют хотя бы 1 гол
        away_itb_05 = sum(p for (gh, ga), p in score_probs.items()
                          if ga > 0)
        markets['away_itb_0_5_prob'] = away_itb_05

        # ИТБ 1.5 Away: гости забьют хотя бы 2 гола
        away_itb_15 = sum(p for (gh, ga), p in score_probs.items()
                          if ga >= 2)
        markets['away_itb_1_5_prob'] = away_itb_15

        # ==============================
        # Азиатская фора 0 (AH0)
        # ==============================
        # AH0 Home = P(home win) + 0.5 * P(draw)
        ah0_home = prob_home + 0.5 * prob_draw
        ah0_away = prob_away + 0.5 * prob_draw
        markets['home_ah0_prob'] = ah0_home
        markets['away_ah0_prob'] = ah0_away

        # ==============================
        # Точные счёта (top-5)
        # ==============================
        sorted_scores = sorted(score_probs.items(),
                               key=lambda x: x[1], reverse=True)[:5]
        exact_scores = {
            f"{h}:{a}": p
            for (h, a), p in sorted_scores
        }
        markets['exact_scores'] = exact_scores

        # ==============================
        # Справедливые коэффициенты (fair odds = 1 / probability)
        # ==============================
        markets['fair_odd_home'] = 1.0 / prob_home if prob_home > 0 else 99.99
        markets['fair_odd_draw'] = 1.0 / prob_draw if prob_draw > 0 else 99.99
        markets['fair_odd_away'] = 1.0 / prob_away if prob_away > 0 else 99.99

        markets['fair_odd_over_2_5'] = 1.0 / over_25 if over_25 > 0 else 99.99
        markets['fair_odd_over_1_5'] = 1.0 / over_15 if over_15 > 0 else 99.99

        markets['fair_odd_btts_yes'] = 1.0 / btts_yes if btts_yes > 0 else 99.99

        markets['fair_odd_home_itb_0_5'] = (1.0 / home_itb_05
                                             if home_itb_05 > 0 else 99.99)
        markets['fair_odd_home_itb_1_5'] = (1.0 / home_itb_15
                                             if home_itb_15 > 0 else 99.99)
        markets['fair_odd_away_itb_0_5'] = (1.0 / away_itb_05
                                             if away_itb_05 > 0 else 99.99)
        markets['fair_odd_away_itb_1_5'] = (1.0 / away_itb_15
                                             if away_itb_15 > 0 else 99.99)

        markets['fair_odd_home_ah0'] = (1.0 / ah0_home
                                        if ah0_home > 0 else 99.99)
        markets['fair_odd_away_ah0'] = (1.0 / ah0_away
                                        if ah0_away > 0 else 99.99)

        # Точные счёта fair odds
        exact_odds = {
            score: 1.0 / prob if prob > 0 else 99.99
            for score, prob in exact_scores.items()
        }
        markets['exact_odds'] = exact_odds

        # ==============================
        # Диагностика (для логирования)
        # ==============================
        markets['lambda_home'] = lambda_h
        markets['lambda_away'] = lambda_a
        markets['rho'] = self.rho

        return markets
