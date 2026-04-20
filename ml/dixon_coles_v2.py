"""
dixon_coles_v2.py — Улучшенная модель Dixon-Coles с оптимизацией rho.

Улучшения по сравнению с V1:
- rho (корреляция) подбирается через scipy.optimize из данных
- Адаптивный max_goals: matrix size = f(lambda), не фиксированный
- Per-league rho: каждая лига имеет свою корреляцию
- Все рынки рассчитаны из единой матрицы счёта:
    1X2, ТБ 1.5, ТБ 2.5, ОЗ, ИТБ 0.5/1.5 Home/Away, AH0
- Валидация входных данных и защита от NaN/inf
- L2 регуляризация rho (penalty=10) + bounds [-0.2, 0.2]
- Tail probability: хвост учитывается ОТДЕЛЬНО в каждом рынке

Нормализация:
  1. tau корректирует матрицу → сумма меняется
  2. Renormalize после tau (матрица = 1.0)
  3. Tail = Poisson CDF хвост (tau не затрагивает высокие счёты)
  4. Матрица масштабируется до (1 - tail), total = 1.0

Author: Hybrid Model V2
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)

# Минимальная матрица — не меньше 10 (121 ячеек)
_MIN_MAX_GOALS = 10


def _adaptive_max_goals(lambda_h: float, lambda_a: float) -> int:
    """
    Адаптивный размер матрицы по lambda.

    Формула: max(10, ceil(lambda + 5*sqrt(lambda)))
    При lambda=1.0: max(10, 6) = 10
    При lambda=1.5: max(10, 7) = 10
    При lambda=2.5: max(10, 10) = 10
    При lambda=3.0: max(10, 11) = 11
    """
    max_lam = max(lambda_h, lambda_a)
    dynamic = int(np.ceil(max_lam + 5 * np.sqrt(max_lam)))
    return max(_MIN_MAX_GOALS, dynamic)


class DixonColesV2:
    """
    Улучшенная модель Диксона-Колза.

    Основная идея: распределение голов по Пуассону с коррекцией
    для низких счётов (0-0, 1-0, 0-1, 1-1), где корреляция между
    голами хозяев и гостей отлична от нуля.

    tau() — функция коррекции:
    - rho > 0: больше 0-0 и 1-1, меньше 0-1 и 1-0
    - rho < 0: больше 0-1 и 1-0, меньше 0-0 и 1-1

    Per-league rho:
    - Каждая лига имеет свою корреляцию
    - Falls back на global rho для неизвестных лиг
    """

    def __init__(self, rho: float = 0.05):
        """
        Args:
            rho: начальное значение глобальной корреляции (0.03-0.15 типично).
                 Будет оптимизировано через optimize_rho().
        """
        self.rho = rho
        self._rho_optimized = False
        # Per-league rho: {league_name: rho_value}
        self.rho_per_league: Dict[str, float] = {}

    def _get_rho(self, league: Optional[str] = None) -> float:
        """Получить rho для лиги (fallback на глобальный)."""
        if league and league in self.rho_per_league:
            return self.rho_per_league[league]
        return self.rho

    def tau(self, goals_h: int, goals_a: int,
            lambda_h: float, lambda_a: float,
            rho_override: Optional[float] = None) -> float:
        """
        Функция коррекции tau из статьи Dixon-Coles (1997).

        Корректирует вероятность счёта для низких голов:
        - (0,0): 1 - lambda_h * lambda_a * rho
        - (0,1): 1 + lambda_h * rho
        - (1,0): 1 + lambda_a * rho
        - (1,1): 1 - rho
        - остальные: 1.0 (без коррекции)

        Guard: tau >= 1e-6 чтобы не сломать вероятности.

        Args:
            rho_override: если передан — использовать вместо self.rho
        """
        rho = rho_override if rho_override is not None else self.rho

        if goals_h == 0 and goals_a == 0:
            tau_val = 1 - lambda_h * lambda_a * rho
        elif goals_h == 0 and goals_a == 1:
            tau_val = 1 + lambda_h * rho
        elif goals_h == 1 and goals_a == 0:
            tau_val = 1 + lambda_a * rho
        elif goals_h == 1 and goals_a == 1:
            tau_val = 1 - rho
        else:
            return 1.0

        # Guard от отрицательного tau
        return max(1e-6, tau_val)

    @staticmethod
    def _safe_poisson_pmf(k: int, lam: float) -> float:
        """
        Безопасный poisson.pmf с защитой от NaN/inf.
        """
        if not np.isfinite(lam) or lam <= 0:
            return 1e-10
        try:
            p = float(poisson.pmf(k, lam))
            if not np.isfinite(p):
                return 1e-10
            return max(p, 1e-10)
        except (ValueError, OverflowError):
            return 1e-10

    def predict_score_probability(self, lambda_h: float, lambda_a: float,
                                  max_goals: Optional[int] = None
                                  ) -> Tuple[Dict[Tuple[int, int], float], float]:
        """
        Матрица вероятностей всех возможных счётов.

        Возвращает:
            (score_probs, tail_prob)
            - score_probs: Dict[(gh, ga), probability] — матрица, сумма = 1 - tail_prob
            - tail_prob: float — вероятность хвоста (gh > max_goals OR ga > max_goals)

        Нормализация:
            1. Poisson * tau для каждой ячейки
            2. Renormalize после tau (sum = 1.0)
            3. Poisson tail (tau не затрагивает высокие счёты)
            4. Scale matrix до (1 - tail_prob), чтобы matrix + tail = 1.0
        """
        # Численная защита от 0/отрицательных/NaN/inf
        if not np.isfinite(lambda_h) or lambda_h <= 0:
            lambda_h = 1e-6
        if not np.isfinite(lambda_a) or lambda_a <= 0:
            lambda_a = 1e-6

        # Адаптивный размер матрицы
        if max_goals is None:
            max_goals = _adaptive_max_goals(lambda_h, lambda_a)

        score_probs = {}

        for g_h in range(max_goals + 1):
            for g_a in range(max_goals + 1):
                prob = self._safe_poisson_pmf(g_h, lambda_h) * \
                       self._safe_poisson_pmf(g_a, lambda_a)

                # Коррекция Диксона-Колза (только для низких счётов)
                prob *= self.tau(g_h, g_a, lambda_h, lambda_a)

                score_probs[(g_h, g_a)] = prob

        # ============================================================
        # ОШИБКА №2 FIX: Renormalize после tau
        # tau перераспределяет вероятность между (0,0), (0,1), (1,0), (1,1)
        # → сумма матрицы меняется
        # → нормализуем, чтобы матрица = 1.0 (временно, до учёта tail)
        # ============================================================
        matrix_sum = sum(score_probs.values())
        if matrix_sum > 0:
            score_probs = {k: v / matrix_sum for k, v in score_probs.items()}

        # ============================================================
        # ОШИБКА №1 FIX: Tail — СЧИТАЕМ ОТДЕЛЬНО, НЕ растворяем!
        # Tail = P(gh > max_goals OR ga > max_goals)
        # tau затрагивает только (0,0)-(1,1), все внутри max_goals >= 10
        # → tail — чисто Poisson, без tau
        # ============================================================
        p_tail_h = 1.0 - float(poisson.cdf(max_goals, lambda_h))
        p_tail_a = 1.0 - float(poisson.cdf(max_goals, lambda_a))
        tail_prob = p_tail_h + p_tail_a - p_tail_h * p_tail_a

        # Safety: если tail аномально большой — warn
        if tail_prob > 0.01:
            logger.warning(
                f"Large tail: {tail_prob:.4f} "
                f"(lambda_h={lambda_h:.2f}, lambda_a={lambda_a:.2f}, max_goals={max_goals})"
            )

        # Scale matrix до (1 - tail_prob): matrix + tail = 1.0
        if tail_prob > 0 and tail_prob < 1.0:
            score_probs = {k: v * (1.0 - tail_prob) for k, v in score_probs.items()}

        return score_probs, tail_prob

    def optimize_rho(self, actual_scores: List[Tuple[int, int]],
                     lambda_h_list: List[float],
                     lambda_a_list: List[float],
                     time_weights: Optional[List[float]] = None,
                     league_labels: Optional[List[str]] = None) -> float:
        """
        Оптимизировать rho (корреляцию) через Maximum Likelihood Estimation.

        Args:
            actual_scores: список реальных счётов [(goals_h, goals_a), ...]
            lambda_h_list: список ожидаемых голов хозяев
            lambda_a_list: список ожидаемых голов гостей
            time_weights: веса матчей по времени (опционально).
            league_labels: список названий лиг (опционально).

        Returns:
            Оптимальное глобальное rho
        """
        # ==============================
        # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
        # ==============================
        n = len(actual_scores)
        if n != len(lambda_h_list):
            raise ValueError(
                f"actual_scores length ({n}) != lambda_h_list length ({len(lambda_h_list)})"
            )
        if n != len(lambda_a_list):
            raise ValueError(
                f"actual_scores length ({n}) != lambda_a_list length ({len(lambda_a_list)})"
            )
        if n < 50:
            logger.warning(f"Слишком мало данных для оптимизации rho: {n} (minimum 50)")
            return self.rho

        if league_labels is not None and len(league_labels) != n:
            raise ValueError(
                f"actual_scores length ({n}) != league_labels length ({len(league_labels)})"
            )

        use_time_decay = time_weights is not None
        if time_weights is None:
            time_weights = [1.0] * n

        # ==============================
        # ЗАЩИТА ОТ NaN/inf В LAMBDA
        # ==============================
        lambda_h_list = [
            max(1e-6, float(x)) if np.isfinite(x) else 1e-6
            for x in lambda_h_list
        ]
        lambda_a_list = [
            max(1e-6, float(x)) if np.isfinite(x) else 1e-6
            for x in lambda_a_list
        ]

        # Нормализация весов (сумма = n)
        w_sum = sum(time_weights)
        if w_sum > 0:
            norm_weights = [w * n / w_sum for w in time_weights]
        else:
            norm_weights = [1.0] * n

        def neg_log_likelihood(rho_val: float) -> float:
            """Взвешенное отрицательное лог-правдоподобие (минимизируем)."""
            self.rho = rho_val
            total_ll = 0.0

            for i, (gh, ga) in enumerate(actual_scores):
                lh = lambda_h_list[i]
                la = lambda_a_list[i]

                prob = self._safe_poisson_pmf(gh, lh) * \
                       self._safe_poisson_pmf(ga, la)
                prob *= self.tau(gh, ga, lh, la, rho_override=rho_val)

                if not np.isfinite(prob) or prob <= 1e-10:
                    total_ll += norm_weights[i] * -23  # log(1e-10) ~ -23
                else:
                    total_ll += norm_weights[i] * np.log(prob)

            return -total_ll

        # ============================================================
        # ОШИБКА №3 FIX: penalty = 10 (было 100 — убивало rho!)
        #
        # penalty=100: rho ≈ 0 всегда → Dixon-Coles вырождается в Poisson
        # penalty=10:  мягкая регуляризация, rho может быть 0.05-0.15
        #
        # Зачем нужен penalty вообще:
        # - Предотвращает прилипание к границам на шумных данных
        # - Если data мало, rho не улетает в экстремумы
        # ============================================================
        RHO_PENALTY = 10.0

        def neg_ll_with_penalty(rho_val: float) -> float:
            base_nll = neg_log_likelihood(rho_val)
            penalty = RHO_PENALTY * rho_val ** 2
            return base_nll + penalty

        # ============================================================
        # ОШИБКА №4 FIX: bounds = (-0.2, 0.2) (было [-0.1, 0.1])
        #
        # Dixon-Coles (1997): типичные rho = 0.03-0.15
        # Некоторые лиги: rho до 0.18-0.20
        # Узкие bounds обрезали реальные значения!
        # ============================================================
        RHO_BOUNDS = (-0.20, 0.20)

        result = minimize_scalar(
            neg_ll_with_penalty,
            bounds=RHO_BOUNDS,
            method='bounded'
        )

        self.rho = result.x
        self._rho_optimized = True

        logger.info(f"Dixon-Coles rho оптимизирован: rho={self.rho:.4f} "
                     f"(neg_ll={result.fun:.2f}, penalty={RHO_PENALTY}, "
                     f"bounds={RHO_BOUNDS}, "
                     f"matches={n}, time_decay={'ON' if use_time_decay else 'OFF'})")

        # ==============================
        # PER-LEAGUE RHO
        # ==============================
        if league_labels is not None:
            self._optimize_per_league_rho(
                actual_scores, lambda_h_list, lambda_a_list,
                norm_weights, league_labels, RHO_PENALTY, RHO_BOUNDS
            )

        return self.rho

    def _optimize_per_league_rho(self, actual_scores, lambda_h_list, lambda_a_list,
                                  norm_weights, league_labels, penalty_strength,
                                  rho_bounds):
        """
        Оптимизировать rho отдельно для каждой лиги (>=30 матчей).
        Меньшие лиги используют глобальный rho.
        """
        from collections import defaultdict

        league_indices = defaultdict(list)
        for i, league in enumerate(league_labels):
            if league:
                league_indices[league].append(i)

        MIN_MATCHES_FOR_LEAGUE_RHO = 30

        for league, indices in league_indices.items():
            if len(indices) < MIN_MATCHES_FOR_LEAGUE_RHO:
                continue  # мало данных — используем глобальный rho

            league_scores = [actual_scores[i] for i in indices]
            league_lh = [lambda_h_list[i] for i in indices]
            league_la = [lambda_a_list[i] for i in indices]
            league_w = [norm_weights[i] for i in indices]

            def league_neg_ll(rho_val):
                self.rho = rho_val
                total_ll = 0.0
                for j, (gh, ga) in enumerate(league_scores):
                    prob = self._safe_poisson_pmf(gh, league_lh[j]) * \
                           self._safe_poisson_pmf(ga, league_la[j])
                    prob *= self.tau(gh, ga, league_lh[j], league_la[j],
                                     rho_override=rho_val)
                    if not np.isfinite(prob) or prob <= 1e-10:
                        total_ll += league_w[j] * -23
                    else:
                        total_ll += league_w[j] * np.log(prob)
                return -total_ll

            def league_neg_ll_penalty(rho_val):
                return league_neg_ll(rho_val) + penalty_strength * rho_val ** 2

            result = minimize_scalar(
                league_neg_ll_penalty,
                bounds=rho_bounds,  # (-0.2, 0.2) — ОШИБКА №4 FIX
                method='bounded'
            )

            self.rho_per_league[league] = round(result.x, 5)
            logger.info(f"  League rho: {league}={result.x:.4f} "
                         f"(matches={len(indices)})")

        logger.info(f"Per-league rho: {len(self.rho_per_league)} лиг")

    def calculate_all_markets(self, score_probs: Dict[Tuple[int, int], float],
                              lambda_h: float, lambda_a: float,
                              league: Optional[str] = None,
                              tail_prob: float = 0.0) -> Dict[str, float]:
        """
        Рассчитать ВСЕ рынки ставок из матрицы + tail.

        Включает:
        - 1X2 (исход матча)
        - ТБ 1.5, ТБ 2.5 (общие тоталы)
        - ОЗ (обе забьют)
        - ИТБ 0.5/1.5 Home и Away (индивидуальные тоталы)
        - AH0 (азиатская фора 0) Home и Away
        - Точные счёта (top-5)

        Args:
            score_probs: матрица от predict_score_probability() (sum = 1 - tail_prob)
            lambda_h: ожидаемые голы хозяев
            lambda_a: ожидаемые голы гостей
            league: название лиги (для per-league rho в диагностике)
            tail_prob: вероятность хвоста (от predict_score_probability)

        Tail distribution:
          Tail = P(gh > max_goals OR ga > max_goals), min score in tail = 11
          → 100% tail → over 1.5, over 2.5
          → ~100% tail → BTTS yes, все ITB (min gh/ga в tail >> 0)
          → 1X2, AH0: распределяем пропорционально
        """
        markets = {}

        # ==============================
        # 1X2 (из матрицы)
        # ==============================
        prob_home = sum(p for (gh, ga), p in score_probs.items() if gh > ga)
        prob_draw = sum(p for (gh, ga), p in score_probs.items() if gh == ga)
        prob_away = sum(p for (gh, ga), p in score_probs.items() if gh < ga)

        # ==============================
        # Тоталы (из матрицы)
        # ==============================
        over_25 = sum(p for (gh, ga), p in score_probs.items()
                      if gh + ga > 2.5)
        over_15 = sum(p for (gh, ga), p in score_probs.items()
                      if gh + ga > 1.5)

        # ==============================
        # BTTS (из матрицы)
        # ==============================
        btts_yes = sum(p for (gh, ga), p in score_probs.items()
                       if gh > 0 and ga > 0)

        # ==============================
        # ITB (из матрицы)
        # ==============================
        home_itb_05 = sum(p for (gh, ga), p in score_probs.items() if gh > 0)
        home_itb_15 = sum(p for (gh, ga), p in score_probs.items() if gh >= 2)
        away_itb_05 = sum(p for (gh, ga), p in score_probs.items() if ga > 0)
        away_itb_15 = sum(p for (gh, ga), p in score_probs.items() if ga >= 2)

        # ==============================
        # ОШИБКА №1 FIX: Учитываем tail в КАЖДОМ рынке
        # ==============================
        if tail_prob > 1e-10:
            # --- Тоталы: ВСЕ tail outcomes > 2.5 и > 1.5 ---
            # Минимальный счёт в tail = max_goals+1 >= 11 >> 2.5
            over_25 += tail_prob
            over_15 += tail_prob

            # --- BTTS yes: ~100% tail ---
            # P(gh=0 в tail) = Poisson(0|λh) * P(ga>max_goals) ≈ 0
            # P(ga=0 в tail) = P(gh>max_goals) * Poisson(0|λa) ≈ 0
            btts_yes += tail_prob

            # --- ITB: ~100% tail ---
            # В tail: gh >= max_goals+1 >= 11 > 2 и ga >= 11 > 2
            # за исключением (0, 11+), (1, 11+) и (11+, 0), (11+, 1)
            # но P этих ≈ 0 при max_goals >= 10
            home_itb_05 += tail_prob
            home_itb_15 += tail_prob
            away_itb_05 += tail_prob
            away_itb_15 += tail_prob

            # --- 1X2: распределяем tail пропорционально ---
            # Tail outcomes — высокие счёты, распределены ~как 1X2 в матрице
            s_1x2 = prob_home + prob_draw + prob_away
            if s_1x2 > 0:
                factor = (s_1x2 + tail_prob) / s_1x2
                prob_home *= factor
                prob_draw *= factor
                prob_away *= factor

        # ==============================
        # Сохраняем вероятности
        # ==============================
        markets['prob_home'] = prob_home
        markets['prob_draw'] = prob_draw
        markets['prob_away'] = prob_away

        markets['over_2_5_prob'] = over_25
        markets['over_1_5_prob'] = over_15

        markets['btts_yes_prob'] = btts_yes
        markets['btts_no_prob'] = 1.0 - btts_yes

        markets['home_itb_0_5_prob'] = home_itb_05
        markets['home_itb_1_5_prob'] = home_itb_15
        markets['away_itb_0_5_prob'] = away_itb_05
        markets['away_itb_1_5_prob'] = away_itb_15

        # ==============================
        # AH0: из СКОРРЕКТИРОВАННОГО 1X2 (после tail!)
        # ==============================
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
        markets['rho'] = self._get_rho(league)
        markets['tail_prob'] = tail_prob

        return markets
