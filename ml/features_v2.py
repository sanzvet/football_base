"""
features_v2.py — Расширенный FeatureEngineer для гибридной модели.

Архитектура (оптимизированная):
- 33 признака
- ОДИН SQL запрос на весь датасет (preload)
- EWMA считается в памяти (без N+1 queries)
- ELO с home advantage (+100)
- Кросс-признаки (отношение атаки к защите)
- Target encoding для лиг (вместо ручного mapping)
- Global means для defaults (вместо хардкода)

Performance:
  Было: 10k матчей x 4 SQL = 40k запросов (~30-60 мин)
  Стало: 1 SQL запрос + всё в памяти (~30-60 сек)

Author: Hybrid Model V2
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineerV2:

    def __init__(self, pool):
        self.pool = pool

        # EWMA параметр: полураспад ~3 месяца (93 дня)
        # alpha=0.80 -> после 30 дней вес=0.80, после 93 дней вес=0.80^3=0.51
        self.ewma_alpha = 0.80

        # Home advantage для ELO (+100 очков)
        self.home_advantage = 100

        # ELO параметры
        self.elo_k = 20
        self.elo_base = 1500

        # Target encoding для лиг (вычисляется из данных, НЕ хардкод!)
        # league_target_enc[league] = средний total_goals в лиге
        self.league_target_enc: Dict[str, float] = {}

        # Global means для defaults (вычисляется из данных, НЕ хардкод!)
        # Используется когда у команды нет истории матчей
        self.global_means: Dict[str, float] = {}

    # ==========================
    # PRELOAD (1 SQL запрос)
    # ==========================
    async def _preload_all_stats(self) -> pd.DataFrame:
        """
        Единственный SQL запрос — загружает ВСЕ матчи + ВСЕ статистики.
        Вместо 40k запросов — 1 запрос, всё остальное в памяти.
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    m.match_id, m.datetime, m.league,
                    m.home_team_id, m.away_team_id,
                    m.goals_h, m.goals_a, m.status,
                    ms_home.xG as home_xg, ms_home.npxG as home_npxg,
                    ms_home.npxGA as home_npxga,
                    ms_home.ppda_att as home_ppda_att,
                    ms_home.ppda_def as home_ppda_def,
                    ms_home.deep as home_deep,
                    ms_home.deep_allowed as home_deep_allowed,
                    ms_away.xG as away_xg, ms_away.npxG as away_npxg,
                    ms_away.npxGA as away_npxga,
                    ms_away.ppda_att as away_ppda_att,
                    ms_away.ppda_def as away_ppda_def,
                    ms_away.deep as away_deep,
                    ms_away.deep_allowed as away_deep_allowed
                FROM matches m
                JOIN match_team_stats ms_home
                    ON m.match_id = ms_home.match_id
                    AND m.home_team_id = ms_home.team_id
                JOIN match_team_stats ms_away
                    ON m.match_id = ms_away.match_id
                    AND m.away_team_id = ms_away.team_id
                WHERE m.status = 'Result'
                ORDER BY m.datetime ASC
            """)

        df = pd.DataFrame([dict(r) for r in rows])
        df['datetime'] = pd.to_datetime(df['datetime'])
        logger.info(f"Preload: {len(df)} матчей загружено (1 SQL запрос)")
        return df

    # ==========================
    # GLOBAL MEANS & TARGET ENCODING
    # ==========================
    def _compute_global_stats(self, df: pd.DataFrame):
        """
        Вычислить global means и target encoding для лиг из данных.
        Никакого хардкода — всё из реальных данных.
        """
        # Global means — средние по всем матчам для defaults
        all_home_stats = df[['home_xg', 'home_npxg', 'home_npxga',
                             'home_ppda_att', 'home_ppda_def',
                             'home_deep', 'home_deep_allowed']].values
        all_away_stats = df[['away_xg', 'away_npxg', 'away_npxga',
                             'away_ppda_att', 'away_ppda_def',
                             'away_deep', 'away_deep_allowed']].values
        all_stats = np.vstack([all_home_stats, all_away_stats])

        stat_names = ['avg_xg', 'avg_npxg', 'avg_npxga',
                      'avg_ppda_att', 'avg_ppda_def',
                      'avg_deep', 'avg_deep_allowed']

        self.global_means = {}
        for i, name in enumerate(stat_names):
            self.global_means[name] = float(np.nanmean(all_stats[:, i]))

        # Goals averages
        self.global_means['avg_goals_scored'] = float(
            np.mean(np.concatenate([df['goals_h'].values, df['goals_a'].values])))
        self.global_means['avg_goals_conceded'] = self.global_means['avg_goals_scored']

        logger.info(f"Global means: xg={self.global_means['avg_xg']:.2f}, "
                     f"goals={self.global_means['avg_goals_scored']:.2f}")

        # Target encoding для лиг: средний total_goals по лиге
        df['total_goals'] = df['goals_h'] + df['goals_a']
        league_means = df.groupby('league')['total_goals'].mean()

        # Глобальный mean как fallback для неизвестных лиг
        global_goal_mean = float(df['total_goals'].mean())

        self.league_target_enc = {}
        for league, mean_goals in league_means.items():
            self.league_target_enc[league] = round(mean_goals, 3)

        self._league_global_mean = global_goal_mean
        logger.info(f"League target encoding: {self.league_target_enc}")

    def _get_league_encoding(self, league: str) -> float:
        """
        Target encoding для лиги: средний total_goals.
        Новые/неизвестные лиги получают global mean.
        """
        if league and league in self.league_target_enc:
            return self.league_target_enc[league]
        return self._league_global_mean

    # ==========================
    # ELO с Home Advantage
    # ==========================
    def calculate_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать ELO рейтинг для всех команд.
        Home advantage: +100 очков хозяевам перед расчётом вероятности.
        Обновление рейтинга происходит БЕЗ home advantage (только чистый рейтинг).
        """
        ratings = {}
        home_elo = []
        away_elo = []

        for _, row in df.sort_values('datetime').iterrows():
            h = row['home_team_id']
            a = row['away_team_id']

            if h not in ratings:
                ratings[h] = self.elo_base
            if a not in ratings:
                ratings[a] = self.elo_base

            # Рейтинг с учётом домашнего преимущества
            Rh = ratings[h] + self.home_advantage
            Ra = ratings[a]

            home_elo.append(Rh)
            away_elo.append(Ra)

            # Результат матча
            if row['goals_h'] > row['goals_a']:
                Sh, Sa = 1, 0
            elif row['goals_h'] < row['goals_a']:
                Sh, Sa = 0, 1
            else:
                Sh, Sa = 0.5, 0.5

            # Ожидаемый результат
            Eh = 1 / (1 + 10 ** ((Ra - Rh) / 400))
            Ea = 1 - Eh

            # Обновление рейтингов (БЕЗ home advantage)
            ratings[h] = ratings[h] + self.elo_k * (Sh - Eh)
            ratings[a] = ratings[a] + self.elo_k * (Sa - Ea)

        df = df.copy()
        df['home_elo'] = home_elo
        df['away_elo'] = away_elo

        return df

    # ==========================
    # EWMA (в памяти, без SQL)
    # ==========================
    @staticmethod
    def _ewma(values: np.ndarray, days_diff: np.ndarray,
              alpha: float) -> float:
        """
        Экспоненциально взвешенное среднее с time decay.

        Формула: weight_i = alpha ^ (days_diff_i / 30)
        Чем свежее матч — тем выше вес.
        """
        if len(values) == 0:
            return 0.0

        # Векторизовано (без цикла)
        days_diff = np.maximum(days_diff, 0)
        weights = alpha ** (days_diff / 30.0)

        total_weight = weights.sum()
        if total_weight == 0:
            return float(values.mean())

        return float(np.average(values, weights=weights))

    # ==========================
    # Форма команды (в памяти, без SQL)
    # ==========================
    def _get_team_form(self, all_matches: pd.DataFrame, team_id: int,
                       reference_date: datetime, venue: str,
                       n: int = 5) -> Dict[str, float]:
        """
        Расчёт формы команды. Всё в памяти — никаких SQL запросов.

        venue:
            'home' - только домашние матчи команды
            'away' - только гостевые матчи команды
            'all'  - все матчи команды
        """
        # Фильтруем: матчи команды ДО reference_date (строго!)
        mask_team = (
            (all_matches['home_team_id'] == team_id) |
            (all_matches['away_team_id'] == team_id)
        )
        mask_date = all_matches['datetime'] < reference_date
        filtered = all_matches[mask_team & mask_date].copy()

        # Safety assert: ни один матч не >= reference_date
        if len(filtered) > 0:
            assert filtered['datetime'].max() < reference_date, (
                f"LEAKAGE DETECTED: team_id={team_id} has match at "
                f"{filtered['datetime'].max()} >= {reference_date}"
            )

        if venue == 'home':
            filtered = filtered[filtered['home_team_id'] == team_id]
        elif venue == 'away':
            filtered = filtered[filtered['away_team_id'] == team_id]

        # Берём последние n матчей (DESC по дате)
        filtered = filtered.sort_values('datetime', ascending=False).head(n)

        if len(filtered) == 0:
            return self.global_means.copy()

        # Векторизованное извлечение значений
        is_home = filtered['home_team_id'] == team_id

        goals_scored = np.where(is_home, filtered['goals_h'].values,
                                filtered['goals_a'].values)
        goals_conceded = np.where(is_home, filtered['goals_a'].values,
                                  filtered['goals_h'].values)

        if venue in ('home', 'away'):
            prefix = 'home_' if venue == 'home' else 'away_'
            xg_vals = filtered[f'{prefix}xg'].values
            npxg_vals = filtered[f'{prefix}npxg'].values
            npxga_vals = filtered[f'{prefix}npxga'].values
            ppda_att_vals = filtered[f'{prefix}ppda_att'].values
            ppda_def_vals = filtered[f'{prefix}ppda_def'].values
            deep_vals = filtered[f'{prefix}deep'].values
            deep_allowed_vals = filtered[f'{prefix}deep_allowed'].values
        else:
            # 'all' — берём stat по принадлежности команды
            xg_vals = np.where(is_home, filtered['home_xg'].values,
                               filtered['away_xg'].values)
            npxg_vals = np.where(is_home, filtered['home_npxg'].values,
                                 filtered['away_npxg'].values)
            npxga_vals = np.where(is_home, filtered['home_npxga'].values,
                                  filtered['away_npxga'].values)
            ppda_att_vals = np.where(is_home, filtered['home_ppda_att'].values,
                                     filtered['away_ppda_att'].values)
            ppda_def_vals = np.where(is_home, filtered['home_ppda_def'].values,
                                     filtered['away_ppda_def'].values)
            deep_vals = np.where(is_home, filtered['home_deep'].values,
                                 filtered['away_deep'].values)
            deep_allowed_vals = np.where(is_home,
                                         filtered['home_deep_allowed'].values,
                                         filtered['away_deep_allowed'].values)

        days_diff = (reference_date - filtered['datetime']).dt.days.values

        return {
            'avg_xg': self._ewma(xg_vals, days_diff, self.ewma_alpha),
            'avg_npxg': self._ewma(npxg_vals, days_diff, self.ewma_alpha),
            'avg_npxga': self._ewma(npxga_vals, days_diff, self.ewma_alpha),
            'avg_goals_scored': self._ewma(goals_scored, days_diff, self.ewma_alpha),
            'avg_goals_conceded': self._ewma(goals_conceded, days_diff, self.ewma_alpha),
            'avg_ppda_att': self._ewma(ppda_att_vals, days_diff, self.ewma_alpha),
            'avg_ppda_def': self._ewma(ppda_def_vals, days_diff, self.ewma_alpha),
            'avg_deep': self._ewma(deep_vals, days_diff, self.ewma_alpha),
            'avg_deep_allowed': self._ewma(deep_allowed_vals, days_diff, self.ewma_alpha),
        }

    # ==========================
    # LOAD TRAINING DATA
    # ==========================
    async def get_training_data(self) -> pd.DataFrame:
        """Загрузить данные для обучения (1 SQL запрос)."""
        return await self._preload_all_stats()

    # ==========================
    # BUILD FEATURE MATRIX (обучение)
    # ==========================
    async def build_feature_matrix(self, df: pd.DataFrame,
                                   reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Построить матрицу признаков для обучения.
        33 признака + 2 target (goals_home, goals_away) + time_weight.

        Все вычисления в памяти — 0 SQL запросов после preload.
        """
        if reference_date is None:
            reference_date = df['datetime'].max()

        logger.info("Строим матрицу признаков V2 (preload + in-memory EWMA)...")

        # Global stats НЕ пересчитываются здесь!
        # Они должны быть установлены через _compute_global_stats(train_raw) ДО вызова.
        # Если не установлены — считаем из переданного df (fallback).
        if not self.global_means:
            logger.warning("global_means не установлены! Считаем из переданного df (возможен leakage!)")
            self._compute_global_stats(df)

        df = self.calculate_elo(df)

        features = []
        total = len(df)
        max_date = df['datetime'].max()

        for idx, (_, row) in enumerate(df.iterrows()):
            if idx % 500 == 0:
                logger.info(f"  Прогресс: {idx}/{total} матчей...")

            match_date = row['datetime']

            # 4 вызова формы — всё в памяти, без SQL
            hf = self._get_team_form(df, row['home_team_id'], match_date, venue='home')
            hf_all = self._get_team_form(df, row['home_team_id'], match_date, venue='all')
            af = self._get_team_form(df, row['away_team_id'], match_date, venue='away')
            af_all = self._get_team_form(df, row['away_team_id'], match_date, venue='all')

            # CROSS FEATURES
            xg_home_vs_npxga_away = (
                hf['avg_xg'] / af['avg_npxga']
                if af['avg_npxga'] > 0 else 1.0)
            ppda_home_vs_away = (
                hf['avg_ppda_att'] / af['avg_ppda_def']
                if af['avg_ppda_def'] > 0 else 1.0)
            deep_home_vs_away = (
                hf['avg_deep'] / af['avg_deep_allowed']
                if af['avg_deep_allowed'] > 0 else 1.0)

            # TIME WEIGHT: НОВЫЕ матчи = БОЛЬШИЙ вес
            days_from_end = (max_date - match_date).days
            time_weight = np.exp(-days_from_end / 365)

            features.append({
                # HOME ATTACK (домашние матчи хозяев)
                'home_avg_xg': hf['avg_xg'],
                'home_avg_npxg': hf['avg_npxg'],
                'home_avg_goals_scored': hf['avg_goals_scored'],
                'home_avg_ppda_att': hf['avg_ppda_att'],
                'home_avg_deep': hf['avg_deep'],
                # HOME DEFENSE
                'home_avg_npxga': hf['avg_npxga'],
                'home_avg_goals_conceded': hf['avg_goals_conceded'],
                'home_avg_ppda_def': hf['avg_ppda_def'],
                'home_avg_deep_allowed': hf['avg_deep_allowed'],
                # HOME OVERALL
                'home_overall_avg_xg': hf_all['avg_xg'],
                'home_overall_avg_npxg': hf_all['avg_npxg'],
                'home_overall_avg_goals_scored': hf_all['avg_goals_scored'],
                'home_overall_avg_goals_conceded': hf_all['avg_goals_conceded'],
                # AWAY ATTACK
                'away_avg_xg': af['avg_xg'],
                'away_avg_npxg': af['avg_npxg'],
                'away_avg_goals_scored': af['avg_goals_scored'],
                'away_avg_ppda_att': af['avg_ppda_att'],
                'away_avg_deep': af['avg_deep'],
                # AWAY DEFENSE
                'away_avg_npxga': af['avg_npxga'],
                'away_avg_goals_conceded': af['avg_goals_conceded'],
                'away_avg_ppda_def': af['avg_ppda_def'],
                'away_avg_deep_allowed': af['avg_deep_allowed'],
                # AWAY OVERALL
                'away_overall_avg_xg': af_all['avg_xg'],
                'away_overall_avg_npxg': af_all['avg_npxg'],
                'away_overall_avg_goals_scored': af_all['avg_goals_scored'],
                'away_overall_avg_goals_conceded': af_all['avg_goals_conceded'],
                # ELO
                'home_elo': row['home_elo'],
                'away_elo': row['away_elo'],
                'elo_diff': row['home_elo'] - row['away_elo'],
                # CROSS FEATURES
                'xg_home_vs_npxga_away': xg_home_vs_npxga_away,
                'ppda_home_vs_away': ppda_home_vs_away,
                'deep_home_vs_away': deep_home_vs_away,
                # LEAGUE (target encoding)
                'league_encoded': self._get_league_encoding(row.get('league', '')),
                # TIME
                'time_weight': time_weight,
                # TARGETS (реальные голы, НЕ xG!)
                'target_goals_home': row['goals_h'],
                'target_goals_away': row['goals_a'],
            })

        feature_df = pd.DataFrame(features).fillna(0)
        logger.info(f"Матрица признаков V2 готова: {feature_df.shape}")
        return feature_df

    # ==========================
    # PREDICTION FEATURES (один матч)
    # ==========================
    async def get_prediction_features(self, home_team_id: int,
                                      away_team_id: int,
                                      league: str = None) -> Dict[str, float]:
        """
        Получить признаки для прогнозирования одного матча.
        1 SQL запрос (preload) + всё в памяти.
        """
        now = datetime.now()

        # Preload (1 запрос вместо 5)
        all_matches = await self._preload_all_stats()

        if len(all_matches) == 0:
            raise ValueError("Нет завершённых матчей в базе данных!")

        # Global means для defaults
        self._compute_global_stats(all_matches)

        # ELO
        all_matches = self.calculate_elo(all_matches)

        home_matches = all_matches[
            (all_matches['home_team_id'] == home_team_id) |
            (all_matches['away_team_id'] == home_team_id)
        ]
        away_matches = all_matches[
            (all_matches['home_team_id'] == away_team_id) |
            (all_matches['away_team_id'] == away_team_id)
        ]

        if len(home_matches) == 0:
            raise ValueError(f"Нет данных о команде хозяев (ID: {home_team_id})")
        if len(away_matches) == 0:
            raise ValueError(f"Нет данных о гостевой команде (ID: {away_team_id})")

        last_home = home_matches.sort_values('datetime').iloc[-1]
        last_away = away_matches.sort_values('datetime').iloc[-1]

        # Форма (в памяти, без SQL)
        hf = self._get_team_form(all_matches, home_team_id, now, venue='home')
        hf_all = self._get_team_form(all_matches, home_team_id, now, venue='all')
        af = self._get_team_form(all_matches, away_team_id, now, venue='away')
        af_all = self._get_team_form(all_matches, away_team_id, now, venue='all')

        # Кросс-признаки
        xg_home_vs_npxga_away = (
            hf['avg_xg'] / af['avg_npxga']
            if af['avg_npxga'] > 0 else 1.0)
        ppda_home_vs_away = (
            hf['avg_ppda_att'] / af['avg_ppda_def']
            if af['avg_ppda_def'] > 0 else 1.0)
        deep_home_vs_away = (
            hf['avg_deep'] / af['avg_deep_allowed']
            if af['avg_deep_allowed'] > 0 else 1.0)

        return {
            'home_avg_xg': hf['avg_xg'],
            'home_avg_npxg': hf['avg_npxg'],
            'home_avg_goals_scored': hf['avg_goals_scored'],
            'home_avg_ppda_att': hf['avg_ppda_att'],
            'home_avg_deep': hf['avg_deep'],
            'home_avg_npxga': hf['avg_npxga'],
            'home_avg_goals_conceded': hf['avg_goals_conceded'],
            'home_avg_ppda_def': hf['avg_ppda_def'],
            'home_avg_deep_allowed': hf['avg_deep_allowed'],
            'home_overall_avg_xg': hf_all['avg_xg'],
            'home_overall_avg_npxg': hf_all['avg_npxg'],
            'home_overall_avg_goals_scored': hf_all['avg_goals_scored'],
            'home_overall_avg_goals_conceded': hf_all['avg_goals_conceded'],
            'away_avg_xg': af['avg_xg'],
            'away_avg_npxg': af['avg_npxg'],
            'away_avg_goals_scored': af['avg_goals_scored'],
            'away_avg_ppda_att': af['avg_ppda_att'],
            'away_avg_deep': af['avg_deep'],
            'away_avg_npxga': af['avg_npxga'],
            'away_avg_goals_conceded': af['avg_goals_conceded'],
            'away_avg_ppda_def': af['avg_ppda_def'],
            'away_avg_deep_allowed': af['avg_deep_allowed'],
            'away_overall_avg_xg': af_all['avg_xg'],
            'away_overall_avg_npxg': af_all['avg_npxg'],
            'away_overall_avg_goals_scored': af_all['avg_goals_scored'],
            'away_overall_avg_goals_conceded': af_all['avg_goals_conceded'],
            'home_elo': last_home['home_elo'],
            'away_elo': last_away['away_elo'],
            'elo_diff': last_home['home_elo'] - last_away['away_elo'],
            'xg_home_vs_npxga_away': xg_home_vs_npxga_away,
            'ppda_home_vs_away': ppda_home_vs_away,
            'deep_home_vs_away': deep_home_vs_away,
            'league_encoded': self._get_league_encoding(league),
            'time_weight': 1.0,
        }

    # ==========================
    # FEATURE LIST
    # ==========================
    def get_feature_columns(self) -> List[str]:
        """Список из 33 признаков для ML модели."""
        return [
            'home_avg_xg', 'home_avg_npxg', 'home_avg_goals_scored',
            'home_avg_ppda_att', 'home_avg_deep',
            'home_avg_npxga', 'home_avg_goals_conceded',
            'home_avg_ppda_def', 'home_avg_deep_allowed',
            'home_overall_avg_xg', 'home_overall_avg_npxg',
            'home_overall_avg_goals_scored', 'home_overall_avg_goals_conceded',
            'away_avg_xg', 'away_avg_npxg', 'away_avg_goals_scored',
            'away_avg_ppda_att', 'away_avg_deep',
            'away_avg_npxga', 'away_avg_goals_conceded',
            'away_avg_ppda_def', 'away_avg_deep_allowed',
            'away_overall_avg_xg', 'away_overall_avg_npxg',
            'away_overall_avg_goals_scored', 'away_overall_avg_goals_conceded',
            'home_elo', 'away_elo', 'elo_diff',
            'xg_home_vs_npxga_away', 'ppda_home_vs_away', 'deep_home_vs_away',
            'league_encoded', 'time_weight',
        ]
