"""
features_v2.py — Расширенный FeatureEngineer для гибридной модели.

Архитектура (оптимизированная):
- 35 признаков (33 + 2 missing flags)
- ОДИН SQL запрос на весь датасет (preload)
- EWMA считается в памяти: exp(-days/tau), tau ≈ 45 дней
- ELO с data-driven home advantage (per-league)
- Rolling target encoding для лиг (без leakage!)
- Global means для defaults (вычисляются ТОЛЬКО на train)
- NaN-safe: EWMA возвращает NaN при пустоте, fillna(global_means)
- Missing flags: модель знает, когда данных нет (не путает с 0!)
- O(N*k) вместо O(N²): precomputed team_histories dict

Performance:
  Было: 10k матчей x 4 SQL = 40k запросов (~30-60 мин)
  Стало: 1 SQL запрос + всё в памяти (~30-60 сек)

Leakage prevention:
  - Global means: computed on train split ONLY (ValueError if not set)
  - League encoding: rolling expanding mean with shift(1)
  - ELO: sort by datetime, reset_index before append
  - Team form: logger.error (not assert) for leakage detection
  - EWMA: returns NaN (not 0.0) for missing data
  - Home advantage: computed from train split only

Author: Hybrid Model V2
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# Маленький epsilon для деления вместо магического 1.0
_EPS = 1e-6

# Ключи формы команды (DRY — используется в 5 местах)
_FORM_KEYS = [
    'avg_xg', 'avg_npxg', 'avg_npxga',
    'avg_goals_scored', 'avg_goals_conceded',
    'avg_ppda_att', 'avg_ppda_def',
    'avg_deep', 'avg_deep_allowed',
]


class FeatureEngineerV2:

    def __init__(self, pool):
        self.pool = pool

        # ОШИБКА №6 FIX: EWMA через tau (стандартное экспоненциальное затухание)
        # exp(-days/tau): после tau дней вес = 1/e ≈ 0.368
        # tau=45: полураспад ≈ 31 день (2*tau/ln2 ≈ 65 дней до 1/e²)
        self.ewma_tau = 45.0

        # ОШИБКА №7 FIX: Home advantage — data-driven (не хардкод 100!)
        # Вычисляется из _compute_home_advantage() на train split
        # Default 65 — стандартное значение для футбольных ELO систем
        self.home_advantage = 65.0

        # Per-league home advantage: {league_name: ha_value}
        # Заполняется из _compute_home_advantage()
        self._home_advantage_per_league: Dict[str, float] = {}
        self._home_advantage_computed = False

        # ELO параметры
        self.elo_k = 20
        self.elo_base = 1500

        # Rolling target encoding для лиг: {match_index: encoding_value}
        # Вычисляется один раз, используется при построении матрицы
        self._league_encoding_map: Dict[int, float] = {}
        self._league_encoding_df_index: List[int] = []

        # Static dict для prediction (last known encoding per league)
        # FIX #3: Инициализируем пустым, иначе _get_league_encoding() падает
        # с AttributeError при prediction (get_prediction_features НЕ вызывает
        # _compute_rolling_league_encoding)
        self.league_target_enc: Dict[str, float] = {}

        # Global means для defaults (вычисляется из данных, НЕ хардкод!)
        # Используется когда у команды нет истории матчей
        self.global_means: Dict[str, float] = {}

        # Глобальный mean тоталов для неизвестных лиг
        self._league_global_mean: float = 2.5

    @staticmethod
    def _nan_form_dict() -> Dict[str, float]:
        """Helper: вернуть dict с NaN для всех ключей формы."""
        return {k: float('nan') for k in _FORM_KEYS}

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
    # GLOBAL MEANS (train only!)
    # ==========================
    def _compute_global_stats(self, df: pd.DataFrame):
        """
        Вычислить global means из данных.

        КРИТИЧЕСКИ: вызывать ТОЛЬКО на train split!
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

        # Глобальный mean тоталов для fallback лиг
        self._league_global_mean = float((df['goals_h'] + df['goals_a']).mean())

        logger.info(f"Global means: xg={self.global_means['avg_xg']:.2f}, "
                     f"goals={self.global_means['avg_goals_scored']:.2f}, "
                     f"league_global_mean={self._league_global_mean:.2f}")

    # ==========================
    # HOME ADVANTAGE (data-driven, per-league)
    # ==========================
    def _compute_home_advantage(self, df: pd.DataFrame):
        """
        Вычислить home advantage из данных (НЕ хардкод!).

        ELO home advantage — сколько очков прибавлять хозяевам.
        Формула: HA = -400 * log10(1/E - 1), где E = home_expected_score
        E = (home_wins + 0.5 * draws) / total

        Типичные значения:
          - EPL/La Liga: 50-70
          - Bundesliga: 60-80
          - Нижние лиги: 40-60
          - Хардкод 100 — ЗАВЫШЕН (64% home win при равных, реально ~46%)

        Вызывать ТОЛЬКО на train split!
        """
        total = len(df)
        if total == 0:
            self.home_advantage = 65.0
            return

        # Global home advantage
        home_wins = int((df['goals_h'] > df['goals_a']).sum())
        draws = int((df['goals_h'] == df['goals_a']).sum())
        home_score = home_wins + 0.5 * draws
        home_expected = home_score / total

        if 0 < home_expected < 1:
            self.home_advantage = round(
                -400 * np.log10(1.0 / home_expected - 1.0), 1
            )
        else:
            self.home_advantage = 65.0

        # Per-league home advantage (>= 30 матчей в лиге)
        self._home_advantage_per_league = {}
        if 'league' in df.columns:
            for league, group in df.groupby('league'):
                n = len(group)
                if n < 30:
                    continue
                hw = int((group['goals_h'] > group['goals_a']).sum())
                dr = int((group['goals_h'] == group['goals_a']).sum())
                hs = hw + 0.5 * dr
                he = hs / n
                if 0 < he < 1:
                    self._home_advantage_per_league[league] = round(
                        -400 * np.log10(1.0 / he - 1.0), 1
                    )

        self._home_advantage_computed = True

        logger.info(
            f"Home advantage: global={self.home_advantage:.1f}, "
            f"per-league: {len(self._home_advantage_per_league)} лиг"
        )

    def _get_home_advantage(self, league: Optional[str] = None) -> float:
        """Получить home advantage для лиги (fallback на глобальный)."""
        if league and league in self._home_advantage_per_league:
            return self._home_advantage_per_league[league]
        return self.home_advantage

    # ==========================
    # ROLLING TARGET ENCODING (без leakage!)
    # ==========================
    def _compute_rolling_league_encoding(self, df: pd.DataFrame):
        """
        Rolling expanding target encoding для лиг.

        Для каждого матча: средний total_goals в этой лиге
        по ВСЕМ ПРЕДЫДУЩИМ матчам (expanding + shift(1)).

        Первый матч в лиге получает global_mean (нет предыдущих).
        Это предотвращает leakage — каждый матч видит только прошлое.
        """
        df = df.sort_values('datetime').copy()
        df['_total_goals'] = df['goals_h'] + df['goals_a']

        # Expanding mean по лиге с shift(1) — текущий матч НЕ включён
        expanding_means = (
            df.groupby('league')['_total_goals']
            .expanding()
            .mean()
            .shift(1)
            .reset_index(level=0, drop=True)
        )

        # Filling: первый матч лиги = global mean
        expanding_means = expanding_means.fillna(self._league_global_mean)

        # Сохраняем: маппинг позиция в df -> encoding value
        # Ключи = df.index ( НЕ enumerate position!)
        self._league_encoding_df_index = list(df.index)
        self._league_encoding_map = {
            idx: round(expanding_means.iloc[i], 3)
            for i, idx in enumerate(df.index)
        }

        # Also keep a simple static dict for prediction (last known encoding)
        # На момент предикта используем последний rolling mean для каждой лиги
        last_per_league = (
            df.groupby('league')['_total_goals']
            .expanding()
            .mean()
            .reset_index(level=0, drop=True)
        )
        self.league_target_enc = {}
        for league in df['league'].unique():
            league_vals = last_per_league[df['league'] == league]
            if len(league_vals) > 0:
                self.league_target_enc[league] = round(float(league_vals.iloc[-1]), 3)

        logger.info(f"Rolling league encoding: {len(self.league_target_enc)} лиг, "
                     f"{len(self._league_encoding_map)} записей в map")

    def _get_league_encoding_by_index(self, df_index: int) -> float:
        """Получить rolling league encoding для конкретного матча по df.index."""
        if df_index in self._league_encoding_map:
            return self._league_encoding_map[df_index]
        return self._league_global_mean

    def _get_league_encoding(self, league: str) -> float:
        """
        Target encoding для лиги (для prediction).
        Использует последний rolling mean для лиги.
        Новые/неизвестные лиги получают global mean.
        """
        if league and league in self.league_target_enc:
            return self.league_target_enc[league]
        return self._league_global_mean

    # ==========================
    # ELO с Home Advantage (векторизованный, без leakage)
    # ==========================
    def calculate_elo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитать ELO рейтинг для всех команд.

        Home advantage: +100 очков хозяевам перед расчётом вероятности.
        Обновление рейтинга происходит БЕЗ home advantage (только чистый рейтинг).

        КРИТИЧЕСКИ: df сортируется по datetime, ELO считается в порядке времени,
        затем results привязываются обратно к оригинальным индексам.
        """
        df = df.copy()

        # Сортируем по времени для корректного расчёта ELO
        df_sorted = df.sort_values('datetime').reset_index(drop=True)

        ratings = {}
        home_elo = []
        away_elo = []

        for idx, row in df_sorted.iterrows():
            h = row['home_team_id']
            a = row['away_team_id']

            if h not in ratings:
                ratings[h] = self.elo_base
            if a not in ratings:
                ratings[a] = self.elo_base

            # ОШИБКА №7 FIX: per-league home advantage (data-driven)
            ha = self._get_home_advantage(row.get('league'))

            # Рейтинг с учётом домашнего преимущества
            Rh = ratings[h] + ha
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

        # Привязываем ELO к отсортированному df, потом восстанавливаем оригинальный порядок
        df_sorted['home_elo'] = home_elo
        df_sorted['away_elo'] = away_elo

        # Возвращаем в оригинальном порядке индексов
        df_sorted = df_sorted.set_index(df.index)
        df['home_elo'] = df_sorted['home_elo']
        df['away_elo'] = df_sorted['away_elo']

        # Сохраняем текущие рейтинги для prediction
        self._current_elo_ratings = ratings

        return df

    # ==========================
    # EWMA (в памяти, без SQL, NaN-safe)
    # ==========================
    @staticmethod
    def _ewma(values: np.ndarray, days_diff: np.ndarray,
              tau: float) -> float:
        """
        Экспоненциально взвешенное среднее с time decay.

        ОШИБКА №6 FIX: стандартная формула exp(-days/tau)
        weight_i = exp(-days_diff_i / tau)

        tau = 45 дней:
          - 0 дней: вес = 1.0
          - 15 дней: вес = 0.72
          - 30 дней: вес = 0.51
          - 45 дней: вес = 0.37 (= 1/e)
          - 90 дней: вес = 0.14
          - 180 дней: вес = 0.018

        Чем больше tau — тем больше «память».
        Была формула alpha^(days/30) — НЕ стандартное затухание.

        Возвращает NaN если нет данных (не 0.0!).
        """
        if len(values) == 0:
            return float('nan')

        # Векторизовано (без цикла)
        days_diff = np.maximum(days_diff, 0)
        # ОШИБКА №6 FIX: правильное экспоненциальное затухание
        weights = np.exp(-days_diff / tau)

        total_weight = weights.sum()
        if total_weight == 0:
            return float('nan')  # Не 0.0 — NaN будет заполнен global_means

        return float(np.average(values, weights=weights))

    # ==========================
    # TEAM HISTORIES CACHE (O(N*k) вместо O(N²))
    # ==========================
    def _precompute_team_histories(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Pre-index матчей по team_id для O(1) lookups.

        Вместо O(N) scan по всему DataFrame для каждого вызова _get_team_form(),
        получаем O(1) lookup + O(k) фильтрация по дате (k = матчей команды, обычно 50-200).

        Performance:
          Было: 4 вызова _get_team_form() x N матчей x O(N) scan = O(4N²)
          Стало: 4 вызова x N матчей x O(k) фильтрация = O(4N*k), k << N
        """
        histories = defaultdict(list)

        for _, row in df.iterrows():
            histories[row['home_team_id']].append(row)
            histories[row['away_team_id']].append(row)

        # Конвертируем в sorted DataFrames (один раз)
        result = {}
        for team_id, rows in histories.items():
            team_df = pd.DataFrame(rows).sort_values('datetime')
            result[team_id] = team_df

        logger.info(f"Team histories precomputed: {len(result)} teams "
                     f"(avg matches/team: {len(df) / max(len(result), 1):.0f})")
        return result

    # ==========================
    # Форма команды (в памяти, без SQL, NaN-safe)
    # ==========================
    def _get_team_form(self, all_matches: Union[pd.DataFrame, Dict[int, pd.DataFrame]],
                       team_id: int, reference_date: datetime,
                       venue: str, n: int = 5) -> Dict[str, float]:
        """
        Расчёт формы команды. Всё в памяти — никаких SQL запросов.

        Args:
            all_matches: либо полный DataFrame (original path для prediction),
                         либо precomputed dict {team_id: DataFrame} (fast path для training)
            team_id: ID команды
            reference_date: дата матча (строго <)
            venue: 'home' | 'away' | 'all'
            n: количество последних матчей

        Возвращает NaN для команд без истории (заполняется global_means позже).
        """
        # ---- Fast path: precomputed dict {team_id: team_df} ----
        if isinstance(all_matches, dict):
            team_df = all_matches.get(team_id)
            if team_df is None or len(team_df) == 0:
                return self._nan_form_dict()

            # Фильтруем по дате (строго < reference_date)
            filtered = team_df[team_df['datetime'] < reference_date].copy()
        else:
            # ---- Original path: full DataFrame (для single prediction) ----
            mask_team = (
                (all_matches['home_team_id'] == team_id) |
                (all_matches['away_team_id'] == team_id)
            )
            mask_date = all_matches['datetime'] < reference_date
            filtered = all_matches[mask_team & mask_date].copy()

        # Safety check: ни один матч не >= reference_date (logger, не assert!)
        if len(filtered) > 0 and filtered['datetime'].max() >= reference_date:
            logger.error(
                f"LEAKAGE DETECTED: team_id={team_id} has match at "
                f"{filtered['datetime'].max()} >= {reference_date}. Skipping."
            )
            return self._nan_form_dict()

        if venue == 'home':
            filtered = filtered[filtered['home_team_id'] == team_id]
        elif venue == 'away':
            filtered = filtered[filtered['away_team_id'] == team_id]

        # Берём последние n матчей (DESC по дате)
        filtered = filtered.sort_values('datetime', ascending=False).head(n)

        if len(filtered) == 0:
            # Возвращаем NaN (не global_means!) — fillna будет позже
            return self._nan_form_dict()

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
            'avg_xg': self._ewma(xg_vals, days_diff, self.ewma_tau),
            'avg_npxg': self._ewma(npxg_vals, days_diff, self.ewma_tau),
            'avg_npxga': self._ewma(npxga_vals, days_diff, self.ewma_tau),
            'avg_goals_scored': self._ewma(goals_scored, days_diff, self.ewma_tau),
            'avg_goals_conceded': self._ewma(goals_conceded, days_diff, self.ewma_tau),
            'avg_ppda_att': self._ewma(ppda_att_vals, days_diff, self.ewma_tau),
            'avg_ppda_def': self._ewma(ppda_def_vals, days_diff, self.ewma_tau),
            'avg_deep': self._ewma(deep_vals, days_diff, self.ewma_tau),
            'avg_deep_allowed': self._ewma(deep_allowed_vals, days_diff, self.ewma_tau),
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
        35 признаков (33 + 2 missing flags) + 2 target (goals_home, goals_away).

        Все вычисления в памяти — 0 SQL запросов после preload.

        КРИТИЧЕСКИ: global_means и rolling league encoding должны быть
        вычислены на train split ДО вызова этого метода!
        Home advantage вычисляется внутри из train split.
        """
        if reference_date is None:
            reference_date = df['datetime'].max()

        # ========================================================
        # BUG 2 FIX: Жёстко требовать предустановленные global_means
        # ========================================================
        if not self.global_means:
            raise ValueError(
                "global_means не установлены! "
                "Вызовите fe._compute_global_stats(train_raw) ДО build_feature_matrix(). "
                "Это предотвращает data leakage."
            )

        logger.info("Строим матрицу признаков V2 (preload + in-memory EWMA)...")

        # ========================================================
        # ОШИБКА №7 FIX: Home advantage (data-driven, до ELO!)
        # Вычисляется ТОЛЬКО ОДИН РАЗ (на train). Val/test не перезаписывают.
        # ========================================================
        if not self._home_advantage_computed:
            self._compute_home_advantage(df)
            self._home_advantage_computed = True

        # ========================================================
        # BUG 1 FIX: Rolling league encoding (вычисляется ДО матрицы)
        # ========================================================
        if not self._league_encoding_map:
            self._compute_rolling_league_encoding(df)

        # ELO (векторизованный, корректная привязка индексов, per-league HA)
        df = self.calculate_elo(df)

        # ========================================================
        # BUG 7 FIX: Precompute team histories для O(N*k)
        # ========================================================
        team_histories = self._precompute_team_histories(df)

        features = []
        total = len(df)
        max_date = df['datetime'].max()

        # Global means dict для быстрого fillna
        gm = self.global_means

        for i, (df_idx, row) in enumerate(df.iterrows()):
            if i % 500 == 0:
                logger.info(f"  Прогресс: {i}/{total} матчей...")

            match_date = row['datetime']

            # 4 вызова формы — через precomputed dict (O(k), не O(N)!)
            hf = self._get_team_form(team_histories, row['home_team_id'], match_date, venue='home')
            hf_all = self._get_team_form(team_histories, row['home_team_id'], match_date, venue='all')
            af = self._get_team_form(team_histories, row['away_team_id'], match_date, venue='away')
            af_all = self._get_team_form(team_histories, row['away_team_id'], match_date, venue='all')

            # CROSS FEATURES (eps вместо магического 1.0)
            xg_home_vs_npxga_away = (
                hf['avg_xg'] / (af['avg_npxga'] + _EPS))
            ppda_home_vs_away = (
                hf['avg_ppda_att'] / (af['avg_ppda_def'] + _EPS))
            deep_home_vs_away = (
                hf['avg_deep'] / (af['avg_deep_allowed'] + _EPS))

            # TIME WEIGHT: НОВЫЕ матчи = БОЛЬШИЙ вес
            days_from_end = (max_date - match_date).days
            time_weight = np.exp(-days_from_end / 365)

            # BUG 1 FIX: Rolling league encoding по df.index (не enumerate!)
            league_enc = self._get_league_encoding_by_index(df_idx)

            # ОШИБКА №5 FIX: Missing flags — модель знает, когда данных нет
            home_form_vals = list(hf.values()) + list(hf_all.values())
            home_missing = any(np.isnan(v) for v in home_form_vals)
            away_form_vals = list(af.values()) + list(af_all.values())
            away_missing = any(np.isnan(v) for v in away_form_vals)

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
                # LEAGUE (rolling target encoding — без leakage!)
                'league_encoded': league_enc,
                # TIME
                'time_weight': time_weight,
                # MISSING FLAGS (ОШИБКА №5 FIX: 0 ≠ "нет данных"!)
                'home_missing_data': 1.0 if home_missing else 0.0,
                'away_missing_data': 1.0 if away_missing else 0.0,
                # TARGETS (реальные голы, НЕ xG!)
                'target_goals_home': row['goals_h'],
                'target_goals_away': row['goals_a'],
            })

        feature_df = pd.DataFrame(features)

        # ============================================================
        # ОШИБКА №5 FIX: fillna с правильными defaults (НЕ 0!)
        #
        # 0 ≠ "нет данных". ELO=0 → супер-слабая команда.
        # xG=0 → команда не умеет создавать моменты.
        # Модель учится неправильным паттернам!
        #
        # Стратегия:
        #   - Stat features → global_means (вычислены на train)
        #   - ELO → elo_base (1500) — нейтральный рейтинг
        #   - Cross features → 1.0 — нейтральное соотношение
        #   - League → _league_global_mean
        #   - Missing flags → 0.0 (уже заполнены в цикле)
        #   - НИКАКОГО blanket fillna(0)!
        # ============================================================

        # 1) Stat features → global_means (key mapping через replace)
        for col in self.get_feature_columns():
            if col in feature_df.columns:
                gm_key = (col
                          .replace('home_', '')
                          .replace('away_', '')
                          .replace('overall_', ''))
                if gm_key in gm:
                    feature_df[col] = feature_df[col].fillna(gm[gm_key])

        # 2) ELO features → elo_base (нейтральный рейтинг, НЕ 0!)
        feature_df['home_elo'] = feature_df['home_elo'].fillna(
            float(self.elo_base + self.home_advantage))
        feature_df['away_elo'] = feature_df['away_elo'].fillna(
            float(self.elo_base))
        feature_df['elo_diff'] = feature_df['elo_diff'].fillna(
            float(self.home_advantage))

        # 3) Cross features → 1.0 (нейтральное соотношение, НЕ 0!)
        feature_df['xg_home_vs_npxga_away'] = feature_df[
            'xg_home_vs_npxga_away'].fillna(1.0)
        feature_df['ppda_home_vs_away'] = feature_df[
            'ppda_home_vs_away'].fillna(1.0)
        feature_df['deep_home_vs_away'] = feature_df[
            'deep_home_vs_away'].fillna(1.0)

        # 4) League → global mean
        feature_df['league_encoded'] = feature_df[
            'league_encoded'].fillna(self._league_global_mean)

        # 5) Time weight → 1.0
        feature_df['time_weight'] = feature_df['time_weight'].fillna(1.0)

        # 6) Missing flags → 0.0 (уже заполнены в цикле, но на всякий случай)
        feature_df['home_missing_data'] = feature_df[
            'home_missing_data'].fillna(0.0)
        feature_df['away_missing_data'] = feature_df[
            'away_missing_data'].fillna(0.0)

        # Проверка: не должно остаться NaN в feature колонках
        feat_cols = self.get_feature_columns()
        nan_count = feature_df[feat_cols].isna().sum().sum()
        if nan_count > 0:
            nan_cols = feature_df[feat_cols].isna().sum()
            nan_cols = nan_cols[nan_cols > 0].to_dict()
            logger.warning(f"После fillna осталось {nan_count} NaN в: {nan_cols}")

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

        BUG 4 FIX: ELO берётся из ratings dict (после calculate_elo),
        а не из .iloc[-1] случайного матча команды.
        """
        now = datetime.now()

        # Preload (1 запрос вместо 5)
        all_matches = await self._preload_all_stats()

        if len(all_matches) == 0:
            raise ValueError("Нет завершённых матчей в базе данных!")

        # Global means для defaults (prediction видит все данные — это нормально)
        self._compute_global_stats(all_matches)

        # Home advantage — считаем из данных (для prediction OK — все данные)
        self._compute_home_advantage(all_matches)

        # ELO — считаем на всех данных
        all_matches = self.calculate_elo(all_matches)

        # ELO из ratings dict (актуальный рейтинг после ВСЕХ матчей)
        # ОШИБКА №7 FIX: per-league home advantage
        ratings = self._current_elo_ratings
        ha = self._get_home_advantage(league)
        home_elo = ratings.get(home_team_id, self.elo_base) + ha
        away_elo = ratings.get(away_team_id, self.elo_base)

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

        # Форма (в памяти, без SQL — оригинальный путь с DataFrame)
        hf = self._get_team_form(all_matches, home_team_id, now, venue='home')
        hf_all = self._get_team_form(all_matches, home_team_id, now, venue='all')
        af = self._get_team_form(all_matches, away_team_id, now, venue='away')
        af_all = self._get_team_form(all_matches, away_team_id, now, venue='all')

        # Кросс-признаки (eps вместо 1.0)
        gm = self.global_means
        xg_home_vs_npxga_away = (
            hf['avg_xg'] / (af['avg_npxga'] + _EPS))
        ppda_home_vs_away = (
            hf['avg_ppda_att'] / (af['avg_ppda_def'] + _EPS))
        deep_home_vs_away = (
            hf['avg_deep'] / (af['avg_deep_allowed'] + _EPS))

        # Fill NaN → global_means для prediction
        def safe_val(val, default_key=None):
            """Заменить NaN на global mean."""
            if val != val:  # NaN check
                if default_key and default_key in gm:
                    return gm[default_key]
                return gm.get('avg_xg', 1.3)
            return val

        result = {
            'home_avg_xg': safe_val(hf['avg_xg'], 'avg_xg'),
            'home_avg_npxg': safe_val(hf['avg_npxg'], 'avg_npxg'),
            'home_avg_goals_scored': safe_val(hf['avg_goals_scored'], 'avg_goals_scored'),
            'home_avg_ppda_att': safe_val(hf['avg_ppda_att'], 'avg_ppda_att'),
            'home_avg_deep': safe_val(hf['avg_deep'], 'avg_deep'),
            'home_avg_npxga': safe_val(hf['avg_npxga'], 'avg_npxga'),
            'home_avg_goals_conceded': safe_val(hf['avg_goals_conceded'], 'avg_goals_conceded'),
            'home_avg_ppda_def': safe_val(hf['avg_ppda_def'], 'avg_ppda_def'),
            'home_avg_deep_allowed': safe_val(hf['avg_deep_allowed'], 'avg_deep_allowed'),
            'home_overall_avg_xg': safe_val(hf_all['avg_xg'], 'avg_xg'),
            'home_overall_avg_npxg': safe_val(hf_all['avg_npxg'], 'avg_npxg'),
            'home_overall_avg_goals_scored': safe_val(hf_all['avg_goals_scored'], 'avg_goals_scored'),
            'home_overall_avg_goals_conceded': safe_val(hf_all['avg_goals_conceded'], 'avg_goals_conceded'),
            'away_avg_xg': safe_val(af['avg_xg'], 'avg_xg'),
            'away_avg_npxg': safe_val(af['avg_npxg'], 'avg_npxg'),
            'away_avg_goals_scored': safe_val(af['avg_goals_scored'], 'avg_goals_scored'),
            'away_avg_ppda_att': safe_val(af['avg_ppda_att'], 'avg_ppda_att'),
            'away_avg_deep': safe_val(af['avg_deep'], 'avg_deep'),
            'away_avg_npxga': safe_val(af['avg_npxga'], 'avg_npxga'),
            'away_avg_goals_conceded': safe_val(af['avg_goals_conceded'], 'avg_goals_conceded'),
            'away_avg_ppda_def': safe_val(af['avg_ppda_def'], 'avg_ppda_def'),
            'away_avg_deep_allowed': safe_val(af['avg_deep_allowed'], 'avg_deep_allowed'),
            'away_overall_avg_xg': safe_val(af_all['avg_xg'], 'avg_xg'),
            'away_overall_avg_npxg': safe_val(af_all['avg_npxg'], 'avg_npxg'),
            'away_overall_avg_goals_scored': safe_val(af_all['avg_goals_scored'], 'avg_goals_scored'),
            'away_overall_avg_goals_conceded': safe_val(af_all['avg_goals_conceded'], 'avg_goals_conceded'),
            # BUG 4 FIX: ELO из ratings dict
            'home_elo': float(home_elo),
            'away_elo': float(away_elo),
            'elo_diff': float(home_elo - away_elo),
            'xg_home_vs_npxga_away': xg_home_vs_npxga_away,
            'ppda_home_vs_away': ppda_home_vs_away,
            'deep_home_vs_away': deep_home_vs_away,
            'league_encoded': self._get_league_encoding(league),
            'time_weight': 1.0,
            # FIX #4: Missing flags для консистентности с get_feature_columns() (35)
            # Для live prediction: команда найдена и форма посчитана — 0.0
            'home_missing_data': 0.0,
            'away_missing_data': 0.0,
            # Метаданные для predict() (не feature!)
            '_league': league,
        }

        return result

    # ==========================
    # FEATURE LIST
    # ==========================
    def get_feature_columns(self) -> List[str]:
        """Список из 35 признаков для ML модели (33 + 2 missing flags)."""
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
            # Missing flags (ОШИБКА №5 FIX)
            'home_missing_data', 'away_missing_data',
        ]
