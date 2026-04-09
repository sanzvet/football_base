import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    # ==========================
    # GET TRAINING DATA (Обязателен для model.train())
    # ==========================
    async def get_training_data(self) -> pd.DataFrame:
        query = '''
            SELECT 
                m.match_id,
                m.datetime,
                m.league,
                m.season,
                m.home_team_id,
                m.away_team_id,
                m.goals_h,
                m.goals_a,
                ms_home.xG as xg_home,
                ms_away.xG as xg_away,
                ms_home.npxG as npxg_home,
                ms_away.npxG as npxg_away,
                ms_home.npxGA as npxga_home,
                ms_away.npxGA as npxga_away,
                ms_home.deep as deep_home,
                ms_away.deep as deep_away,
                ms_home.ppda_att as ppda_att_home,
                ms_away.ppda_att as ppda_att_away
            FROM matches m
            JOIN match_team_stats ms_home 
                ON m.match_id = ms_home.match_id AND m.home_team_id = ms_home.team_id
            JOIN match_team_stats ms_away 
                ON m.match_id = ms_away.match_id AND m.away_team_id = ms_away.team_id
            WHERE m.status = 'Result'
              AND m.goals_h IS NOT NULL 
              AND m.goals_a IS NOT NULL
              AND ms_home.xG IS NOT NULL
              AND ms_away.xG IS NOT NULL
            ORDER BY m.datetime ASC
        '''
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query)
        
        if not rows:
            logger.warning("⚠️ Нет данных для обучения модели!")
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(row) for row in rows])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"✅ Загружено {len(df)} матчей для обучения")
        return df

    # ==========================
    # TIME DECAY (Детерминированный)
    # ==========================
    def calculate_time_decay_weight(self, match_date: datetime, reference_date: datetime) -> float:
        days_ago = (reference_date - match_date).days
        decay = 0.005
        weight = np.exp(-decay * max(0, days_ago))
        return max(0.1, min(1.0, weight))

    # ==========================
    # ФОРМА КОМАНДЫ (Venue-split)
    # ==========================
    async def calculate_team_form(
        self, 
        team_id: int, 
        match_date: datetime, 
        n_games: int = 5,
        venue: Optional[str] = None  # 'home', 'away' или None для всех
    ) -> Dict[str, float]:
        query = '''
            SELECT 
                m.datetime,
                m.home_team_id,
                m.away_team_id,
                m.goals_h,
                m.goals_a,
                ms.xG as xg,
                ms.npxG as npxg,
                ms.npxGA as npxga,
                ms.deep as deep,
                ms.ppda_att as ppda,
                CASE WHEN m.home_team_id = $1 THEN 'home' ELSE 'away' END as venue
            FROM matches m
            JOIN match_team_stats ms ON m.match_id = ms.match_id AND ms.team_id = $1
            WHERE m.status = 'Result'
              AND m.datetime < $2
              AND (m.home_team_id = $1 OR m.away_team_id = $1)
            ORDER BY m.datetime DESC
            LIMIT $3
        '''
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, team_id, match_date, n_games)
            
        if not rows:
            return self._empty_form()
            
        df = pd.DataFrame([dict(r) for r in rows])
        
        if venue:
            df = df[df['venue'] == venue]
            
        if len(df) == 0:
            return self._empty_form()
            
        goals_scored = df.apply(lambda r: r['goals_h'] if r['venue'] == 'home' else r['goals_a'], axis=1).sum()
        goals_conceded = df.apply(lambda r: r['goals_a'] if r['venue'] == 'home' else r['goals_h'], axis=1).sum()
        
        return {
            'form_xg': df['xg'].mean(),
            'form_npxg': df['npxg'].mean(),
            'form_npxga': df['npxga'].mean(),
            'form_deep': df['deep'].mean(),
            'form_ppda': df['ppda'].mean(),
            'form_goals_scored': goals_scored / len(df),
            'form_goals_conceded': goals_conceded / len(df),
            'form_games': len(df)
        }

    def _empty_form(self) -> Dict[str, float]:
        return {
            'form_xg': 1.0, 'form_npxg': 1.0, 'form_npxga': 1.0,
            'form_deep': 10.0, 'form_ppda': 10.0,
            'form_goals_scored': 1.0, 'form_goals_conceded': 1.0, 'form_games': 0
        }

    # ==========================
    # BUILD FEATURE MATRIX (Обучение)
    # ==========================
    async def build_feature_matrix(self, df: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
        if reference_date is None:
            reference_date = df['datetime'].max()
            
        logger.info("🔧 Строим матрицу признаков...")
        features = []
        league_map = {'La_liga': 0, 'EPL': 1, 'Bundesliga': 2, 'Serie_A': 3, 'Ligue_1': 4, 'RFPL': 5}

        for idx, row in df.iterrows():
            hf = await self.calculate_team_form(row['home_team_id'], row['datetime'], 5)
            af = await self.calculate_team_form(row['away_team_id'], row['datetime'], 5)
            hf_home = await self.calculate_team_form(row['home_team_id'], row['datetime'], 5, venue='home')
            af_away = await self.calculate_team_form(row['away_team_id'], row['datetime'], 5, venue='away')
            
            tw = self.calculate_time_decay_weight(row['datetime'], reference_date)
            le = league_map.get(row.get('league', ''), 1)
            
            features.append({
                'league_encoded': le,
                'xg_home': row['xg_home'], 'xg_away': row['xg_away'],
                'xg_diff': row['xg_home'] - row['xg_away'], 'xg_total': row['xg_home'] + row['xg_away'],
                'npxg_home': row['npxg_home'], 'npxg_away': row['npxg_away'],
                'npxg_diff': row['npxg_home'] - row['npxg_away'], 'npxg_total': row['npxg_home'] + row['npxg_away'],
                'penalty_xg_home': row['xg_home'] - row['npxg_home'],
                'penalty_xg_away': row['xg_away'] - row['npxg_away'],
                'penalty_xg_diff': (row['xg_home'] - row['npxg_home']) - (row['xg_away'] - row['npxg_away']),
                'npxga_home': row['npxga_home'], 'npxga_away': row['npxga_away'],
                'npxga_diff': row['npxga_home'] - row['npxga_away'],
                'home_form_xg': hf['form_xg'], 'away_form_xg': af['form_xg'],
                'home_form_xg_diff': hf['form_xg'] - af['form_xg'],
                'home_form_npxg': hf['form_npxg'], 'away_form_npxg': af['form_npxg'],
                'home_form_npxg_diff': hf['form_npxg'] - af['form_npxg'],
                'home_form_home_xg': hf_home['form_xg'], 'away_form_away_xg': af_away['form_xg'],
                'home_form_home_npxg': hf_home['form_npxg'], 'away_form_away_npxg': af_away['form_npxg'],
                'home_form_goals_scored': hf['form_goals_scored'], 'home_form_goals_conceded': hf['form_goals_conceded'],
                'away_form_goals_scored': af['form_goals_scored'], 'away_form_goals_conceded': af['form_goals_conceded'],
                'goal_diff_form': hf['form_goals_scored'] - af['form_goals_conceded'],
                'home_deep': row.get('deep_home', 0), 'away_deep': row.get('deep_away', 0),
                'home_ppda': row.get('ppda_att_home', 0), 'away_ppda': row.get('ppda_att_away', 0),
                'home_form_deep': hf['form_deep'], 'away_form_deep': af['form_deep'],
                'home_form_ppda': hf['form_ppda'], 'away_form_ppda': af['form_ppda'],
                'time_weight': tw,
                'result_home': 1 if row['goals_h'] > row['goals_a'] else 0,
                'result_draw': 1 if row['goals_h'] == row['goals_a'] else 0,
                'result_away': 1 if row['goals_h'] < row['goals_a'] else 0,
                'goals_h': row['goals_h'], 'goals_a': row['goals_a'],
                'total_goals': row['goals_h'] + row['goals_a'],
                'over_2_5': 1 if (row['goals_h'] + row['goals_a']) > 2.5 else 0,
                'btts': 1 if row['goals_h'] > 0 and row['goals_a'] > 0 else 0
            })
            
        feature_df = pd.DataFrame(features).fillna(0)
        logger.info(f"✅ Матрица признаков: {feature_df.shape}")
        return feature_df

    # ==========================
    # GET PREDICTION FEATURES (Инференс)
    # ==========================
    async def get_prediction_features(
        self, 
        home_team_id: int, 
        away_team_id: int, 
        league: str = 'EPL',
        match_datetime: Optional[datetime] = None
    ) -> Dict[str, float]:
        if match_datetime is None:
            match_datetime = datetime.utcnow()
            
        hf = await self.calculate_team_form(home_team_id, match_datetime, 5)
        af = await self.calculate_team_form(away_team_id, match_datetime, 5)
        hf_home = await self.calculate_team_form(home_team_id, match_datetime, 5, venue='home')
        af_away = await self.calculate_team_form(away_team_id, match_datetime, 5, venue='away')
        
        league_map = {'La_liga': 0, 'EPL': 1, 'Bundesliga': 2, 'Serie_A': 3, 'Ligue_1': 4, 'RFPL': 5}
        league_encoded = league_map.get(league, 1)
        
        return {
            'league_encoded': league_encoded,
            'xg_home': hf['form_xg'], 'xg_away': af['form_xg'],
            'xg_diff': hf['form_xg'] - af['form_xg'], 'xg_total': hf['form_xg'] + af['form_xg'],
            'npxg_home': hf['form_npxg'], 'npxg_away': af['form_npxg'],
            'npxg_diff': hf['form_npxg'] - af['form_npxg'], 'npxg_total': hf['form_npxg'] + af['form_npxg'],
            'penalty_xg_home': hf['form_xg'] - hf['form_npxg'],
            'penalty_xg_away': af['form_xg'] - af['form_npxg'],
            'penalty_xg_diff': (hf['form_xg'] - hf['form_npxg']) - (af['form_xg'] - af['form_npxg']),
            'npxga_home': hf['form_npxga'], 'npxga_away': af['form_npxga'],
            'npxga_diff': hf['form_npxga'] - af['form_npxga'],
            'home_form_xg': hf['form_xg'], 'away_form_xg': af['form_xg'],
            'home_form_xg_diff': hf['form_xg'] - af['form_xg'],
            'home_form_npxg': hf['form_npxg'], 'away_form_npxg': af['form_npxg'],
            'home_form_npxg_diff': hf['form_npxg'] - af['form_npxg'],
            'home_form_home_xg': hf_home['form_xg'], 'away_form_away_xg': af_away['form_xg'],
            'home_form_home_npxg': hf_home['form_npxg'], 'away_form_away_npxg': af_away['form_npxg'],
            'home_form_goals_scored': hf['form_goals_scored'], 'home_form_goals_conceded': hf['form_goals_conceded'],
            'away_form_goals_scored': af['form_goals_scored'], 'away_form_goals_conceded': af['form_goals_conceded'],
            'goal_diff_form': hf['form_goals_scored'] - af['form_goals_conceded'],
            'home_deep': hf['form_deep'], 'away_deep': af['form_deep'],
            'home_ppda': hf['form_ppda'], 'away_ppda': af['form_ppda'],
            'home_form_deep': hf['form_deep'], 'away_form_deep': af['form_deep'],
            'home_form_ppda': hf['form_ppda'], 'away_form_ppda': af['form_ppda'],
            'time_weight': 1.0
        }

    def get_feature_columns(self) -> List[str]:
        return [
            'league_encoded', 'xg_home', 'xg_away', 'xg_diff', 'xg_total',
            'npxg_home', 'npxg_away', 'npxg_diff', 'npxg_total',
            'penalty_xg_home', 'penalty_xg_away', 'penalty_xg_diff',
            'npxga_home', 'npxga_away', 'npxga_diff',
            'home_form_xg', 'away_form_xg', 'home_form_xg_diff',
            'home_form_npxg', 'away_form_npxg', 'home_form_npxg_diff',
            'home_form_home_xg', 'away_form_away_xg',
            'home_form_home_npxg', 'away_form_away_npxg',
            'home_form_goals_scored', 'home_form_goals_conceded',
            'away_form_goals_scored', 'away_form_goals_conceded',
            'goal_diff_form',
            'home_deep', 'away_deep', 'home_ppda', 'away_ppda',
            'home_form_deep', 'away_form_deep',
            'home_form_ppda', 'away_form_ppda', 'time_weight'
        ]

    def get_target_columns(self) -> List[str]:
        return [
            'result_home', 'result_draw', 'result_away',
            'goals_h', 'goals_a', 'total_goals',
            'over_2_5', 'btts'
        ]