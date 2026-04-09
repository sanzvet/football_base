import asyncpg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Извлечение и агрегация признаков для ML-модели.
    Использует обе метрики: xG (с пенальти) и npxG (без пенальти).
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def get_training_data(self) -> pd.DataFrame:
        """
        Получить данные для обучения модели.
        Только завершённые матчи (status = 'Result').
        """
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
                
                -- xG (полный, с пенальти)
                ms_home.xG as xg_home,
                ms_away.xG as xg_away,
                
                -- npxG (без пенальти, для формы)
                ms_home.npxG as npxg_home,
                ms_away.npxG as npxg_away,
                
                -- npxGA (пропущенные без пенальти)
                ms_home.npxGA as npxga_home,
                ms_away.npxGA as npxga_away,
                
                -- Глубокие проходы
                ms_home.deep as deep_home,
                ms_away.deep as deep_away,
                
                -- Прессинг (PPDA)
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
            logger.warning("Нет данных для обучения модели!")
            return pd.DataFrame()
        
        df = pd.DataFrame([dict(row) for row in rows])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        logger.info(f"✅ Загружено {len(df)} матчей для обучения")
        return df
    
    async def calculate_team_form(
        self, 
        team_id: int, 
        match_date: datetime, 
        n_games: int = 5,
        home_only: bool = False
    ) -> Dict[str, float]:
        """
        Рассчитать форму команды за последние n матчей.
        """
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
                CASE 
                    WHEN m.home_team_id = $1 THEN 'home'
                    ELSE 'away'
                END as venue
            FROM matches m
            JOIN match_team_stats ms 
                ON m.match_id = ms.match_id AND ms.team_id = $1
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
        
        df = pd.DataFrame([dict(row) for row in rows])
        
        if home_only:
            df = df[df['venue'] == 'home']
        
        if len(df) == 0:
            return self._empty_form()
        
        return {
            'form_xg': df['xg'].mean(),
            'form_npxg': df['npxg'].mean(),
            'form_npxga': df['npxga'].mean(),
            'form_deep': df['deep'].mean(),
            'form_ppda': df['ppda'].mean(),
            'form_goals_scored': (df[df['venue'] == 'home']['goals_h'].sum() + 
                                df[df['venue'] == 'away']['goals_a'].sum()) / len(df),
            'form_goals_conceded': (df[df['venue'] == 'home']['goals_a'].sum() + 
                                    df[df['venue'] == 'away']['goals_h'].sum()) / len(df),
            'form_games': len(df)
        }
    
    def _empty_form(self) -> Dict[str, float]:
        """Вернуть пустую форму (значения по умолчанию)"""
        return {
            'form_xg': 1.0,
            'form_npxg': 1.0,
            'form_npxga': 1.0,
            'form_deep': 10.0,
            'form_ppda': 10.0,
            'form_goals_scored': 1.0,
            'form_goals_conceded': 1.0,
            'form_games': 0
        }
    
    def calculate_time_decay_weight(self, match_date: datetime, reference_date: datetime = None) -> float:
        """
        Рассчитать вес матча на основе времени (Time Decay).
        Новые матчи имеют больший вес.
        
        Формула: weight = exp(-decay * days_ago)
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        days_ago = (reference_date - match_date).days
        
        # Параметр затухания (0.005 = половина веса за ~140 дней)
        decay = 0.005
        
        weight = np.exp(-decay * max(0, days_ago))
        
        return max(0.1, min(1.0, weight))  # Ограничиваем от 0.1 до 1.0
    
    async def build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Построить матрицу признаков для обучения.
        """
        logger.info("🔧 Строим матрицу признаков...")
        
        features = []
        
        for idx, row in df.iterrows():
            # Форма домашней команды
            home_form = await self.calculate_team_form(
                row['home_team_id'], 
                row['datetime'], 
                n_games=5
            )
            
            # Форма гостевой команды
            away_form = await self.calculate_team_form(
                row['away_team_id'], 
                row['datetime'], 
                n_games=5
            )
            
            # Форма дома для домашней команды
            home_form_home = await self.calculate_team_form(
                row['home_team_id'], 
                row['datetime'], 
                n_games=5,
                home_only=True
            )
            
            # Форма на выезде для гостевой команды
            away_form_away = await self.calculate_team_form(
                row['away_team_id'], 
                row['datetime'], 
                n_games=5,
                home_only=True
            )
            
            # Time Decay вес
            time_weight = self.calculate_time_decay_weight(row['datetime'])
            
            # Кодирование лиги
            league_map = {
                'La_liga': 0, 'EPL': 1, 'Bundesliga': 2, 
                'Serie_A': 3, 'Ligue_1': 4, 'RFPL': 5
            }
            league_encoded = league_map.get(row['league'], 0)
            
            # ========== ПРИЗНАКИ С xG И npxG ==========
            feature_row = {
                'match_id': row['match_id'],
                'datetime': row['datetime'],
                'league_encoded': league_encoded,
                
                # --- xG (полный, с пенальти) ---
                'xg_home': row['xg_home'],
                'xg_away': row['xg_away'],
                'xg_diff': row['xg_home'] - row['xg_away'],
                'xg_total': row['xg_home'] + row['xg_away'],
                
                # --- npxG (без пенальти) ---
                'npxg_home': row['npxg_home'],
                'npxg_away': row['npxg_away'],
                'npxg_diff': row['npxg_home'] - row['npxg_away'],
                'npxg_total': row['npxg_home'] + row['npxg_away'],
                
                # --- Пенальти-тенденция (xG - npxG) ---
                'penalty_xg_home': row['xg_home'] - row['npxg_home'],
                'penalty_xg_away': row['xg_away'] - row['npxg_away'],
                'penalty_xg_diff': (row['xg_home'] - row['npxg_home']) - (row['xg_away'] - row['npxg_away']),
                
                # --- Оборона (npxGA) ---
                'npxga_home': row['npxga_home'],
                'npxga_away': row['npxga_away'],
                'npxga_diff': row['npxga_home'] - row['npxga_away'],
                
                # --- Форма команд (xG) ---
                'home_form_xg': home_form['form_xg'],
                'away_form_xg': away_form['form_xg'],
                'home_form_xg_diff': home_form['form_xg'] - away_form['form_xg'],
                
                # --- Форма команд (npxG) ---
                'home_form_npxg': home_form['form_npxg'],
                'away_form_npxg': away_form['form_npxg'],
                'home_form_npxg_diff': home_form['form_npxg'] - away_form['form_npxg'],
                
                # --- Форма дома/выезд ---
                'home_form_home_xg': home_form_home['form_xg'],
                'away_form_away_xg': away_form_away['form_xg'],
                'home_form_home_npxg': home_form_home['form_npxg'],
                'away_form_away_npxg': away_form_away['form_npxg'],
                
                # --- Голы ---
                'home_form_goals_scored': home_form['form_goals_scored'],
                'home_form_goals_conceded': home_form['form_goals_conceded'],
                'away_form_goals_scored': away_form['form_goals_scored'],
                'away_form_goals_conceded': away_form['form_goals_conceded'],
                'goal_diff_form': home_form['form_goals_scored'] - away_form['form_goals_conceded'],
                
                # --- Deep & PPDA ---
                'home_deep': row['deep_home'],
                'away_deep': row['deep_away'],
                'home_ppda': row['ppda_att_home'],
                'away_ppda': row['ppda_att_away'],
                'home_form_deep': home_form['form_deep'],
                'away_form_deep': away_form['form_deep'],
                'home_form_ppda': home_form['form_ppda'],
                'away_form_ppda': away_form['form_ppda'],
                
                # --- Time Decay ---
                'time_weight': time_weight,
                
                # --- Целевые переменные ---
                'goals_h': row['goals_h'],
                'goals_a': row['goals_a'],
                'goal_diff': row['goals_h'] - row['goals_a'],
                'total_goals': row['goals_h'] + row['goals_a'],
                'result_home': 1 if row['goals_h'] > row['goals_a'] else 0,
                'result_draw': 1 if row['goals_h'] == row['goals_a'] else 0,
                'result_away': 1 if row['goals_h'] < row['goals_a'] else 0,
                'over_2_5': 1 if (row['goals_h'] + row['goals_a']) > 2.5 else 0,
                'btts': 1 if (row['goals_h'] > 0 and row['goals_a'] > 0) else 0
            }
            
            features.append(feature_row)
            
            if (idx + 1) % 500 == 0:
                logger.info(f"✅ Обработано {idx + 1}/{len(df)} матчей")
        
        feature_df = pd.DataFrame(features)
        logger.info(f"✅ Матрица признаков: {feature_df.shape}")
        
        return feature_df
    
    async def get_prediction_features(
        self, 
        home_team_id: int, 
        away_team_id: int, 
        league: str = 'EPL',
        match_datetime: datetime = None
    ) -> Dict[str, float]:
        """
        Получить признаки для прогноза конкретного матча.
        """
        if match_datetime is None:
            match_datetime = datetime.now()
        
        home_form = await self.calculate_team_form(home_team_id, match_datetime, n_games=5)
        away_form = await self.calculate_team_form(away_team_id, match_datetime, n_games=5)
        home_form_home = await self.calculate_team_form(home_team_id, match_datetime, n_games=5, home_only=True)
        away_form_away = await self.calculate_team_form(away_team_id, match_datetime, n_games=5, home_only=True)
        
        # Кодирование лиги
        league_map = {
            'La_liga': 0, 'EPL': 1, 'Bundesliga': 2, 
            'Serie_A': 3, 'Ligue_1': 4, 'RFPL': 5
        }
        league_encoded = league_map.get(league, 1)
        
        return {
            'match_id': 0,
            'datetime': match_datetime,
            'league_encoded': league_encoded,
            
            # xG (полный)
            'xg_home': home_form['form_xg'],
            'xg_away': away_form['form_xg'],
            'xg_diff': home_form['form_xg'] - away_form['form_xg'],
            'xg_total': home_form['form_xg'] + away_form['form_xg'],
            
            # npxG (без пенальти)
            'npxg_home': home_form['form_npxg'],
            'npxg_away': away_form['form_npxg'],
            'npxg_diff': home_form['form_npxg'] - away_form['form_npxg'],
            'npxg_total': home_form['form_npxg'] + away_form['form_npxg'],
            
            # Пенальти-тенденция
            'penalty_xg_home': home_form['form_xg'] - home_form['form_npxg'],
            'penalty_xg_away': away_form['form_xg'] - away_form['form_npxg'],
            'penalty_xg_diff': (home_form['form_xg'] - home_form['form_npxg']) - 
                               (away_form['form_xg'] - away_form['form_npxg']),
            
            # Оборона
            'npxga_home': home_form['form_npxga'],
            'npxga_away': away_form['form_npxga'],
            'npxga_diff': home_form['form_npxga'] - away_form['form_npxga'],
            
            # Форма
            'home_form_xg': home_form['form_xg'],
            'away_form_xg': away_form['form_xg'],
            'home_form_xg_diff': home_form['form_xg'] - away_form['form_xg'],
            'home_form_npxg': home_form['form_npxg'],
            'away_form_npxg': away_form['form_npxg'],
            'home_form_npxg_diff': home_form['form_npxg'] - away_form['form_npxg'],
            'home_form_home_xg': home_form_home['form_xg'],
            'away_form_away_xg': away_form_away['form_xg'],
            'home_form_home_npxg': home_form_home['form_npxg'],
            'away_form_away_npxg': away_form_away['form_npxg'],
            'home_form_goals_scored': home_form['form_goals_scored'],
            'home_form_goals_conceded': home_form['form_goals_conceded'],
            'away_form_goals_scored': away_form['form_goals_scored'],
            'away_form_goals_conceded': away_form['form_goals_conceded'],
            'goal_diff_form': home_form['form_goals_scored'] - away_form['form_goals_conceded'],
            
            # Deep & PPDA
            'home_deep': home_form['form_deep'],
            'away_deep': away_form['form_deep'],
            'home_ppda': home_form['form_ppda'],
            'away_ppda': away_form['form_ppda'],
            'home_form_deep': home_form['form_deep'],
            'away_form_deep': away_form['form_deep'],
            'home_form_ppda': home_form['form_ppda'],
            'away_form_ppda': away_form['form_ppda'],
            
            # Time Decay
            'time_weight': 1.0
        }
    
    def get_feature_columns(self) -> List[str]:
        """
        Вернуть список колонок признаков для модели.
        """
        return [
            'league_encoded',
            'xg_home', 'xg_away', 'xg_diff', 'xg_total',
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
            'home_form_ppda', 'away_form_ppda',
            'time_weight'
        ]
    
    def get_target_columns(self) -> List[str]:
        """
        Вернуть список целевых переменных.
        """
        return [
            'result_home', 'result_draw', 'result_away',
            'goals_h', 'goals_a', 'total_goals',
            'over_2_5', 'btts'
        ]