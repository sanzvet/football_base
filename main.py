"""
main.py — ОБНОВЛЁННАЯ ВЕРСИЯ с гибридной моделью.

Что изменилось:
1. Инициализация HybridModel (новая) + FootballModel (старая как fallback)
2. API /api/predict возвращает справедливые кэфы для ИТБ 0.5 и ИТБ 1.5
3. API /api/train_hybrid — endpoint для обучения гибридной модели
4. В predict() сначала пробуем HybridModel, при ошибке — fallback на FootballModel
5. АВТО-РЕЗОЛЬВЕНЦИЯ ставок при загрузке истории
6. Эндпоинт сброса неправильно разрешённых ставок

ИНСТРУКЦИЯ ПО ИНТЕГРАЦИИ:
1. Скопируйте этот файл ВМЕСТО вашего текущего main.py
2. Убедитесь, что файлы ml/features_v2.py, ml/dixon_coles_v2.py, ml/hybrid_model.py
   лежат в директории ml/ вашего проекта
3. Для обучения вызовите: POST /api/train_hybrid
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form, Query, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from database import init_db, get_pool
from config import settings
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import subprocess
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import math
import os

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# LIFESPAN (startup/shutdown) — FIX #1: замена deprecated on_event
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan: инициализация при старте, очистка при остановке."""
    # === STARTUP ===
    logger.info("Запуск Football Predictor Pro V2...")
    await init_db()
    asyncio.create_task(init_ml_models_background())
    setup_scheduler()
    logger.info("Приложение готово! Откройте http://127.0.0.1:8000")

    yield  # Приложение работает

    # === SHUTDOWN ===
    scheduler.shutdown()
    logger.info("Приложение остановлено")


app = FastAPI(title="Football Predictor Pro", version="2.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
scheduler = AsyncIOScheduler()

# Глобальные переменные для ML-моделей
hybrid_model: Optional[Any] = None         # HybridModel (V2 — основная)
feature_engineer_v2: Optional[Any] = None  # FeatureEngineerV2 (V2)

parser_running = False


# ============================================
# ИНИЦИАЛИЗАЦИЯ ML-МОДЕЛЕЙ
# ============================================

async def init_ml_models(pool):
    """Инициализировать ML-модель V2 при старте приложения"""
    global hybrid_model, feature_engineer_v2

    try:
        from ml.features_v2 import FeatureEngineerV2
        from ml.hybrid_model import HybridModel

        feature_engineer_v2 = FeatureEngineerV2(pool)
        hybrid_model = HybridModel(model_path="ml/hybrid_model.pkl")

        if hybrid_model.load():
            logger.info("V2: Гибридная модель загружена (основная)")
        else:
            logger.info("V2: Гибридная модель не найдена — нужна тренировка")
    except Exception as e:
        logger.warning(f"V2: Ошибка инициализации: {e}")


# ============================================
# Pydantic модель для ставки
# ============================================

class BetCreate(BaseModel):
    league: str
    home_team: str
    away_team: str
    match_datetime: datetime
    bet_type: str
    bet_description: str
    bet_category: str
    bookmaker: str
    probability: float
    fair_odd: float
    bookmaker_odd: float
    stake: float
    status: str = "pending"
    result_goals_h: Optional[int] = None
    result_goals_a: Optional[int] = None
    payout: Optional[float] = None
    profit: Optional[float] = None
    notes: Optional[str] = None


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/api/leagues")
async def get_leagues():
    """Получить список всех лиг из базы данных"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT DISTINCT league
            FROM matches
            WHERE league IS NOT NULL
            ORDER BY league
        ''')

    leagues = [row['league'] for row in rows]
    return {"leagues": leagues, "count": len(leagues)}


@app.get("/api/teams")
async def get_teams(league: str = Query(..., description="Название лиги")):
    """Получить список команд для выбранной лиги"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT DISTINCT t.team_id, t.team_name
            FROM teams t
            JOIN matches m ON t.team_id = m.home_team_id OR t.team_id = m.away_team_id
            WHERE m.league = $1
            ORDER BY t.team_name
        ''', league)

    teams = [{"id": row['team_id'], "name": row['team_name']} for row in rows]
    return {"teams": teams, "count": len(teams), "league": league}


@app.get("/api/predict")
async def get_prediction(
    home_id: int = Query(..., description="ID домашней команды"),
    away_id: int = Query(..., description="ID гостевой команды"),
    league: str = Query("EPL", description="Название лиги")
):
    """
    Получить полный прогноз для матча.
    Приоритет: HybridModel (V2) -> FootballModel (V1) -> Mock data
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    # Проверка команд
    async with pool.acquire() as conn:
        home_team = await conn.fetchrow(
            'SELECT team_name FROM teams WHERE team_id = $1', home_id)
        away_team = await conn.fetchrow(
            'SELECT team_name FROM teams WHERE team_id = $1', away_id)

        if not home_team or not away_team:
            raise HTTPException(status_code=404, detail="Команда не найдена")

    # === ГИБРИДНАЯ МОДЕЛЬ (V2) ===
    if hybrid_model and hybrid_model.is_trained and feature_engineer_v2:
        try:
            features = await feature_engineer_v2.get_prediction_features(
                home_team_id=home_id,
                away_team_id=away_id,
                league=league
            )
            prediction = hybrid_model.predict(features)

            return {
                "home_team": home_team['team_name'],
                "away_team": away_team['team_name'],
                **prediction
            }

        except Exception as e:
            logger.error(f"V2 ошибка прогноза: {e}")

    # === MOCK DATA (если нет моделей) ===
    logger.warning("Модели не обучены — возвращаем тестовые данные")
    return {
        "home_team": home_team['team_name'],
        "away_team": away_team['team_name'],
        "model_status": "mock_data",
        "prob_home": 0.45, "prob_draw": 0.28, "prob_away": 0.27,
        "fair_odd_home": 2.22, "fair_odd_draw": 3.57, "fair_odd_away": 3.70,
        "fair_odd_home_ah0": 2.50, "fair_odd_away_ah0": 1.80,
        "home_ah0_prob": 0.73, "away_ah0_prob": 0.55,
        "xg_home": 1.5, "xg_away": 1.2,
        "home_over_05": 77.7, "home_over_15": 44.2,
        "away_over_05": 69.9, "away_over_15": 38.5,
        "over25_prob": 50.6, "over25_odd": 1.97,
        "over15_prob": 75.3, "over15_odd": 1.33,
        "btts_prob": 53.7, "btts_odd": 1.86,
        "home_itb05_prob": 77.7, "home_itb05_odd": 1.29,
        "home_itb15_prob": 44.2, "home_itb15_odd": 2.26,
        "away_itb05_prob": 69.9, "away_itb05_odd": 1.43,
        "away_itb15_prob": 38.5, "away_itb15_odd": 2.60,
        "exact_scores": {"1:1": 11.5, "1:0": 10.7, "2:1": 9.1, "0:1": 8.7, "2:0": 7.6},
    }


@app.post("/api/train_hybrid")
async def train_hybrid_model():
    """
    Обучить гибридную модель V2.
    Endpoint для ручного запуска обучения.
    Время: ~10-30 минут (зависит от объёма данных).
    """
    global hybrid_model, feature_engineer_v2

    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    try:
        from ml.features_v2 import FeatureEngineerV2
        from ml.hybrid_model import HybridModel

        feature_engineer_v2 = FeatureEngineerV2(pool)
        hybrid_model = HybridModel(model_path="ml/hybrid_model.pkl")

        metrics = await hybrid_model.train(pool)

        if "error" in metrics:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": metrics["error"]}
            )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Гибридная модель обучена!",
                "metrics": metrics
            }
        )

    except Exception as e:
        logger.error(f"Ошибка обучения гибридной модели: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


@app.post("/api/update_db")
async def update_database():
    """Запустить парсер для обновления базы данных"""
    global parser_running

    if parser_running:
        return JSONResponse(
            status_code=202,
            content={"status": "running", "message": "Парсер уже запущен"}
        )

    parser_running = True

    try:
        process = subprocess.run(
            ["python", "stable_parse.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if process.returncode == 0:
            logger.info("Парсер успешно завершён")
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": "База данных обновлена!"}
            )
        else:
            logger.error(f"Ошибка парсера: {process.stderr}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"Ошибка: {process.stderr[:500]}"}
            )

    except subprocess.TimeoutExpired:
        logger.error("Таймаут парсера")
        return JSONResponse(
            status_code=500,
            content={"status": "timeout", "message": "Превышено время ожидания"}
        )

    except Exception as e:
        logger.error(f"Исключение: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

    finally:
        parser_running = False


@app.get("/api/matches")
async def get_matches(
    league: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    status: Optional[str] = Query("Result", description="Статус матча")
):
    """Получить список ближайших матчей"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    query = '''
        SELECT m.match_id, m.datetime, m.league, m.status,
               th.team_name as home_team, ta.team_name as away_team,
               m.goals_h, m.goals_a
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.status = $1
    '''
    params = [status]

    if league:
        query += ' AND m.league = $2'
        params.append(league)

    query += ' ORDER BY m.datetime DESC LIMIT $' + str(len(params) + 1)
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    matches = [
        {
            "match_id": row['match_id'],
            "datetime": row['datetime'].isoformat() if row['datetime'] else None,
            "league": row['league'],
            "status": row['status'],
            "home_team": row['home_team'],
            "away_team": row['away_team'],
            "goals_h": row['goals_h'],
            "goals_a": row['goals_a']
        }
        for row in rows
    ]

    return {"matches": matches, "count": len(matches)}


# ============================================
# API ДЛЯ СТАВОК (bet_records)
# ============================================

@app.post("/api/bet_records")
async def create_bet_record(bet: BetCreate):
    """Сохранить новую ставку в таблицу bet_records"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        query = '''
            INSERT INTO bet_records (
                league, home_team, away_team, match_datetime,
                bet_type, bet_description, bet_category, bookmaker,
                probability, fair_odd, bookmaker_odd, stake,
                status, result_goals_h, result_goals_a, payout, profit, notes
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
            RETURNING id
        '''
        row = await conn.fetchrow(query,
            bet.league, bet.home_team, bet.away_team, bet.match_datetime,
            bet.bet_type, bet.bet_description, bet.bet_category, bet.bookmaker,
            bet.probability, bet.fair_odd, bet.bookmaker_odd, bet.stake,
            bet.status, bet.result_goals_h, bet.result_goals_a, bet.payout, bet.profit, bet.notes
        )
        return {"id": row["id"], "message": "Ставка сохранена"}


# ============================================
# АВТО-РЕЗОЛЬВЕНЦИЯ СТАВОК
# ============================================

def resolve_bet_outcome(bet_type: str, bet_description: str, gh: int, ga: int) -> str:
    """
    Определить исход ставки по финальному счёту.
    Возвращает: 'won', 'lost', 'void'

    gh = goals_h (голы хозяев), ga = goals_a (голы гостей)
    """
    desc_lower = bet_description.lower().strip()
    btype = bet_type.strip()

    # ========== 1X2 + короткие формы (home/away/draw) ==========
    if btype in ('1X2', 'home', 'away', 'draw', '1', 'x', '2', 'п1', 'п2', 'x'):
        # Проверяем ЛЮБЫЕ варианты описания победы хозяев
        if any(k in desc_lower for k in ['хозяев', 'хозяева', 'хозяин', 'home', 'п1', '=== 1']):
            return 'won' if gh > ga else 'lost'
        # Ничья
        elif any(k in desc_lower for k in ['ничья', 'draw', 'нечья', '=== x']):
            return 'won' if gh == ga else 'lost'
        # Проверяем ЛЮБЫЕ варианты описания победы гостей
        elif any(k in desc_lower for k in ['гостей', 'гость', 'гостев', 'away', 'п2', '=== 2']):
            return 'won' if ga > gh else 'lost'
        # Фоллбэк: определяем по самому bet_type
        elif btype in ('home', '1', 'п1'):
            return 'won' if gh > ga else 'lost'
        elif btype in ('draw', 'x', 'х'):
            return 'won' if gh == ga else 'lost'
        elif btype in ('away', '2', 'п2'):
            return 'won' if ga > gh else 'lost'

    # ========== Ф(0) / DNB / AH0 ==========
    elif btype in ('Ф(0)', 'AH0', 'DNB'):
        if any(k in desc_lower for k in ['хозяев', 'хозяева', 'хозяин', 'home', 'п1', '1']):
            if gh > ga: return 'won'
            elif gh == ga: return 'void'
            else: return 'lost'
        elif any(k in desc_lower for k in ['гост', 'away', 'п2', '2']):
            if ga > gh: return 'won'
            elif gh == ga: return 'void'
            else: return 'lost'

    # ========== ИТБ ==========
    elif btype.startswith('ИТБ'):
        threshold = 0.5
        if any(d in desc_lower for d in ['1.5', '2.5', '3.5']):
            for t in ['1.5', '2.5', '3.5']:
                if t in desc_lower:
                    threshold = float(t)
                    break
        elif '1.5' in btype:
            threshold = 1.5
        elif '2.5' in btype:
            threshold = 2.5

        if any(k in desc_lower for k in ['домашн', 'home', 'хозяев', 'хозяева']):
            return 'won' if gh > threshold else 'lost'
        elif any(k in desc_lower for k in ['гостев', 'away', 'гост', 'гостей']):
            return 'won' if ga > threshold else 'lost'

    # ========== ИТМ ==========
    elif btype.startswith('ИТМ'):
        threshold = 0.5
        if any(d in desc_lower for d in ['1.5', '2.5', '3.5']):
            for t in ['1.5', '2.5', '3.5']:
                if t in desc_lower:
                    threshold = float(t)
                    break
        elif '1.5' in btype:
            threshold = 1.5
        elif '2.5' in btype:
            threshold = 2.5

        if any(k in desc_lower for k in ['домашн', 'home', 'хозяев', 'хозяева']):
            return 'won' if gh < threshold else 'lost'
        elif any(k in desc_lower for k in ['гостев', 'away', 'гост', 'гостей']):
            return 'won' if ga < threshold else 'lost'

    # ========== Тотал ==========
    elif btype == 'Тотал':
        if any(k in desc_lower for k in ['тб 2.5', 'tb 2.5', 'over 2.5', 'больше 2.5']):
            return 'won' if (gh + ga) >= 3 else 'lost'
        elif any(k in desc_lower for k in ['тб 1.5', 'tb 1.5', 'over 1.5', 'больше 1.5']):
            return 'won' if (gh + ga) >= 2 else 'lost'
        elif any(k in desc_lower for k in ['тм 2.5', 'tm 2.5', 'under 2.5', 'меньше 2.5']):
            return 'won' if (gh + ga) < 3 else 'lost'
        elif any(k in desc_lower for k in ['тм 1.5', 'tm 1.5', 'under 1.5', 'меньше 1.5']):
            return 'won' if (gh + ga) < 2 else 'lost'
        elif 'тб' in desc_lower or 'tb' in desc_lower or 'over' in desc_lower or 'больше' in desc_lower:
            return 'won' if (gh + ga) >= 3 else 'lost'
        elif 'тм' in desc_lower or 'tm' in desc_lower or 'under' in desc_lower or 'меньше' in desc_lower:
            return 'won' if (gh + ga) < 3 else 'lost'

    # ========== ОЗ / BTTS ==========
    elif btype in ('ОЗ', 'BTTS'):
        if any(k in desc_lower for k in ['обе забь', 'btts', 'yes', 'да']):
            return 'won' if (gh >= 1 and ga >= 1) else 'lost'
        elif any(k in desc_lower for k in ['обе не забь', 'no', 'нет']):
            return 'won' if (gh == 0 or ga == 0) else 'lost'
        elif btype in ('ОЗ', 'BTTS'):
            return 'won' if (gh >= 1 and ga >= 1) else 'lost'

    # Если ничего не совпало — логируем и возвращаем lost
    logger.warning(f"Не распознан паттерн: type='{bet_type}', desc='{bet_description}', score={gh}:{ga}")
    return 'lost'


async def resolve_pending_bets():
    """
    Автоматически разрешить все pending ставки.
    Ищет завершённые матчи по названиям команд + дате матча.
    """
    pool = await get_pool()
    if not pool:
        return

    async with pool.acquire() as conn:
        pending_bets = await conn.fetch('''
            SELECT id, league, home_team, away_team, match_datetime,
                   bet_type, bet_description, bookmaker_odd, stake
            FROM bet_records
            WHERE status = 'pending'
            ORDER BY match_datetime
        ''')

        if not pending_bets:
            return

        resolved_count = 0

        for bet in pending_bets:
            home_norm = bet['home_team'].strip().lower()
            away_norm = bet['away_team'].strip().lower()

            # Ищем завершённый матч по названиям команд (самый свежий)
            match_row = await conn.fetchrow('''
                SELECT m.goals_h, m.goals_a, m.status, m.datetime,
                       th.team_name as home_name, ta.team_name as away_name
                FROM matches m
                JOIN teams th ON m.home_team_id = th.team_id
                JOIN teams ta ON m.away_team_id = ta.team_id
                WHERE LOWER(th.team_name) = $1
                  AND LOWER(ta.team_name) = $2
                  AND m.status = 'Result'
                  AND m.goals_h IS NOT NULL
                ORDER BY m.datetime DESC
                LIMIT 1
            ''', home_norm, away_norm)

            if not match_row:
                logger.info(f"  Матч не найден: {bet['home_team']} vs {bet['away_team']}")
                continue

            gh = match_row['goals_h']
            ga = match_row['goals_a']

            # ЛОГИРОВАНИЕ перед резольвацией
            logger.info(f"  Ставка #{bet['id']}: {bet['bet_type']} '{bet['bet_description']}' | "
                        f"Матч: {match_row['home_name']} {gh}:{ga} {match_row['away_name']}")

            outcome = resolve_bet_outcome(bet['bet_type'], bet['bet_description'], gh, ga)

            stake = bet['stake']
            odd = bet['bookmaker_odd']

            if outcome == 'won':
                payout = round(stake * odd, 2)
                profit = round(payout - stake, 2)
            elif outcome == 'void':
                payout = round(stake, 2)
                profit = 0.0
            else:
                payout = 0.0
                profit = -round(stake, 2)

            await conn.execute('''
                UPDATE bet_records
                SET status = $1,
                    result_goals_h = $2,
                    result_goals_a = $3,
                    payout = $4,
                    profit = $5
                WHERE id = $6
            ''', outcome, gh, ga, payout, profit, bet['id'])

            resolved_count += 1
            emoji = '✅' if outcome == 'won' else ('🔄' if outcome == 'void' else '❌')
            logger.info(f"  {emoji} -> {outcome} ({gh}:{ga}), profit={profit}")

        if resolved_count > 0:
            logger.info(f"Авто-резольвация: {resolved_count} ставок обновлено")


@app.get("/api/bet_records")
async def get_bet_records(
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500)
):
    """Получить список ставок с авто-резольвацией pending ставок"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    # АВТО-РЕЗОЛЬВЕНЦИЯ: сначала обновляем pending ставки
    await resolve_pending_bets()

    query = '''
        SELECT id, created_at, league, home_team, away_team, match_datetime,
               bet_type, bet_description, bet_category, bookmaker,
               probability, fair_odd, bookmaker_odd, stake,
               status, result_goals_h, result_goals_a, payout, profit, notes
        FROM bet_records
    '''
    params = []
    if status:
        query += " WHERE status = $1"
        params.append(status)
    query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    records = [dict(row) for row in rows]
    for rec in records:
        if rec.get("match_datetime"):
            rec["match_datetime"] = rec["match_datetime"].isoformat()
        if rec.get("created_at"):
            rec["created_at"] = rec["created_at"].isoformat()
    return {"records": records, "count": len(records)}


@app.put("/api/bet_records/{record_id}")
async def update_bet_record(
    record_id: int,
    status: str,
    result_goals_h: Optional[int] = None,
    result_goals_a: Optional[int] = None,
    payout: Optional[float] = None,
    profit: Optional[float] = None
):
    """Обновить результат ставки"""
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        await conn.execute('''
            UPDATE bet_records
            SET status = $1,
                result_goals_h = $2,
                result_goals_a = $3,
                payout = $4,
                profit = $5
            WHERE id = $6
        ''', status, result_goals_h, result_goals_a, payout, profit, record_id)
    return {"message": "Результат обновлён"}


@app.post("/api/bet_records/reset_resolved")
async def reset_resolved_bets():
    """
    Сбросить все won/lost ставки обратно в pending,
    чтобы они были повторно разрешены авто-резольвацией.
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        result = await conn.execute('''
            UPDATE bet_records
            SET status = 'pending',
                result_goals_h = NULL,
                result_goals_a = NULL,
                payout = NULL,
                profit = NULL
            WHERE status IN ('won', 'lost', 'void')
        ''')

    count = int(result.split()[-1]) if result else 0
    logger.info(f"Сброшено {count} ставок в pending")
    return {"message": f"Сброшено {count} ставок в pending для повторной резольвации", "count": count}


@app.get("/api/bet_records/debug")
async def debug_bet_records():
    """
    ДИАГНОСТИКА: Показать что хранится в базе для каждой ставки
    и что нашла авто-резольвенция.
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")

    async with pool.acquire() as conn:
        bets = await conn.fetch('''
            SELECT id, home_team, away_team, match_datetime,
                   bet_type, bet_description, bookmaker_odd, stake,
                   status, result_goals_h, result_goals_a, payout, profit
            FROM bet_records
            ORDER BY id
        ''')

    debug_info = []

    for bet in bets:
        bet_dict = dict(bet)
        # Пробуем найти матч
        match_found = None
        home_norm = bet['home_team'].strip().lower()
        away_norm = bet['away_team'].strip().lower()

        async with pool.acquire() as conn:
            if bet['match_datetime']:
                match_row = await conn.fetchrow('''
                    SELECT m.goals_h, m.goals_a, m.status, m.datetime,
                           th.team_name as home_name, ta.team_name as away_name
                    FROM matches m
                    JOIN teams th ON m.home_team_id = th.team_id
                    JOIN teams ta ON m.away_team_id = ta.team_id
                    WHERE LOWER(th.team_name) = $1
                      AND LOWER(ta.team_name) = $2
                      AND m.status = 'Result'
                      AND m.goals_h IS NOT NULL
                      AND m.datetime >= ($3::timestamp - INTERVAL '1 day')
                      AND m.datetime <= ($3::timestamp + INTERVAL '1 day')
                    ORDER BY m.datetime DESC
                    LIMIT 1
                ''', home_norm, away_norm, bet['match_datetime'])
            else:
                match_row = await conn.fetchrow('''
                    SELECT m.goals_h, m.goals_a, m.status, m.datetime,
                           th.team_name as home_name, ta.team_name as away_name
                    FROM matches m
                    JOIN teams th ON m.home_team_id = th.team_id
                    JOIN teams ta ON m.away_team_id = ta.team_id
                    WHERE LOWER(th.team_name) = $1
                      AND LOWER(ta.team_name) = $2
                      AND m.status = 'Result'
                      AND m.goals_h IS NOT NULL
                    ORDER BY m.datetime DESC
                    LIMIT 1
                ''', home_norm, away_norm)

        if match_row:
            gh = match_row['goals_h']
            ga = match_row['goals_a']
            outcome = resolve_bet_outcome(bet['bet_type'], bet['bet_description'], gh, ga)

            debug_info.append({
                "id": bet['id'],
                "home_team": bet['home_team'],
                "away_team": bet['away_team'],
                "bet_type": bet['bet_type'],
                "bet_description": bet['bet_description'],
                "current_status": bet['status'],
                "match_found": {
                    "home_name": match_row['home_name'],
                    "away_name": match_row['away_name'],
                    "goals_h": gh,
                    "goals_a": ga,
                    "match_date": match_row['datetime'].isoformat() if match_row['datetime'] else None
                },
                "calculated_outcome": outcome,
                "correct": (outcome == bet['status']) or (bet['status'] == 'pending')
            })
        else:
            debug_info.append({
                "id": bet['id'],
                "home_team": bet['home_team'],
                "away_team": bet['away_team'],
                "bet_type": bet['bet_type'],
                "bet_description": bet['bet_description'],
                "current_status": bet['status'],
                "match_found": None,
                "calculated_outcome": None,
                "correct": bet['status'] == 'pending'
            })

    return {"bets": debug_info, "total": len(debug_info)}


@app.get("/health")
async def health_check():
    """Проверка здоровья приложения"""
    pool = await get_pool()
    db_status = "connected" if pool else "disconnected"

    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            db_status = "ok"
        except:
            db_status = "error"

    v2_status = "loaded" if hybrid_model and hybrid_model.is_trained else "not_loaded"

    return {
        "status": "healthy",
        "database": db_status,
        "ml_model_v2": v2_status,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# WEB PAGES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница приложения"""
    pool = await get_pool()

    stats = {"matches": 0, "teams": 0, "leagues": 0}

    if pool:
        async with pool.acquire() as conn:
            stats["matches"] = await conn.fetchval('SELECT COUNT(*) FROM matches')
            stats["teams"] = await conn.fetchval('SELECT COUNT(*) FROM teams')
            stats["leagues"] = await conn.fetchval('SELECT COUNT(DISTINCT league) FROM matches')

    return templates.TemplateResponse("index.html", {
        "request": request,
        "stats": stats
    })


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    """Страница с историей ставок"""
    return templates.TemplateResponse("history.html", {"request": request})


# ============================================
# SCHEDULER
# ============================================

async def scheduled_parser_run():
    """Автоматический запуск парсера по расписанию"""
    logger.info("Запуск планового обновления базы данных...")
    await update_database()


def setup_scheduler():
    """Настроить расписание обновлений (Пн и Чт в 10:00)"""
    scheduler.add_job(
        scheduled_parser_run,
        'cron',
        hour=10,
        minute=0,
        day_of_week='mon,thu',
        id='weekly_update',
        replace_existing=True
    )
    scheduler.start()
    logger.info("Планировщик запущен: Пн и Чт в 10:00")


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )


async def init_ml_models_background():
    """Фоновая задача: инициализация ML моделей после старта."""
    pool = await get_pool()
    if pool:
        await init_ml_models(pool)
