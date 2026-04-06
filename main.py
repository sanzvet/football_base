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

# ML импорты
from ml.features import FeatureEngineer
from ml.model import FootballModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Football Predictor Pro", version="1.0.0")
templates = Jinja2Templates(directory="templates")
scheduler = AsyncIOScheduler()

# Глобальные переменные для ML-моделей
football_model: Optional[FootballModel] = None
feature_engineer: Optional[FeatureEngineer] = None

# Флаг, чтобы избежать повторного запуска парсера
parser_running = False


# ============================================
# 🔄 ИНИЦИАЛИЗАЦИЯ ML-МОДЕЛЕЙ
# ============================================

async def init_ml_models(pool):
    """Инициализировать ML-модели при старте приложения"""
    global football_model, feature_engineer
    
    try:
        feature_engineer = FeatureEngineer(pool)
        football_model = FootballModel(model_path="ml/ml_model.pkl")
        
        if football_model.load():
            logger.info("✅ ML-модель успешно загружена")
        else:
            logger.warning("⚠️ ML-модель не найдена — будут использоваться тестовые данные")
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации ML-моделей: {e}")


# ============================================
# 📊 API ENDPOINTS
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
    
    Возвращает:
    - Исход 1X2 (вероятности и справедливые коэффициенты)
    - Ожидаемые голы (xG)
    - Индивидуальные голы (0, 1+, 2+)
    - Рынки ставок (ТБ 2.5, ОЗ)
    - Точные счёта (Топ-5)
    """
    pool = await get_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database connection not initialized")
    
    # Проверка существования команд
    async with pool.acquire() as conn:
        home_team = await conn.fetchrow('SELECT team_name FROM teams WHERE team_id = $1', home_id)
        away_team = await conn.fetchrow('SELECT team_name FROM teams WHERE team_id = $1', away_id)
        
        if not home_team or not away_team:
            raise HTTPException(status_code=404, detail="Команда не найдена")
    
    # === Если модель не загружена — возвращаем тестовые данные ===
    if not football_model or not football_model.is_trained:
        logger.warning("⚠️ Модель не обучена — возвращаем тестовые данные")
        return {
            "home_team": home_team['team_name'],
            "away_team": away_team['team_name'],
            "prob_home": 0.45, "prob_draw": 0.28, "prob_away": 0.27,
            "fair_odd_home": 2.22, "fair_odd_draw": 3.57, "fair_odd_away": 3.70,
            "xg_home": 1.5, "xg_away": 1.2,
            "home_0": 22.3, "home_1plus": 77.7, "home_2plus": 44.2,
            "away_0": 30.1, "away_1plus": 69.9, "away_2plus": 38.5,
            "over25_prob": 50.6, "over25_odd": 1.97,
            "btts_prob": 53.7, "btts_odd": 1.86,
            "exact_scores": {"1:1": 11.5, "1:0": 10.7, "2:1": 9.1, "0:1": 8.7, "2:0": 7.6},
            "model_status": "mock_data"
        }
    
    # === Реальный прогноз через ML-модель ===
    try:
        # Получаем признаки для матча
        features = await feature_engineer.get_prediction_features(
            home_team_id=home_id,
            away_team_id=away_id,
            league=league
        )
        
        # Делаем прогноз
        prediction = football_model.predict(features)
        
        return {
            "home_team": home_team['team_name'],
            "away_team": away_team['team_name'],
            "model_status": "ml_model",
            **prediction
        }
        
    except Exception as e:
        logger.error(f"Ошибка прогноза: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {str(e)}")


@app.post("/api/update_db")
async def update_database():
    """Запустить парсер для обновления базы данных"""
    global parser_running
    
    if parser_running:
        return JSONResponse(
            status_code=202,
            content={"status": "running", "message": "⏳ Парсер уже запущен"}
        )
    
    parser_running = True
    
    try:
        # Запускаем парсер в отдельном процессе
        process = subprocess.run(
            ["python", "stable_parse.py"],
            capture_output=True,
            text=True,
            timeout=600,  # 10 минут максимум
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if process.returncode == 0:
            logger.info("✅ Парсер успешно завершён")
            return JSONResponse(
                status_code=200,
                content={"status": "success", "message": "✅ База данных обновлена!"}
            )
        else:
            logger.error(f"❌ Ошибка парсера: {process.stderr}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"❌ Ошибка: {process.stderr[:500]}"}
            )
    
    except subprocess.TimeoutExpired:
        logger.error("⏰ Таймаут парсера")
        return JSONResponse(
            status_code=500,
            content={"status": "timeout", "message": "⏰ Превышено время ожидания"}
        )
    
    except Exception as e:
        logger.error(f"❌ Исключение: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"❌ {str(e)}"}
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
    
    model_status = "loaded" if football_model and football_model.is_trained else "not_loaded"
    
    return {
        "status": "healthy",
        "database": db_status,
        "ml_model": model_status,
        "timestamp": datetime.now().isoformat()
    }


# ============================================
# 🌐 WEB PAGES
# ============================================

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Главная страница приложения"""
    pool = await get_pool()
    
    # Получаем статистику для отображения
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


# ============================================
# 🔄 SCHEDULER (Автоматическое обновление)
# ============================================

async def scheduled_parser_run():
    """Автоматический запуск парсера по расписанию"""
    logger.info("🔄 Запуск планового обновления базы данных...")
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
    logger.info("📅 Планировщик запущен: Пн и Чт в 10:00")


# ============================================
# 🚀 STARTUP / SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения"""
    logger.info("🚀 Запуск Football Predictor Pro...")
    await init_db()
    
    # Инициализация ML-моделей (после подключения к БД)
    pool = await get_pool()
    if pool:
        await init_ml_models(pool)
    
    setup_scheduler()
    logger.info("✅ Приложение готово! Откройте http://127.0.0.1:8000")


@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при остановке приложения"""
    scheduler.shutdown()
    logger.info("👋 Приложение остановлено")


# ============================================
# 🎯 MAIN ENTRY POINT
# ============================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )