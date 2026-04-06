import asyncpg
from config import settings

pool = None

async def init_db():
    global pool
    pool = await asyncpg.create_pool(
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
        port=settings.DB_PORT
    )
    await create_predictions_table()

async def get_pool():
    return pool

async def create_predictions_table():
    async with pool.acquire() as conn:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                match_id INTEGER PRIMARY KEY REFERENCES matches(match_id),
                prob_home FLOAT,
                prob_draw FLOAT,
                prob_away FLOAT,
                xg_home FLOAT,
                xg_away FLOAT,
                fair_odd_home FLOAT,
                fair_odd_draw FLOAT,
                fair_odd_away FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS bookmaker_odds (
                match_id INTEGER REFERENCES matches(match_id),
                odd_home FLOAT,
                odd_draw FLOAT,
                odd_away FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')