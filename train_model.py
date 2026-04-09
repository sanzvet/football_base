import asyncio
import asyncpg
from ml.model import FootballModel
from config import settings

async def train():
    pool = await asyncpg.create_pool(
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
        port=settings.DB_PORT
    )
    model = FootballModel("ml/ml_model.pkl")
    metrics = await model.train(pool)  # ← здесь создаётся файл модели
    print("Обучение завершено, метрики:", metrics)
    await pool.close()

if __name__ == "__main__":
    asyncio.run(train())