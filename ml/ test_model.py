import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from model import FootballModel

load_dotenv()

async def test():
    pool = await asyncpg.create_pool(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ ML-МОДЕЛИ (LightGBM)")
    print("=" * 70)
    
    model = FootballModel(model_path="ml/ml_model.pkl")
    
    # 1. Обучение
    print("\n🚀 Обучение модели...")
    metrics = await model.train(pool)
    
    if 'error' in metrics:
        print(f"❌ Ошибка: {metrics['error']}")
    else:
        print(f"\n✅ Метрики:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 2. Важность признаков
        print(f"\n📊 Топ-10 признаков:")
        importance_df = model.get_feature_importance()
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']}")
        
        # 3. Тестовый прогноз
        print(f"\n🔮 Тестовый прогноз:")
        test_features = {col: 1.0 for col in model.feature_columns}
        prediction = model.predict(test_features)
        
        print(f"  П1: {prediction['prob_home']*100:.1f}% (Кэф: {prediction['fair_odd_home']:.2f})")
        print(f"  X:  {prediction['prob_draw']*100:.1f}% (Кэф: {prediction['fair_odd_draw']:.2f})")
        print(f"  П2: {prediction['prob_away']*100:.1f}% (Кэф: {prediction['fair_odd_away']:.2f})")
        print(f"  xG Дом: {prediction['xg_home']:.2f}")
        print(f"  xG Гость: {prediction['xg_away']:.2f}")
    
    await pool.close()
    print("\n✅ Тест модели завершён!")

if __name__ == "__main__":
    asyncio.run(test())