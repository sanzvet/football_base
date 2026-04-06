import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from features import FeatureEngineer

load_dotenv()

async def test():
    pool = await asyncpg.create_pool(
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_NAME'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    
    fe = FeatureEngineer(pool)
    
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ FEATURE ENGINEERING")
    print("=" * 70)
    
    # 1. Загрузка данных
    df = await fe.get_training_data()
    print(f"\n✅ Загружено матчей: {len(df)}")
    
    if len(df) > 0:
        # 2. Построение матрицы (тест на 100 матчах для скорости)
        feature_df = await fe.build_feature_matrix(df.head(100))
        print(f"✅ Матрица признаков: {feature_df.shape}")
        
        # 3. Показать колонки
        print(f"\n📊 Признаки ({len(fe.get_feature_columns())}):")
        for col in fe.get_feature_columns()[:10]:
            print(f"  - {col}")
        print(f"  ... и ещё {len(fe.get_feature_columns()) - 10}")
        
        # 4. Time Decay тест
        from datetime import datetime
        weight_old = fe.calculate_time_decay_weight(datetime(2020, 1, 1))
        weight_new = fe.calculate_time_decay_weight(datetime(2025, 1, 1))
        print(f"\n⏰ Time Decay: старый матч={weight_old:.3f}, новый={weight_new:.3f}")
        
        # 5. Статистика признаков
        print(f"\n📈 Статистика признаков:")
        print(f"  xg_home: mean={feature_df['xg_home'].mean():.3f}, std={feature_df['xg_home'].std():.3f}")
        print(f"  npxg_home: mean={feature_df['npxg_home'].mean():.3f}, std={feature_df['npxg_home'].std():.3f}")
        print(f"  penalty_xg_home: mean={feature_df['penalty_xg_home'].mean():.3f}")
    
    await pool.close()
    print("\n✅ Тест завершён!")

if __name__ == "__main__":
    asyncio.run(test())