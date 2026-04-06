import asyncio
import asyncpg
import os
from dotenv import load_dotenv
from features import FeatureEngineer
from model import FootballModel

load_dotenv()

async def test():
    pool = None
    try:
        pool = await asyncpg.create_pool(
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )
        
        print("=" * 70)
        print("🧪 ФИНАЛЬНЫЙ ТЕСТ: Полный пайплайн")
        print("=" * 70)
        
        # Инициализация
        fe = FeatureEngineer(pool)
        model = FootballModel(model_path="ml/ml_model.pkl")
        
        if not model.load():
            print("⚠️ Модель не найдена — пропускаем тест прогноза")
            return
        
        print("\n✅ Модель загружена")
        
        # Тестовые команды (замени на реальные ID из твоей БД)
        # Сначала получим любые команды из БД
        async with pool.acquire() as conn:
            teams = await conn.fetch('SELECT team_id, team_name FROM teams LIMIT 2')
            if len(teams) >= 2:
                home_id = teams[0]['team_id']
                away_id = teams[1]['team_id']
                home_name = teams[0]['team_name']
                away_name = teams[1]['team_name']
            else:
                print("⚠️ Недостаточно команд в БД")
                return
        
        league = "EPL"
        
        print(f"\n🔮 Прогноз: {home_name} vs {away_name} ({league})")
        
        # Получаем признаки
        features = await fe.get_prediction_features(home_id, away_id, league)
        
        # Делаем прогноз
        prediction = model.predict(features)
        
        # Вывод результатов
        print(f"\n📊 Исход 1X2:")
        print(f"  П1: {prediction['prob_home']*100:.1f}% (Кэф: {prediction['fair_odd_home']:.2f})")
        print(f"  X:  {prediction['prob_draw']*100:.1f}% (Кэф: {prediction['fair_odd_draw']:.2f})")
        print(f"  П2: {prediction['prob_away']*100:.1f}% (Кэф: {prediction['fair_odd_away']:.2f})")
        
        print(f"\n⚽ Ожидаемые голы (xG):")
        print(f"  Дом: {prediction['xg_home']:.2f}")
        print(f"  Гость: {prediction['xg_away']:.2f}")
        
        print(f"\n🎯 Индивидуальные голы:")
        print(f"  Дом 0: {prediction['home_0']:.1f}% | 1+: {prediction['home_1plus']:.1f}% | 2+: {prediction['home_2plus']:.1f}%")
        print(f"  Гость 0: {prediction['away_0']:.1f}% | 1+: {prediction['away_1plus']:.1f}% | 2+: {prediction['away_2plus']:.1f}%")
        
        print(f"\n💰 Рынки ставок:")
        print(f"  ТБ 2.5: {prediction['over25_prob']:.1f}% (Кэф: {prediction['over25_odd']:.2f})")
        print(f"  ОЗ Да:  {prediction['btts_prob']:.1f}% (Кэф: {prediction['btts_odd']:.2f})")
        
        print(f"\n🎲 Точные счёта (Топ-5):")
        for score, prob in prediction['exact_scores'].items():
            print(f"  {score}: {prob:.2f}%")
        
        print("\n✅ Полный тест завершён!")
        
    except Exception as e:
        print(f"\n❌ Ошибка теста: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Корректное закрытие пула
        if pool:
            try:
                await pool.close()
                print("🔒 Пул соединений закрыт")
            except Exception as e:
                print(f"⚠️ Предупреждение при закрытии пула: {e}")

if __name__ == "__main__":
    asyncio.run(test())