from dixon_coles import DixonColes
import pprint

def test():
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ DIXON-COLES CORRECTION")
    print("=" * 70)
    
    dc = DixonColes(rho=0.05)
    
    # Тестовые xG из модели
    lambda_h = 1.5  # Ожидаемые голы хозяев
    lambda_a = 1.2  # Ожидаемые голы гостей
    
    print(f"\n📊 Входные данные:")
    print(f"  xG Дом: {lambda_h}")
    print(f"  xG Гость: {lambda_a}")
    
    # Расчёт рынков
    markets = dc.calculate_market_probabilities(lambda_h, lambda_a)
    
    print(f"\n💰 Вероятности рынков:")
    print(f"  ТБ 2.5: {markets['over_2_5']*100:.1f}%")
    print(f"  ТМ 2.5: {markets['under_2_5']*100:.1f}%")
    print(f"  ОЗ Да:  {markets['btts_yes']*100:.1f}%")
    print(f"  ОЗ Нет: {markets['btts_no']*100:.1f}%")
    print(f"  П1:     {markets['result_home']*100:.1f}%")
    print(f"  X:      {markets['result_draw']*100:.1f}%")
    print(f"  П2:     {markets['result_away']*100:.1f}%")
    
    print(f"\n🎯 Точные счёта (Топ-10):")
    exact_scores = {k: v for k, v in markets.items() if k.startswith('exact_')}
    sorted_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for score, prob in sorted_scores:
        g_h, g_a = score.replace('exact_', '').split('_')
        print(f"  {g_h}:{g_a} — {prob*100:.2f}%")
    
    # Справедливые коэффициенты
    odds = dc.get_fair_odds(markets)
    
    print(f"\n📈 Справедливые коэффициенты:")
    print(f"  ТБ 2.5: {odds['over_2_5_odd']:.2f}")
    print(f"  ОЗ Да:  {odds['btts_yes_odd']:.2f}")
    print(f"  П1:     {odds['result_home_odd']:.2f}")
    print(f"  X:      {odds['result_draw_odd']:.2f}")
    print(f"  П2:     {odds['result_away_odd']:.2f}")
    
    print("\n✅ Тест Dixon-Coles завершён!")

if __name__ == "__main__":
    test()