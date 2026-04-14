"""
parse_historical.py — V2: Парсер предыдущих сезонов с understat.com

Назначение: Одноразовый запуск для загрузки исторических данных
(2 предыдущих сезона) в существующую базу данных.

Сезоны: 2023/2024 и 2024/2025
URL формат: https://understat.com/league/{league}/{year}

Отличия от stable_parse.py:
- Парсит конкретные сезоны (не только текущий)
- НЕ собирает shots (не используются в модели, экономит часы)
- НЕ проверяет last_parsed_date (загружает ВСЕ матчи сезона)
- Season берётся из параметра, а не из даты матча

Использование:
    python parse_historical.py

Время: ~15-30 минут (6 лиг × 2 сезона, без shots)
Результат: ~3,200 дополнительных матчей в БД

После запуска:
    curl -X POST http://127.0.0.1:8000/api/train_hybrid
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import asyncpg
from dotenv import load_dotenv
from aiolimiter import AsyncLimiter
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from tqdm import tqdm
from decimal import Decimal

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------- Настройки -------------------

# Сезоны для загрузки (год начала сезона)
SEASONS = [2023, 2024]

LEAGUES = [
    {'name': 'La_liga', 'slug': 'La_liga'},
    {'name': 'EPL',     'slug': 'EPL'},
    {'name': 'Bundesliga', 'slug': 'Bundesliga'},
    {'name': 'Serie_A', 'slug': 'Serie_A'},
    {'name': 'Ligue_1', 'slug': 'Ligue_1'},
    {'name': 'RFPL',    'slug': 'RFPL'},
]

# ------------------- Хелперы -------------------

def to_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None

def to_decimal(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value)) if value is not None else None
    except:
        return None

# ------------------- Работа с БД -------------------

class HistoricalDatabase:
    """
    Работа с существующей БД.
    НЕ создаёт таблицы — они должны существовать от stable_parse.py
    """
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def __aenter__(self):
        self.pool = await asyncpg.create_pool(
            user=os.getenv('DB_USER', 'myuser'),
            password=os.getenv('DB_PASSWORD', 'pass'),
            database=os.getenv('DB_NAME', 'understat'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 5432)),
        )
        logger.info("Подключение к PostgreSQL установлено")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()
            logger.info("Подключение к PostgreSQL закрыто")

    async def save_teams_bulk(self, teams: List[Tuple[int, str]], conn=None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO teams (team_id, team_name) VALUES ($1, $2)
            ON CONFLICT (team_id) DO UPDATE SET team_name = EXCLUDED.team_name
        ''', teams)

    async def save_matches_bulk(self, matches: List[Tuple], conn=None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO matches (match_id, league, season, datetime, home_team_id, away_team_id, goals_h, goals_a, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (match_id) DO UPDATE SET
                status = EXCLUDED.status,
                goals_h = EXCLUDED.goals_h,
                goals_a = EXCLUDED.goals_a,
                season = EXCLUDED.season
        ''', matches)

    async def save_match_team_stats_bulk(self, stats: List[Tuple], conn=None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO match_team_stats (
                match_id, team_id, xG, npxG, npxGA, deep, deep_allowed,
                ppda_att, ppda_def, ppda_allowed_att, ppda_allowed_def
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (match_id, team_id) DO UPDATE SET
                xG = EXCLUDED.xG,
                npxG = EXCLUDED.npxG,
                npxGA = EXCLUDED.npxGA,
                deep = EXCLUDED.deep,
                deep_allowed = EXCLUDED.deep_allowed,
                ppda_att = EXCLUDED.ppda_att,
                ppda_def = EXCLUDED.ppda_def,
                ppda_allowed_att = EXCLUDED.ppda_allowed_att,
                ppda_allowed_def = EXCLUDED.ppda_allowed_def
        ''', stats)

    async def count_matches(self, league: str, season: str) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                'SELECT COUNT(*) FROM matches WHERE league = $1 AND season = $2',
                league, season
            )

    async def count_all_matches(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval('SELECT COUNT(*) FROM matches')

    async def count_completed_matches(self) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM matches WHERE status = 'Result'")


# ------------------- Парсер -------------------

class HistoricalParser:
    def __init__(self, headless: bool = True, concurrency: int = 2):
        self.headless = headless
        self.concurrency = concurrency
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.limiter = AsyncLimiter(3, 1)  # 3 запроса в секунду

    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080}
        )
        await self.context.route("**/*", self._block_unnecessary_resources)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _block_unnecessary_resources(self, route):
        resource_type = route.request.resource_type
        if resource_type in ["image", "stylesheet", "font", "media"]:
            await route.abort()
        else:
            await route.continue_()

    @staticmethod
    async def _wait_and_extract(page: Page, var_name: str) -> Any:
        try:
            await page.wait_for_function(
                f"() => typeof window.{var_name} !== 'undefined'",
                timeout=30000
            )
            return await page.evaluate(f"window.{var_name}")
        except Exception as e:
            logger.error(f"JS extraction failed for {var_name}: {e}")
            return None

    async def parse_season(self, league_cfg: Dict, year: int,
                           db: HistoricalDatabase) -> Tuple[int, int]:
        """
        Парсит один сезон одной лиги.

        URL: https://understat.com/league/{slug}/{year}
        Season: {year}/{year+1}

        Returns: (new_matches, total_matches_in_season)
        """
        slug = league_cfg['slug']
        league_name = league_cfg['name']
        season_str = f"{year}/{year + 1}"
        url = f"https://understat.com/league/{slug}/{year}"

        logger.info(f"Парсинг: {league_name} — сезон {season_str}")
        logger.info(f"URL: {url}")

        page = await self.context.new_page()

        try:
            async with self.limiter:
                response = await page.goto(url, wait_until='load', timeout=60000)
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")

            # Ждём загрузку данных
            dates_data = await self._wait_and_extract(page, 'datesData')
            teams_data = await self._wait_and_extract(page, 'teamsData')

            if not dates_data or not teams_data:
                raise ValueError("datesData или teamsData не найдены на странице")

            total_in_page = len(dates_data)
            logger.info(f"Найдено матчей на странице: {total_in_page}")

            teams_batch = set()
            matches_batch = []
            stats_batch = []

            # ========== Сбор матчей ==========
            for entry in dates_data:
                h_id = int(entry['h']['id'])
                a_id = int(entry['a']['id'])
                m_id = int(entry['id'])
                dt = datetime.strptime(entry['datetime'], '%Y-%m-%d %H:%M:%S')

                teams_batch.add((h_id, entry['h']['title']))
                teams_batch.add((a_id, entry['a']['title']))

                matches_batch.append((
                    m_id,
                    league_name,
                    season_str,
                    dt,
                    h_id,
                    a_id,
                    to_int(entry['goals']['h']),
                    to_int(entry['goals']['a']),
                    'Result' if entry['isResult'] else 'Timing'
                ))

            # ========== Сбор статистики команд ==========
            match_lookup = {}
            for entry in dates_data:
                h_id = int(entry['h']['id'])
                a_id = int(entry['a']['id'])
                dt = datetime.strptime(entry['datetime'], '%Y-%m-%d %H:%M:%S')
                date_key = dt.strftime('%Y-%m-%d')
                m_id = int(entry['id'])
                match_lookup[(h_id, date_key)] = m_id
                match_lookup[(a_id, date_key)] = m_id

            for team_id_str, team_info in teams_data.items():
                t_id = int(team_id_str)
                for h in team_info.get('history', []):
                    h_date = h.get('date', '').split(' ')[0]
                    m_id = match_lookup.get((t_id, h_date))
                    if m_id:
                        stats_batch.append((
                            m_id,
                            t_id,
                            to_decimal(h.get('xG')),
                            to_decimal(h.get('npxG')),
                            to_decimal(h.get('npxGA')),
                            to_int(h.get('deep')),
                            to_int(h.get('deep_allowed')),
                            to_int(h.get('ppda', {}).get('att')),
                            to_int(h.get('ppda', {}).get('def')),
                            to_int(h.get('ppda_allowed', {}).get('att')),
                            to_int(h.get('ppda_allowed', {}).get('def'))
                        ))

            # ========== Сохранение в БД ==========
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    await db.save_teams_bulk(list(teams_batch), conn=conn)
                    if matches_batch:
                        await db.save_matches_bulk(matches_batch, conn=conn)
                    if stats_batch:
                        await db.save_match_team_stats_bulk(stats_batch, conn=conn)

            # Подсчитываем сколько реально новых (не было в БД)
            existing = await db.count_matches(league_name, season_str)
            new_count = max(0, total_in_page - existing + len(matches_batch))
            # Проще: считаем total_matches_in_page как upper bound

            logger.info(
                f"  {league_name} {season_str}: "
                f"сохранено матчей={len(matches_batch)}, "
                f"статистик={len(stats_batch)}, "
                f"команд={len(teams_batch)}"
            )

            return len(matches_batch), total_in_page

        except Exception as e:
            logger.error(f"  Ошибка парсинга {league_name} {season_str}: {e}")
            return 0, 0

        finally:
            await page.close()

    async def run(self):
        """Основной цикл парсинга всех лиг и сезонов."""
        total_matches = 0
        total_leagues = len(LEAGUES) * len(SEASONS)
        progress = 0

        async with HistoricalDatabase() as db:
            before = await db.count_all_matches()
            before_completed = await db.count_completed_matches()
            logger.info(f"═══════════════════════════════════════════════")
            logger.info(f"  ИСТОРИЧЕСКИЙ ПАРСЕР V2")
            logger.info(f"  Сезоны: {SEASONS}")
            logger.info(f"  Лиги: {[l['name'] for l in LEAGUES]}")
            logger.info(f"  До начала: {before} матчей ({before_completed} завершённых)")
            logger.info(f"  Shots: НЕ собираются (не используются в модели)")
            logger.info(f"═══════════════════════════════════════════════")

            for league in LEAGUES:
                for year in SEASONS:
                    progress += 1
                    season_str = f"{year}/{year + 1}"
                    # Убрано — лог теперь внутри try/except после парсинга
                    logger.info(f"[{progress}/{total_leagues}] ✅ {league['name']} {season_str}")

                    try:
                        count, total = await self.parse_season(league, year, db)
                        total_matches += count
                    except Exception as e:
                        logger.error(f"Критическая ошибка {league['name']} {season_str}: {e}")

                    # Пауза между запросами (уважение к серверу)
                    await asyncio.sleep(2)

            after = await db.count_all_matches()
            after_completed = await db.count_completed_matches()
            added = after - before
            added_completed = after_completed - before_completed

            logger.info(f"\n═══════════════════════════════════════════════")
            logger.info(f"  РЕЗУЛЬТАТ:")
            logger.info(f"  До:     {before} матчей ({before_completed} завершённых)")
            logger.info(f"  После:  {after} матчей ({after_completed} завершённых)")
            logger.info(f"  Добавлено: {added} матчей ({added_completed} завершённых)")
            logger.info(f"═══════════════════════════════════════════════")

            if added > 0:
                logger.info(f"\n✅ Данные загружены! Перетренируйте модель:")
                logger.info(f"   curl -X POST http://127.0.0.1:8000/api/train_hybrid")
            else:
                logger.info(f"\n⚠️  Новые матчи не добавлены (возможно, уже есть в БД)")


async def main():
    parser = HistoricalParser(headless=True, concurrency=2)
    async with parser:
        await parser.run()


if __name__ == "__main__":
    asyncio.run(main())
