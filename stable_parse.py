import asyncio
import logging
import os
from datetime import datetime, timedelta
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

def get_season(dt: datetime) -> str:
    year = dt.year
    return f"{year}/{year + 1}" if dt.month >= 7 else f"{year - 1}/{year}"

# ------------------- Работа с БД -------------------

class UnderstatPostgresDatabase:
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
        await self._create_tables()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            await self.pool.close()

    async def _create_tables(self):
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS teams (
                        team_id INTEGER PRIMARY KEY,
                        team_name TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS players (
                        player_id INTEGER PRIMARY KEY,
                        player_name TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS matches (
                        match_id INTEGER PRIMARY KEY,
                        league TEXT,
                        season TEXT,
                        datetime TIMESTAMP,
                        home_team_id INTEGER REFERENCES teams(team_id) ON DELETE CASCADE,
                        away_team_id INTEGER REFERENCES teams(team_id) ON DELETE CASCADE,
                        goals_h INTEGER,
                        goals_a INTEGER, 
                        status TEXT,
                        shots_failed BOOLEAN DEFAULT FALSE
                    );
                    CREATE TABLE IF NOT EXISTS shots (
                        shot_id INTEGER PRIMARY KEY,
                        match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
                        team_id INTEGER REFERENCES teams(team_id) ON DELETE CASCADE,
                        player_id INTEGER REFERENCES players(player_id) ON DELETE CASCADE,
                        x NUMERIC(6,4),
                        y NUMERIC(6,4),
                        xG NUMERIC(6,4),
                        minute INTEGER,
                        result TEXT,
                        situation TEXT,
                        assisted_by TEXT,
                        shot_type TEXT
                    );
                    CREATE TABLE IF NOT EXISTS match_team_stats (
                        match_id INTEGER REFERENCES matches(match_id) ON DELETE CASCADE,
                        team_id INTEGER REFERENCES teams(team_id) ON DELETE CASCADE,
                        xG FLOAT,
                        npxG FLOAT,
                        npxGA FLOAT,
                        deep INTEGER,
                        deep_allowed INTEGER,
                        ppda_att INTEGER,
                        ppda_def INTEGER,
                        ppda_allowed_att INTEGER,
                        ppda_allowed_def INTEGER,
                        PRIMARY KEY (match_id, team_id)
                    );
                    CREATE TABLE IF NOT EXISTS logs (
                        id SERIAL PRIMARY KEY,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT,
                        league_processed TEXT,
                        message TEXT
                    );
                ''')

                await self._ensure_column(conn, 'matches', 'season', 'TEXT')
                await self._ensure_column(conn, 'shots', 'player_id', 'INTEGER')
                await self._ensure_column(conn, 'shots', 'shot_type', 'TEXT')
                await self._ensure_foreign_key(conn, 'shots', 'player_id', 'players', 'player_id')

                await conn.execute('CREATE INDEX IF NOT EXISTS idx_shots_match_id ON shots(match_id);')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_shots_player_id ON shots(player_id);')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_matches_league_datetime ON matches(league, datetime);')
                await conn.execute('CREATE INDEX IF NOT EXISTS idx_match_team_stats_match_id ON match_team_stats(match_id);')

    async def _ensure_column(self, conn: asyncpg.Connection, table: str, column: str, data_type: str):
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = $1 AND column_name = $2
            )
        """, table, column)
        if not exists:
            logger.info(f"Добавляю колонку {column} в таблицу {table}")
            await conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {data_type};")

    async def _ensure_foreign_key(self, conn: asyncpg.Connection, table: str, column: str,
                                  ref_table: str, ref_column: str):
        exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.table_constraints
                WHERE constraint_type = 'FOREIGN KEY'
                  AND table_name = $1
            )
        """, table)
        if not exists:
            await conn.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.table_constraints
                        WHERE constraint_type = 'FOREIGN KEY'
                          AND table_name = '{table}'
                          AND constraint_name LIKE '%_{column}_fkey'
                    ) THEN
                        EXECUTE 'ALTER TABLE {table} ADD FOREIGN KEY ({column}) REFERENCES {ref_table}({ref_column})';
                    END IF;
                END $$;
            """)

    # ========== НОВЫЙ МЕТОД: Получение последней даты парсинга ==========
    async def get_last_parsed_date(self, league: str) -> Optional[datetime]:
        """Получить дату последнего загруженного матча для лиги"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT MAX(datetime) as last_date 
                FROM matches 
                WHERE league = $1
            ''', league)
            return row['last_date'] if row and row['last_date'] else None

    # ========== НОВЫЙ МЕТОД: Проверка существования матча ==========
    async def match_exists(self, match_id: int) -> bool:
        """Проверить, существует ли матч в базе"""
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval(
                'SELECT EXISTS(SELECT 1 FROM matches WHERE match_id = $1)',
                match_id
            )
            return exists

    # ========== Методы сохранения ==========
    async def save_teams_bulk(self, teams: List[Tuple[int, str]], conn: asyncpg.Connection = None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO teams (team_id, team_name) VALUES ($1, $2)
            ON CONFLICT (team_id) DO UPDATE SET team_name = EXCLUDED.team_name
        ''', teams)

    async def save_players_bulk(self, players: List[Tuple[int, str]], conn: asyncpg.Connection = None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO players (player_id, player_name) VALUES ($1, $2)
            ON CONFLICT (player_id) DO UPDATE SET player_name = EXCLUDED.player_name
        ''', players)

    async def save_matches_bulk(self, matches: List[Tuple], conn: asyncpg.Connection = None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO matches (match_id, league, season, datetime, home_team_id, away_team_id, goals_h, goals_a, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (match_id) DO UPDATE SET
                status = EXCLUDED.status,
                goals_h = EXCLUDED.goals_h,
                goals_a = EXCLUDED.goals_a
        ''', matches)

    async def save_match_team_stats_bulk(self, stats: List[Tuple], conn: asyncpg.Connection = None):
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

    async def save_shots_bulk(self, shots: List[Tuple], conn: asyncpg.Connection = None):
        target = conn or self.pool
        await target.executemany('''
            INSERT INTO shots (
                shot_id, match_id, team_id, player_id, x, y, xG, minute,
                result, situation, assisted_by, shot_type
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            ON CONFLICT (shot_id) DO NOTHING
        ''', shots)

    # ========== Вспомогательные запросы ==========
    async def get_matches_without_shots(self) -> List[int]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch('''
                SELECT m.match_id FROM matches m
                LEFT JOIN shots s ON m.match_id = s.match_id
                WHERE s.shot_id IS NULL
                  AND m.shots_failed = FALSE
                  AND m.status = 'Result'
            ''')
            return [r['match_id'] for r in rows]

    async def mark_match_shots_failed(self, match_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute('UPDATE matches SET shots_failed = TRUE WHERE match_id = $1', match_id)

    async def log_start(self, league: str) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                'INSERT INTO logs (league_processed, status) VALUES ($1, $2) RETURNING id',
                league, 'RUNNING'
            )

    async def log_finish(self, log_id: int, status: str, message: str = None):
        async with self.pool.acquire() as conn:
            await conn.execute(
                'UPDATE logs SET end_time = CURRENT_TIMESTAMP, status = $1, message = $2 WHERE id = $3',
                status, message, log_id
            )

# ------------------- Парсер -------------------

class UnderstatParser:
    LEAGUES = [
        {'name': 'La_liga', 'url': 'https://understat.com/league/La_liga'},
        {'name': 'EPL', 'url': 'https://understat.com/league/EPL'},
        {'name': 'Bundesliga', 'url': 'https://understat.com/league/Bundesliga'},
        {'name': 'Serie_A', 'url': 'https://understat.com/league/Serie_A'},
        {'name': 'Ligue_1', 'url': 'https://understat.com/league/Ligue_1'},
        {'name': 'RFPL', 'url': 'https://understat.com/league/RFPL'},
    ]

    def __init__(self, headless: bool = True, concurrency: int = 3):
        self.headless = headless
        self.concurrency = concurrency
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.limiter = AsyncLimiter(2, 1)

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
        if resource_type in ["image", "stylesheet", "font"]:
            await route.abort()
        else:
            await route.continue_()

    @staticmethod
    async def _wait_and_extract(page: Page, var_name: str) -> Any:
        try:
            await page.wait_for_function(f"() => typeof window.{var_name} !== 'undefined'", timeout=20000)
            return await page.evaluate(f"window.{var_name}")
        except Exception as e:
            logger.error(f"JS extraction failed for {var_name}: {e}")
            return None

    # ========== ОБНОВЛЁННЫЙ МЕТОД: Парсинг с проверкой даты ==========
    async def parse_league(self, league_cfg: Dict, db: UnderstatPostgresDatabase) -> Tuple[int, int]:
        """
        Парсит лигу с оптимизацией:
        - Проверяет последнюю дату в БД
        - Пропускает уже загруженные матчи
        - Возвращает (новые_матчи, обновлённые_матчи)
        """
        page = await self.context.new_page()
        
        # Получаем последнюю дату парсинга для этой лиги
        last_parsed_date = await db.get_last_parsed_date(league_cfg['name'])
        
        if last_parsed_date:
            logger.info(f"Лига {league_cfg['name']}: последняя дата в БД = {last_parsed_date.strftime('%d.%m.%Y %H:%M')}")
        else:
            logger.info(f"Лига {league_cfg['name']}: первая загрузка (нет данных в БД)")
        
        try:
            async with self.limiter:
                response = await page.goto(league_cfg['url'], wait_until='load', timeout=60000)
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}")

            dates_data = await self._wait_and_extract(page, 'datesData')
            teams_data = await self._wait_and_extract(page, 'teamsData')
            
            if not dates_data or not teams_data:
                raise ValueError("Missing datesData or teamsData")

            teams_batch = set()
            matches_batch = []
            stats_batch = []
            
            new_matches = 0
            updated_matches = 0
            skipped_matches = 0

            for entry in dates_data:
                h_id = int(entry['h']['id'])
                a_id = int(entry['a']['id'])
                m_id = int(entry['id'])
                dt = datetime.strptime(entry['datetime'], '%Y-%m-%d %H:%M:%S')
                date_key = dt.strftime('%Y-%m-%d')

                teams_batch.add((h_id, entry['h']['title']))
                teams_batch.add((a_id, entry['a']['title']))

                # ========== ПРОВЕРКА: Пропускаем старые матчи ==========
                if last_parsed_date and dt <= last_parsed_date:
                    # Проверяем, нужно ли обновить статус (Timing -> Result)
                    if entry['isResult']:
                        # Матч завершён, обновляем статус и счёт
                        matches_batch.append((
                            m_id,
                            league_cfg['name'],
                            get_season(dt),
                            dt,
                            h_id,
                            a_id,
                            to_int(entry['goals']['h']),
                            to_int(entry['goals']['a']),
                            'Result'
                        ))
                        updated_matches += 1
                    else:
                        skipped_matches += 1
                    continue

                # Новый матч — добавляем в батч
                matches_batch.append((
                    m_id,
                    league_cfg['name'],
                    get_season(dt),
                    dt,
                    h_id,
                    a_id,
                    to_int(entry['goals']['h']),
                    to_int(entry['goals']['a']),
                    'Result' if entry['isResult'] else 'Timing'
                ))
                new_matches += 1

            # Обработка статистики команд
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

            # Сохранение в БД
            async with db.pool.acquire() as conn:
                async with conn.transaction():
                    await db.save_teams_bulk(list(teams_batch), conn=conn)
                    if matches_batch:
                        await db.save_matches_bulk(matches_batch, conn=conn)
                    if stats_batch:
                        await db.save_match_team_stats_bulk(stats_batch, conn=conn)

            logger.info(f"Лига {league_cfg['name']}: новые={new_matches}, обновлённые={updated_matches}, пропущено={skipped_matches}")
            return new_matches, updated_matches

        finally:
            await page.close()

    async def _process_match_shots(self, match_id: int, db: UnderstatPostgresDatabase, 
                                    semaphore: asyncio.Semaphore, page: Page):
        async with semaphore:
            attempt = 1
            max_retries = 3
            success = False

            while attempt <= max_retries and not success:
                try:
                    async with self.limiter:
                        response = await page.goto(
                            f"https://understat.com/match/{match_id}",
                            wait_until='load',
                            timeout=60000
                        )

                    if response and response.status == 404:
                        logger.warning(f"Матч {match_id} вернул 404 – возможно, удалён")
                        break

                    shots_data = await self._wait_and_extract(page, 'shotsData')
                    if not shots_data:
                        raise ValueError("shotsData not found")

                    players_batch = {}
                    shots_batch = []

                    async with db.pool.acquire() as conn:
                        match_row = await conn.fetchrow(
                            "SELECT home_team_id, away_team_id FROM matches WHERE match_id = $1",
                            match_id
                        )
                    if not match_row:
                        raise ValueError(f"Match {match_id} not found in database")
                    
                    home_team_id = match_row['home_team_id']
                    away_team_id = match_row['away_team_id']

                    for side in ['h', 'a']:
                        for s in shots_data.get(side, []):
                            p_id = int(s['player_id'])
                            players_batch[p_id] = s['player']
                            team_id = home_team_id if side == 'h' else away_team_id

                            shots_batch.append((
                                int(s['id']),
                                match_id,
                                team_id,
                                p_id,
                                to_decimal(s['X']),
                                to_decimal(s['Y']),
                                to_decimal(s['xG']),
                                to_int(s['minute']),
                                s['result'],
                                s['situation'],
                                s.get('player_assisted'),
                                s.get('shotType')
                            ))

                    async with db.pool.acquire() as conn:
                        async with conn.transaction():
                            if players_batch:
                                await db.save_players_bulk(list(players_batch.items()), conn=conn)
                            if shots_batch:
                                await db.save_shots_bulk(shots_batch, conn=conn)

                    logger.info(f"Матч {match_id}: сохранено {len(shots_batch)} ударов")
                    success = True

                except Exception as e:
                    logger.warning(f"Матч {match_id}, попытка {attempt}/{max_retries}: {e}")
                    if attempt == max_retries:
                        await db.mark_match_shots_failed(match_id)
                        logger.error(f"Матч {match_id} помечен как ошибочный")
                    else:
                        await asyncio.sleep(5 * attempt)
                    attempt += 1

    async def collect_shots(self, db: UnderstatPostgresDatabase):
        match_ids = await db.get_matches_without_shots()
        if not match_ids:
            logger.info("Нет матчей, требующих загрузки ударов.")
            return

        logger.info(f"Начинаем параллельный сбор ударов для {len(match_ids)} матчей (concurrency={self.concurrency})")
        semaphore = asyncio.Semaphore(self.concurrency)
        pages = [await self.context.new_page() for _ in range(self.concurrency)]

        pbar = tqdm(total=len(match_ids), desc="Сбор ударов", unit="матчей")

        async def process_with_pbar(match_id, page):
            try:
                await self._process_match_shots(match_id, db, semaphore, page)
            finally:
                pbar.update(1)

        tasks = [
            process_with_pbar(mid, pages[i % len(pages)])
            for i, mid in enumerate(match_ids)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        pbar.close()
        for page in pages:
            await page.close()
        logger.info("Сбор ударов завершён.")

    async def run(self):
        async with UnderstatPostgresDatabase() as db:
            logger.info("Начинаем парсинг лиг...")
            
            total_new = 0
            total_updated = 0
            
            for idx, league in enumerate(self.LEAGUES, 1):
                log_id = await db.log_start(league['name'])
                try:
                    new_count, updated_count = await self.parse_league(league, db)
                    total_new += new_count
                    total_updated += updated_count
                    await db.log_finish(log_id, 'SUCCESS', f"Новых: {new_count}, Обновлённых: {updated_count}")
                    logger.info(f"[{idx}/{len(self.LEAGUES)}] Лига {league['name']} завершена")
                except Exception as e:
                    logger.error(f"Лига {league['name']} завершилась с ошибкой: {e}")
                    await db.log_finish(log_id, 'ERROR', str(e))

            logger.info(f"════════════════════════════════════════")
            logger.info(f"ВСЕГО: Новые матчи={total_new}, Обновлённые={total_updated}")
            logger.info(f"════════════════════════════════════════")

            await self.collect_shots(db)

async def main():
    parser = UnderstatParser(headless=True, concurrency=3)
    async with parser:
        await parser.run()

if __name__ == "__main__":
    asyncio.run(main())