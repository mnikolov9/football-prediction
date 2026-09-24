"""Настройки на проекта."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # кеш на свалените файлове
LOG_DIR = DATA_DIR / "log"          # история на публикуваните прогнози (track record)
BOOST_DIR = DATA_DIR / "boost"      # обучените LightGBM модели (създават се от `run.py tune`)
SITE_DIR = ROOT / "site"            # генериран статичен сайт
TEMPLATES_DIR = ROOT / "templates"

# Държави -> дивизии (кодовете са на football-data.co.uk).
# Всяка държава се моделира съвместно във всички дивизии, така отборите,
# които влизат/изпадат, запазват рейтинга си.
COUNTRIES = {
    "Англия": {"E0": "Премиър лийг", "E1": "Чемпиъншип", "E2": "Лига 1", "E3": "Лига 2"},
    "Испания": {"SP1": "Ла Лига", "SP2": "Сегунда дивисион"},
    "Италия": {"I1": "Серия А", "I2": "Серия Б"},
    "Германия": {"D1": "Бундеслига", "D2": "2. Бундеслига"},
    "Франция": {"F1": "Лига 1", "F2": "Лига 2"},
}

# Шампионска лига – от football-data.org (нужен е безплатен API ключ
# в променлива на средата FOOTBALL_DATA_ORG_KEY).
CL_CODE = "CL"
# Програма на първите дивизии от football-data.org (със същия ключ). Нужна е,
# защото fixtures.csv съдържа само мачовете за идващия уикенд/кръг и се
# обновява във вторник и петък.
FDORG_LEAGUES = {"E0": "PL", "E1": "ELC", "SP1": "PD", "I1": "SA", "D1": "BL1", "F1": "FL1"}
CL_NAME = "Шампионска лига"

# Лига на нациите – модел за националните отбори, обучен върху всички
# международни мачове (github.com/martj42/international_results).
NL_CODE = "UNL"
NL_NAME = "Лига на нациите"
INTL_YEARS = 4               # години назад (националните отбори играят рядко)
INTL_TIME_DECAY_XI = 0.0010
FRIENDLY_WEIGHT = 0.5        # приятелските мачове тежат наполовина
# Ако автоматичният източник няма програмата, мачовете могат да се добавят
# ръчно в този файл (колони: date,home,away,neutral – имената на английски).
NL_MANUAL_FIXTURES = ROOT / "nations_league_fixtures.csv"
# мачове без дата във файла се показват в отделна (сгъната) секция „Дата предстои“
NL_SHOW_UNDATED = True

DIV_NAMES = {d: n for divs in COUNTRIES.values() for d, n in divs.items()}
DIV_NAMES[CL_CODE] = CL_NAME
DIV_COUNTRY = {d: c for c, divs in COUNTRIES.items() for d in divs}
DIV_COUNTRY[CL_CODE] = "Европа"
DIV_NAMES[NL_CODE] = NL_NAME
DIV_COUNTRY[NL_CODE] = "Национални отбори"

# Колко сезона назад да ползваме за обучение
N_SEASONS = 3

# Параметри на модела
TIME_DECAY_XI = 0.0019     # тегло = exp(-xi * дни); ~ половин тегло след ~1 година
L2_PENALTY = 1.0           # регуляризация на силите на отборите
MAX_GOALS = 10

# xG от Understat – има ги само за първите дивизии на петте държави.
# За останалите първенства характеристиките с xG остават празни.
UNDERSTAT_LEAGUES = {"E0": "EPL", "SP1": "La_liga", "I1": "Serie_A", "D1": "Bundesliga", "F1": "Ligue_1"}

# Слой с gradient boosting (LightGBM) върху Dixon–Coles.
# Коригира очакваните голове според формата, xG и почивката. Включва се за
# дадено първенство само ако в бектеста има по-нисък log loss от Dixon–Coles.
BOOST_ENABLED = True
BOOST_TRAIN_DAYS = 600     # период с прогнози „извън извадката“, върху който се учи
BOOST_VALID_SHARE = 0.35   # последната част от периода – за проверка срещу Dixon–Coles
BOOST_MIN_GAIN = 0.002     # минимално подобрение на log loss 1X2, за да се включи

# Линии за пазарите
GOAL_LINES = [1.5, 2.5, 3.5]
CORNER_LINES = [8.5, 9.5, 10.5, 11.5]

# Value залози
# При търсене на value вероятността на модела се смесва с тази на пазара.
# Пазарът е много точен; смесването рязко намалява фалшивите сигнали.
VALUE_MODEL_WEIGHT = 0.5
VALUE_MIN_EDGE = 0.05      # минимум 5% очаквана печалба
VALUE_MAX_ODDS = 8.0       # игнорираме много високи коефициенти (голяма несигурност)
VALUE_MIN_PROB = 0.15
KELLY_FRACTION = 0.25      # четвърт Кели за препоръчителен залог (% от банката)

# Колко дни напред да показваме мачове
DAYS_AHEAD = 10
