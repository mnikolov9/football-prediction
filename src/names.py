"""Български имена на националните отбори от УЕФА."""
NATIONS_BG = {
    "Albania": "Албания", "Andorra": "Андора", "Armenia": "Армения", "Austria": "Австрия",
    "Azerbaijan": "Азербайджан", "Belarus": "Беларус", "Belgium": "Белгия",
    "Bosnia and Herzegovina": "Босна и Херцеговина", "Bulgaria": "България", "Croatia": "Хърватия",
    "Cyprus": "Кипър", "Czech Republic": "Чехия", "Denmark": "Дания", "England": "Англия",
    "Estonia": "Естония", "Faroe Islands": "Фарьорски острови", "Finland": "Финландия",
    "France": "Франция", "Georgia": "Грузия", "Germany": "Германия", "Gibraltar": "Гибралтар",
    "Greece": "Гърция", "Hungary": "Унгария", "Iceland": "Исландия", "Israel": "Израел",
    "Italy": "Италия", "Kazakhstan": "Казахстан", "Kosovo": "Косово", "Latvia": "Латвия",
    "Liechtenstein": "Лихтенщайн", "Lithuania": "Литва", "Luxembourg": "Люксембург",
    "Malta": "Малта", "Moldova": "Молдова", "Montenegro": "Черна гора", "Netherlands": "Нидерландия",
    "North Macedonia": "Северна Македония", "Northern Ireland": "Северна Ирландия",
    "Norway": "Норвегия", "Poland": "Полша", "Portugal": "Португалия",
    "Republic of Ireland": "Ирландия", "Romania": "Румъния", "Russia": "Русия",
    "San Marino": "Сан Марино", "Scotland": "Шотландия", "Serbia": "Сърбия", "Slovakia": "Словакия",
    "Slovenia": "Словения", "Spain": "Испания", "Sweden": "Швеция", "Switzerland": "Швейцария",
    "Turkey": "Турция", "Ukraine": "Украйна", "Wales": "Уелс",
}


def display(name: str) -> str:
    return NATIONS_BG.get(name, name)
