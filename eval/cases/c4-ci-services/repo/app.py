import os


def database_url() -> str:
    return os.environ["DATABASE_URL"]


def total(prices: list[float]) -> float:
    return round(sum(prices), 2)
