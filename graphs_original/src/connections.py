"""Параметры подключений к БД (не бизнес-настройки пайплайна)."""

import os

from dotenv import load_dotenv

load_dotenv('/home/jovyan/fraud_groups_v.3.0/.env')

oisuu_conn = {
    'host': os.getenv('OISUU_CONN_HOST', default='oisuu-db-biz.vsk.ru'),
    'username': os.getenv('OISUU_CONN_USERNAME'),
    'password': os.getenv('OISUU_CONN_PASSWORD'),
    'database': os.getenv('OISUU_CONN_DATABASE', default='OISUU_report'),
}

actuar_conn = {
    'host': os.getenv('ACTUAR_CONN_HOST', default='actuar2.vsk.ru'),
    'username': os.getenv('ACTUAR_CONN_USERNAME'),
    'password': os.getenv('ACTUAR_CONN_PASSWORD'),
    'database': os.getenv('ACTUAR_CONN_DATABASE', default='Motor'),
}
