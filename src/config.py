import numpy as np
import os
from dotenv import load_dotenv


X_token = ""

# Аномалии в ФИО людей
wrong_people_val = ['НЕ УСТАНОВЛЕН', 'НЕИЗВЕСТНЫЙ -', 'НЕУСТАНОВЛЕННЫЙ ВОДИТЕЛЬ', 'НЕИЗВЕСТНЫЙ ВОДИТЕЛЬ', \
                    'СКРЫЛСЯ', 'НЕУСТАНОВЛЕН НЕУСТАНОВЛЕН НЕУСТАНОВЛЕН', 'ОТСУТСТВОВАЛ НА МЕСТЕ ДТП', 'ОТСУТСТВОВАЛ -', \
                    'НЕ УПРАВЛЯЛОСЬ', '-----------------', '11111111111111111', '00000000000000000', 'ОТСУТСТВУЕТ',]

# Аномалии в VIN-номерах авто
wrong_auto_val = {'А001АА': np.nan, 'ЖИЗНЬ': np.nan, 'ВРЕДЗДОРОВЬЮВ': np.nan, 'ЗДОРОВЬЕ': np.nan, \
                  'НУ': np.nan, 'ЖИЗНЬИЗДОРОВЬ': np.nan, 'МОТОЦИКЛ': np.nan, 'СМЕРТЬ ПОСЛЕ ДТП': np.nan,}

# Минимальное количество человек в группе (default = 15) (группы с меньшим количеством человек исключаются из отчета)
bound = 15



load_dotenv("/home/jovyan/fraud_groups_v.3.0/.env")

oisuu_conn = {
    'host': os.getenv("OISUU_CONN_HOST", default='oisuu-db-biz.vsk.ru'),
    'username': os.getenv("OISUU_CONN_USERNAME"),
    'password': os.getenv("OISUU_CONN_PASSWORD"),
    'database': os.getenv("OISUU_CONN_DATABASE", default='OISUU_report')
}

actuar_conn = {
    'host': os.getenv("ACTUAR_CONN_HOST", default='actuar2.vsk.ru'),
    'username': os.getenv("ACTUAR_CONN_USERNAME"),
    'password': os.getenv("ACTUAR_CONN_PASSWORD"),
    'database': os.getenv("ACTUAR_CONN_DATABASE", default='Motor')
}
