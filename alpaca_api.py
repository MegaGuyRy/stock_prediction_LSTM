# alpaca_api.py

import alpaca_trade_api as tradeapi

API_KEY = 'PKB1UY8JYZJS5EMLR2JS'
API_SECRET = '07dMk3HRAQjtv2ntWiKE50J1hzKTAos0Ce7FaLFQ'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, API_SECRET, base_url=BASE_URL, api_version='v2')
