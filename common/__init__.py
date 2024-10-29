from enum import Enum
from typing import List
from typing import List
import os
import logging
from datetime import datetime
import uuid
from typing import List, Optional
from pandas import Series, DataFrame
from abc import ABC, abstractmethod
from typing import List

from datetime import date
from enum  import Enum
from functools import reduce
import math
from typing import List, Optional, Tuple
from unicodedata import numeric
import pandas as pd

class Resolution(Enum):
    MONTH = 'month',
    WEEK = 'w',
    DAY = 'd'
    HOUR = 'h',
    MINUTE = 'm'


class RunOptions:
    """
    Universal configuration for running a strategy
    """

    def __init__(
        self,
        instrument: str,
        date_from: date,
        date_to: date,
        contra: str = 'USD',
        transaction_cost = 0.003,
        resolution: Tuple[int,Resolution] = (24, Resolution.HOUR),
        analysis_date: Optional[date] = None
    ) -> None:
        self.instrument = instrument
        self.date_from = date_from
        self.date_to = date_to
        self.contra = contra
        self.transaction_cost = transaction_cost
        self.resolution = resolution
        self.analysis_date = analysis_date if analysis_date else self.date_from

    @property
    def pair(self):
        return f"{self.instrument}{self.contra}"

    def __str__(self) -> str:
            return str(vars(self))

    def __repr__(self) -> str:
        return str(self)
    
    
def join_all(dfs: List[pd.DataFrame]):
    return reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how="outer"
        ),
        dfs,
    )
    

def log_to_aritmetic(log_value):
    return math.exp(log_value) - 1

def aritmetic_to_log(arithmetic_value):
    return math.log(arithmetic_value)

usd_equivs = ['USDT', 'BUSD', 'USDC', 'TUSD', 'PAX', 'USDS', 'DAI']

class OrderType(Enum):
    LIMIT = 'LIMIT'
    MARKET = 'MARKET'
    STOP_LOSS = 'STOP_LOSS'
    STOP_LOSS_LIMIT = 'STOP_LOSS_LIMIT'
    TAKE_PROFIT = 'TAKE_PROFIT'
    TAKE_PROFIT_LIMIT = 'TAKE_PROFIT_LIMIT'
    LIMIT_MAKER = 'LIMIT_MAKER'
    
class Side(Enum):
    BUY = 'BUY'
    SELL = 'SELL'
    
class TransactionType(Enum):
    OPEN = 'OPEN'
    CLOSE = 'CLOSE'
    ADJUSTMENT = 'ADJUSTMENT'
    
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.__str__()
    
class  AccountInfo:
    
    def __init__(self,
                 balance: float,
                 pnl: float,
                 available_balance:float,
                 utilizable_balance:float = 0) -> None:
        self.balance = balance
        self.pnl = pnl
        self.available_balance = available_balance
        self.utilizable_balance = utilizable_balance
    
    def __str__(self) -> str:
        return f"AccountInfo(balance={self.balance}, pnl={self.pnl}, available_balance={self.available_balance})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> dict:
        return self.__dict__.copy()
        
class Position:
    
    def __init__(self, trade, contra, symbol, amount, notional, counterparty, leverage=None, initial_margin=None, maintenance_margin=None, entry_price=None, cpty_position_exposure=None):
        self.trade_asset = trade
        self.contra_asset = contra
        self.symbol = symbol
        self.counterparty = counterparty    
        self.amount = amount
        self.notional = notional
        self.leverage = leverage
        self.initial_margin = initial_margin
        self.maintenance_margin = maintenance_margin
        self.entry_price = entry_price
        self.cpty_position_exposure = cpty_position_exposure
        
    def __str__(self) -> str:
        return f"Position(trade_asset={self.trade_asset}, contra_asset={self.contra_asset}, symbol={self.symbol}, amount={self.amount}, notional={self.notional}, leverage={self.leverage}, initial_margin={self.initial_margin}, maintenance_margin={self.maintenance_margin}, entry_price={self.entry_price}, counterparty={self.counterparty})"
    
    def __repr__(self) -> str:  
        return self.__str__()
       
    def to_dict(self) -> dict:
        return self.__dict__.copy()

class Trade:
    
    def __init__(self,
                    type: str,
                    trade_asset: str,
                    contra_asset: str,
                    timestamp: int,
                    price: float,
                    amount: float,
                    side: str,
                    fees: float,
                    execution_order_id: str,
                    meta_data: object) -> None:
        self.type = type
        self.trade_asset = trade_asset
        self.contra_asset = contra_asset
        self.timestamp = timestamp
        self.price = price
        self.amount = amount
        self.side = side
        self.fees = fees
        self.execution_order_id = execution_order_id
        self.meta_data = meta_data
    
    def __str__(self) -> str:
        return f"Trade(type={self.type}, trade_asset={self.trade_asset}, contra_asset={self.contra_asset}, timestamp={self.timestamp}, price={self.price}, amount={self.amount}, side={self.side}, fees={self.fees}, execution_order_id={self.execution_order_id}, meta_data={self.meta_data})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self, include_metadata=False) -> dict:
        trade = self.__dict__.copy()
        if not include_metadata:
            del trade['meta_data']
            
        return trade
    
    
class TradeOrder:
    def __init__(self, trade_symbol: str, 
                 contra_symbol: str, 
                 side: Side, 
                 quantity: float, 
                 order_type: OrderType, 
                 counterparty: str,
                 transaction_type: TransactionType,
                 price: float = None, 
                 stop_price: float = None):
        self.trade_symbol = trade_symbol
        self.contra_symbol = contra_symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.counterparty = counterparty
        self.price = price
        self.stop_price = stop_price
        self.transaction_type = transaction_type

def extract_asset_from_symbol(pair: str, filter_list: List[str] = usd_equivs) -> str:
    for asset in filter_list:
        if pair.endswith(asset):
            return pair[:-len(asset)]
        if pair.startswith(asset):
            return pair[len(asset):]
        
    raise ValueError(f"Could not extract asset from {pair} with filter list {filter_list}")

def extract_stable_from_symbol(pair: str, filter_list: List[str] = usd_equivs) -> str:
    for asset in filter_list:
        if pair.endswith(asset):
            return asset
        
    raise ValueError(f"Could not extract contra asset from {pair} with filter list {filter_list}")

class CalibrationDataPoint():
    def __init__(self, timestamp: datetime, 
                 portfolio: str, 
                 approval_status: str, 
                 approval_user: str, 
                 calibration_value: float, 
                 data: DataFrame ) -> None:
        self.timestamp = timestamp
        self.portfolio = portfolio
        self.approval_status = approval_status
        self.approval_user = approval_user
        self.calibration_value = calibration_value
        self.data = data
        
    def to_dict(self):
        return {
            'id':str(uuid.uuid4()), #Needed for cosmosdb
            'timestamp': str(self.timestamp),
            'portfolio': self.portfolio,
            'approval_status': self.approval_status,
            'approval_user': self.approval_user,
            'calibration_value': self.calibration_value,
            'data': self.data.fillna(0).to_dict(orient='records')
        }

class ModelEnum:
    RV = 'RV'
    SB = 'Smart_Beta'
    CTA = 'CTA'
    
class Exchange:
    BINANCE = 'binance'
    

class SignalDataPoint():
    def __init__(self, timestamp: datetime, label: str, data: Series ) -> None:
        self.timestamp = timestamp
        self.label = label
        self.data = data
        
    def to_dict(self):
        return {
            'id':str(uuid.uuid4()), #Needed for cosmosdb
            'timestamp': str(self.timestamp),
            'label':self.label,
            'data': self.data.fillna(0).to_dict(orient='records')
        }

class Strategy:
    def __init__(self, stratlet: str, account: str, universe: List[str], defaultUniverseSize: Optional[int] = None):
        self.stratlet = stratlet
        self.account = account
        self.universe = universe
        self.defaultUniverseSize = defaultUniverseSize
    
    def __repr__(self) -> str:
        return self.__dict__.__repr__()

class Portfolio:
    def __init__(self, id: str, name: str, isActive: bool, strategyAccountName: str, counterpartyAccountName: str, treasuryAccountName: str, strategies: List[Strategy]):
        self.id = id
        self.name = name
        self.isActive = isActive
        self.strategyAccountName = strategyAccountName
        self.counterpartyAccountName = counterpartyAccountName
        self.treasuryAccountName = treasuryAccountName
        self.strategies = strategies
        
    def get_strategy(self, stratlet: str) -> Strategy:
        return next(filter(lambda x: x.stratlet == stratlet, self.strategies), None)
        
    def __repr__(self) -> str:
        return self.__dict__.__repr__()

def map_dict_to_class(data: dict) -> Portfolio:
    strategies = [Strategy(**strategy_data) for strategy_data in data['strategies']]
    return Portfolio(
        id=data['id'],
        name=data['name'],
        isActive=data['isActive'],
        strategyAccountName=data['strategyAccountName'],
        counterpartyAccountName=data['counterpartyAccountName'],
        treasuryAccountName=data['treasuryAccountName'],
        strategies=strategies,
    )

class PositionData:
    def __init__(self, positions, trades, accounts, trades_by_account, combined_account, exposures):
        self.positions = positions
        self.trades = trades
        self.accounts = accounts
        self.trades_by_account = trades_by_account
        self.combined_account = combined_account
        self.exposures = exposures



from common import AccountInfo, Trade, Position, TradeOrder

class TradeExecutor(ABC):
    
    @abstractmethod
    def place_futures_order(self, trade: TradeOrder) -> Trade:
        pass

class ExchangeDataProvider(ABC):
    
    @abstractmethod
    def get_id(self) -> str:
        pass
    
    @abstractmethod
    def get_futures_positions(self) -> List[Position]:
        pass
    
    @abstractmethod
    def get_futures_account(self) -> AccountInfo:
        pass
    
    @abstractmethod
    def get_futures_account_transaction_history(self, from_date: datetime) -> List[Trade]:
        pass
    
    @abstractmethod
    def get_live_prices(self, trade_assets: List[str]=None, contra_assets:List[str]=None) -> dict:
        pass
    
    @abstractmethod
    def get_trade_executor(self) -> TradeExecutor:
        pass
    
    