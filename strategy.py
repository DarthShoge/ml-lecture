from abc import ABC, abstractmethod, abstractproperty
from typing import List, Tuple
from numpy import sign
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, ppo
from ta.trend import PSARIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands
from common import RunOptions

class Factor(ABC):
    @abstractmethod
    async def get_signal(self, inputs : RunOptions, price_data: pd.Series) -> Tuple[ pd.Series, pd.Series]:
        """Calculates single factor using inputs and price data 

        Args:
            inputs (RunOptions): contains inputs to run signal caluclations such as instrument and timeframe.
            price_data (pd.DataFrame): closing prices for corresponding instrument in run options

        Returns:
            pd.Series: single series from the range of {-1,1} where -1 is 100% short and 1 is 100% long. any values greater than this range represent leveraged trade e.g 2 signal would represent 2x levered position
        """        
        pass
    
    @abstractproperty
    def name(self):
        """Name of the strategy. Should include any parameters

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()
    

def long_short_label(long_value: float, short_value: float):
    long_label = 'LNG' if long_value > 0 or short_value > 0 else None
    short_label = 'SHORT' if short_value < 0 or long_value < 0 else None
    only_label = None if long_label and short_label else 'ONLY'
    labels = [long_label, short_label, only_label]
    return '_'.join([x for x in labels if x is not None])

class MovingAverageCrossFactor(Factor):

    def __init__(self, fast_window: int, slow_window: int, upper_cross_pos: float = 1, lower_cross_pos: float = 0) -> None:
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.upper_cross_pos = upper_cross_pos
        self.lower_cross_pos = lower_cross_pos
        
    def name(self) -> str:
        ls_label = long_short_label(self.upper_cross_pos, self.lower_cross_pos) 
        return f'{type(self).__name__}-SLW-{self.slow_window}-FST-{self.fast_window}-{ls_label}'

    async def get_signal(self, price_data: pd.Series) -> pd.Series:
        price_ser = price_data
        fast_avg_ser: pd.Series = price_ser.rolling(self.fast_window).mean()
        fast_avg_ser.name = str(self.fast_window) + ' MA'
        slow_avg_ser = price_ser.rolling(self.slow_window).mean()
        slow_avg_ser.name = str(self.slow_window) + ' MA'
        all_data = pd.concat([price_ser, fast_avg_ser, slow_avg_ser], axis=1)
        title = f'{self.fast_window}/{self.slow_window} Cross'
        all_data[title] = (fast_avg_ser > slow_avg_ser).apply(
            lambda x: self.upper_cross_pos if x else self.lower_cross_pos)
        return all_data[title]
    

class EWMACrossFactor(Factor):

    def __init__(self, fast_window: int, slow_window: int, upper_cross_pos: float = 1, lower_cross_pos: float = 0) -> None:
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.upper_cross_pos = upper_cross_pos
        self.lower_cross_pos = lower_cross_pos
        
    def name(self):
        ls_label = long_short_label(self.upper_cross_pos, self.lower_cross_pos)
        return f'{type(self).__name__}-SLW-{self.slow_window}-FST-{self.fast_window}-{ls_label}'

    async def get_signal(self, price_data: pd.Series) -> pd.Series:
        price_ser = price_data
        fast_avg_ser: pd.Series = price_ser.ewm(self.fast_window).mean()
        fast_avg_ser.name = str(self.fast_window) + ' MA'
        slow_avg_ser = price_ser.ewm(self.slow_window).mean()
        slow_avg_ser.name = str(self.slow_window) + ' MA'
        all_data = pd.concat([price_ser, fast_avg_ser, slow_avg_ser], axis=1)
        title = f'{self.fast_window}/{self.slow_window} Cross'
        all_data[title] = (fast_avg_ser > slow_avg_ser).apply(
            lambda x: self.upper_cross_pos if x else self.lower_cross_pos)
        return all_data[title].fillna(0)


class RsiFactor(Factor):

    def __init__(self, window: int = 14, ovr_bought_lvl: float = 80.0, ovr_sold_lvl: float = 40.0, ovr_bought_pos: float = 1, ovr_sold_pos: float = 0) -> None:
        super().__init__()
        self.window = window
        self.ovr_bought_lvl = ovr_bought_lvl
        self.ovr_sold_lvl = ovr_sold_lvl
        self.ovr_bought_pos = ovr_bought_pos
        self.ovr_sold_pos = ovr_sold_pos
    
    def name(self):
        ls_label = long_short_label(self.ovr_bought_pos, self.ovr_sold_pos) 
        mode = 'MREV' if self.ovr_sold_pos > self.ovr_bought_pos else 'MOM'
        return f'{type(self).__name__}-OB-{self.ovr_bought_lvl}-OS-{self.ovr_sold_lvl}-{mode}-{ls_label}'

    async def get_signal(self, price_df: pd.Series) -> pd.Series:
        rsi_indicator = RSIIndicator(
            close=price_df, window=self.window)
        rsi = rsi_indicator.rsi()
        is_buy = rsi < self.ovr_sold_lvl
        is_buy.name = 'oversold'

        is_sell = rsi > self.ovr_bought_lvl
        is_sell.name = 'overbought'

        rsi_df = pd.concat([rsi, is_buy, is_sell], axis=1)
        signal = rsi_df.apply(
            lambda x: self.ovr_sold_pos if x['oversold'] else self.ovr_bought_pos if x['overbought'] else pd.NA, axis=1).ffill().fillna(0)
        return signal.fillna(0)



class BollingerBandFactor(Factor):
    def __init__(self, window: float = 20, deviation: float = 2, low_pos: float = -1, hi_pos: float = 1, hold_pos: float = pd.NA) -> None:
        super().__init__()
        self.window = window
        self.deviation = deviation
        self.low_pos = low_pos
        self.hi_pos = hi_pos
        self.hold_pos = hold_pos
        
    def name(self):
        ls_label = long_short_label(self.hi_pos, self.low_pos)
        return f'BollingerFactor-DEV-{self.deviation}-WND-{self.window}-{ls_label}'

    async def get_signal(self, price_data: pd.Series) -> pd.Series:
        bollinger_vals = BollingerBands(
            price_data, self.window, self.deviation)
        bollinger_df = pd.DataFrame({
            'bb_hband': bollinger_vals.bollinger_hband(),
            'bb_lband': bollinger_vals.bollinger_lband(),
            'bb_lband_indicator': bollinger_vals.bollinger_lband_indicator(),
            'bb_hband_indicator': bollinger_vals.bollinger_hband_indicator(),
            'price_data': price_data,
            'bb_mavg': bollinger_vals.bollinger_mavg()
        })
        signal = bollinger_df.apply(
            lambda x: self.low_pos if x['bb_lband'] else self.hi_pos if x['bb_hband'] else self.hold_pos, axis=1).ffill()
        signal.name = 'bband_signal'
        return signal.fillna(0)


class MACDDifferenceFactor(Factor):
    def __init__(self, slow_window: int = 26, fast_window: int = 12, signal_window: int = 9, rescale_period: int = 10, is_binary: bool = False, long_signal: int = 1,short_signal : int=-1) -> None:
        super().__init__()
        self.slow_win = slow_window
        self.fast_win = fast_window
        self.sign_win = signal_window
        self.rescale_period = rescale_period
        self.is_binary = is_binary
        self.short_sig = short_signal
        self.long_sig = long_signal
        
    def name(self):
        ls_label = long_short_label(self.long_sig, self.short_sig) 
        return f'{type(self).__name__}-SLW-{self.slow_win}-FST-{self.fast_win}-{ls_label}'

    async def get_signal(self, price_data: pd.Series) -> pd.Series:
        macd_vals = MACD(price_data,
                         self.slow_win, self.fast_win, self.sign_win)
        macd_df = pd.DataFrame({
            f'MACD_{self.fast_win}_{self.slow_win}': macd_vals.macd(),
            f'MACD_signal_{self.fast_win}_{self.slow_win}': macd_vals.macd_signal(),
            'price_data': price_data,
            f'MACD_diff_{self.fast_win}_{self.slow_win}': macd_vals.macd_diff()
        })
        diff_df = macd_df[f'MACD_diff_{self.fast_win}_{self.slow_win}']
        rescaled_signal = diff_df.rolling(self.rescale_period).apply(
            lambda x: x[0] / x.abs().max()) if not self.is_binary else diff_df.apply(lambda x: self.long_sig if x > 0 else self.short_sig)
        rescaled_signal.name = 'MACD_Signal'
        return rescaled_signal.fillna(0)
