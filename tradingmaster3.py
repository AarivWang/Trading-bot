#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web3 Crypto 多维度信号融合轮动策略（收益最大化版）
合规约束：仅现货/无杠杆/无做空/低频交易
核心：链上+舆情+技术+波动率 四维信号共振
"""
import os
import time
import json
import hmac
import hashlib
from decimal import Decimal, ROUND_DOWN, getcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm

# 环境配置
load_dotenv()
getcontext().prec = 28

# API配置
ROOSTOO_BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://api.roostoo.com")
ROOSTOO_API_KEY = os.getenv("ROOSTOO_API_KEY")
ROOSTOO_SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY")
HORUS_API_KEY = os.getenv("HORUS_API_KEY")
HORUS_BASE_URL = "https://api.horusdata.xyz/v1"

# 策略参数（可根据回测优化）
INITIAL_CAPITAL = Decimal("10000.0")  # 初始资金
MAX_POSITIONS = 8  # 最大持仓数
OPEN_THRESHOLD = 80  # 开仓信号得分阈值
CLOSE_THRESHOLD = 60  # 平仓信号得分阈值
COOLDOWN_SEC = 180  # 单标的冷却期（秒）
MAX_DAILY_TRADES = 10  # 单日最大交易次数
MAX_DRAWNDOWN = Decimal("0.15")  # 最大回撤熔断
WARNING_DRAWNDOWN = Decimal("0.12")  # 预警回撤（减持）
MIN_HOLD_SEC = 300  # 最短持仓时间（秒）
FIXED_TAKE_PROFIT = Decimal("0.15")  # 基础止盈比例

# 日志配置
TRADES_LOG = "max_profit_trades.csv"
EQUITY_LOG = "max_profit_equity.csv"

# 会话初始化
SESSION = requests.Session()

# ---------------- 数据类定义 ----------------
@dataclass
class PairMeta:
    """交易对元信息"""
    price_prec: int  # 价格精度
    amt_prec: int    # 数量精度
    mini_order: Decimal  # 最小订单金额
    liquidity_rank: int  # 流动性排名（1-50）

@dataclass
class Position:
    """持仓信息"""
    pair: str
    entry_price: Decimal
    quantity: Decimal
    entry_ts: float
    current_price: Decimal
    max_price: Decimal  # 持仓期间最高价
    signal_score: float  # 开仓时信号得分
    target_position: Decimal  # 目标仓位（USD）

@dataclass
class SignalScores:
    """信号得分"""
    on_chain: float  # 链上信号分（0-25）
    sentiment: float  # 舆情信号分（0-35）
    technical: float  # 技术信号分（0-40）
    volatility: float  # 波动率信号分（0-10）
    total: float  # 总分（0-110）
    confidence: float  # 信号置信度（0-2）

# ---------------- API工具函数 ----------------
def _now_ms() -> int:
    """获取毫秒级时间戳"""
    return int(time.time() * 1000)

def _sign_payload(payload: Dict) -> Tuple[Dict, str]:
    """生成Roostoo API签名"""
    payload["timestamp"] = _now_ms()
    # 按ASCII升序排序参数
    sorted_keys = sorted(payload.keys())
    params_str = "&".join(f"{k}={v}" for k, v in payload.items() if k in sorted_keys)
    # HMAC SHA256签名
    signature = hmac.new(
        ROOSTOO_SECRET_KEY.encode("utf-8"),
        params_str.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()
    headers = {
        "RST-API-KEY": ROOSTOO_API_KEY,
        "MSG-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    return headers, params_str

def roostoo_api_balance() -> Dict:
    """获取Roostoo账户余额"""
    url = f"{ROOSTOO_BASE_URL}/v3/balance"
    headers, params = _sign_payload({})
    response = SESSION.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def roostoo_api_exchange_info() -> Dict:
    """获取交易对元信息"""
    url = f"{ROOSTOO_BASE_URL}/v3/exchangeInfo"
    response = SESSION.get(url, timeout=15)
    response.raise_for_status()
    return response.json()

def roostoo_api_ticker(pair: Optional[str] = None) -> Dict:
    """获取行情数据"""
    url = f"{ROOSTOO_BASE_URL}/v3/ticker"
    params = {"timestamp": _now_ms()}
    if pair:
        params["pair"] = pair
    headers = {"RST-API-KEY": ROOSTOO_API_KEY}
    response = SESSION.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def roostoo_api_place_order(pair: str, side: str, quantity: str, price: Optional[str] = None) -> Dict:
    """下单（Limit订单优先）"""
    url = f"{ROOSTOO_BASE_URL}/v3/place_order"
    payload = {
        "pair": pair,
        "side": side.upper(),
        "type": "LIMIT" if price else "MARKET",
        "quantity": quantity
    }
    if price:
        payload["price"] = price
    headers, data = _sign_payload(payload)
    response = SESSION.post(url, headers=headers, data=data, timeout=15)
    response.raise_for_status()
    return response.json()

def horus_api_on_chain(pair: str) -> Dict:
    """获取Horus链上数据"""
    url = f"{HORUS_BASE_URL}/on-chain"
    headers = {"X-API-KEY": HORUS_API_KEY}
    params = {"pair": pair, "period": "7d"}
    response = SESSION.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def horus_api_sentiment(pair: str) -> Dict:
    """获取Horus舆情数据"""
    url = f"{HORUS_BASE_URL}/sentiment"
    headers = {"X-API-KEY": HORUS_API_KEY}
    params = {"pair": pair, "period": "24h"}
    response = SESSION.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

def horus_api_volatility(pair: str) -> Dict:
    """获取Horus波动率数据"""
    url = f"{HORUS_BASE_URL}/volatility"
    headers = {"X-API-KEY": HORUS_API_KEY}
    params = {"pair": pair, "period": "90d"}
    response = SESSION.get(url, headers=headers, params=params, timeout=15)
    response.raise_for_status()
    return response.json()

# ---------------- 策略工具函数 ----------------
def calculate_ema(prices: List[Decimal], period: int) -> Decimal:
    """计算EMA"""
    if len(prices) < period:
        return prices[-1] if prices else Decimal("0")
    alpha = Decimal("2") / (Decimal(period) + Decimal("1"))
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (Decimal("1") - alpha) * ema
    return ema

def calculate_rsi(prices: List[Decimal], period: int = 14) -> Decimal:
    """计算RSI"""
    if len(prices) < period + 1:
        return Decimal("50")
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d for d in deltas if d > 0]
    losses = [-d for d in deltas if d < 0]
    avg_gain = sum(gains[:period]) / Decimal(period) if gains else Decimal("0")
    avg_loss = sum(losses[:period]) / Decimal(period) if losses else Decimal("0")
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))
    return rsi

def get_pair_meta() -> Dict[str, PairMeta]:
    """获取交易对元信息（过滤流动性Top50的*/USD）"""
    info = roostoo_api_exchange_info()
    pairs_meta = {}
    # 先获取所有可交易的*/USD对
    usd_pairs = []
    for pair_name, pair_data in info.get("TradePairs", {}).items():
        if pair_name.endswith("/USD") and pair_data.get("CanTrade", True):
            usd_pairs.append((
                pair_name,
                pair_data.get("PricePrecision", 2),
                pair_data.get("AmountPrecision", 6),
                Decimal(str(pair_data.get("MiniOrder", "1.0"))),
                pair_data.get("LiquidityRank", 999)
            ))
    # 按流动性排序，取Top50
    usd_pairs.sort(key=lambda x: x[4])
    top50_pairs = usd_pairs[:50]
    # 构建PairMeta字典
    for name, price_prec, amt_prec, mini_order, liq_rank in top50_pairs:
        pairs_meta[name] = PairMeta(
            price_prec=price_prec,
            amt_prec=amt_prec,
            mini_order=mini_order,
            liquidity_rank=liq_rank
        )
    return pairs_meta

def calculate_signal_scores(pair: str, price_history: List[Decimal]) -> SignalScores:
    """计算标的信号得分"""
    # 1. 链上信号分（0-25）
    on_chain_score = 0.0
    try:
        on_chain_data = horus_api_on_chain(pair)
        # 大额净流入（交易所→个人钱包占比≥60%）
        large_transfer_ratio = on_chain_data.get("largeTransferRatio", 0.0)
        # 持币地址7天增长≥15%
        holder_growth = on_chain_data.get("holderGrowth7d", 0.0)
        # TVL 24h增长≥10%（仅DeFi代币）
        tvl_growth = on_chain_data.get("tvlGrowth24h", 0.0)
        if large_transfer_ratio >= 60 or holder_growth >= 15 or tvl_growth >= 10:
            on_chain_score = 25.0
    except Exception as e:
        print(f"[链上信号] {pair} 数据获取失败：{e}")
    
    # 2. 舆情信号分（0-35）
    sentiment_score = 0.0
    try:
        sentiment_data = horus_api_sentiment(pair)
        # 利好情绪占比≥70%，置信度≥0.8
        bullish_ratio = sentiment_data.get("bullishRatio", 0.0)
        confidence = sentiment_data.get("confidence", 0.0)
        if bullish_ratio >= 70 and confidence >= 0.8:
            sentiment_score += 25.0
        # 利好关键词热度环比增长≥100%
        keyword_growth = sentiment_data.get("keywordGrowth", 0.0)
        if keyword_growth >= 100:
            sentiment_score += 10.0
        sentiment_score = min(sentiment_score, 35.0)
    except Exception as e:
        print(f"[舆情信号] {pair} 数据获取失败：{e}")
    
    # 3. 技术信号分（0-40）
    technical_score = 0.0
    if len(price_history) >= 120:  # 至少120个价格点（满足最长EMA周期）
        # EMA多头排列（EMA20>EMA60>EMA120）
        ema20 = calculate_ema(price_history, 20)
        ema60 = calculate_ema(price_history, 60)
        ema120 = calculate_ema(price_history, 120)
        if ema20 > ema60 > ema120:
            technical_score += 20.0
        # 突破30日高点，量能≥20日均量1.5倍
        price_30d_high = max(price_history[-30:])
        current_price = price_history[-1]
        if current_price >= price_30d_high * Decimal("0.995"):  允许0.5%误差
            volume_ratio = 1.6  # 假设量能达标（实际需从API获取）
            if volume_ratio >= 1.5:
                technical_score += 15.0
        # RSI<70
        rsi = calculate_rsi(price_history)
        if rsi < 70:
            technical_score += 5.0
    
    # 4. 波动率信号分（0-10）
    volatility_score = 0.0
    try:
        vol_data = horus_api_volatility(pair)
        current_vol = vol_data.get("volatility24h", 0.0)
        vol_90d = vol_data.get("volatility90d", [])
        if vol_90d:
            vol_percentile = percentileofscore(vol_90d, current_vol)
            # 波动率分位30%-70%
            if 30 <= vol_percentile <= 70:
                volatility_score = 10.0
    except Exception as e:
        print(f"[波动率信号] {pair} 数据获取失败：{e}")
    
    # 信号置信度（链上置信度+舆情置信度，0-2）
    confidence = (on_chain_score / 25) + (sentiment_score / 35)
    
    return SignalScores(
        on_chain=on_chain_score,
        sentiment=sentiment_score,
        technical=technical_score,
        volatility=volatility_score,
        total=on_chain_score + sentiment_score + technical_score + volatility_score,
        confidence=confidence
    )

def calculate_position_size(pair: str, signal_scores: SignalScores, available_usd: Decimal, pair_meta: PairMeta) -> Tuple[Optional[str], Optional[str]]:
    """计算下单数量和价格（Limit订单）"""
    # 获取当前行情
    ticker = roostoo_api_ticker(pair)
    current_price = Decimal(str(ticker.get("Data", {}).get(pair, {}).get("LastPrice", 0)))
    if current_price <= 0:
        return None, None
    
    # 基础仓位（初始资金/最大持仓数）
    base_position = INITIAL_CAPITAL / Decimal(str(MAX_POSITIONS))
    # 仓位调整因子（信号得分/110 × (1 - 波动率分位/100)）
    try:
        vol_data = horus_api_volatility(pair)
        current_vol = vol_data.get("volatility24h", 0.0)
        vol_90d = vol_data.get("volatility90d", [])
        vol_percentile = percentileofscore(vol_90d, current_vol) if vol_90d else 50
    except:
        vol_percentile = 50
    adjust_factor = (signal_scores.total / 110) * (1 - vol_percentile / 100)
    adjust_factor = max(0.1, min(0.5, adjust_factor))  # 限制调整因子0.1-0.5
    
    # 信号置信度因子（0-2）
    confidence_factor = min(2.0, signal_scores.confidence)
    
    # 目标仓位
    target_position = base_position * Decimal(str(adjust_factor)) * Decimal(str(confidence_factor))
    target_position = min(target_position, available_usd * Decimal("0.3"))  # 单标的最大30%可用资金
    target_position = max(target_position, INITIAL_CAPITAL * Decimal("0.05"))  # 最小5%初始资金
    
    # 计算下单数量（Limit订单，挂买一价-0.1%，提高Maker概率）
    bid_price = current_price * Decimal("0.999")
    bid_price_str = f"{bid_price:.{pair_meta.price_prec}f}"
    quantity = (target_position / bid_price).quantize(
        Decimal("10") ** (-pair_meta.amt_prec),
        rounding=ROUND_DOWN
    )
    # 检查最小订单限制
    order_amount = quantity * bid_price
    if order_amount < pair_meta.mini_order:
        quantity = (pair_meta.mini_order / bid_price).quantize(
            Decimal("10") ** (-pair_meta.amt_prec),
            rounding=ROUND_DOWN
        )
        # 确保满足最小订单金额
        if quantity * bid_price < pair_meta.mini_order:
            quantity += Decimal("10") ** (-pair_meta.amt_prec)
        if quantity * bid_price > available_usd:
            return None, None
    
    quantity_str = f"{quantity:.{pair_meta.amt_prec}f}"
    return quantity_str, bid_price_str

def check_exit_conditions(position: Position, current_scores: SignalScores, price_history: List[Decimal]) -> Tuple[bool, str]:
    """检查退出条件"""
    current_price = price_history[-1]
    position.current_price = current_price
    # 更新持仓期间最高价
    if current_price > position.max_price:
        position.max_price = current_price
    
    # 1. 固定止盈（基础止盈或信号得分对应的止盈）
    score_based_tp = Decimal(str(position.signal_score * 0.001))
    take_profit = max(FIXED_TAKE_PROFIT, score_based_tp)
    profit_rate = (current_price - position.entry_price) / position.entry_price
    if profit_rate >= take_profit:
        return True, f"止盈触发（收益{profit_rate*100:.2f}%≥目标{take_profit*100:.2f}%）"
    
    # 2. 移动止损
    stop_loss_price = Decimal("0")
    if profit_rate < Decimal("0.05"):
        # 收益<5%：固定止损5%
        stop_loss_price = position.entry_price * Decimal("0.95")
    elif profit_rate < Decimal("0.10"):
        # 收益5%-10%：回撤50%盈利
        stop_loss_price = position.entry_price + (current_price - position.entry_price) * Decimal("0.5")
    else:
        # 收益≥10%：追踪止损10%
        stop_loss_price = position.max_price * Decimal("0.90")
    if current_price <= stop_loss_price:
        return True, f"止损触发（价格{current_price:.2f}≤止损价{stop_loss_price:.2f}）"
    
    # 3. 信号反转（得分<平仓阈值）
    if current_scores.total < CLOSE_THRESHOLD:
        return True, f"信号反转（得分{current_scores.total:.1f}<阈值{CLOSE_THRESHOLD}）"
    
    # 4. 持仓时间不足
    if time.time() - position.entry_ts < MIN_HOLD_SEC:
        return False, "持仓时间不足"
    
    return False, "无退出条件"

# ---------------- 策略主类 ----------------
class MaxProfitStrategy:
    def __init__(self):
        self.pair_meta = get_pair_meta()  # 交易对元信息
        self.positions: Dict[str, Position] = {}  # 当前持仓
        self.price_history: Dict[str, List[Decimal]] = {pair: [] for pair in self.pair_meta.keys()}  # 价格历史
        self.daily_trades = 0  # 当日交易次数
        self.last_trade_date = time.strftime("%Y-%m-%d")  # 最后交易日期
        self.equity_peak = INITIAL_CAPITAL  # 净值峰值
        self.available_usd = INITIAL_CAPITAL  # 可用USD
        
        # 初始化日志
        self._init_logs()
    
    def _init_logs(self):
        """初始化日志文件"""
        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG, "w", encoding="utf-8") as f:
                f.write("ts,pair,side,quantity,price,signal_score,reason,success,error\n")
        if not os.path.exists(EQUITY_LOG):
            with open(EQUITY_LOG, "w", encoding="utf-8") as f:
                f.write("ts,equity,available_usd,hold_value,max_drawdown\n")
    
    def _update_price_history(self):
        """更新所有标的价格历史"""
        ticker_data = roostoo_api_ticker()
        for pair in self.pair_meta.keys():
            price = ticker_data.get("Data", {}).get(pair, {}).get("LastPrice")
            if price:
                self.price_history[pair].append(Decimal(str(price)))
                # 保留最近200个价格点（足够计算所有技术指标）
                if len(self.price_history[pair]) > 200:
                    self.price_history[pair].pop(0)
    
    def _calculate_equity(self) -> Tuple[Decimal, Decimal, Decimal]:
        """计算组合净值、持仓价值、最大回撤"""
        # 获取最新余额
        balance_data = roostoo_api_balance()
        usd_free = Decimal(str(balance_data.get("Wallet", {}).get("USD", {}).get("Free", 0)))
        self.available_usd = usd_free
        
        # 计算持仓价值
        hold_value = Decimal("0")
        for pair, pos in self.positions.items():
            ticker = roostoo_api_ticker(pair)
            current_price = Decimal(str(ticker.get("Data", {}).get(pair, {}).get("LastPrice", 0)))
            hold_value += pos.quantity * current_price
            pos.current_price = current_price
            # 更新最高价
            if current_price > pos.max_price:
                pos.max_price = current_price
        
        # 组合净值
        equity = usd_free + hold_value
        # 更新净值峰值
        if equity > self.equity_peak:
            self.equity_peak = equity
        # 最大回撤
        max_drawdown = (equity - self.equity_peak) / self.equity_peak if self.equity_peak > 0 else Decimal("0")
        
        # 记录净值日志
        with open(EQUITY_LOG, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())},{equity},{usd_free},{hold_value},{max_drawdown}\n")
        
        return equity, hold_value, max_drawdown
    
    def _reset_daily_trades(self):
        """重置当日交易次数"""
        current_date = time.strftime("%Y-%m-%d")
        if current_date != self.last_trade_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
    
    def _log_trade(self, pair: str, side: str, quantity: str, price: str, signal_score: float, reason: str, success: bool, error: str = ""):
        """记录交易日志"""
        with open(TRADES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())},{pair},{side},{quantity},{price},{signal_score},{reason},{int(success)},{error}\n")
    
    def _execute_buy(self, pair: str, signal_scores: SignalScores):
        """执行买入"""
        # 检查交易频率
        if self.daily_trades >= MAX_DAILY_TRADES:
            print(f"[买入] {pair} 当日交易次数已达上限（{MAX_DAILY_TRADES}笔），跳过")
            return
        # 检查冷却期
        if pair in self.positions:
            print(f"[买入] {pair} 已有持仓，跳过")
            return
        last_trade_ts = self._get_last_trade_ts(pair)
        if last_trade_ts and time.time() - last_trade_ts < COOLDOWN_SEC:
            print(f"[买入] {pair} 冷却期未到（剩余{COOLDOWN_SEC - (time.time() - last_trade_ts):.0f}秒），跳过")
            return
        # 检查持仓数
        if len(self.positions) >= MAX_POSITIONS:
            print(f"[买入] 持仓数已达上限（{MAX_POSITIONS}个），跳过")
            return
        
        # 计算仓位
        pair_meta = self.pair_meta[pair]
        quantity_str, price_str = calculate_position_size(pair, signal_scores, self.available_usd, pair_meta)
        if not quantity_str or not price_str:
            print(f"[买入] {pair} 计算仓位失败，跳过")
            return
        
        # 执行下单
        try:
            print(f"[买入] {pair} 信号得分{signal_scores.total:.1f}，下单：{quantity_str} @ {price_str} USD")
            response = roostoo_api_place_order(pair, "BUY", quantity_str, price_str)
            success = response.get("Success", False)
            if success:
                # 记录持仓
                entry_price = Decimal(price_str)
                self.positions[pair] = Position(
                    pair=pair,
                    entry_price=entry_price,
                    quantity=Decimal(quantity_str),
                    entry_ts=time.time(),
                    current_price=entry_price,
                    max_price=entry_price,
                    signal_score=signal_scores.total,
                    target_position=Decimal(quantity_str) * entry_price
                )
                self.daily_trades += 1
                self._log_trade(pair, "BUY", quantity_str, price_str, signal_scores.total, "信号共振开仓", True)
                print(f"[买入] {pair} 成功，持仓数量：{quantity_str}")
            else:
                error = response.get("ErrMsg", "未知错误")
                self._log_trade(pair, "BUY", quantity_str, price_str, signal_scores.total, "信号共振开仓", False, error)
                print(f"[买入] {pair} 失败：{error}")
        except Exception as e:
            self._log_trade(pair, "BUY", quantity_str, price_str, signal_scores.total, "信号共振开仓", False, str(e))
            print(f"[买入] {pair} 异常：{e}")
    
    def _execute_sell(self, pair: str, position: Position, reason: str):
        """执行卖出"""
        # 检查最短持仓时间
        if time.time() - position.entry_ts < MIN_HOLD_SEC:
            print(f"[卖出] {pair} 持仓时间不足（{MIN_HOLD_SEC}秒），跳过")
            return
        # 检查交易频率
        if self.daily_trades >= MAX_DAILY_TRADES:
            print(f"[卖出] {pair} 当日交易次数已达上限（{MAX_DAILY_TRADES}笔），跳过")
            return
        
        # 计算卖出价格（Limit订单，挂卖一价+0.1%）
        ticker = roostoo_api_ticker(pair)
        current_price = Decimal(str(ticker.get("Data", {}).get(pair, {}).get("LastPrice", 0)))
        ask_price = current_price * Decimal("1.001")
        pair_meta = self.pair_meta[pair]
        ask_price_str = f"{ask_price:.{pair_meta.price_prec}f}"
        quantity_str = f"{position.quantity:.{pair_meta.amt_prec}f}"
        
        # 执行下单
        try:
            print(f"[卖出] {pair} 理由：{reason}，下单：{quantity_str} @ {ask_price_str} USD")
            response = roostoo_api_place_order(pair, "SELL", quantity_str, ask_price_str)
            success = response.get("Success", False)
            if success:
                # 移除持仓
                del self.positions[pair]
                self.daily_trades += 1
                # 计算实际收益
                fill_price = Decimal(str(response.get("OrderDetail", {}).get("FilledAverPrice", ask_price_str)))
                profit = (fill_price - position.entry_price) * position.quantity
                profit_rate = (fill_price - position.entry_price) / position.entry_price
                self._log_trade(pair, "SELL", quantity_str, str(fill_price), position.signal_score, reason, True)
                print(f"[卖出] {pair} 成功，收益：{profit:.2f} USD（{profit_rate*100:.2f}%）")
            else:
                error = response.get("ErrMsg", "未知错误")
                self._log_trade(pair, "SELL", quantity_str, ask_price_str, position.signal_score, reason, False, error)
                print(f"[卖出] {pair} 失败：{error}")
        except Exception as e:
            self._log_trade(pair, "SELL", quantity_str, ask_price_str, position.signal_score, reason, False, str(e))
            print(f"[卖出] {pair} 异常：{e}")
    
    def _get_last_trade_ts(self, pair: str) -> Optional[float]:
        """获取标的最后交易时间戳"""
        try:
            df = pd.read_csv(TRADES_LOG)
            pair_trades = df[df["pair"] == pair].sort_values("ts", ascending=False)
            if not pair_trades.empty:
                return float(pair_trades.iloc[0]["ts"])
        except:
            pass
        return None
    
    def _check_portfolio_risk(self) -> bool:
        """检查组合风险（熔断机制）"""
        _, _, max_drawdown = self._calculate_equity()
        if max_drawdown <= -MAX_DRAWNDOWN:
            # 触发熔断，清仓所有持仓
            print(f"[风险熔断] 最大回撤{max_drawdown*100:.2f}%≥{MAX_DRAWNDOWN*100:.2f}%，清仓所有持仓")
            for pair, pos in list(self.positions.items()):
                self._execute_sell(pair, pos, f"熔断清仓（回撤{max_drawdown*100:.2f}%）")
            return True
        elif max_drawdown <= -WARNING_DRAWNDOWN:
            # 触发预警，减持50%仓位
            print(f"[风险预警] 最大回撤{max_drawdown*100:.2f}%≥{WARNING_DRAWNDOWN*100:.2f}%，减持50%仓位")
            positions_to_reduce = list(self.positions.items())[:len(self.positions)//2]
            for pair, pos in positions_to_reduce:
                self._execute_sell(pair, pos, f"预警减持（回撤{max_drawdown*100:.2f}%）")
            return True
        return False
    
    def run(self):
        """策略主循环"""
        print("=== 多维度信号融合轮动策略启动 ===")
        print(f"初始资金：{INITIAL_CAPITAL} USD")
        print(f"最大持仓数：{MAX_POSITIONS}")
        print(f"开仓信号阈值：{OPEN_THRESHOLD}分")
        print(f"熔断最大回撤：{MAX_DRAWNDOWN*100:.2f}%")
        print("="*50)
        
        while True:
            try:
                # 1. 重置当日交易次数
                self._reset_daily_trades()
                
                # 2. 计算净值和风险检查
                equity, hold_value, max_drawdown = self._calculate_equity()
                print(f"\n[状态] 净值：{equity:.2f} USD | 可用资金：{self.available_usd:.2f} USD | 持仓价值：{hold_value:.2f} USD | 最大回撤：{max_drawdown*100:.2f}%")
                
                # 3. 风险熔断检查
                if self._check_portfolio_risk():
                    time.sleep(60)
                    continue
                
                # 4. 更新价格历史
                self._update_price_history()
                
                # 5. 处理现有持仓退出
                for pair, pos in list(self.positions.items()):
                    price_history = self.price_history[pair]
                    if len(price_history) < 30:
                        continue
                    # 计算当前信号得分
                    current_scores = calculate_signal_scores(pair, price_history)
                    # 检查退出条件
                    should_exit, reason = check_exit_conditions(pos, current_scores, price_history)
                    if should_exit:
                        self._execute_sell(pair, pos, reason)
                
                # 6. 寻找新的开仓机会
                if len(self.positions) < MAX_POSITIONS and self.available_usd >= INITIAL_CAPITAL * Decimal("0.05"):
                    # 遍历所有标的，计算信号得分
                    candidates = []
                    for pair in tqdm(self.pair_meta.keys(), desc="计算信号得分"):
                        price_history = self.price_history[pair]
                        if len(price_history) < 120:  # 至少120个价格点（满足EMA120）
                            continue
                        # 计算信号得分
                        scores = calculate_signal_scores(pair, price_history)
                        if scores.total >= OPEN_THRESHOLD:
                            candidates.append((pair, scores))
                    # 按信号得分降序排序
                    candidates.sort(key=lambda x: x[1].total, reverse=True)
                    # 执行买入（优先选择得分最高的标的）
                    for pair, scores in candidates:
                        if len(self.positions) >= MAX_POSITIONS or self.daily_trades >= MAX_DAILY_TRADES:
                            break
                        self._execute_buy(pair, scores)
                
                # 7. 轮询间隔（15分钟，避免高频）
                print(f"\n[等待] 下一轮轮询将在15分钟后执行")
                time.sleep(15 * 60)
                
            except Exception as e:
                print(f"[主循环异常] {e}")
                time.sleep(60)

# ---------------- 启动策略 ----------------
if __name__ == "__main__":
    # 检查API密钥
    if not ROOSTOO_API_KEY or not ROOSTOO_SECRET_KEY or not HORUS_API_KEY:
        raise ValueError("请配置环境变量：ROOSTOO_API_KEY、ROOSTOO_SECRET_KEY、HORUS_API_KEY")
    
    # 启动策略
    strategy = MaxProfitStrategy()
    strategy.run()
