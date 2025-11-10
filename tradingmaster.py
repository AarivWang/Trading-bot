
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roostoo OMEGA GrandMaster — 多资产「大模型」交易机器人（终极版）
================================================================
目标：在 Roostoo 模拟交易所，追求高收益/受控回撤。
核心：
  • 多资产动态选股（流动性+强度）
  • 多信号集成（EMA趋势 / 突破 / 动量 / 斜率 / 点差过滤）
  • 自适应资金管理（波动缩放 + 风险/仓位上限）
  • 在线学习分配（Softmax/UCBlike 交易后反馈）
  • 三重退出（止损/止盈/追踪）+ EMA 反转 + 排名跌出
  • 组合级风控（最大回撤阈值、杀死开关）
  • 状态持久化（断线复活）

依赖：
  pip install requests

环境变量：
  export ROOSTOO_API_KEY="你的API_KEY"
  export ROOSTOO_SECRET_KEY="你的SECRET_KEY"

快速启动（示例参数，可微调）：
  python roostoo_omega_grandmaster.py --interval 3 --universe-size 40 --topk 6 \
    --risk-per-trade 0.02 --budget-perc 0.9 --per-asset-cap 0.25 \
    --fast 12 --slow 26 --breakout 60 --roc 30 --slope 20 \
    --vol-floor 0.002 --min-stop 0.01 --take-mul 2.0 --trail-mul 1.5 \
    --entry-threshold 0.0 --min-unit-volume 20000000 --cooldown 20

仅观测（不下单）：
  python roostoo_omega_grandmaster.py --dry-run

⚠️ 免责声明：教育/比赛用示例，不构成投资建议。请自控风险。
"""
import os
import time
import json
import hmac
import hashlib
import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_DOWN, ROUND_UP, getcontext
from typing import Dict, Optional, Tuple, List

import requests

getcontext().prec = 28

BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.getenv("ROOSTOO_API_KEY", "ttdLE0ZRfWAZeaBoaq8SyEhH6QSQ8yr6laZzTgKNTTYTkogK1cUWdT9P6r59Jd3S")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "8mYHLPiVxrTDx8bTe5UUumVOmZ07DBAE1ZovuxgyJk7ny1oYrDECdWc1GbuMX1fN")

STATE_FILE = "omega_state.json"
TRADES_LOG = "omega_trades.csv"
EQUITY_LOG = "omega_equity.csv"

# ---------------- 签名与 API 封装 ----------------
SESSION = requests.Session()

def _now_ms_str() -> str:
    return str(int(time.time() * 1000))

def _ensure_str_vals(payload: Dict) -> Dict[str, str]:
    return {str(k): str(v) for k, v in payload.items()}

def _total_params_str(payload: Dict[str, str]) -> str:
    keys_sorted = sorted(payload.keys())
    return "&".join(f"{k}={payload[k]}" for k in keys_sorted)

def _hmac_sha256_hex(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

def sign_for_get(payload: Optional[Dict]=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    payload = {} if payload is None else dict(payload)
    payload["timestamp"] = _now_ms_str()
    payload = _ensure_str_vals(payload)
    total_params = _total_params_str(payload)
    signature = _hmac_sha256_hex(SECRET_KEY, total_params)
    headers = {"RST-API-KEY": API_KEY, "MSG-SIGNATURE": signature}
    return headers, payload

def sign_for_post(payload: Optional[Dict]=None) -> Tuple[Dict[str, str], str]:
    payload = {} if payload is None else dict(payload)
    payload["timestamp"] = _now_ms_str()
    payload = _ensure_str_vals(payload)
    total_params = _total_params_str(payload)
    signature = _hmac_sha256_hex(SECRET_KEY, total_params)
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RST-API-KEY": API_KEY,
        "MSG-SIGNATURE": signature
    }
    return headers, total_params

def api_server_time():
    url = f"{BASE_URL}/v3/serverTime"
    r = SESSION.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def api_exchange_info():
    url = f"{BASE_URL}/v3/exchangeInfo"
    r = SESSION.get(url, timeout=15)
    r.raise_for_status()
    return r.json()

def api_ticker_all():
    url = f"{BASE_URL}/v3/ticker"
    params = {"timestamp": _now_ms_str()}
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def api_balance():
    url = f"{BASE_URL}/v3/balance"
    headers, params = sign_for_get({})
    r = SESSION.get(url, headers=headers, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def api_place_market(pair: str, side: str, qty_str: str):
    url = f"{BASE_URL}/v3/place_order"
    payload = {"pair": pair, "side": side.upper(), "type": "MARKET", "quantity": qty_str}
    headers, body = sign_for_post(payload)
    r = SESSION.post(url, headers=headers, data=body, timeout=15)
    r.raise_for_status()
    return r.json()

# ---------------- 工具/数学 ----------------
def quantize_down(x: Decimal, prec: int) -> Decimal:
    q = Decimal(1).scaleb(-prec)
    return x.quantize(q, rounding=ROUND_DOWN)

class EMA:
    def __init__(self, n: int):
        self.n = n
        self.alpha = Decimal(2) / (Decimal(n) + Decimal(1))
        self.v: Optional[Decimal] = None
    def update(self, p: Decimal) -> Decimal:
        if self.v is None:
            self.v = p
        else:
            self.v = self.alpha * p + (Decimal(1) - self.alpha) * self.v
        return self.v

def stdev(xs: List[Decimal]) -> Decimal:
    n = len(xs)
    if n < 2:
        return Decimal("0")
    m = sum(xs) / Decimal(n)
    var = sum((x - m) * (x - m) for x in xs) / Decimal(n - 1)
    return (var if var > 0 else Decimal("0")).sqrt()

def linreg_slope(prices: List[Decimal]) -> Optional[Decimal]:
    n = len(prices)
    if n < 3:
        return None
    # y vs x=0..n-1 的最小二乘斜率
    x_sum = Decimal(n * (n - 1) / 2)
    x2_sum = Decimal((n - 1) * n * (2 * n - 1) / 6)
    y_sum = sum(prices)
    xy_sum = sum(Decimal(i) * prices[i] for i in range(n))
    denom = Decimal(n) * x2_sum - x_sum * x_sum
    if denom == 0:
        return None
    slope = (Decimal(n) * xy_sum - x_sum * y_sum) / denom
    return slope

# ---------------- 数据结构 ----------------
@dataclass
class PairMeta:
    price_prec: int
    amt_prec: int
    mini_order: Decimal

class SymState:
    def __init__(self, name: str, meta: PairMeta, fast: int, slow: int, breakout_n: int, roc_n: int, slope_n: int):
        self.name = name
        self.meta = meta
        self.fast = EMA(fast)
        self.slow = EMA(slow)
        self.breakout_n = breakout_n
        self.roc_n = roc_n
        self.slope_n = slope_n
        self.prices: deque = deque(maxlen=max(breakout_n, roc_n, slope_n, slow*3, 80))
        self.returns: deque = deque(maxlen=max(roc_n, 20))
        self.last_fast: Optional[Decimal] = None
        self.last_slow: Optional[Decimal] = None
        self.qty: Decimal = Decimal("0")
        self.entry_price: Optional[Decimal] = None
        self.peak_after_entry: Optional[Decimal] = None
        self.last_trade_ts: float = 0.0
        # 在线学习（交易后反馈）：均值回报 & 次数
        self.reward_mean: float = 0.0
        self.trades: int = 0

    def update(self, p: Decimal):
        if self.prices:
            prev = self.prices[-1]
            ret = (p - prev) / prev if prev != 0 else Decimal("0")
            self.returns.append(ret)
        self.prices.append(p)
        f = self.fast.update(p)
        s = self.slow.update(p)
        cu = (self.last_fast is not None and self.last_slow is not None and self.last_fast <= self.last_slow and f > s)
        cd = (self.last_fast is not None and self.last_slow is not None and self.last_fast >= self.last_slow and f < s)
        self.last_fast, self.last_slow = f, s
        return f, s, cu, cd

    def highest(self, n: int) -> Optional[Decimal]:
        if len(self.prices) < n:
            return None
        return max(self.prices[-n:])

    def roc(self, n: int) -> Optional[Decimal]:
        if len(self.prices) < n + 1:
            return None
        a = self.prices[-1]
        b = self.prices[-1-n]
        if b == 0:
            return None
        return (a - b) / b

    def vol(self) -> Decimal:
        if len(self.returns) < 5:
            return Decimal("0")
        return stdev(list(self.returns))

    def slope(self) -> Optional[Decimal]:
        if len(self.prices) < self.slope_n:
            return None
        return linreg_slope(list(self.prices)[-self.slope_n:])

# ---------------- 机器人主体 ----------------
class OmegaBot:
    def __init__(self, args):
        self.args = args
        self.meta_by_pair: Dict[str, PairMeta] = {}
        self.syms: Dict[str, SymState] = {}
        self.universe: List[str] = []
        self.equity_peak: Decimal = Decimal("0")
        self.kill_switch: bool = False

        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG, "w", encoding="utf-8") as f:
                f.write("ts,pair,side,qty,price,reason,ok,err\n")
        if not os.path.exists(EQUITY_LOG):
            with open(EQUITY_LOG, "w", encoding="utf-8") as f:
                f.write("ts,equity,usd_free,holding_value,dd\n")

        self._init_meta()
        self._load_state()

    # ---- 初始化交易对精度信息 ----
    def _init_meta(self):
        info = api_exchange_info()
        tps = info.get("TradePairs", {})
        for name, d in tps.items():
            if not d.get("CanTrade", True):
                continue
            meta = PairMeta(
                price_prec=int(d.get("PricePrecision", 2)),
                amt_prec=int(d.get("AmountPrecision", 6)),
                mini_order=Decimal(str(d.get("MiniOrder", "1.0")))
            )
            self.meta_by_pair[name] = meta

    # ---- 状态持久化 ----
    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.equity_peak = Decimal(s.get("equity_peak", "0"))
            # 恢复 per-symbol 持仓与学习统计
            for name, st in s.get("symbols", {}).items():
                if name in self.meta_by_pair:
                    ss = SymState(name, self.meta_by_pair[name],
                                  self.args.fast, self.args.slow, self.args.breakout, self.args.roc, self.args.slope)
                    ss.qty = Decimal(st.get("qty", "0"))
                    ep = st.get("entry_price", None)
                    ss.entry_price = Decimal(str(ep)) if ep is not None else None
                    ss.peak_after_entry = Decimal(str(st.get("peak_after_entry"))) if st.get("peak_after_entry") else None
                    ss.reward_mean = float(st.get("reward_mean", 0.0))
                    ss.trades = int(st.get("trades", 0))
                    self.syms[name] = ss
        except Exception as e:
            print(f"[STATE] 加载失败：{e}")

    def _save_state(self):
        try:
            symbols_dump = {}
            for name, ss in self.syms.items():
                symbols_dump[name] = {
                    "qty": str(ss.qty),
                    "entry_price": str(ss.entry_price) if ss.entry_price is not None else None,
                    "peak_after_entry": str(ss.peak_after_entry) if ss.peak_after_entry is not None else None,
                    "reward_mean": ss.reward_mean,
                    "trades": ss.trades
                }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump({"equity_peak": str(self.equity_peak), "symbols": symbols_dump}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[STATE] 保存失败：{e}")

    # ---- 组合估值与风控 ----
    def _portfolio_value(self, data: Dict) -> Tuple[Decimal, Decimal, Decimal]:
        usd = Decimal(str(api_balance().get("Wallet", {}).get("USD", {}).get("Free", "0")))
        hold_value = Decimal("0")
        for name, ss in self.syms.items():
            if ss.qty > 0:
                px = Decimal(str(data.get(name, {}).get("LastPrice", "0")))
                hold_value += ss.qty * px
        equity = usd + hold_value
        if equity > self.equity_peak:
            self.equity_peak = equity
        dd = Decimal("0")
        if self.equity_peak > 0:
            dd = (equity - self.equity_peak) / self.equity_peak
        # 记录
        with open(EQUITY_LOG, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())},{equity},{usd},{hold_value},{dd}\n")
        return equity, usd, hold_value

    # ---- 选股（流动性+强度） ----
    def _select_universe(self, data: Dict) -> List[str]:
        L = []
        for name, d in data.items():
            if name not in self.meta_by_pair:
                continue
            utv = Decimal(str(d.get("UnitTradeValue", "0")))  # 24h USD 成交额
            if utv < Decimal(self.args.min_unit_volume):
                continue
            lp = d.get("LastPrice")
            if lp is None:
                continue
            L.append((name, utv, Decimal(str(d.get("Change", 0)))))
        # 优先流动性，再兼顾涨幅
        L.sort(key=lambda x: (x[1], x[2]), reverse=True)
        universe = [x[0] for x in L[: self.args.universe_size]]
        return universe

    # ---- 评分（多信号集成 + 学习加权） ----
    def _score(self, ss: SymState, p: Decimal, d: Dict) -> float:
        # Spread 过滤：点差太大直接降权
        bid = d.get("MaxBid"); ask = d.get("MinAsk")
        spread_penalty = 0.0
        if bid is not None and ask is not None and (bid > 0 and ask > 0):
            mid = (Decimal(str(bid)) + Decimal(str(ask))) / Decimal(2)
            spr = (Decimal(str(ask)) - Decimal(str(bid))) / mid if mid > 0 else Decimal("0")
            spread_penalty = float(min(Decimal("0.05"), spr))  # 上限 5%

        # 组件：EMA gap、突破、动量、线性斜率、24h change
        if ss.last_fast is None or ss.last_slow is None:
            return -1e9
        ema_gap = (ss.last_fast - ss.last_slow) / (ss.last_slow if ss.last_slow != 0 else Decimal("1"))
        ema_gap_f = float(ema_gap)

        hi = ss.highest(self.args.breakout)
        breakout = 1.0 if (hi is not None and p >= hi) else 0.0

        roc_v = ss.roc(self.args.roc)
        roc_f = float(roc_v) if roc_v is not None else 0.0

        slope_v = ss.slope()
        slope_f = float(slope_v) if slope_v is not None else 0.0

        ch24 = max(0.0, float(Decimal(str(d.get("Change", 0)))))

        # 基础分
        base = 0.35*ema_gap_f + 0.25*roc_f + 0.2*breakout + 0.15*slope_f + 0.05*ch24 - 0.5*spread_penalty

        # 在线学习奖励：近期该标的平均单笔收益，温度缩放
        learn = ss.reward_mean
        score = base + 0.5 * learn  # 可调系数
        return score

    # ---- 下单数量 ----
    def _qty_from_budget(self, pair: str, usd_budget: Decimal, price: Decimal) -> Optional[str]:
        meta = self.meta_by_pair[pair]
        qty = quantize_down(usd_budget / price, meta.amt_prec)
        if qty <= 0:
            return None
        if qty * price <= meta.mini_order:
            need = quantize_down((meta.mini_order / price) * Decimal("1.001"), meta.amt_prec)
            if need * price > usd_budget:
                return None
            qty = need
        return f"{qty:.{meta.amt_prec}f}"

    # ---- 交易记录 ----
    def _log_trade(self, pair: str, side: str, qty: str, price: str, reason: str, ok: bool, err: str):
        with open(TRADES_LOG, "a", encoding="utf-8") as f:
            f.write(f"{int(time.time())},{pair},{side},{qty},{price},{reason},{int(ok)},{err}\n")

    # ---- 买入/卖出 ----
    def _enter_long(self, pair: str, ss: SymState, price: Decimal, usd_free: Decimal, vol_i: Decimal, now_ts: float, reason: str) -> Tuple[Decimal, bool]:
        if ss.qty > 0 or self.kill_switch:
            return usd_free, False
        if now_ts - ss.last_trade_ts < self.args.cooldown:
            return usd_free, False

        # 风险预算
        risk_usd = max(Decimal("1"), usd_free * Decimal(self.args.risk_per_trade))
        stop_dist = max(vol_i * Decimal(self.args.stop_mul), Decimal(self.args.min_stop))
        if stop_dist <= 0:
            stop_dist = Decimal("0.01")
        target_notional = risk_usd / stop_dist
        per_asset_cap = usd_free * Decimal(self.args.per_asset_cap)
        use_notional = min(target_notional, per_asset_cap, usd_free) * Decimal(self.args.budget_perc)
        if use_notional < Decimal("1"):
            return usd_free, False

        qty_str = self._qty_from_budget(pair, use_notional, price)
        if qty_str is None:
            return usd_free, False

        if self.args.dry_run:
            ss.qty = Decimal(qty_str)
            ss.entry_price = price
            ss.peak_after_entry = price
            ss.last_trade_ts = now_ts
            self._log_trade(pair, "BUY", qty_str, str(price), "[DRY] " + reason, True, "")
            print(f"[DRY BUY] {pair} qty={qty_str} price={price} reason={reason}")
            return usd_free - Decimal(qty_str)*price, True

        try:
            res = api_place_market(pair, "BUY", qty_str)
            ok = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail", {}).get("FilledAverPrice", price)))
            err = "" if ok else res.get("ErrMsg", "unknown")
            if ok:
                ss.qty = Decimal(str(res["OrderDetail"]["FilledQuantity"]))
                ss.entry_price = fill
                ss.peak_after_entry = fill
                ss.last_trade_ts = now_ts
                usd_free = max(Decimal("0"), usd_free - ss.qty * fill)
            self._log_trade(pair, "BUY", qty_str, str(fill), reason, ok, err)
            print(f"[BUY] {pair} qty={qty_str} fill={fill} ok={ok} err={err} reason={reason}")
            return usd_free, ok
        except requests.HTTPError as e:
            body = ""
            try: body = e.response.text
            except: pass
            self._log_trade(pair, "BUY", qty_str, str(price), reason, False, f"{e} {body}")
            print(f"[BUY ERR] {pair} {e} {body}")
            return usd_free, False

    def _exit_long(self, pair: str, ss: SymState, price: Decimal, now_ts: float, reason: str) -> bool:
        if ss.qty <= 0 or now_ts - ss.last_trade_ts < self.args.cooldown:
            return False
        qty = quantize_down(ss.qty, self.meta_by_pair[pair].amt_prec)
        if qty <= 0:
            return False
        qty_str = f"{qty:.{self.meta_by_pair[pair].amt_prec}f}"

        # 计算这笔的收益（用于在线学习）
        trade_ret = 0.0
        if ss.entry_price and ss.entry_price > 0:
            trade_ret = float((price - ss.entry_price) / ss.entry_price)

        if self.args.dry_run:
            self._log_trade(pair, "SELL", qty_str, str(price), "[DRY] " + reason, True, "")
            print(f"[DRY SELL] {pair} qty={qty_str} price={price} reason={reason}")
            self._update_learning(ss, trade_ret)
            ss.qty = Decimal("0"); ss.entry_price=None; ss.peak_after_entry=None; ss.last_trade_ts = now_ts
            return True

        try:
            res = api_place_market(pair, "SELL", qty_str)
            ok = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail", {}).get("FilledAverPrice", price)))
            self._log_trade(pair, "SELL", qty_str, str(fill), reason, ok, "" if ok else res.get("ErrMsg",""))
            print(f"[SELL] {pair} qty={qty_str} fill={fill} ok={ok} reason={reason}")
            # 用成交价计算回报
            if ss.entry_price and ss.entry_price > 0:
                trade_ret = float((fill - ss.entry_price) / ss.entry_price)
            self._update_learning(ss, trade_ret)
            ss.qty = Decimal("0"); ss.entry_price=None; ss.peak_after_entry=None; ss.last_trade_ts = now_ts
            return ok
        except requests.HTTPError as e:
            body = ""
            try: body = e.response.text
            except: pass
            self._log_trade(pair, "SELL", qty_str, str(price), reason, False, f"{e} {body}")
            print(f"[SELL ERR] {pair} {e} {body}")
            return False

    def _update_learning(self, ss: SymState, r: float):
        # 指数加权的均值回报，限制范围 [-1, +1] 以稳健
        r = max(-1.0, min(1.0, r))
        beta = 0.2  # 学习率
        ss.reward_mean = (1 - beta) * ss.reward_mean + beta * r
        ss.trades += 1

    # ---- 主循环 ----
    def run(self):
        print("[环境] API_KEY：", "来自环境变量" if os.getenv("ROOSTOO_API_KEY") else "脚本内置示例（仅演示）")
        print("[校时] serverTime =", api_server_time())

        # 预热：拉一次ticker，搭建初始 universe
        tk = api_ticker_all()
        data = tk.get("Data", {})
        self.universe = self._select_universe(data)
        print(f"[UNIVERSE] 初始选出 {len(self.universe)} 个：{self.universe}")

        # 初始化每个标的的状态容器
        for name in self.universe:
            if name not in self.syms:
                self.syms[name] = SymState(name, self.meta_by_pair[name],
                                           self.args.fast, self.args.slow, self.args.breakout, self.args.roc, self.args.slope)

        last_minute = int(time.time() // 60)
        trades_this_minute = 0

        while True:
            loop_ts = time.time()
            try:
                # 重置每分钟的交易上限
                cur_minute = int(loop_ts // 60)
                if cur_minute != last_minute:
                    trades_this_minute = 0
                    last_minute = cur_minute

                # 拉行情
                tk = api_ticker_all()
                data = tk.get("Data", {})

                # 组合估值 & 全局风控
                equity, usd_free, holding_value = self._portfolio_value(data)
                if self.equity_peak > 0:
                    dd = float((equity - self.equity_peak) / self.equity_peak)
                    if dd <= -abs(self.args.max_drawdown):
                        self.kill_switch = True
                        print(f"[KILL] 组合回撤达 {dd*100:.2f}%，触发全局熔断。")

                # 动态 Universe
                self.universe = self._select_universe(data)
                # 确保每个标的有状态容器
                for name in self.universe:
                    if name not in self.syms:
                        self.syms[name] = SymState(name, self.meta_by_pair[name],
                                                   self.args.fast, self.args.slow, self.args.breakout, self.args.roc, self.args.slope)

                # 更新指标与打分
                candidates: List[Tuple[str, float, Decimal, Dict]] = []
                for name in self.universe:
                    d = data.get(name, {})
                    lp = d.get("LastPrice", None)
                    if lp is None:
                        continue
                    p = Decimal(str(lp))
                    ss = self.syms[name]
                    f, s, cu, cd = ss.update(p)
                    # 波动过滤
                    vol_i = ss.vol()
                    if vol_i < Decimal(self.args.vol_floor):
                        continue
                    # 只做多：趋势确立（快>慢 且 价>慢线）
                    if ss.last_fast is None or ss.last_slow is None or not (ss.last_fast > ss.last_slow and p > ss.last_slow):
                        continue
                    sc = self._score(ss, p, d)
                    candidates.append((name, sc, p, d))

                # 排名 & 选择 top-k
                candidates.sort(key=lambda x: x[1], reverse=True)
                long_list = [c for c in candidates if c[1] > self.args.entry_threshold]
                top_targets = long_list[: self.args.topk]

                # 先处理退出：不在目标 or 风险触发
                for name, ss in list(self.syms.items()):
                    if ss.qty <= 0:
                        continue
                    # 当前价格
                    d = data.get(name, {})
                    lp = d.get("LastPrice", None)
                    if lp is None:
                        continue
                    p = Decimal(str(lp))
                    # 追踪峰值
                    if ss.peak_after_entry is None or p > ss.peak_after_entry:
                        ss.peak_after_entry = p
                    # 风控判定
                    exit_reasons = []
                    if ss.entry_price and ss.entry_price > 0:
                        r = (p - ss.entry_price) / ss.entry_price
                        vol_i = ss.vol()
                        stop_dist = max(vol_i * Decimal(self.args.stop_mul), Decimal(self.args.min_stop))
                        take_dist = max(vol_i * Decimal(self.args.take_mul), Decimal(self.args.min_take))
                        trail_dist = max(vol_i * Decimal(self.args.trail_mul), Decimal(self.args.min_trail))
                        if r <= -stop_dist:
                            exit_reasons.append(f"stop {float(stop_dist)*100:.2f}%")
                        if r >= take_dist:
                            exit_reasons.append(f"take {float(take_dist)*100:.2f}%")
                        if ss.peak_after_entry and ss.peak_after_entry > 0:
                            drawdown = (p - ss.peak_after_entry) / ss.peak_after_entry
                            if drawdown <= -trail_dist:
                                exit_reasons.append(f"trail {float(trail_dist)*100:.2f}%")
                    # EMA 反转
                    if ss.last_fast is not None and ss.last_slow is not None and ss.last_fast < ss.last_slow:
                        exit_reasons.append("ema_cross_down")
                    # 不在 top
                    if name not in [t[0] for t in top_targets]:
                        exit_reasons.append("drop_from_top")

                    if exit_reasons and trades_this_minute < self.args.max_trades_per_minute:
                        self._exit_long(name, ss, p, loop_ts, ";".join(exit_reasons))
                        trades_this_minute += 1

                # 再处理进入
                usd_for_buy = usd_free * Decimal(self.args.budget_perc)
                for name, sc, p, d in top_targets:
                    if trades_this_minute >= self.args.max_trades_per_minute:
                        break
                    ss = self.syms[name]
                    if ss.qty > 0:
                        continue
                    vol_i = ss.vol()
                    usd_for_buy, entered = self._enter_long(name, ss, p, usd_for_buy, vol_i, loop_ts, reason=f"score={sc:.4f}")
                    if entered:
                        trades_this_minute += 1

                # 打印心跳
                hold = {k: str(v.qty) for k, v in self.syms.items() if v.qty > 0}
                print(f"[HB] top={[ (t[0], round(t[1],4)) for t in top_targets ]} hold={hold} usd≈{usd_free} equity≈{equity} trades/min={trades_this_minute}/{self.args.max_trades_per_minute} kill={self.kill_switch}")

                # 保存状态
                if int(loop_ts) % 10 == 0:  # 低频保存
                    self._save_state()

            except requests.HTTPError as e:
                body = ""
                try: body = e.response.text
                except: pass
                print(f"[HTTP] {e} | {body}")
            except Exception as e:
                print(f"[ERR] {e}")

            time.sleep(self.args.interval)

# ---------------- CLI ----------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Roostoo OMEGA GrandMaster — 多资产集成趋势交易机器人（终极版）")
    # 信号参数
    ap.add_argument("--fast", type=int, default=12, help="EMA 快线周期")
    ap.add_argument("--slow", type=int, default=26, help="EMA 慢线周期（需 > fast）")
    ap.add_argument("--breakout", type=int, default=60, help="突破窗口")
    ap.add_argument("--roc", type=int, default=30, help="动量窗口")
    ap.add_argument("--slope", type=int, default=20, help="线性斜率窗口")
    ap.add_argument("--entry-threshold", type=float, default=0.0, help="进入阈值")

    # 风控 & 资金
    ap.add_argument("--risk-per-trade", type=float, default=0.02, help="单笔风险（占可用资金比例）")
    ap.add_argument("--budget-perc", type=float, default=0.9, help="下单动用的可用现金比例")
    ap.add_argument("--per-asset-cap", type=float, default=0.25, help="单资产最大占比（相对可用现金）")
    ap.add_argument("--vol-floor", type=float, default=0.002, help="最低波动过滤阈值")
    ap.add_argument("--min-stop", type=float, default=0.01, help="最小止损比例")
    ap.add_argument("--min-take", type=float, default=0.02, help="最小止盈比例")
    ap.add_argument("--min-trail", type=float, default=0.01, help="最小追踪止损回撤比例")
    ap.add_argument("--stop-mul", type=float, default=1.0, help="止损倍数 * 局部波动")
    ap.add_argument("--take-mul", type=float, default=2.0, help="止盈倍数 * 局部波动")
    ap.add_argument("--trail-mul", type=float, default=1.5, help="追踪止损倍数 * 局部波动")
    ap.add_argument("--max-drawdown", type=float, default=0.30, help="组合最大回撤（超出触发 Kill Switch）")

    # 市场筛选
    ap.add_argument("--universe-size", type=int, default=30, help="流动性 Top N")
    ap.add_argument("--topk", type=int, default=5, help="最多建仓标的数")
    ap.add_argument("--min-unit-volume", type=float, default=2e7, help="最小 USD 成交额过滤")

    # 运行
    ap.add_argument("--interval", type=int, default=3, help="轮询秒数")
    ap.add_argument("--max-trades-per-minute", type=int, default=10, help="每分钟最大成交次数")
    ap.add_argument("--cooldown", type=int, default=20, help="单标的交易冷却（秒）")
    ap.add_argument("--dry-run", action="store_true", help="只观测，不下单")
    return ap.parse_args()

def main():
    args = parse_args()
    print("[环境] 使用 API_KEY：", "来自环境变量" if os.getenv("ROOSTOO_API_KEY") else "脚本内置示例（仅演示）")
    print("[校时] serverTime =", api_server_time())
    bot = OmegaBot(args)
    bot.run()

if __name__ == "__main__":
    main()
