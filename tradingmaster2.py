
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roostoo OMEGA-SPOT Rotation — 低频、现货、费用感知 的多资产轮动机器人
===================================================================
遵循规则：
  • 仅现货（spot），仅 USD 基准：买=USD→币，卖=币→USD；不做空、不保证金/杠杆；不做做市/套利。
  • 交易费：0.1%/笔（买+卖回合计 ~0.2%），策略在开仓/平仓判断中显式纳入费用缓冲。
  • 频率限制：避免高频；默认每分钟最多 2 笔，下单后同一标的冷却期≥120s，持仓最短持有≥300s。
  • 撮合：仅交易 */USD 交易对；不进行“币→币”直换，轮动时先卖出旧持仓为 USD，再买入新标的。

策略核心（非单纯趋势）：
  • Cross-sectional Rotation：多资产相对强度轮动（过去 M 分钟 ROC + 斜率 + 突破 + 24h 强度）
  • 波动缩放仓位 + 费用阈值（edge 必须 > 费用+冗余）
  • 三重退出（止损/止盈/追踪） + 排名跌出 + 最短持有检查
  • 组合级风控（最大回撤 Kill Switch）
  • 状态持久化（断线续跑）

使用：
  pip install requests
  export ROOSTOO_API_KEY="你的API_KEY"
  export ROOSTOO_SECRET_KEY="你的SECRET_KEY"

  python roostoo_omega_spot.py --interval 10 --universe-size 30 --topk 5 \
    --risk-per-trade 0.02 --budget-perc 0.8 --per-asset-cap 0.25 \
    --roc 36 --slope 24 --breakout 72 --vol-floor 0.002 \
    --min-hold-sec 300 --cooldown 120 --max-trades-per-minute 2

仅观测：
  python roostoo_omega_spot.py --dry-run
"""
import os
import time
import json
import hmac
import hashlib
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import requests

getcontext().prec = 28

BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.getenv("ROOSTOO_API_KEY", " ttdLE0ZRfWAZeaBoaq8SyEhH6QSQ8yr6laZzTgKNTTYTkogK1cUWdT9P6r59Jd3S")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "8mYHLPiVxrTDx8bTe5UUumVOmZ07DBAE1ZovuxgyJk7ny1oYrDECdWc1GbuMX1fN")

FEE_RATE = Decimal("0.001")  # 0.1% 每笔
ROUND_TRIP_FEE = FEE_RATE * Decimal("2.0")

STATE_FILE = "spot_state.json"
TRADES_LOG = "spot_trades.csv"
EQUITY_LOG = "spot_equity.csv"

SESSION = requests.Session()

# ---------------- 签名 & API ----------------
def _now_ms_str() -> str:
    return str(int(time.time() * 1000))

def _ensure_str_vals(payload: Dict) -> Dict[str, str]:
    return {str(k): str(v) for k, v in payload.items()}

def _total_params_str(payload: Dict[str, str]) -> str:
    keys = sorted(payload.keys())
    return "&".join(f"{k}={payload[k]}" for k in keys)

def _hmac_sha256_hex(secret: str, message: str) -> str:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()

def sign_for_get(payload: Optional[Dict]=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    payload = {} if payload is None else dict(payload)
    payload["timestamp"] = _now_ms_str()
    payload = _ensure_str_vals(payload)
    total = _total_params_str(payload)
    sig = _hmac_sha256_hex(SECRET_KEY, total)
    headers = {"RST-API-KEY": API_KEY, "MSG-SIGNATURE": sig}
    return headers, payload

def sign_for_post(payload: Optional[Dict]=None) -> Tuple[Dict[str, str], str]:
    payload = {} if payload is None else dict(payload)
    payload["timestamp"] = _now_ms_str()
    payload = _ensure_str_vals(payload)
    total = _total_params_str(payload)
    sig = _hmac_sha256_hex(SECRET_KEY, total)
    headers = {"Content-Type": "application/x-www-form-urlencoded", "RST-API-KEY": API_KEY, "MSG-SIGNATURE": sig}
    return headers, total

def api_ticker_all():
    url = f"{BASE_URL}/v3/ticker"
    params = {"timestamp": _now_ms_str()}
    r = SESSION.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()

def api_exchange_info():
    url = f"{BASE_URL}/v3/exchangeInfo"
    r = SESSION.get(url, timeout=15)
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

# ---------------- 数学工具 ----------------
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
            self.v = self.alpha * p + (Decimal(1)-self.alpha) * self.v
        return self.v

def stdev(xs: List[Decimal]) -> Decimal:
    n = len(xs)
    if n < 2:
        return Decimal("0")
    m = sum(xs)/Decimal(n)
    var = sum((x-m)*(x-m) for x in xs)/Decimal(n-1)
    return (var if var>0 else Decimal("0")).sqrt()

def linreg_slope(prices: List[Decimal]) -> Optional[Decimal]:
    n = len(prices)
    if n < 3:
        return None
    x_sum = Decimal(n*(n-1)/2)
    x2_sum = Decimal((n-1)*n*(2*n-1)/6)
    y_sum = sum(prices)
    xy_sum = sum(Decimal(i)*prices[i] for i in range(n))
    denom = Decimal(n)*x2_sum - x_sum*x_sum
    if denom == 0:
        return None
    return (Decimal(n)*xy_sum - x_sum*y_sum) / denom

# ---------------- 元信息与状态 ----------------
@dataclass
class PairMeta:
    price_prec: int
    amt_prec: int
    mini_order: Decimal

class Sym:
    def __init__(self, name: str, meta: PairMeta, roc_n: int, slope_n: int, breakout_n: int):
        self.name = name
        self.meta = meta
        self.roc_n = roc_n
        self.slope_n = slope_n
        self.breakout_n = breakout_n
        self.prices: deque = deque(maxlen=max(roc_n, slope_n, breakout_n, 120))
        self.returns: deque = deque(maxlen=max(roc_n, 20))
        self.qty: Decimal = Decimal("0")
        self.entry_price: Optional[Decimal] = None
        self.peak_after_entry: Optional[Decimal] = None
        self.last_trade_ts: float = 0.0
        self.hold_start_ts: Optional[float] = None

        self.ema_fast = EMA(12)
        self.ema_slow = EMA(26)
        self.last_fast: Optional[Decimal] = None
        self.last_slow: Optional[Decimal] = None

        # 在线奖励（近似 bandit）：最近交易的平均单笔收益
        self.reward_mean: float = 0.0
        self.trades: int = 0

    def update(self, p: Decimal):
        if self.prices:
            prev = self.prices[-1]
            ret = (p - prev)/prev if prev != 0 else Decimal("0")
            self.returns.append(ret)
        self.prices.append(p)
        f = self.ema_fast.update(p)
        s = self.ema_slow.update(p)
        self.last_fast, self.last_slow = f, s
        return f, s

    def roc(self) -> Optional[Decimal]:
        n = self.roc_n
        if len(self.prices) < n+1:
            return None
        a = self.prices[-1]
        b = self.prices[-1-n]
        if b == 0: return None
        return (a-b)/b

    def slope(self) -> Optional[Decimal]:
        if len(self.prices) < self.slope_n: return None
        return linreg_slope(list(self.prices)[-self.slope_n:])

    def vol(self) -> Decimal:
        if len(self.returns) < 5: return Decimal("0")
        return stdev(list(self.returns))

# ---------------- 机器人主体 ----------------
class SpotRotationBot:
    def __init__(self, args):
        self.args = args
        self.meta: Dict[str, PairMeta] = {}
        self.syms: Dict[str, Sym] = {}
        self.universe: List[str] = []
        self.equity_peak: Decimal = Decimal("0")
        self.kill: bool = False

        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG,"w",encoding="utf-8") as f:
                f.write("ts,pair,side,qty,price,reason,ok,err\n")
        if not os.path.exists(EQUITY_LOG):
            with open(EQUITY_LOG,"w",encoding="utf-8") as f:
                f.write("ts,equity,usd_free,holding_value,dd\n")

        self._init_meta()
        self._load_state()

    def _init_meta(self):
        info = api_exchange_info()
        tps = info.get("TradePairs",{})
        for name, d in tps.items():
            if not d.get("CanTrade", True): continue
            if not name.endswith("/USD"): continue  # 只允许 USD 基准
            self.meta[name] = PairMeta(
                price_prec=int(d.get("PricePrecision",2)),
                amt_prec=int(d.get("AmountPrecision",6)),
                mini_order=Decimal(str(d.get("MiniOrder","1.0")))
            )

    def _load_state(self):
        if not os.path.exists(STATE_FILE): return
        try:
            with open(STATE_FILE,"r",encoding="utf-8") as f:
                s=json.load(f)
            self.equity_peak = Decimal(s.get("equity_peak","0"))
            for name, st in s.get("symbols",{}).items():
                if name in self.meta:
                    sym = Sym(name, self.meta[name], self.args.roc, self.args.slope, self.args.breakout)
                    sym.qty = Decimal(st.get("qty","0"))
                    ep = st.get("entry_price")
                    sym.entry_price = Decimal(str(ep)) if ep is not None else None
                    sym.peak_after_entry = Decimal(str(st.get("peak_after_entry"))) if st.get("peak_after_entry") else None
                    sym.last_trade_ts = float(st.get("last_trade_ts",0))
                    sym.hold_start_ts = float(st.get("hold_start_ts",0)) if st.get("hold_start_ts") else None
                    sym.reward_mean = float(st.get("reward_mean",0))
                    sym.trades = int(st.get("trades",0))
                    self.syms[name] = sym
        except Exception as e:
            print("[STATE] 载入失败：",e)

    def _save_state(self):
        try:
            dump={
                "equity_peak": str(self.equity_peak),
                "symbols": {}
            }
            for name, sym in self.syms.items():
                dump["symbols"][name]={
                    "qty": str(sym.qty),
                    "entry_price": str(sym.entry_price) if sym.entry_price is not None else None,
                    "peak_after_entry": str(sym.peak_after_entry) if sym.peak_after_entry is not None else None,
                    "last_trade_ts": sym.last_trade_ts,
                    "hold_start_ts": sym.hold_start_ts,
                    "reward_mean": sym.reward_mean,
                    "trades": sym.trades
                }
            with open(STATE_FILE,"w",encoding="utf-8") as f:
                json.dump(dump,f,ensure_ascii=False,indent=2)
        except Exception as e:
            print("[STATE] 保存失败：",e)

    # ---- 估值 & 风控 ----
    def _portfolio_value(self, data: Dict) -> Tuple[Decimal, Decimal, Decimal]:
        bal = api_balance()
        usd_free = Decimal(str(bal.get("Wallet",{}).get("USD",{}).get("Free","0")))
        hold = Decimal("0")
        for name, sym in self.syms.items():
            if sym.qty>0:
                px = Decimal(str(data.get(name,{}).get("LastPrice","0")))
                hold += sym.qty*px
        eq = usd_free + hold
        if eq>self.equity_peak: self.equity_peak = eq
        dd = (eq - self.equity_peak)/self.equity_peak if self.equity_peak>0 else Decimal("0")
        with open(EQUITY_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{eq},{usd_free},{hold},{dd}\n")
        if dd <= -abs(Decimal(str(self.args.max_drawdown))):
            self.kill = True
            print(f"[KILL] 回撤 {float(dd)*100:.2f}% 触发熔断")
        return eq, usd_free, hold

    # ---- Universe 选择 ----
    def _select_universe(self, data: Dict) -> List[str]:
        L=[]
        for name, d in data.items():
            if name not in self.meta: continue
            utv = Decimal(str(d.get("UnitTradeValue","0")))
            if utv < Decimal(self.args.min_unit_volume): continue
            if d.get("LastPrice") is None: continue
            L.append((name, utv, Decimal(str(d.get("Change",0)))))
        L.sort(key=lambda x:(x[1],x[2]), reverse=True)
        return [x[0] for x in L[: self.args.universe_size]]

    # ---- 打分（费用感知 + 相对强度） ----
    def _score(self, sym: Sym, p: Decimal, d: Dict) -> float:
        # 组件：短期 ROC、线性斜率、突破、24h 强度、在线奖励
        roc_v = sym.roc()
        roc_f = float(roc_v) if roc_v is not None else 0.0
        slope_v = sym.slope()
        slope_f = float(slope_v) if slope_v is not None else 0.0
        hi = max(list(sym.prices)[-sym.breakout_n:]) if len(sym.prices)>=sym.breakout_n else None
        breakout = 1.0 if (hi is not None and p>=hi) else 0.0
        ch24 = max(0.0, float(Decimal(str(d.get("Change",0)))))
        base = 0.4*roc_f + 0.25*slope_f + 0.25*breakout + 0.10*ch24
        # 费用缓冲：需要足够的“正动量”才值得开仓
        # 将 ROUND_TRIP_FEE 作为最低 edge，附加 0.1% 冗余
        fee_buffer = float(ROUND_TRIP_FEE + Decimal("0.001"))
        score = base - fee_buffer + 0.4*sym.reward_mean
        return score

    # ---- 下单数量（现货，仅 USD） ----
    def _qty_from_budget(self, name: str, usd: Decimal, price: Decimal) -> Optional[str]:
        m = self.meta[name]
        qty = quantize_down(usd/price, m.amt_prec)
        if qty<=0: return None
        if qty*price <= m.mini_order:
            need = quantize_down((m.mini_order/price)*Decimal("1.001"), m.amt_prec)
            if need*price > usd: return None
            qty = need
        return f"{qty:.{m.amt_prec}f}"

    # ---- 交易记录 ----
    def _log_trade(self, pair, side, qty, price, reason, ok, err):
        with open(TRADES_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{pair},{side},{qty},{price},{reason},{int(ok)},{err}\n")

    # ---- 买入/卖出（Spot USD） ----
    def _buy(self, name: str, sym: Sym, price: Decimal, usd_free: Decimal, now_ts: float, reason: str) -> Tuple[Decimal,bool]:
        if self.kill or sym.qty>0: return usd_free, False
        if now_ts - sym.last_trade_ts < self.args.cooldown: return usd_free, False

        # 风险与仓位上限
        risk_usd = max(Decimal("1"), usd_free*Decimal(self.args.risk_per_trade))
        vol_i = sym.vol()
        stop_dist = max(vol_i*Decimal(self.args.stop_mul), Decimal(self.args.min_stop)) + ROUND_TRIP_FEE  # 费用纳入止损距离
        if stop_dist <= 0: stop_dist = Decimal("0.01")
        target_notional = risk_usd / stop_dist
        cap = usd_free * Decimal(self.args.per_asset_cap)
        use = min(target_notional, cap, usd_free) * Decimal(self.args.budget_perc)
        if use < Decimal("1"): return usd_free, False

        qty_str = self._qty_from_budget(name, use, price)
        if qty_str is None: return usd_free, False

        if self.args.dry_run:
            sym.qty = Decimal(qty_str)
            sym.entry_price = price*(Decimal("1.0")+FEE_RATE)  # 买入含费成本
            sym.peak_after_entry = price
            sym.last_trade_ts = now_ts
            sym.hold_start_ts = now_ts
            self._log_trade(name,"BUY",qty_str,str(price),"[DRY] "+reason,True,"")
            print(f"[DRY BUY] {name} {qty_str} @ {price} reason={reason}")
            return usd_free - Decimal(qty_str)*price, True

        try:
            res = api_place_market(name,"BUY",qty_str)
            ok = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail",{}).get("FilledAverPrice", price)))
            err = "" if ok else res.get("ErrMsg","")
            if ok:
                sym.qty = Decimal(str(res["OrderDetail"]["FilledQuantity"]))
                sym.entry_price = fill*(Decimal("1.0")+FEE_RATE)
                sym.peak_after_entry = fill
                sym.last_trade_ts = now_ts
                sym.hold_start_ts = now_ts
                usd_free = max(Decimal("0"), usd_free - sym.qty*fill)
            self._log_trade(name,"BUY",qty_str,str(fill),reason,ok,err)
            print(f"[BUY] {name} {qty_str} @ {fill} ok={ok} reason={reason}")
            return usd_free, ok
        except requests.HTTPError as e:
            body = ""
            try: body=e.response.text
            except: pass
            self._log_trade(name,"BUY",qty_str,str(price),reason,False,f"{e} {body}")
            print(f"[BUY ERR] {name} {e} {body}")
            return usd_free, False

    def _sell(self, name: str, sym: Sym, price: Decimal, now_ts: float, reason: str) -> bool:
        if sym.qty<=0: return False
        if now_ts - sym.last_trade_ts < self.args.cooldown: return False
        if sym.hold_start_ts and now_ts - sym.hold_start_ts < self.args.min_hold_sec:
            return False  # 最短持有期

        qty = quantize_down(sym.qty, self.meta[name].amt_prec)
        if qty<=0: return False
        qty_str = f"{qty:.{self.meta[name].amt_prec}f}"

        # 交易收益用于在线学习（净价考虑卖出手续费）
        trade_ret = 0.0
        cost = sym.entry_price if sym.entry_price else price
        eff_price = price*(Decimal("1.0")-FEE_RATE)
        if cost>0:
            trade_ret = float((eff_price - cost)/cost)

        if self.args.dry_run:
            self._log_trade(name,"SELL",qty_str,str(price),"[DRY] "+reason,True,"")
            print(f"[DRY SELL] {name} {qty_str} @ {price} reason={reason}")
            self._update_reward(sym, trade_ret)
            sym.qty = Decimal("0"); sym.entry_price=None; sym.peak_after_entry=None; sym.last_trade_ts = now_ts; sym.hold_start_ts=None
            return True

        try:
            res = api_place_market(name,"SELL",qty_str)
            ok = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail",{}).get("FilledAverPrice", price)))
            eff_fill = fill*(Decimal("1.0")-FEE_RATE)
            if sym.entry_price and sym.entry_price>0:
                trade_ret = float((eff_fill - sym.entry_price)/sym.entry_price)
            self._log_trade(name,"SELL",qty_str,str(fill),reason,ok,"" if ok else res.get("ErrMsg",""))
            print(f"[SELL] {name} {qty_str} @ {fill} ok={ok} reason={reason}")
            self._update_reward(sym, trade_ret)
            sym.qty = Decimal("0"); sym.entry_price=None; sym.peak_after_entry=None; sym.last_trade_ts = now_ts; sym.hold_start_ts=None
            return ok
        except requests.HTTPError as e:
            body=""
            try: body=e.response.text
            except: pass
            self._log_trade(name,"SELL",qty_str,str(price),reason,False,f"{e} {body}")
            print(f"[SELL ERR] {name} {e} {body}")
            return False

    def _update_reward(self, sym: Sym, r: float):
        r = max(-1.0, min(1.0, r))
        beta = 0.2
        sym.reward_mean = (1-beta)*sym.reward_mean + beta*r
        sym.trades += 1

    # ---- 主循环 ----
    def run(self):
        print("[规则] 仅现货 USD，费率 0.1%/笔，低频轮动；不做空/杠杆/套利/做市。")
        tk = api_ticker_all()
        data = tk.get("Data",{})
        self.universe = self._select_universe(data)
        for name in self.universe:
            if name not in self.syms:
                self.syms[name]=Sym(name,self.meta[name],self.args.roc,self.args.slope,self.args.breakout)

        last_minute = int(time.time()//60)
        trades_minute = 0

        while True:
            loop_ts = time.time()
            try:
                cur_minute = int(loop_ts//60)
                if cur_minute != last_minute:
                    trades_minute = 0
                    last_minute = cur_minute

                tk = api_ticker_all()
                data = tk.get("Data",{})
                equity, usd_free, hold_val = self._portfolio_value(data)

                # 动态 Universe（保持低频选股）
                self.universe = self._select_universe(data)
                for name in self.universe:
                    if name not in self.syms:
                        self.syms[name]=Sym(name,self.meta[name],self.args.roc,self.args.slope,self.args.breakout)

                # 更新状态并打分
                candidates: List[Tuple[str,float,Decimal,Dict]] = []
                for name in self.universe:
                    d = data.get(name,{})
                    lp = d.get("LastPrice",None)
                    if lp is None: continue
                    p = Decimal(str(lp))
                    sym = self.syms[name]
                    sym.update(p)
                    # 仅做多：价>慢 EMA
                    if sym.last_slow is None or p <= sym.last_slow: continue
                    # 波动过滤
                    if sym.vol() < Decimal(self.args.vol_floor): continue
                    sc = self._score(sym, p, d)
                    if sc > 0:  # 必须跨过费用缓冲
                        candidates.append((name, sc, p, d))

                candidates.sort(key=lambda x:x[1], reverse=True)
                targets = candidates[: self.args.topk]

                # 退出：不在 targets 或 风控
                for name, sym in list(self.syms.items()):
                    if sym.qty <= 0: continue
                    d = data.get(name,{})
                    lp = d.get("LastPrice",None)
                    if lp is None: continue
                    p = Decimal(str(lp))
                    # 更新峰值
                    if sym.peak_after_entry is None or p > sym.peak_after_entry: sym.peak_after_entry = p

                    # 风控：考虑费用后的阈值
                    r = Decimal("0")
                    if sym.entry_price and sym.entry_price>0:
                        eff = p*(Decimal("1.0")-FEE_RATE)
                        r = (eff - sym.entry_price)/sym.entry_price
                    vol_i = sym.vol()
                    stop_dist = max(vol_i*Decimal(self.args.stop_mul), Decimal(self.args.min_stop)) + ROUND_TRIP_FEE
                    take_dist = max(vol_i*Decimal(self.args.take_mul), Decimal(self.args.min_take)) + ROUND_TRIP_FEE
                    trail_dist = max(vol_i*Decimal(self.args.trail_mul), Decimal(self.args.min_trail)) + ROUND_TRIP_FEE

                    exit_reasons=[]
                    if r <= -stop_dist: exit_reasons.append(f"stop {float(stop_dist)*100:.2f}% (fee-aware)")
                    if r >= take_dist: exit_reasons.append(f"take {float(take_dist)*100:.2f}% (fee-aware)")
                    if sym.peak_after_entry and sym.peak_after_entry>0:
                        drawdown = (p - sym.peak_after_entry)/sym.peak_after_entry
                        if drawdown <= -trail_dist:
                            exit_reasons.append(f"trail {float(trail_dist)*100:.2f}% (fee-aware)")
                    if name not in [t[0] for t in targets]:
                        exit_reasons.append("drop_from_targets")

                    if exit_reasons and trades_minute < self.args.max_trades_per_minute:
                        self._sell(name, sym, p, loop_ts, ";".join(exit_reasons))
                        trades_minute += 1

                # 进入：按 targets 顺序，受速率限制与现金约束
                cash_for_buys = usd_free * Decimal(self.args.budget_perc)
                for name, sc, p, d in targets:
                    if trades_minute >= self.args.max_trades_per_minute: break
                    sym = self.syms[name]
                    if sym.qty > 0: continue
                    cash_for_buys, entered = self._buy(name, sym, p, cash_for_buys, loop_ts, reason=f"score={sc:.4f}")
                    if entered:
                        trades_minute += 1

                # 心跳
                holding = {n:str(s.qty) for n,s in self.syms.items() if s.qty>0}
                print(f"[HB] top={[ (t[0], round(t[1],4)) for t in targets ]} hold={holding} usd≈{usd_free} eq≈{equity} trades/min={trades_minute}/{self.args.max_trades_per_minute}")

                if int(loop_ts) % 10 == 0:
                    self._save_state()

            except requests.HTTPError as e:
                body=""
                try: body=e.response.text
                except: pass
                print("[HTTP]", e, body)
            except Exception as e:
                print("[ERR]", e)

            time.sleep(self.args.interval)

# ---------------- CLI ----------------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Roostoo OMEGA-SPOT Rotation — 低频现货轮动机器人（费用感知）")
    # 信号
    ap.add_argument("--roc", type=int, default=36, help="ROC 窗口（样本）")
    ap.add_argument("--slope", type=int, default=24, help="线性斜率窗口（样本）")
    ap.add_argument("--breakout", type=int, default=72, help="突破窗口（样本）")
    # 资金 & 风控
    ap.add_argument("--risk-per-trade", type=float, default=0.02, help="单笔风险占可用现金比例")
    ap.add_argument("--budget-perc", type=float, default=0.8, help="用于买入的可用现金比例")
    ap.add_argument("--per-asset-cap", type=float, default=0.25, help="单资产最大占比（相对可用现金）")
    ap.add_argument("--vol-floor", type=float, default=0.002, help="最小波动过滤")
    ap.add_argument("--min-stop", type=float, default=0.01, help="最小止损比例")
    ap.add_argument("--min-take", type=float, default=0.02, help="最小止盈比例")
    ap.add_argument("--min-trail", type=float, default=0.01, help="最小追踪止损比例")
    ap.add_argument("--stop-mul", type=float, default=1.0, help="止损倍数 * 波动")
    ap.add_argument("--take-mul", type=float, default=2.0, help="止盈倍数 * 波动")
    ap.add_argument("--trail-mul", type=float, default=1.5, help="追踪止损倍数 * 波动")
    ap.add_argument("--max-drawdown", type=float, default=0.3, help="最大组合回撤（触发熔断）")
    ap.add_argument("--min-hold-sec", type=int, default=300, help="最短持仓时间（秒）")

    # Universe
    ap.add_argument("--universe-size", type=int, default=30, help="流动性 Top N")
    ap.add_argument("--topk", type=int, default=5, help="最多持仓数量")
    ap.add_argument("--min-unit-volume", type=float, default=2e7, help="最小USD成交额")

    # 运行节奏
    ap.add_argument("--interval", type=int, default=10, help="轮询间隔秒数（避免HFT）")
    ap.add_argument("--max-trades-per-minute", type=int, default=2, help="每分钟最下单次数上限")
    ap.add_argument("--cooldown", type=int, default=120, help="单标的交易冷却（秒）")
    ap.add_argument("--dry-run", action="store_true", help="只观测不下单")
    return ap.parse_args()

def main():
    args = parse_args()
    print("[环境] 使用 API_KEY：", "来自环境变量" if os.getenv("ROOSTOO_API_KEY") else "脚本内置示例（仅演示）")
    bot = SpotRotationBot(args)
    bot.run()

if __name__ == "__main__":
    main()
