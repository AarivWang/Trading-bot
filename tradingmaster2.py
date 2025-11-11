
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Roostoo OMEGA-SPOT (Rate-Limited + Report Edition)
==================================================
• 现货（spot）/USD，低频轮动，费用感知；内置限流 + 指数退避
• 新增：--report / --report-h 输出“回测/绩效报告”（基于交易日志与当前资产）

快速：
  pip install requests
  export ROOSTOO_API_KEY="你的API_KEY"; export ROOSTOO_SECRET_KEY="你的SECRET_KEY"
  python roostoo_omega_spot_rl.py --dry-run    # 观测
  python roostoo_omega_spot_rl.py --report     # 只输出报告并退出
"""
import os
import time
import json
import hmac
import hashlib
import random
import math
from collections import deque
from decimal import Decimal, ROUND_DOWN, getcontext
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import requests

getcontext().prec = 28

BASE_URL = "https://mock-api.roostoo.com"
API_KEY = os.getenv("ROOSTOO_API_KEY", "yN6cF2VtW9xJ4qPbeK1mZ7YruA3dH0GiP5oL8wQnrR2tC6XfD8vB1kMeZ4gU7SaJ")
SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY", "tG5yH7uJ9iK1oL3pA6sD8fF0gH2jK4lZ7xC9vB1nM3qW5eR7tY9uI1oP3aS5")

FEE_RATE = Decimal("0.001")
ROUND_TRIP_FEE = FEE_RATE * Decimal(2)

STATE_FILE  = "spot_state.json"
TRADES_LOG  = "spot_trades.csv"
EQUITY_LOG  = "spot_equity.csv"

SESSION = requests.Session()

# ---------- 签名与通用 ----------
def _now_ms_str() -> str:
    return str(int(time.time() * 1000))

def _ensure_str_vals(d: Dict) -> Dict[str, str]:
    return {str(k): str(v) for k, v in d.items()}

def _total_params_str(d: Dict[str, str]) -> str:
    keys = sorted(d.keys())
    return "&".join(f"{k}={d[k]}" for k in keys)

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

# ---------- 限流退避 ----------
_last_call_ts = {"ticker": 0.0, "balance": 0.0, "order": 0.0}

def _rate_limited_get(url: str, params: Dict, kind: str, min_interval: float, max_retries: int, max_backoff: float, headers: Optional[Dict]=None):
    now = time.time()
    delta = now - _last_call_ts.get(kind, 0.0)
    if delta < min_interval:
        time.sleep(min_interval - delta)

    for attempt in range(max_retries):
        r = SESSION.get(url, params=params, headers=headers, timeout=20)
        if r.status_code in (429, 500, 502, 503, 504):
            backoff = min((2 ** attempt), max_backoff) + random.random()
            print(f"[RATE] {r.status_code} on {kind} → sleep {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(backoff)
            continue
        r.raise_for_status()
        _last_call_ts[kind] = time.time()
        return r.json()
    r.raise_for_status()

def _rate_limited_post(url: str, data: str, headers: Dict[str,str], kind: str, min_interval: float, max_retries: int, max_backoff: float):
    now = time.time()
    delta = now - _last_call_ts.get(kind, 0.0)
    if delta < min_interval:
        time.sleep(min_interval - delta)

    for attempt in range(max_retries):
        r = SESSION.post(url, headers=headers, data=data, timeout=20)
        if r.status_code in (429, 500, 502, 503, 504):
            backoff = min((2 ** attempt), max_backoff) + random.random()
            print(f"[RATE] {r.status_code} on {kind} → sleep {backoff:.2f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(backoff)
            continue
        r.raise_for_status()
        _last_call_ts[kind] = time.time()
        return r.json()
    r.raise_for_status()

# ---------- API 封装 ----------
def api_exchange_info():
    url = f"{BASE_URL}/v3/exchangeInfo"
    return _rate_limited_get(url, {}, "exchangeInfo", 5.0, 3, 8.0)

def api_ticker_all(min_interval: float, max_retries: int, max_backoff: float):
    url = f"{BASE_URL}/v3/ticker"
    params = {"timestamp": _now_ms_str()}
    return _rate_limited_get(url, params, "ticker", min_interval, max_retries, max_backoff)

def api_balance(min_interval: float, max_retries: int, max_backoff: float):
    url = f"{BASE_URL}/v3/balance"
    headers, params = sign_for_get({})
    return _rate_limited_get(url, params, "balance", min_interval, max_retries, max_backoff, headers=headers)

def api_place_market(pair: str, side: str, qty_str: str, min_interval: float, max_retries: int, max_backoff: float):
    url = f"{BASE_URL}/v3/place_order"
    payload = {"pair": pair, "side": side.upper(), "type": "MARKET", "quantity": qty_str}
    headers, body = sign_for_post(payload)
    return _rate_limited_post(url, body, headers, "order", min_interval, max_retries, max_backoff)

# ---------- 工具 ----------
def quantize_down(x: Decimal, prec: int) -> Decimal:
    q = Decimal(1).scaleb(-prec)
    return x.quantize(q, rounding=ROUND_DOWN)

def stdev(xs: List[Decimal]) -> Decimal:
    n = len(xs)
    if n < 2: return Decimal("0")
    m = sum(xs)/Decimal(n)
    var = sum((x-m)*(x-m) for x in xs)/Decimal(n-1)
    return (var if var>0 else Decimal("0")).sqrt()

def linreg_slope(prices: List[Decimal]) -> Optional[Decimal]:
    n = len(prices)
    if n < 3: return None
    x_sum  = Decimal(n*(n-1)/2)
    x2_sum = Decimal((n-1)*n*(2*n-1)/6)
    y_sum  = sum(prices)
    xy_sum = sum(Decimal(i)*prices[i] for i in range(n))
    denom  = Decimal(n)*x2_sum - x_sum*x_sum
    if denom == 0: return None
    return (Decimal(n)*xy_sum - x_sum*y_sum) / denom

# ---------- 元信息与状态 ----------
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
        self.prices: deque = deque(maxlen=max(roc_n, slope_n, breakout_n, 180))
        self.returns: deque = deque(maxlen=roc_n if roc_n>10 else 20)
        self.qty: Decimal = Decimal("0")
        self.entry_price: Optional[Decimal] = None
        self.peak_after_entry: Optional[Decimal] = None
        self.last_trade_ts: float = 0.0
        self.hold_start_ts: Optional[float] = None
        self.reward_mean: float = 0.0
        self.trades: int = 0

    def update(self, p: Decimal):
        if self.prices:
            prev = self.prices[-1]
            ret  = (p - prev)/prev if prev != 0 else Decimal("0")
            self.returns.append(ret)
        self.prices.append(p)

    def roc(self) -> Optional[Decimal]:
        n = self.roc_n
        if len(self.prices) < n+1: return None
        a = self.prices[-1]; b = self.prices[-1-n]
        if b == 0: return None
        return (a-b)/b

    def slope(self) -> Optional[Decimal]:
        if len(self.prices) < self.slope_n: return None
        return linreg_slope(list(self.prices)[-self.slope_n:])

    def vol(self) -> Decimal:
        if len(self.returns) < 5: return Decimal("0")
        return stdev(list(self.returns))

# ---------- 机器人主体 ----------
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
            if not name.endswith("/USD"):   continue
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
                    sym.last_trade_ts = float(st.get("last_trade_ts",0.0))
                    sym.hold_start_ts = float(st.get("hold_start_ts",0.0)) if st.get("hold_start_ts") else None
                    sym.reward_mean = float(st.get("reward_mean",0.0))
                    sym.trades = int(st.get("trades",0))
                    self.syms[name]=sym
        except Exception as e:
            print("[STATE] 加载失败：", e)

    def _save_state(self):
        try:
            dump = {"equity_peak": str(self.equity_peak), "symbols": {}}
            for name, sym in self.syms.items():
                dump["symbols"][name] = {
                    "qty": str(sym.qty),
                    "entry_price": str(sym.entry_price) if sym.entry_price is not None else None,
                    "peak_after_entry": str(sym.peak_after_entry) if sym.peak_after_entry is not None else None,
                    "last_trade_ts": sym.last_trade_ts,
                    "hold_start_ts": sym.hold_start_ts,
                    "reward_mean": sym.reward_mean,
                    "trades": sym.trades
                }
            with open(STATE_FILE,"w",encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[STATE] 保存失败：", e)

    # ---- 估值与组合风控 ----
    def _portfolio_value(self, data: Dict) -> Tuple[Decimal, Decimal, Decimal]:
        bal = api_balance(self.args.min_balance_interval, self.args.max_retries, self.args.max_backoff)
        usd_free = Decimal(str(bal.get("Wallet",{}).get("USD",{}).get("Free","0")))
        hold_val = Decimal("0")
        for name, sym in self.syms.items():
            if sym.qty>0:
                px = Decimal(str(data.get(name,{}).get("LastPrice","0")))
                hold_val += sym.qty * px
        eq = usd_free + hold_val
        if eq > self.equity_peak: self.equity_peak = eq
        dd = (eq - self.equity_peak)/self.equity_peak if self.equity_peak>0 else Decimal("0")
        with open(EQUITY_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{eq},{usd_free},{hold_val},{dd}\n")
        if dd <= -abs(Decimal(str(self.args.max_drawdown))):
            self.kill = True
            print(f"[KILL] 回撤 {float(dd)*100:.2f}% 触发熔断")
        return eq, usd_free, hold_val

    # ---- Universe ----
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

    # ---- 打分（费用感知） ----
    def _score(self, sym: Sym, p: Decimal, d: Dict) -> float:
        roc_v = sym.roc(); roc_f = float(roc_v) if roc_v is not None else 0.0
        slope_v = sym.slope(); slope_f = float(slope_v) if slope_v is not None else 0.0
        hi = max(list(sym.prices)[-sym.breakout_n:]) if len(sym.prices)>=sym.breakout_n else None
        breakout = 1.0 if (hi is not None and p>=hi) else 0.0
        ch24 = max(0.0, float(Decimal(str(d.get("Change",0)))))
        base = 0.45*roc_f + 0.25*slope_f + 0.20*breakout + 0.10*ch24
        fee_buffer = float(ROUND_TRIP_FEE + Decimal("0.001"))
        score = base - fee_buffer + 0.4*sym.reward_mean
        return score

    def _qty_from_budget(self, name: str, usd: Decimal, price: Decimal) -> Optional[str]:
        m = self.meta[name]
        qty = quantize_down(usd/price, m.amt_prec)
        if qty <= 0: return None
        if qty*price <= m.mini_order:
            need = quantize_down((m.mini_order/price)*Decimal("1.001"), m.amt_prec)
            if need*price > usd: return None
            qty = need
        return f"{qty:.{m.amt_prec}f}"

    def _log_trade(self, pair, side, qty, price, reason, ok, err):
        with open(TRADES_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{pair},{side},{qty},{price},{reason},{int(ok)},{err}\n")

    # ---- 买入/卖出 ----
    def _buy(self, name: str, sym: Sym, price: Decimal, usd_free: Decimal, now_ts: float, reason: str) -> Tuple[Decimal,bool]:
        if self.kill or sym.qty>0: return usd_free, False
        if now_ts - sym.last_trade_ts < self.args.cooldown: return usd_free, False

        risk_usd = max(Decimal("1"), usd_free*Decimal(self.args.risk_per_trade))
        vol_i = sym.vol()
        stop_dist = max(vol_i*Decimal(self.args.stop_mul), Decimal(self.args.min_stop)) + ROUND_TRIP_FEE
        if stop_dist <= 0: stop_dist = Decimal("0.01")
        target_notional = risk_usd / stop_dist
        cap = usd_free * Decimal(self.args.per_asset_cap)
        use = min(target_notional, cap, usd_free) * Decimal(self.args.budget_perc)
        if use < Decimal("1"): return usd_free, False

        qty_str = self._qty_from_budget(name, use, price)
        if qty_str is None: return usd_free, False

        if self.args.dry_run:
            sym.qty = Decimal(qty_str)
            sym.entry_price = price*(Decimal("1.0")+FEE_RATE)
            sym.peak_after_entry = price
            sym.last_trade_ts = now_ts
            sym.hold_start_ts = now_ts
            self._log_trade(name,"BUY",qty_str,str(price),"[DRY] "+reason,True,"")
            print(f"[DRY BUY] {name} {qty_str} @ {price} reason={reason}")
            return usd_free - Decimal(qty_str)*price, True

        try:
            res  = api_place_market(name,"BUY",qty_str, self.args.min_order_interval, self.args.max_retries, self.args.max_backoff)
            ok   = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail",{}).get("FilledAverPrice", price)))
            err  = "" if ok else res.get("ErrMsg","")
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
            body=""; 
            try: body=e.response.text
            except: pass
            self._log_trade(name,"BUY",qty_str,str(price),reason,False,f"{e} {body}")
            print(f"[BUY ERR] {name} {e} {body}")
            return usd_free, False

    def _sell(self, name: str, sym: Sym, price: Decimal, now_ts: float, reason: str) -> bool:
        if sym.qty<=0: return False
        if now_ts - sym.last_trade_ts < self.args.cooldown: return False
        if sym.hold_start_ts and now_ts - sym.hold_start_ts < self.args.min_hold_sec: return False

        qty = quantize_down(sym.qty, self.meta[name].amt_prec)
        if qty<=0: return False
        qty_str = f"{qty:.{self.meta[name].amt_prec}f}"

        trade_ret = 0.0
        cost = sym.entry_price if sym.entry_price else price
        eff_price = price*(Decimal("1.0")-FEE_RATE)
        if cost and cost>0:
            trade_ret = float((eff_price - cost)/cost)

        if self.args.dry_run:
            self._log_trade(name,"SELL",qty_str,str(price),"[DRY] "+reason,True,"")
            print(f"[DRY SELL] {name} {qty_str} @ {price} reason={reason}")
            self._update_reward(sym, trade_ret)
            sym.qty=Decimal("0"); sym.entry_price=None; sym.peak_after_entry=None; sym.last_trade_ts=now_ts; sym.hold_start_ts=None
            return True

        try:
            res  = api_place_market(name,"SELL",qty_str, self.args.min_order_interval, self.args.max_retries, self.args.max_backoff)
            ok   = bool(res.get("Success"))
            fill = Decimal(str(res.get("OrderDetail",{}).get("FilledAverPrice", price)))
            eff  = fill*(Decimal("1.0")-FEE_RATE)
            if sym.entry_price and sym.entry_price>0:
                trade_ret = float((eff - sym.entry_price)/sym.entry_price)
            self._log_trade(name,"SELL",qty_str,str(fill),reason,ok,"" if ok else res.get("ErrMsg",""))
            print(f"[SELL] {name} {qty_str} @ {fill} ok={ok} reason={reason}")
            self._update_reward(sym, trade_ret)
            sym.qty=Decimal("0"); sym.entry_price=None; sym.peak_after_entry=None; sym.last_trade_ts=now_ts; sym.hold_start_ts=None
            return ok
        except requests.HTTPError as e:
            body=""; 
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
        print("[规则] 仅现货 USD；费率 0.1%/笔；限流 + 退避；低频轮动。")

        tk = api_ticker_all(self.args.min_ticker_interval, self.args.max_retries, self.args.max_backoff)
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
                cur_min = int(loop_ts//60)
                if cur_min != last_minute:
                    trades_minute = 0
                    last_minute = cur_min

                tk = api_ticker_all(self.args.min_ticker_interval, self.args.max_retries, self.args.max_backoff)
                data = tk.get("Data",{})

                equity, usd_free, hold_val = self._portfolio_value(data)

                # Universe
                self.universe = self._select_universe(data)
                for name in self.universe:
                    if name not in self.syms:
                        self.syms[name]=Sym(name,self.meta[name],self.args.roc,self.args.slope,self.args.breakout)

                # candidates
                cands: List[Tuple[str,float,Decimal,Dict]] = []
                for name in self.universe:
                    d = data.get(name,{})
                    lp = d.get("LastPrice",None)
                    if lp is None: continue
                    p = Decimal(str(lp))
                    s = self.syms[name]
                    s.update(p)
                    if s.vol() < Decimal(self.args.vol_floor): continue
                    sc = self._score(s,p,d)
                    if sc > 0: cands.append((name,sc,p,d))

                cands.sort(key=lambda x:x[1], reverse=True)
                targets = cands[: self.args.topk]

                # exit
                for name, s in list(self.syms.items()):
                    if s.qty <= 0: continue
                    d  = data.get(name,{}); lp = d.get("LastPrice",None)
                    if lp is None: continue
                    p  = Decimal(str(lp))
                    if s.peak_after_entry is None or p > s.peak_after_entry: s.peak_after_entry = p

                    r = Decimal("0")
                    if s.entry_price and s.entry_price>0:
                        eff = p*(Decimal("1.0")-FEE_RATE)
                        r = (eff - s.entry_price)/s.entry_price
                    vol_i = s.vol()
                    stop_dist  = max(vol_i*Decimal(self.args.stop_mul),  Decimal(self.args.min_stop))  + ROUND_TRIP_FEE
                    take_dist  = max(vol_i*Decimal(self.args.take_mul),  Decimal(self.args.min_take))  + ROUND_TRIP_FEE
                    trail_dist = max(vol_i*Decimal(self.args.trail_mul), Decimal(self.args.min_trail)) + ROUND_TRIP_FEE

                    reasons=[]
                    if r <= -stop_dist: reasons.append("stop(fee-aware)")
                    if r >=  take_dist: reasons.append("take(fee-aware)")
                    if s.peak_after_entry and s.peak_after_entry>0:
                        dd = (p - s.peak_after_entry)/s.peak_after_entry
                        if dd <= -trail_dist: reasons.append("trail(fee-aware)")
                    if name not in [t[0] for t in targets]: reasons.append("drop_from_targets")

                    if reasons and trades_minute < self.args.max_trades_per_minute:
                        self._sell(name, s, p, loop_ts, ";".join(reasons))
                        trades_minute += 1

                # enter
                cash_for_buys = usd_free * Decimal(self.args.budget_perc)
                for name, sc, p, d in targets:
                    if trades_minute >= self.args.max_trades_per_minute: break
                    s = self.syms[name]
                    if s.qty > 0: continue
                    cash_for_buys, entered = self._buy(name, s, p, cash_for_buys, loop_ts, reason=f"score={sc:.4f}")
                    if entered: trades_minute += 1

                hold = {n:str(s.qty) for n,s in self.syms.items() if s.qty>0}
                print(f"[HB] top={[ (t[0], round(t[1],4)) for t in targets ]} hold={hold} trades/min={trades_minute}/{self.args.max_trades_per_minute}")

                if int(loop_ts) % 10 == 0:
                    self._save_state()

            except requests.HTTPError as e:
                body = ""
                try: body = e.response.text
                except: pass
                print("[HTTP]", e, body)
            except Exception as e:
                print("[ERR]", e)

            time.sleep(self.args.interval)

# ---------- 报告（回测/绩效汇总） ----------
def _safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def generate_report(hours: Optional[int]=None):
    """从 spot_trades.csv / equity 与当前资产，输出收益汇总。"""
    since_ts = None
    if hours is not None and hours > 0:
        since_ts = time.time() - hours*3600

    # 1) 初始资金
    try:
        info = api_exchange_info()
        init_usd = Decimal(str(info.get("InitialWallet",{}).get("USD","50000")))
    except Exception:
        init_usd = Decimal("50000")

    # 2) 解析成交日志（只看 ok=1 的真实成交）
    trades = []
    if os.path.exists(TRADES_LOG):
        with open(TRADES_LOG,"r",encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8: continue
                ts, pair, side, qty, price, reason, ok, err = parts[:8]
                if ok.strip() != "1": continue
                if since_ts and int(ts) < since_ts: continue
                trades.append({
                    "ts": int(ts), "pair": pair, "side": side, 
                    "qty": Decimal(qty), "price": Decimal(price)
                })

    # 3) 基于成交重建仓位与已实现收益（含费）
    realized = Decimal("0")
    wins = 0; losses = 0; n_sells = 0
    # 简单 FIFO/加权平均（只做多）
    pos: Dict[str, Dict[str, Decimal]] = {}  # pair -> {"qty":..., "avg_cost":...}

    for t in trades:
        pair = t["pair"]
        side = t["side"].upper()
        qty  = t["qty"]
        px   = t["price"]
        if side == "BUY":
            cost = px * (Decimal("1.0")+FEE_RATE)
            if pair not in pos: pos[pair] = {"qty": Decimal(0), "avg": Decimal(0)}
            old_q = pos[pair]["qty"]; old_avg = pos[pair]["avg"]
            new_q = old_q + qty
            new_avg = (old_q*old_avg + qty*cost) / (new_q if new_q>0 else Decimal(1))
            pos[pair]["qty"] = new_q; pos[pair]["avg"] = new_avg
        elif side == "SELL":
            n_sells += 1
            eff = px * (Decimal("1.0")-FEE_RATE)
            if pair not in pos or pos[pair]["qty"] <= 0:
                # 假设空仓卖出（不应发生），略过
                continue
            use_qty = min(qty, pos[pair]["qty"])
            pnl = use_qty * (eff - pos[pair]["avg"])
            realized += pnl
            pos[pair]["qty"] -= use_qty
            # 胜负统计
            if pnl > 0: wins += 1
            elif pnl < 0: losses += 1

    # 4) 未实现收益（按当前价格估值，考虑潜在卖出费）
    tk = api_ticker_all(2.5, 3, 10.0)
    data = tk.get("Data",{})
    unreal = Decimal("0")
    for pair, st in pos.items():
        q = st["qty"]
        if q <= 0: continue
        last = Decimal(str(data.get(pair,{}).get("LastPrice","0")))
        if last <= 0: continue
        eff = last * (Decimal("1.0")-FEE_RATE)  # 假设此刻卖出
        unreal += q * (eff - st["avg"])

    # 5) 当前权益（从 API 读取）
    bal = api_balance(5.0, 3, 10.0)
    usd_free = Decimal(str(bal.get("Wallet",{}).get("USD",{}).get("Free","0")))
    # 估值其它币（Wallet 只有币种余额，不含价格；用 ticker 估）
    hold_val = Decimal("0")
    for coin, w in bal.get("Wallet",{}).items():
        if coin == "USD": continue
        free = Decimal(str(w.get("Free","0")))
        if free <= 0: continue
        pair = f"{coin}/USD"
        px = Decimal(str(data.get(pair,{}).get("LastPrice","0")))
        if px > 0:
            hold_val += free * px
    equity_now = usd_free + hold_val

    # 6) 权益曲线变化（如有 equity log）
    eq_first = None; eq_last = None
    if os.path.exists(EQUITY_LOG):
        with open(EQUITY_LOG,"r",encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                ts, eq, usd, hold, dd = line.strip().split(",")
                if since_ts and int(ts) < since_ts:
                    continue
                val = Decimal(eq)
                if eq_first is None: eq_first = val
                eq_last = val

    # 7) 汇总输出
    total_pnl = realized + unreal
    # 基准：若有 equity log，则以 eq_first 作为起点，否则以初始钱包
    base_equity = eq_first if eq_first is not None else init_usd
    roi = ((equity_now - base_equity) / base_equity * Decimal(100)) if base_equity > 0 else Decimal("0")

    print("\n================= 绩效/回测报告 =================")
    if hours: print(f"区间：最近 {hours} 小时")
    print(f"初始资金（参考）：${base_equity:.2f}")
    print(f"当前权益（API）：  ${equity_now:.2f}")
    print(f"已实现收益：       ${realized:.2f}")
    print(f"未实现收益：       ${unreal:.2f}")
    print(f"总收益（估）：     ${total_pnl:.2f}")
    print(f"胜率：             {wins}/{max(1,n_sells)} = {wins/max(1,n_sells)*100:.2f}%")
    if eq_first is not None and eq_last is not None:
        print(f"权益曲线变动：     ${eq_first:.2f} → ${eq_last:.2f}（Δ={eq_last-eq_first:.2f}）")
    print(f"ROI（相对基准）：  {roi:.2f}%")
    print("=================================================\n")

# ---------- CLI ----------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Roostoo OMEGA-SPOT (Rate-Limited + Report Edition)")
    # 信号
    ap.add_argument("--roc", type=int, default=36)
    ap.add_argument("--slope", type=int, default=24)
    ap.add_argument("--breakout", type=int, default=72)
    # 资金/风控
    ap.add_argument("--risk-per-trade", type=float, default=0.02)
    ap.add_argument("--budget-perc", type=float, default=0.8)
    ap.add_argument("--per-asset-cap", type=float, default=0.25)
    ap.add_argument("--vol-floor", type=float, default=0.002)
    ap.add_argument("--min-stop", type=float, default=0.01)
    ap.add_argument("--min-take", type=float, default=0.02)
    ap.add_argument("--min-trail", type=float, default=0.01)
    ap.add_argument("--stop-mul", type=float, default=1.0)
    ap.add_argument("--take-mul", type=float, default=2.0)
    ap.add_argument("--trail-mul", type=float, default=1.5)
    ap.add_argument("--max-drawdown", type=float, default=0.30)
    ap.add_argument("--min-hold-sec", type=int, default=300)
    # Universe
    ap.add_argument("--universe-size", type=int, default=30)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min-unit-volume", type=float, default=2e7)
    # 节奏与限流
    ap.add_argument("--interval", type=int, default=10)
    ap.add_argument("--max-trades-per-minute", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=120)
    ap.add_argument("--min-ticker-interval", type=float, default=float(os.getenv("MIN_TICKER_INTERVAL", "2.5")))
    ap.add_argument("--min-balance-interval", type=float, default=5.0)
    ap.add_argument("--min-order-interval", type=float, default=2.0)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--max-backoff", type=float, default=10.0)
    # 报告
    ap.add_argument("--report", action="store_true", help="只输出回测/绩效报告并退出")
    ap.add_argument("--report-h", type=int, default=None, help="报告窗口（最近 N 小时）")
    # 观测
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    print("[环境] 使用 API_KEY：", "来自环境变量" if os.getenv("ROOSTOO_API_KEY") else "脚本内置示例（仅演示）")
    if args.report:
        generate_report(args.report_h)
        return
    print("[限流] min_ticker_interval=%.2fs, max_retries=%d" % (args.min_ticker_interval, args.max_retries))
    bot = SpotRotationBot(args)
    bot.run()

if __name__ == "__main__":
    main()
