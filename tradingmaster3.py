# === 一键部署 & 启动（可直接粘贴回车）========================================
set -e
mkdir -p ~/Trading-bot && cd ~/Trading-bot

# 1) 环境变量（按你的要求直接内置）
export ROOSTOO_BASE_URL="https://mock-api.roostoo.com"
export ROOSTOO_API_KEY="yN6cF2VtW9xJ4qPbeK1mZ7YruA3dH0GiP5oL8wQnrR2tC6XfD8vB1kMeZ4gU7SaJ"
export ROOSTOO_SECRET_KEY="tG5yH7uJ9iK1oL3pA6sD8fF0gH2jK4lZ7xC9vB1nM3qW5eR7tY9uI1oP3aS5"
# 可选的外部情绪/链上数据KEY（没有也能跑，缺失则相关分数按0处理）
export HORUS_API_KEY="${HORUS_API_KEY:-}"

# 2) 依赖
python3 -m pip install -q --user requests python-dotenv pandas numpy scipy tqdm >/dev/null

# 3) 交易脚本（增强但不改策略内核）
cat > maxprofit_bot.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, time, json, hmac, hashlib, sys, random
from decimal import Decimal, ROUND_DOWN, getcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import requests
import pandas as pd
from scipy.stats import percentileofscore
from tqdm import tqdm
getcontext().prec = 28
load_dotenv()
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------- 基本配置（保持策略不变） ----------------
ROOSTOO_BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")
ROOSTOO_API_KEY = os.getenv("ROOSTOO_API_KEY")
ROOSTOO_SECRET_KEY = os.getenv("ROOSTOO_SECRET_KEY")
HORUS_API_KEY = os.getenv("HORUS_API_KEY")
HORUS_BASE_URL = "https://api.horusdata.xyz/v1"

INITIAL_CAPITAL = Decimal("10000.0")
MAX_POSITIONS = 8
OPEN_THRESHOLD = 80
CLOSE_THRESHOLD = 60
COOLDOWN_SEC = 180
MAX_DAILY_TRADES = 10
MAX_DRAWNDOWN = Decimal("0.15")
WARNING_DRAWNDOWN = Decimal("0.12")
MIN_HOLD_SEC = 300
FIXED_TAKE_PROFIT = Decimal("0.15")

FEE = Decimal("0.001")  # 手续费 0.1%/笔（仅用于打印/报表，不改变你的信号与阈值）
TRADES_LOG = "max_profit_trades.csv"
EQUITY_LOG = "max_profit_equity.csv"
PAUSE_FILE = "PAUSE"

S = requests.Session()

# ---------- 带退避的请求（429/5xx 自动重试，不改变策略逻辑） ----------
def _get(url, *, params=None, headers=None, kind="GET", min_interval=2.5, retries=6):
    time.sleep(min_interval)
    for a in range(retries):
        r = S.get(url, params=params, headers=headers, timeout=20)
        if r.status_code in (429, 500, 502, 503, 504):
            sl = min(2**a, 10.0) + random.random()
            print(f"[RATE] {r.status_code} {kind} backoff {sl:.2f}s")
            time.sleep(sl); continue
        r.raise_for_status(); return r
    r.raise_for_status()

def _post(url, *, data=None, headers=None, kind="POST", min_interval=2.0, retries=6):
    time.sleep(min_interval)
    for a in range(retries):
        r = S.post(url, data=data, headers=headers, timeout=20)
        if r.status_code in (429, 500, 502, 503, 504):
            sl = min(2**a, 10.0) + random.random()
            print(f"[RATE] {r.status_code} {kind} backoff {sl:.2f}s")
            time.sleep(sl); continue
        r.raise_for_status(); return r
    r.raise_for_status()

# ---------------- 数据类 ----------------
@dataclass
class PairMeta:
    price_prec: int
    amt_prec: int
    mini_order: Decimal
    liquidity_rank: int

@dataclass
class Position:
    pair: str
    entry_price: Decimal
    quantity: Decimal
    entry_ts: float
    current_price: Decimal
    max_price: Decimal
    signal_score: float
    target_position: Decimal

@dataclass
class SignalScores:
    on_chain: float
    sentiment: float
    technical: float
    volatility: float
    total: float
    confidence: float

# ---------------- API工具 ----------------
def _now_ms() -> int: return int(time.time()*1000)

def _sign_payload(payload: Dict) -> Tuple[Dict, str]:
    payload["timestamp"] = _now_ms()
    keys = sorted(payload.keys())
    params_str = "&".join(f"{k}={payload[k]}" for k in keys)
    sig = hmac.new(ROOSTOO_SECRET_KEY.encode("utf-8"), params_str.encode("utf-8"), hashlib.sha256).hexdigest()
    headers = {"RST-API-KEY": ROOSTOO_API_KEY, "MSG-SIGNATURE": sig, "Content-Type":"application/x-www-form-urlencoded"}
    return headers, params_str

def roostoo_api_balance() -> Dict:
    h, q = _sign_payload({})
    r = _get(f"{ROOSTOO_BASE_URL}/v3/balance", params=dict(x.split("=") for x in q.split("&")), headers=h, kind="balance", min_interval=5.0)
    return r.json()

def roostoo_api_exchange_info() -> Dict:
    return _get(f"{ROOSTOO_BASE_URL}/v3/exchangeInfo", kind="exInfo", min_interval=5.0).json()

def roostoo_api_ticker(pair: Optional[str]=None) -> Dict:
    p = {"timestamp": _now_ms()}
    if pair: p["pair"] = pair
    return _get(f"{ROOSTOO_BASE_URL}/v3/ticker", params=p, kind="ticker", min_interval=2.5).json()

def roostoo_api_place_order(pair: str, side: str, quantity: str, price: Optional[str]=None) -> Dict:
    payload={"pair":pair,"side":side.upper(),"type":"LIMIT" if price else "MARKET","quantity":quantity}
    if price: payload["price"]=price
    h, body = _sign_payload(payload)
    return _post(f"{ROOSTOO_BASE_URL}/v3/place_order", data=body, headers=h, kind="place_order", min_interval=2.0).json()

def horus_get(path: str, params: Dict) -> Dict:
    if not HORUS_API_KEY:  # 没有外部KEY时，返回空，让分数为0（不改策略分配，只是缺省为0）
        return {}
    h={"X-API-KEY": HORUS_API_KEY}
    try:
        return _get(f"{HORUS_BASE_URL}/{path}", params=params, headers=h, kind=f"horus:{path}", min_interval=0.0).json()
    except Exception as e:
        print(f"[Horus] {path} 获取失败：{e}")
        return {}

def horus_api_on_chain(pair:str)->Dict: return horus_get("on-chain", {"pair":pair,"period":"7d"})
def horus_api_sentiment(pair:str)->Dict: return horus_get("sentiment", {"pair":pair,"period":"24h"})
def horus_api_volatility(pair:str)->Dict: return horus_get("volatility", {"pair":pair,"period":"90d"})

# ---------------- 技术指标（保持不变） ----------------
def calculate_ema(prices: List[Decimal], period: int) -> Decimal:
    if len(prices) < period: return prices[-1] if prices else Decimal("0")
    alpha = Decimal("2")/(Decimal(period)+Decimal("1"))
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha*price + (Decimal(1)-alpha)*ema
    return ema

def calculate_rsi(prices: List[Decimal], period:int=14)->Decimal:
    if len(prices) < period+1: return Decimal("50")
    deltas=[prices[i]-prices[i-1] for i in range(1,len(prices))]
    gains=[d for d in deltas if d>0]; losses=[-d for d in deltas if d<0]
    avg_gain=(sum(gains[:period])/Decimal(period)) if gains else Decimal("0")
    avg_loss=(sum(losses[:period])/Decimal(period)) if losses else Decimal("0")
    if avg_loss==0: return Decimal("100")
    rs=avg_gain/avg_loss
    return Decimal("100")-(Decimal("100")/(Decimal("1")+rs))

# ---------------- 元信息与信号（保持原意） ----------------
def get_pair_meta()->Dict[str,PairMeta]:
    info=roostoo_api_exchange_info()
    usd=[]
    for name,d in info.get("TradePairs",{}).items():
        if name.endswith("/USD") and d.get("CanTrade",True):
            usd.append((name, d.get("PricePrecision",2), d.get("AmountPrecision",6),
                        Decimal(str(d.get("MiniOrder","1.0"))), d.get("LiquidityRank",999)))
    usd.sort(key=lambda x:x[4])
    top50=usd[:50]
    return {n:PairMeta(price_prec=pp, amt_prec=ap, mini_order=mo, liquidity_rank=lr) for n,pp,ap,mo,lr in top50}

def calculate_signal_scores(pair:str, price_history:List[Decimal])->SignalScores:
    on_chain_score=0.0
    try:
        oc=horus_api_on_chain(pair)
        if oc:
            if oc.get("largeTransferRatio",0.0) >= 60 or oc.get("holderGrowth7d",0.0) >= 15 or oc.get("tvlGrowth24h",0.0) >= 10:
                on_chain_score=25.0
    except Exception as e:
        print(f"[链上信号] {pair} 失败：{e}")

    sentiment_score=0.0
    try:
        se=horus_api_sentiment(pair)
        if se:
            if se.get("bullishRatio",0.0) >= 70 and se.get("confidence",0.0) >= 0.8:
                sentiment_score += 25.0
            if se.get("keywordGrowth",0.0) >= 100:
                sentiment_score += 10.0
            sentiment_score=min(sentiment_score,35.0)
    except Exception as e:
        print(f"[舆情信号] {pair} 失败：{e}")

    technical_score=0.0
    if len(price_history)>=120:
        ema20,ema60,ema120 = calculate_ema(price_history,20), calculate_ema(price_history,60), calculate_ema(price_history,120)
        if ema20>ema60>ema120: technical_score += 20.0
        price_30d_high=max(price_history[-30:])
        current_price=price_history[-1]
        # 允许0.5%误差
        if current_price >= price_30d_high*Decimal("0.995"):
            volume_ratio=1.6
            if volume_ratio>=1.5: technical_score += 15.0
        if calculate_rsi(price_history)<70: technical_score += 5.0

    volatility_score=0.0
    try:
        vd=horus_api_volatility(pair)
        cur=vd.get("volatility24h",0.0); window=vd.get("volatility90d",[])
        if window:
            pct=percentileofscore(window, cur)
            if 30<=pct<=70: volatility_score=10.0
    except Exception as e:
        print(f"[波动率信号] {pair} 失败：{e}")

    confidence=(on_chain_score/25)+(sentiment_score/35)
    return SignalScores(on_chain=on_chain_score, sentiment=sentiment_score, technical=technical_score,
                        volatility=volatility_score, total=on_chain_score+sentiment_score+technical_score+volatility_score,
                        confidence=confidence)

def calculate_position_size(pair:str, scores:SignalScores, available_usd:Decimal, meta:PairMeta)->Tuple[Optional[str],Optional[str]]:
    tk=roostoo_api_ticker(pair)
    cur=Decimal(str(tk.get("Data",{}).get(pair,{}).get("LastPrice",0)))
    if cur<=0: return None,None
    base=INITIAL_CAPITAL/Decimal(str(MAX_POSITIONS))
    try:
        vol=horus_api_volatility(pair); curv=vol.get("volatility24h",0.0); window=vol.get("volatility90d",[])
        pct=percentileofscore(window,curv) if window else 50
    except: pct=50
    adj=(scores.total/110)*(1-pct/100); adj=max(0.1, min(0.5, adj))
    conf=min(2.0, scores.confidence)
    target=base*Decimal(str(adj))*Decimal(str(conf))
    target=min(target, available_usd*Decimal("0.3"))
    target=max(target, INITIAL_CAPITAL*Decimal("0.05"))
    bid=cur*Decimal("0.999")
    bid_s=f"{bid:.{meta.price_prec}f}"
    qty=(target/bid).quantize(Decimal("10")**(-meta.amt_prec), rounding=ROUND_DOWN)
    if qty*bid < meta.mini_order:
        qty=(meta.mini_order/bid).quantize(Decimal("10")**(-meta.amt_prec), rounding=ROUND_DOWN)
        if qty*bid < meta.mini_order: qty += Decimal("10")**(-meta.amt_prec)
        if qty*bid > available_usd: return None,None
    return f"{qty:.{meta.amt_prec}f}", bid_s

def check_exit_conditions(position:Position, cur_scores:SignalScores, price_history:List[Decimal])->Tuple[bool,str]:
    cur = price_history[-1]; position.current_price=cur
    if cur > position.max_price: position.max_price = cur
    score_based_tp = Decimal(str(position.signal_score * 0.001))
    take_profit = max(FIXED_TAKE_PROFIT, score_based_tp)
    pr = (cur - position.entry_price)/position.entry_price
    if pr >= take_profit: return True, f"止盈触发（收益{pr*100:.2f}%≥目标{take_profit*100:.2f}%）"
    if pr < Decimal("0.05"):
        stop_price = position.entry_price*Decimal("0.95")
    elif pr < Decimal("0.10"):
        stop_price = position.entry_price + (cur-position.entry_price)*Decimal("0.5")
    else:
        stop_price = position.max_price*Decimal("0.90")
    if cur <= stop_price: return True, f"止损触发（价格{cur:.4f}≤止损价{stop_price:.4f}）"
    if cur_scores.total < CLOSE_THRESHOLD: return True, f"信号反转（得分{cur_scores.total:.1f}<阈值{CLOSE_THRESHOLD}）"
    if time.time()-position.entry_ts < MIN_HOLD_SEC: return False, "持仓时间不足"
    return False, "无退出条件"

# ---------------- 主策略（打印每笔、文件开关暂停、报表） ----------------
class MaxProfitStrategy:
    def __init__(self):
        self.pair_meta = get_pair_meta()
        self.positions: Dict[str, Position] = {}
        self.price_history: Dict[str, List[Decimal]] = {p:[] for p in self.pair_meta.keys()}
        self.daily_trades = 0
        self.last_trade_date = time.strftime("%Y-%m-%d")
        self.equity_peak = INITIAL_CAPITAL
        self.available_usd = INITIAL_CAPITAL
        self._init_logs()

    def _init_logs(self):
        if not os.path.exists(TRADES_LOG):
            with open(TRADES_LOG,"w",encoding="utf-8") as f:
                f.write("ts,pair,side,quantity,price,signal_score,reason,success,error\n")
        if not os.path.exists(EQUITY_LOG):
            with open(EQUITY_LOG,"w",encoding="utf-8") as f:
                f.write("ts,equity,available_usd,hold_value,max_drawdown\n")

    def _update_price_history(self):
        t=roostoo_api_ticker()
        for p in self.pair_meta.keys():
            px=t.get("Data",{}).get(p,{}).get("LastPrice")
            if px:
                arr=self.price_history[p]; arr.append(Decimal(str(px)))
                if len(arr)>200: arr.pop(0)

    def _calculate_equity(self)->Tuple[Decimal,Decimal,Decimal]:
        bal=roostoo_api_balance()
        usd_free=Decimal(str(bal.get("Wallet",{}).get("USD",{}).get("Free",0)))
        self.available_usd=usd_free
        hold=Decimal("0")
        for p,pos in self.positions.items():
            tk=roostoo_api_ticker(p)
            cur=Decimal(str(tk.get("Data",{}).get(p,{}).get("LastPrice",0)))
            hold += pos.quantity*cur
            pos.current_price=cur
            if cur>pos.max_price: pos.max_price=cur
        eq=usd_free+hold
        if eq>self.equity_peak: self.equity_peak=eq
        mdd=(eq-self.equity_peak)/self.equity_peak if self.equity_peak>0 else Decimal("0")
        with open(EQUITY_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{eq},{usd_free},{hold},{mdd}\n")
        return eq, hold, mdd

    def _reset_daily_trades(self):
        d=time.strftime("%Y-%m-%d")
        if d!=self.last_trade_date:
            self.daily_trades=0; self.last_trade_date=d

    def _log_trade(self, pair, side, qty, price, score, reason, ok, err=""):
        with open(TRADES_LOG,"a",encoding="utf-8") as f:
            f.write(f"{int(time.time())},{pair},{side},{qty},{price},{score},{reason},{int(ok)},{err}\n")

    def _execute_buy(self, pair:str, scores:SignalScores):
        if self.daily_trades>=MAX_DAILY_TRADES: 
            print(f"[买入] {pair} 达到当日上限{MAX_DAILY_TRADES}"); return
        if pair in self.positions:
            print(f"[买入] {pair} 已持有，跳过"); return
        ts=self._get_last_trade_ts(pair)
        if ts and time.time()-ts<COOLDOWN_SEC:
            print(f"[买入] {pair} 冷却中 {int(COOLDOWN_SEC-(time.time()-ts))}s"); return
        if len(self.positions)>=MAX_POSITIONS:
            print(f"[买入] 持仓满 {MAX_POSITIONS}"); return

        meta=self.pair_meta[pair]
        qty_s, price_s = calculate_position_size(pair, scores, self.available_usd, meta)
        if not qty_s or not price_s:
            print(f"[买入] {pair} 计算仓位失败"); return

        try:
            print(f"[买入] {pair} 信号{scores.total:.1f} 下单 {qty_s}@{price_s}")
            res=roostoo_api_place_order(pair,"BUY",qty_s,price_s)
            ok=res.get("Success",False)
            if ok:
                entry=Decimal(price_s); q=Decimal(qty_s)
                notional=q*entry; fee=notional*FEE
                self.positions[pair]=Position(pair=pair, entry_price=entry, quantity=q, entry_ts=time.time(),
                                              current_price=entry, max_price=entry, signal_score=scores.total,
                                              target_position=q*entry)
                self.daily_trades+=1
                self._log_trade(pair,"BUY",qty_s,price_s,scores.total,"信号共振开仓",True)
                print(f"  ✅ 成功：名义额=${notional:.2f} 手续费=${fee:.2f}")
            else:
                err=res.get("ErrMsg","未知错误")
                self._log_trade(pair,"BUY",qty_s,price_s,scores.total,"信号共振开仓",False,err)
                print(f"  ❌ 失败：{err}")
        except Exception as e:
            self._log_trade(pair,"BUY",qty_s,price_s,scores.total,"信号共振开仓",False,str(e))
            print(f"  ❌ 异常：{e}")

    def _execute_sell(self, pair:str, position:Position, reason:str):
        if time.time()-position.entry_ts<MIN_HOLD_SEC:
            print(f"[卖出] {pair} 持仓未满{MIN_HOLD_SEC}s"); return
        if self.daily_trades>=MAX_DAILY_TRADES:
            print(f"[卖出] {pair} 达到当日上限"); return
        tk=roostoo_api_ticker(pair)
        cur=Decimal(str(tk.get("Data",{}).get(pair,{}).get("LastPrice",0)))
        meta=self.pair_meta[pair]
        ask=(cur*Decimal("1.001"))
        ask_s=f"{ask:.{meta.price_prec}f}"
        qty_s=f"{position.quantity:.{meta.amt_prec}f}"

        try:
            print(f"[卖出] {pair} 理由:{reason} 下单 {qty_s}@{ask_s}")
            res=roostoo_api_place_order(pair,"SELL",qty_s,ask_s)
            ok=res.get("Success",False)
            if ok:
                fill=Decimal(str(res.get("OrderDetail",{}).get("FilledAverPrice", ask_s)))
                q=position.quantity
                gross_out=q*fill; fee_out=gross_out*FEE
                cost_in=q*position.entry_price*(Decimal(1)+FEE)
                pnl = gross_out - fee_out - cost_in
                ret = pnl / cost_in * Decimal(100) if cost_in>0 else Decimal(0)
                self._log_trade(pair,"SELL",qty_s,str(fill),position.signal_score,reason,True)
                self.daily_trades+=1
                del self.positions[pair]
                print(f"  ✅ 成功：PNL=${pnl:.2f}  收益率={ret:.2f}%  手续费(卖)=${fee_out:.2f}")
            else:
                err=res.get("ErrMsg","未知错误")
                self._log_trade(pair,"SELL",qty_s,ask_s,position.signal_score,reason,False,err)
                print(f"  ❌ 失败：{err}")
        except Exception as e:
            self._log_trade(pair,"SELL",qty_s,ask_s,position.signal_score,reason,False,str(e))
            print(f"  ❌ 异常：{e}")

    def _get_last_trade_ts(self,pair:str)->Optional[float]:
        try:
            df=pd.read_csv(TRADES_LOG)
            x=df[df["pair"]==pair].sort_values("ts",ascending=False)
            if not x.empty: return float(x.iloc[0]["ts"])
        except Exception: pass
        return None

    def _check_portfolio_risk(self)->bool:
        _,_,mdd=self._calculate_equity()
        if mdd <= -MAX_DRAWNDOWN:
            print(f"[熔断] 回撤{mdd*100:.2f}% ≥ {MAX_DRAWNDOWN*100:.2f}%，清仓")
            for p,pos in list(self.positions.items()):
                self._execute_sell(p,pos,f"熔断清仓({mdd*100:.2f}%)")
            return True
        elif mdd <= -WARNING_DRAWNDOWN:
            print(f"[预警] 回撤{mdd*100:.2f}% ≥ {WARNING_DRAWNDOWN*100:.2f}%，减半仓位")
            half=list(self.positions.items())[:max(1,len(self.positions)//2)]
            for p,pos in half:
                self._execute_sell(p,pos,f"预警减持({mdd*100:.2f}%)")
            return True
        return False

    def run(self):
        print("=== 多维度信号融合轮动策略启动 ===")
        print(f"初始资金：{INITIAL_CAPITAL} USD | 最大持仓：{MAX_POSITIONS} | 开仓阈值：{OPEN_THRESHOLD}")
        print("="*56)
        while True:
            # 文件开关暂停
            if os.path.exists(PAUSE_FILE):
                print("[PAUSE] 检测到 PAUSE 文件，策略暂停中…(每30s检查一次)")
                for _ in range(30): 
                    if not os.path.exists(PAUSE_FILE): break
                    time.sleep(1)
                continue
            try:
                self._reset_daily_trades()
                eq, hold, mdd = self._calculate_equity()
                print(f"[状态] Equity={eq:.2f}  USD可用={self.available_usd:.2f}  持仓={hold:.2f}  MaxDD={mdd*100:.2f}%")
                if self._check_portfolio_risk():
                    time.sleep(60); continue
                self._update_price_history()

                # 退出检查
                for p,pos in list(self.positions.items()):
                    hist=self.price_history[p]
                    if len(hist)<30: continue
                    scores=calculate_signal_scores(p, hist)
                    go, why = check_exit_conditions(pos, scores, hist)
                    if go: self._execute_sell(p,pos,why)

                # 开仓机会
                if len(self.positions)<MAX_POSITIONS and self.available_usd >= INITIAL_CAPITAL*Decimal("0.05"):
                    cands=[]
                    for p in tqdm(self.pair_meta.keys(), desc="计算信号得分", leave=False):
                        hist=self.price_history[p]
                        if len(hist)<120: continue
                        s=calculate_signal_scores(p, hist)
                        if s.total>=OPEN_THRESHOLD: cands.append((p,s))
                    cands.sort(key=lambda x:x[1].total, reverse=True)
                    for p,s in cands:
                        if len(self.positions)>=MAX_POSITIONS or self.daily_trades>=MAX_DAILY_TRADES: break
                        self._execute_buy(p,s)

                print("[等待] 下次轮询：15 分钟")
                time.sleep(15*60)
            except Exception as e:
                print(f"[主循环异常] {e}")
                time.sleep(60)

# ---------------- 报表：读交易日志计算 实现/未实现/总收益 ----------------
def report():
    if not os.path.exists(TRADES_LOG):
        print("尚无成交记录"); return
    df=pd.read_csv(TRADES_LOG)
    df=df[df["success"]==1] if "success" in df.columns else df
    df=df.sort_values("ts")
    realized=Decimal("0")
    pos={}
    # 拉一次全市场价格
    data=roostoo_api_ticker().get("Data",{})
    for _,r in df.iterrows():
        pair=r["pair"]; side=r["side"].upper()
        q=Decimal(str(r["quantity"])); px=Decimal(str(r["price"]))
        if side=="BUY":
            cost=q*px*(Decimal(1)+FEE)
            st=pos.get(pair, {"q":Decimal("0"),"avg":Decimal("0")})
            newq=st["q"]+q
            st["avg"] = (st["avg"]*st["q"] + cost)/newq if newq>0 else Decimal("0")
            st["q"]=newq; pos[pair]=st
        elif side=="SELL":
            st=pos.get(pair); 
            if not st or st["q"]<=0: continue
            use=min(q, st["q"])
            eff_out = px*(Decimal(1)-FEE)
            realized += use*(eff_out - st["avg"])
            st["q"]-=use; pos[pair]=st
    unreal=Decimal("0")
    for p,st in pos.items():
        if st["q"]<=0: continue
        cur=Decimal(str(data.get(p,{}).get("LastPrice","0") or "0"))
        if cur>0:
            eff_out=cur*(Decimal(1)-FEE)
            unreal += st["q"]*(eff_out - st["avg"])
    total=realized+unreal
    print("\n===== 报告 =====")
    print(f"已实现PnL: ${realized:.2f}")
    print(f"未实现PnL: ${unreal:.2f}")
    print(f"总计PnL  : ${total:.2f}")
    print("================\n")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser("maxprofit_bot")
    ap.add_argument("--report", action="store_true", help="仅输出报表后退出")
    args=ap.parse_args()
    if not ROOSTOO_API_KEY or not ROOSTOO_SECRET_KEY:
        raise SystemExit("缺少 ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY")
    if args.report:
        report(); sys.exit(0)
    MaxProfitStrategy().run()
PY
chmod +x maxprofit_bot.py

# 4) 管理脚本：start/stop/pause/resume/tail/report/status
cat > maxprofit_manage.sh <<'SH'
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PIDF=bot.pid
case "$1" in
  start)
    rm -f PAUSE
    nohup python3 -u maxprofit_bot.py > run.log 2>&1 & echo $! > "$PIDF"
    echo "🔥 启动完成 PID=$(cat $PIDF) 日志=run.log"
    ;;
  stop)
    if [ -f "$PIDF" ]; then kill "$(cat $PIDF)" 2>/dev/null || true; rm -f "$PIDF"; echo "✅ 已停止"; else echo "未运行"; fi
    ;;
  pause)  touch PAUSE; echo "⏸️ 已暂停（删除 PAUSE 恢复）";;
  resume) rm -f PAUSE; echo "▶️ 已恢复";;
  tail)   tail -n 200 -f run.log;;
  status)
    if [ -f "$PIDF" ] && ps -p "$(cat $PIDF)" >/dev/null 2>&1; then
      echo "✅ 运行中 PID=$(cat $PIDF)"; else echo "❌ 未运行"; fi
    [ -f PAUSE ] && echo "⏸️  当前状态：PAUSE 生效"
    ;;
  report)
    python3 maxprofit_bot.py --report
    ;;
  *)
    echo "用法: $0 {start|stop|pause|resume|tail|status|report}"; exit 1;;
esac
SH
chmod +x maxprofit_manage.sh

# 5) 重启并实时看日志
./maxprofit_manage.sh stop || true
./maxprofit_manage.sh start
./maxprofit_manage.sh tail
# ======================================================================
