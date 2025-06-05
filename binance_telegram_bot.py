import time
import requests
import pandas as pd
import openai
import config
import hmac
import hashlib
from urllib.parse import urlencode

client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

##########################
# BINANCE SIGNED REQUESTS#
##########################
def get_binance_signed_request(endpoint, params, api_key, api_secret, base_url):
    params['timestamp'] = int(time.time() * 1000)
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MBX-APIKEY': api_key}
    url = f'{base_url}{endpoint}?{query_string}&signature={signature}'
    response = requests.get(url, headers=headers)
    return response.json()

def get_spot_balances():
    endpoint = '/api/v3/account'
    params = {}
    data = get_binance_signed_request(endpoint, params, config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY, 'https://api.binance.com')
    nonzero = [a for a in data['balances'] if float(a['free']) > 0.0001 or float(a['locked']) > 0.0001]
    assets = []
    for a in nonzero:
        assets.append({
            'asset': a['asset'],
            'free': float(a['free']),
            'locked': float(a['locked'])
        })
    return assets

def get_futures_positions():
    endpoint = '/fapi/v2/positionRisk'
    params = {}
    data = get_binance_signed_request(endpoint, params, config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY, 'https://fapi.binance.com')
    open_positions = [p for p in data if abs(float(p['positionAmt'])) > 0.0001]
    return open_positions

def get_futures_balance():
    endpoint = '/fapi/v2/account'
    params = {}
    data = get_binance_signed_request(endpoint, params, config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY, 'https://fapi.binance.com')
    return float(data['totalWalletBalance'])
def get_binance_klines(symbol="BTCUSDT", interval="1h", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades', 'taker_base',
        'taker_quote', 'ignore'
    ])
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calc_ema(prices, period=21):
    return round(prices.ewm(span=period, adjust=False).mean().iloc[-1], 2)

def calc_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_val = macd.iloc[-1]
    signal_val = signal_line.iloc[-1]
    hist = macd_val - signal_val
    return round(macd_val,2), round(signal_val,2), round(hist,2)

def calc_bollinger(prices, period=20):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (2 * std)
    lower = sma - (2 * std)
    return round(upper.iloc[-1],2), round(lower.iloc[-1],2), round(sma.iloc[-1],2)

def ai_analysis(symbol, interval, closes, rsi, ema, macd, macd_signal, macd_hist, boll_upper, boll_lower, boll_mid):
    prompt = f"""
Aşağıda {symbol} için son {len(closes)} {interval} periyot kapanış fiyatı verileri ve temel teknik göstergeler bulunmaktadır:
Fiyatlar: {', '.join([str(round(p,2)) for p in closes])}
Son fiyat: {closes[-1]}

RSI(14): {rsi}
EMA(21): {ema}
MACD: {macd}, MACD Sinyal: {macd_signal}, MACD Histogram: {macd_hist}
Bollinger Bantları (20): Üst {boll_upper}, Alt {boll_lower}, Orta {boll_mid}

Sen bir kripto teknik analiz uzmanısın. Göstergeleri birlikte değerlendir ve KISA KISA gerekçe göstererek büyük harflerle kesin bir AL, SAT veya BEKLE önerisi ver. Lütfen sonuçta sadece öneri ve ardından bir iki cümlelik kısa teknik neden belirt.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen bir kripto para teknik analiz uzmanısın."},
            {"role": "user", "content": prompt}
        ]
    )
    reply = response.choices[0].message.content
    return reply.strip()

import re

def split_message_smart(message, max_len=4000):
    """Uzun mesajı cümle/satır sonlarında bölerek parçalara ayırır."""
    parts = []
    while len(message) > max_len:
        # Sınırdan önce en son nokta, ünlem, soru işareti veya satır sonunu bul
        split_pos = max(
            [message.rfind(x, 0, max_len) for x in ['. ', '! ', '? ', '\n']]
        )
        if split_pos <= 0:
            split_pos = max_len
        else:
            split_pos += 1
        parts.append(message[:split_pos].strip())
        message = message[split_pos:]
    if message.strip():
        parts.append(message.strip())
    return parts

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    MAX_LEN = 4000
    parts = split_message_smart(message, MAX_LEN)
    for idx, part in enumerate(parts):
        data = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": part,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, data=data)
        print(f"Telegram mesajı {idx+1}/{len(parts)} gönderildi, kod: {resp.status_code}")
        if resp.status_code != 200:
            print("Telegram Hata:", resp.status_code, resp.text)

def analyze_and_report():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # Coin listesi (istediğiniz gibi güncelleyebilirsiniz)
    intervals = ["1h", "4h", "1d"]  # Saatlik, 4 saatlik, günlük

    for symbol in symbols:
        per_symbol_messages = []
        for interval in intervals:
            try:
                data = get_binance_klines(symbol=symbol, interval=interval)
                closes = data['close']
                rsi = calc_rsi(closes)
                ema = calc_ema(closes)
                closes_list = closes.tolist()
                analysis = ai_analysis(symbol, interval, closes_list, rsi, ema)
                per_symbol_messages.append(f"{interval}: {analysis}")
                time.sleep(5)  # OpenAI API için istekler arası kısa bekleme
            except Exception as e:
                per_symbol_messages.append(f"{interval}: HATA: {e}")
        # Her sembol için tek mesaj olarak gönder
        message = f"{symbol} önerileri:\n" + "\n".join(per_symbol_messages)
        send_telegram_message(message)
        print(message)

def get_symbol_price(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    r = requests.get(url)
    return float(r.json()['price'])


def pnl_percent(entryPrice, markPrice, positionAmt, positionSide):
    try:
        entryPrice = float(entryPrice)
        markPrice = float(markPrice)
        positionAmt = float(positionAmt)
        # LONG: (mark-entry)/entry*100, SHORT: (entry-mark)/entry*100
        if positionSide == "LONG" or positionAmt > 0:
            return round(((markPrice - entryPrice) / entryPrice) * 100, 2)
        else:
            return round(((entryPrice - markPrice) / entryPrice) * 100, 2)
    except:
        return 0.0

def get_futures_position_table():
    positions = get_futures_positions()
    pos_table = []
    for pos in positions:
        symbol = pos['symbol']
        amt = float(pos['positionAmt'])
        if abs(amt)<1e-6:
            continue
        entry = float(pos['entryPrice'])
        mark = float(pos['markPrice'])
        side = "LONG" if amt>0 else "SHORT"
        pnl = pnl_percent(entry, mark, amt, side)
        lev = pos['leverage']
        unreal = float(pos['unRealizedProfit'])
        pos_table.append({
            'symbol':symbol,
            'amount':amt,
            'side':side,
            'entry':entry,
            'mark':mark,
            'leverage':lev,
            'pnl_pct':pnl,
            'upnl': unreal
        })
    return pos_table

def spot_and_futures_report():
    # SPOT:
    spot_assets = get_spot_balances()
    spot_lines = []
    total_usd = 0
    for a in spot_assets:
        symbol = a['asset'] + "USDT" if a['asset'] != "USDT" else "USDT"
        bal = a['free']+a['locked']
        try:
            price = get_symbol_price(symbol) if a['asset'] != "USDT" else 1
            val = bal*price
        except:
            price = 0
            val = 0
        total_usd += val
        spot_lines.append(f"{a['asset']}: {round(bal,4)} (≈ {round(val,2)} USDT)")
    spot_report = (
        f"💰 SPOT CÜZDAN\n"
        + "\n".join(spot_lines)
        + f"\nToplam: ≈ {round(total_usd, 2)} USDT\n"
    )

    # FUTURES:
    fut_bal = get_futures_balance()
    fut_pos = get_futures_position_table()
    if fut_pos:
        fut_lines = [
            f"- {p['symbol']} | {p['side']} x{p['leverage']} | {round(p['amount'],4)} @ {p['entry']} → {p['mark']} | K/Z: {p['pnl_pct']}% ({round(p['upnl'], 2)} USDT)"
            for p in fut_pos
        ]
    else:
        fut_lines = ["Açık pozisyon yok."]
    fut_report = (
        f"📊 FUTURES CÜZDAN\nBakiye: {fut_bal} USDT\n\nAçık Pozisyonlar:\n"
        + "\n".join(fut_lines)
    )
    return spot_report, fut_report, fut_pos


def ai_futures_advice(fut_pos):
    if not fut_pos:
        return "Herhangi bir açık pozisyon yok."
    # Her pozisyonu özetle
    positions_text = "\n".join([f"{p['symbol']} {p['side']} x{p['leverage']} miktar: {p['amount']} giriş: {p['entry']} son: {p['mark']}, Kar/Zarar: {p['pnl_pct']}%" for p in fut_pos])
    prompt = f"""
Aşağıda vadeli (futures) Binance hesabımdaki açık pozisyonlar verilmiştir:
{positions_text}

Lütfen, her bir pozisyon için ayrı ayrı kısa teknik değerlendirme ve TUT/SAT/ARTTIR/KAPAT önerisi ver.
Her öneriyi ayrı satırda yanına neden yazarak ilet. Ek olarak, genel portföy riskini kısaca özetle.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen profesyonel bir kripto para teknik analiz uzmanısın."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def technical_analysis_report():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "4h", "1d"]
    lines = ["📈 TEKNİK ANALİZ"]
    for symbol in symbols:
        lines.append(f"\n<b>{symbol}</b>:")
        for interval in intervals:
            try:
                df = get_binance_klines(symbol=symbol, interval=interval)
                closes = df['close']
                rsi = calc_rsi(closes)
                ema = calc_ema(closes)
                macd, macd_signal, macd_hist = calc_macd(closes)
                boll_upper, boll_lower, boll_mid = calc_bollinger(closes)
                closes_list = closes.tolist()
                analysis = ai_analysis(
                    symbol, interval, closes_list,
                    rsi, ema,
                    macd, macd_signal, macd_hist,
                    boll_upper, boll_lower, boll_mid
                )
                import html
                analysis_str = analysis.strip()
                words = analysis_str.split()
                if len(words) > 1:
                    main_word = words[0].upper()
                    rest_html = html.escape(' '.join(words[1:]))
                else:
                    main_word = analysis_str.upper()
                    rest_html = ""
                if main_word == "AL":
                    analysis_str = f"✅ <b>AL</b> {rest_html}"
                elif main_word == "SAT":
                    analysis_str = f"❌ <b>SAT</b> {rest_html}"
                elif main_word == "BEKLE":
                    analysis_str = f"🟡 <b>BEKLE</b> {rest_html}"
                else:
                    analysis_str = html.escape(analysis_str)
                lines.append(f"  {interval}: {analysis_str}")
                time.sleep(5)
            except Exception as e:
                lines.append(f"  {interval}: HATA: {e}")
    return "\n".join(lines)

def report_to_telegram():
    spot, fut, fut_pos = spot_and_futures_report()
    ai = ai_futures_advice(fut_pos)
    tech = technical_analysis_report()

    msg = (
        "✅ Binance Varlık ve Analiz Raporu\n" +
        "-------------------------------\n" +
        spot +
        "-------------------------------\n" +
        fut +
        "\n-------------------------------\n" +
        "🤖 AI Futures Analizi\n" +
        ai +
        "\n-------------------------------\n" +
        tech
    )
    send_telegram_message(msg)
    print(msg)


def main():
    while True:
        report_to_telegram()
        print("Bir sonraki rapor için 1 saat bekleniyor...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
