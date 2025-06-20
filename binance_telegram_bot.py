import matplotlib
matplotlib.use('Agg')
import time
import requests
import pandas as pd
import openai
import config
import json
import hmac
import hashlib
from urllib.parse import urlencode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import io
import feedparser
import os
import sqlite3
from cryptography.fernet import Fernet

# NOT: 're' modülü iki kez import ediliyordu, biri kaldırıldı.
# NOT: 'matplotlib.dates as mdates' ve 'matplotlib.ticker as MaxNLocator' iki kez import ediliyordu, biri kaldırıldı.

fernet = Fernet(config.FERNET_KEY)
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

# Kullanıcıya özel API key/secret yönetimi (JSON bazlı)
def get_db():
    return sqlite3.connect('portfoy_logs.sqlite3')

def set_user_apikey(chat_id, api_key, api_secret):
    encoded_api_key = fernet.encrypt(api_key.encode()).decode()
    encoded_api_secret = fernet.encrypt(api_secret.encode()).decode()
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_api (
                 chat_id INTEGER PRIMARY KEY,
                 api_key TEXT,
                 api_secret TEXT,
                 last_report_date TEXT,
                 daily_report_count INTEGER,
                 user_type TEXT
            )
        """)
        c.execute("""
            INSERT INTO user_api (chat_id, api_key, api_secret)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET api_key=excluded.api_key, api_secret=excluded.api_secret
        """, (chat_id, encoded_api_key, encoded_api_secret))
        conn.commit()
    finally:
        conn.close()

def get_user_apikey(chat_id):
    conn = get_db()
    try:
        c = conn.cursor()
        c.execute('SELECT api_key, api_secret FROM user_api WHERE chat_id=?', (chat_id,))
        row = c.fetchone()
        if row:
            api_key = fernet.decrypt(row[0].encode()).decode()
            api_secret = fernet.decrypt(row[1].encode()).decode()
            return api_key, api_secret
    except Exception as e:
        print(f"API anahtarı çözülemedi! {e}")
    finally:
        conn.close()
    return None, None

##########################
# BINANCE SIGNED REQUESTS#
##########################
def summarize_and_translate_news(news):
    # news: [{'title': ..., 'summary': ..., ...}, ...]
    articles = [f"{n['title']}\n{n['summary']}" for n in news]
    content = "\n\n".join(articles)
    prompt = (
        "Aşağıda İngilizce kripto haber başlıkları ve özetleri var.\n"
        "Hepsini kısa ve sade Türkçeyle, önemli detayları aktaracak şekilde tek paragraflık haber özeti yap:\n\n"
        "her haber özetinin arasına bir satır boşluk (paragraf) bırak:\n\n"
        f"{content}"
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen profesyonel bir haber redaktörüsün."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def strip_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def fetch_all_crypto_news(sources, limit=2):
    all_news = []
    for url in sources:
        feed = feedparser.parse(url)
        print(f"{url}: {len(feed.entries)} haber bulundu")
        for entry in feed.entries[:limit]:
            summary = entry.summary if hasattr(entry, 'summary') else ''
            all_news.append({
                "title": entry.title,
                "summary": strip_html_tags(summary),
                "link": entry.link,
                "published": getattr(entry, 'published', ''),
                "source": url
            })
    return all_news
def get_binance_signed_request(endpoint, params, api_key, api_secret, base_url):
    """
    Binance API'ye imzalı GET isteği gönderir, spot/futures gibi erişimlerde kullanılır.
    endpoint: API'nın uç noktası (ör. /api/v3/account)
    params: Sorgu parametreleri (dict)
    api_key, api_secret: API Key/Secret
    base_url: https://api.binance.com veya https://fapi.binance.com
    return: JSON veri
    """
    params['timestamp'] = int(time.time() * 1000)
    query_string = urlencode(params)
    signature = hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    headers = {'X-MBX-APIKEY': api_key}
    url = f'{base_url}{endpoint}?{query_string}&signature={signature}'
    response = requests.get(url, headers=headers)
    return response.json()

def get_spot_balances(chat_id):
    """
    Kullanıcının Binance spot cüzdan varlıklarını (pozitif bakiyede olanları) çeker.
    return: List dict [{'asset': ... , 'free': ..., 'locked': ...}, ...]
    """
    api_key, api_secret = get_user_apikey(chat_id)
    if api_key is None or api_secret is None:
        return None
    endpoint = '/api/v3/account'
    params = {}
    data = get_binance_signed_request(endpoint, params, api_key, api_secret, 'https://api.binance.com')
    nonzero = [a for a in data['balances'] if float(a['free']) > 0.0001 or float(a['locked']) > 0.0001]
    assets = []
    for a in nonzero:
        assets.append({
            'asset': a['asset'],
            'free': float(a['free']),
            'locked': float(a['locked'])
        })
    return assets

def get_futures_positions(chat_id):
    """
    Kullanıcının Binance Futures USDT-M cüzdanındaki açık pozisyonlarını döndürür.
    return: List[dict] (pozisyon datası)
    """
    api_key, api_secret = get_user_apikey(chat_id)
    if api_key is None or api_secret is None:
        return None
    endpoint = '/fapi/v2/positionRisk'
    params = {}
    data = get_binance_signed_request(endpoint, params, api_key, api_secret, 'https://fapi.binance.com')
    open_positions = [p for p in data if abs(float(p['positionAmt'])) > 0.0001]
    return open_positions

def get_futures_balance(chat_id):
    """
    Kullanıcının Binance Futures cüzdan bakiyesini döndürür.
    return: float (USDT)
    """
    api_key, api_secret = get_user_apikey(chat_id)
    if api_key is None or api_secret is None:
        return None
    endpoint = '/fapi/v2/account'
    params = {}
    data = get_binance_signed_request(endpoint, params, api_key, api_secret, 'https://fapi.binance.com')
    return float(data['totalWalletBalance'])

def get_binance_klines(chat_id, symbol="BTCUSDT", interval="1h", limit=100):
    """
    Kullanıcının Binance API key/secret'ı ile o coinin OHLCV datasını çeker.
    """
    api_key, api_secret = get_user_apikey(chat_id)
    if api_key is None or api_secret is None:
        return None
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = requests.get(url, params=params, headers={"X-MBX-APIKEY": api_key})
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
    """Verilen fiyat dizisinin RSI(14) değerini hesaplar."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)

def calc_ema(prices, period=21):
    """Kapanış fiyatlarından EMA (üstel hareketli ortalama) hesaplar."""
    return round(prices.ewm(span=period, adjust=False).mean().iloc[-1], 2)

def calc_macd(prices, fast=12, slow=26, signal=9):
    """
    Kapanış fiyatlarından MACD, MACD Sinyal ve Histogram hesaplar.
    return: tuple (macd, signal, hist)
    """
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    macd_val = macd.iloc[-1]
    signal_val = signal_line.iloc[-1]
    hist = macd_val - signal_val
    return round(macd_val,2), round(signal_val,2), round(hist,2)

def calc_bollinger(prices, period=20):
    """
    Kapanış fiyatlarından Bollinger bandı üst, alt ve orta çizgi hesaplar.
    return: tuple (üst, alt, orta)
    """
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


def send_telegram_message(message, chat_id):
    """Belirli bir kullanıcıya telegram mesajı gönderir."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    MAX_LEN = 4000
    parts = split_message_smart(message, MAX_LEN)
    for idx, part in enumerate(parts):
        data = {
            "chat_id": chat_id,
            "text": part,
            "parse_mode": "HTML"
        }
        resp = requests.post(url, data=data)
        print(f"Telegram mesajı {idx+1}/{len(parts)} gönderildi, kod: {resp.status_code}")
        if resp.status_code != 200:
            print("Telegram Hata:", resp.status_code, resp.text)

def send_telegram_photo(image_path, chat_id, caption=None):
    """Belirli bir kullanıcıya Telegram fotoğrafı gönderir."""
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as img:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
            data["parse_mode"] = "HTML"
        resp = requests.post(url, data=data, files={"photo": img})
        print(f"Telegram FOTO Kodu: {resp.status_code}")
        if resp.status_code != 200:
            print("Telegram Photo Hata:", resp.status_code, resp.text)
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
        send_telegram_message(message,chat_id)
        print(message)


import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from mplfinance.original_flavor import candlestick_ohlc

def plot_ohlc_ema_rsi_macd(chat_id,symbol, interval="1h", limit=20, ema_period=9):
    """
    Klasik borsa tarzında, üstte candlestick (mum), üstünde EMA çizgisi, altta RSI ve MACD gösteren görsel (.png) üretir.
    return: Kaydedilen dosya yolu (str)
    """
    df = get_binance_klines(chat_id,symbol=symbol, interval=interval, limit=limit)
    df['candle_time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['num_time'] = mdates.date2num(df['candle_time'])
    closes = df['close']
    opens = df['open']
    highs = df['high']
    lows = df['low']
    ema = closes.ewm(span=ema_period, adjust=False).mean()
    # RSI
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # MACD
    macd_line = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    # Mum datası
    ohlc = list(zip(df['num_time'], opens, highs, lows, closes))

    plt.close('all')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,7), sharex=True,
                            gridspec_kw={'height_ratios': [2.2, 0.8, 0.8]})
    ax1, ax2, ax3 = axs
    # Candlestick (Mum) grafiği çiz
    candlestick_ohlc(ax1, ohlc, width=0.04, colorup='g', colordown='r', alpha=0.9)
    ax1.plot(df['num_time'], ema, label=f"EMA({ema_period})", color='orange', lw=1.3, ls='--')
    ax1.set_title(f"{symbol} - Son {limit} mum ({interval})", fontsize=13)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, which='both', ls=':', alpha=0.35)
    ax1.set_ylabel('Fiyat')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
    # RSI Alt panel
    ax2.plot(df['num_time'], rsi, label="RSI(14)", color='purple')
    ax2.axhline(70, ls=':', color='red', lw=1)
    ax2.axhline(30, ls=':', color='green', lw=1)
    ax2.set_ylabel('RSI')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.grid(True, which='both', ls=':', alpha=0.38)
    # MACD Alt panel
    ax3.plot(df['num_time'], macd_line, label="MACD", color='tab:blue')
    ax3.plot(df['num_time'], macd_signal, label="Sinyal", color='tab:orange')
    ax3.bar(df['num_time'], macd_hist, label='Histogram', color='gray', width=0.04, alpha=0.68)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.set_ylabel('MACD')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax3.grid(True, which='both', ls=':', alpha=0.38)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.04)
    img_path = f"{symbol}_grafik.png"
    plt.savefig(img_path, dpi=165)
    plt.close(fig)
    return img_path

def get_symbol_price(chat_id, symbol="BTCUSDT"):
    api_key, api_secret = get_user_apikey(chat_id)
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    response = requests.get(url, headers={"X-MBX-APIKEY": api_key})
    r = response.json()
    try:
        return float(r['price'])
    except Exception as e:
        print(f"Spot için fiyat çekilemedi: {symbol}, hata: {e} -- Gelen response: {r}")
        return 0

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

def get_futures_position_table(chat_id):
    """Kullanıcıya özel pozisyon risk datasını döndürür."""
    positions = get_futures_positions(chat_id)
    pos_table = []
    if positions is None:
        return []
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

def spot_and_futures_report(chat_id):
    """Kullanıcı bazlı spot ve futures raporu döndürür."""
    # SPOT:
    spot_assets = get_spot_balances(chat_id)
    spot_lines = []
    total_usd = 0
    if spot_assets is None:
        return "Kayıtlı API bulunamadı! Lütfen önce /apikey <KEY> <SECRET> ile kayıt olun.", "", []
    for a in spot_assets:
        symbol = a['asset'] + "USDT" if a['asset'] != "USDT" else "USDT"
        bal = a['free']+a['locked']
        try:
            price = get_symbol_price(chat_id, symbol) if a['asset'] != "USDT" else 1
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
    fut_bal = get_futures_balance(chat_id)
    fut_pos = get_futures_position_table(chat_id)
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


def ai_futures_advice(chat_id,fut_pos):
    import html
    if not fut_pos:
        return "Herhangi bir açık pozisyon yok."
    positions_full = []
    for p in fut_pos:
        symbol = p['symbol']
        # 1h kapanış verisi ve indikatörler
        try:
            df = get_binance_klines(chat_id,symbol=symbol, interval='1h')
            closes = df['close']
            rsi = calc_rsi(closes)
            ema = calc_ema(closes)
            macd, macd_signal, macd_hist = calc_macd(closes)
            boll_upper, boll_lower, boll_mid = calc_bollinger(closes)
            ind_str = f"RSI: {rsi}, EMA: {ema}, MACD: {macd}, Sinyal: {macd_signal}, Bollinger(üst/alt/orta): {boll_upper}/{boll_lower}/{boll_mid}"
        except Exception as e:
            ind_str = f"indikatör HATASI: {e}"
        positions_full.append(
            f"{symbol} {p['side']} x{p['leverage']} miktar: {p['amount']} giriş: {p['entry']} son: {p['mark']}, Kar/Zarar: {p['pnl_pct']}%. {ind_str}"
        )
    positions_text = "\n".join(positions_full)
    prompt = f"""
Aşağıda vadeli (futures) Binance hesabımdaki açık pozisyonlar verilmiştir:
{positions_text}

Her pozisyonun teknik indikatörlerini de dikkate alarak teknik olarak, pozisyon short ise ona göre long ise ona göre kısaca değerlendir, tut/sat/kapalı önerini belirt. En sonda ise genel risk ve yönetim önerisini yalnızca 2-3 cümleyle, tek paragraf halinde özetle.
Kesinlikle başlık, maddeleme, paragraf veya kategori ekleme. Sadece kısa ve bütünleşik bir teknik özet ve öneri ver.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen profesyonel bir kripto para teknik analiz uzmanısın."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def technical_analysis_report(chat_id):
    import html
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "4h", "1d"]
    lines = ["📈 TEKNİK ANALİZ (Genel öneri her coin için tek yorumdur)"]
    for symbol in symbols:
        per_interval_metrics = []
        for interval in intervals:
            try:
                df = get_binance_klines(chat_id,symbol=symbol, interval=interval)
                closes = df['close']
                rsi = calc_rsi(closes)
                ema = calc_ema(closes)
                macd, macd_signal, macd_hist = calc_macd(closes)
                boll_upper, boll_lower, boll_mid = calc_bollinger(closes)
                closes_list = closes.tolist()
                per_interval_metrics.append(
                    f"<b>{interval}</b>: Fiyat: {round(closes_list[-1],2)}, RSI: {rsi}, EMA: {ema}, MACD: {macd}, Sinyal: {macd_signal}, Bollinger(üst/alt/orta): {boll_upper}/{boll_lower}/{boll_mid}"
                )
            except Exception as e:
                per_interval_metrics.append(f"{interval}: HATA: {e}")
        # Tüm veriyi tek seferde analiz ettirelim
        prompt = (
            f"Aşağıda {symbol} için 1 saatlik, 4 saatlik ve günlük teknik veriler listelenmiştir:\n" +
            "\n".join(per_interval_metrics) +
            "\n\nYalnızca bu verileri göz önüne alarak genel piyasa trendini ve olası yönü TEK CÜMLEYLE teknik analiz uzmanı gibi özetle.\n" +
            "Net ve baskın bir sinyal varsa büyük harfle ve emojiyle (✅ AL / ❌ SAT / 🟡 BEKLE) yaz, ardından bir iki cümle teknik kısa sebep belirt, uzatma!"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sen bir kripto para teknik analiz uzmanısın."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = response.choices[0].message.content.strip()
        analysis = html.escape(analysis)  # Emniyet için escape
        # En baştaki emoji ve öneriyi özel yapıyoruz:
        # Kalıpları bulup <b>...</b> ile güçlendir
        if analysis.upper().startswith("AL"):
            analysis = f"✅ <b>AL</b> {analysis[2:].strip()}"
        elif analysis.upper().startswith("SAT"):
            analysis = f"❌ <b>SAT</b> {analysis[2:].strip()}"
        elif analysis.upper().startswith("BEKLE"):
            analysis = f"🟡 <b>BEKLE</b> {analysis[4:].strip()}"
        # Coin başlığı + analiz satırı
        lines.append(f"\n<b>{symbol}</b> önerisi: {analysis}")
    return "\n".join(lines)


def send_telegram_media_group(image_paths, symbols,chat_id):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMediaGroup"
    media_data = []
    files = {}
    file_handles = []
    for idx, img_path in enumerate(image_paths):
        f = open(img_path, "rb")
        files[f"photo{idx}"] = f
        file_handles.append(f)
        caption = f"<b> 1 saatlik Son 20 mum </b>" if idx == 0 else ""
        media_data.append({
            "type": "photo",
            "media": f"attach://photo{idx}",
            "caption": caption,
            "parse_mode": "HTML"
        })
    import json
    data = {
        "chat_id": chat_id,
        "media": json.dumps(media_data)
    }
    resp = requests.post(url, data=data, files=files)
    print(f"Media group status: {resp.status_code}")
    if resp.status_code != 200:
        print("Telegram MediaGroup Hata:", resp.status_code, resp.text)
    # Önce dosya objelerini kapat:
    for f in file_handles:
        try:
            f.close()
        except Exception as ex:
            print(f"file kapatılamadı: {ex}")
    # Sonra dosyaları sil:
    import os
    for fp in image_paths:
        try:
            os.remove(fp)
        except Exception as ex:
            print(f"{fp} silinemedi: {ex}")


def report_to_telegram(chat_id, news_block=None, image_cache=None, user_symbols=None, tech_cache=None):
    import re
    from datetime import datetime
    # -- SPOT VE FUTURES RAPOR VE DB YE KAYIT --
    spot, fut, fut_pos = spot_and_futures_report(chat_id)
    spot_total = None
    fut_balance = None
    spot_match = re.search(r"Toplam: ≈ ([\d\.]+) USDT", spot)
    fut_match = re.search(r"Bakiye: ([\d\.]+) USDT", fut)
    if spot_match: spot_total = float(spot_match.group(1))
    if fut_match: fut_balance = float(fut_match.group(1))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # --- SQLITE KULLAN: ---
    try:
        with sqlite3.connect('portfoy_logs.sqlite3') as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS pnl_log (
                    timestamp TEXT,
                    chat_id INTEGER,
                    spot_total REAL,
                    futures_balance REAL
                )
            """)
            if spot_total is not None and fut_balance is not None:
                c.execute("INSERT INTO pnl_log (timestamp, chat_id, spot_total, futures_balance) VALUES (?, ?, ?, ?)",
                    (now, chat_id, spot_total, fut_balance))
            conn.commit()
    except Exception as e:
        print(f"[SQLITE LOG] hata: {e}")
    ai = ai_futures_advice(chat_id, fut_pos)
    tech = tech_cache if tech_cache else technical_analysis_report(chat_id)
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
        tech +
        (news_block if news_block else "")
    )
    send_telegram_message(msg, chat_id)
    print(msg)
    symbols = user_symbols[chat_id] if user_symbols and chat_id in user_symbols else set()
    image_paths = [image_cache[sym] for sym in symbols if image_cache and sym in image_cache]
    symbol_list = [sym for sym in symbols if image_cache and sym in image_cache]
    if image_paths:
        send_telegram_media_group(image_paths, symbol_list, chat_id)

# -- ZAMANSAL PNL GRAFİĞİ GÖNDERİMİ --
def generate_portfolio_curve_figure(df):
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(df['timestamp'], df['spot_total'], label="Spot Toplam", marker='o', color='orange')
    ax.plot(df['timestamp'], df['futures_balance'], label="Futures Bakiye", marker='o', color='navy')
    ax.set_title("Portföy Zaman Serisi (USDT)")
    ax.set_xticks(range(0, len(df['timestamp']), max(1, len(df['timestamp'])//10)))
    ax.set_xticklabels(df['timestamp'][::max(1, len(df['timestamp'])//10)], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("Tutar (USDT)")
    ax.legend()
    plt.tight_layout()
    img_path = "portfoy_curve.png"
    plt.savefig(img_path, dpi=150)
    plt.close(fig)
    return img_path

# Her döngü sonunda klasik sync fonksiyon ile grafik gönder
# Bu, otomasyon döngüsünde çalışmaya devam edecek

def send_portfolio_curve(chat_id, log_db="portfoy_logs.sqlite3"):
    conn = sqlite3.connect(log_db)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM pnl_log WHERE chat_id = ? ORDER BY timestamp",
            conn, params=(chat_id,)
        )
    finally:
        conn.close()
    if df.empty or len(df) < 2:
        print("Kullanıcıya ait yeterli log yok veya hiç yok!")
        return
    img_path = generate_portfolio_curve_figure(df)
    send_telegram_photo(img_path, chat_id, caption="Portföy zaman serisi (log)")
    try:
        os.remove(img_path)
    except: pass
# Otomatik loop için klasik fonksiyonu her kullanıcı için chat_id ile çağırman gerekir!
def get_all_user_ids():
    import sqlite3
    conn = sqlite3.connect('portfoy_logs.sqlite3')
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_api (
                chat_id INTEGER PRIMARY KEY,
                api_key TEXT,
                api_secret TEXT,
                last_report_date TEXT,
                daily_report_count INTEGER,
                user_type TEXT
            )
        ''')
        c.execute('SELECT chat_id FROM user_api')
        results = c.fetchall()
        return [int(row[0]) for row in results]
    except Exception as e:
        print(f"get_all_user_ids sqlite hata: {e}")
        return []
    finally:
        conn.close()

def technical_analysis_report_batch(chat_id_any):
    import html
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "4h", "1d"]
    results = []
    for symbol in symbols:
        per_interval_metrics = []
        for interval in intervals:
            try:
                df = get_binance_klines(chat_id_any,symbol=symbol, interval=interval)
                closes = df['close']
                rsi = calc_rsi(closes)
                ema = calc_ema(closes)
                macd, macd_signal, macd_hist = calc_macd(closes)
                boll_upper, boll_lower, boll_mid = calc_bollinger(closes)
                closes_list = closes.tolist()
                per_interval_metrics.append(
                    f"<b>{interval}</b>: Fiyat: {round(closes_list[-1],2)}, RSI: {rsi}, EMA: {ema}, MACD: {macd}, Sinyal: {macd_signal}, Bollinger(üst/alt/orta): {boll_upper}/{boll_lower}/{boll_mid}"
                )
            except Exception as e:
                per_interval_metrics.append(f"{interval}: HATA: {e}")
        prompt = (
            f"Aşağıda {symbol} için 1 saatlik, 4 saatlik ve günlük teknik veriler listelenmiştir:\n" +
            "\n".join(per_interval_metrics) +
            "\n\nYalnızca bu verileri göz önüne alarak genel piyasa trendini ve olası yönü TEK CÜMLEYLE teknik analiz uzmanı gibi özetle.\n" +
            "Net ve baskın bir sinyal varsa büyük harfle ve emojiyle (✅ AL / ❌ SAT / 🟡 BEKLE) yaz, ardından bir iki cümle teknik kısa sebep belirt, uzatma!"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sen bir kripto para teknik analiz uzmanısın."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = response.choices[0].message.content.strip()
        analysis = html.escape(analysis)
        if analysis.upper().startswith("AL"):
            analysis = f"✅ <b>AL</b> {analysis[2:].strip()}"
        elif analysis.upper().startswith("SAT"):
            analysis = f"❌ <b>SAT</b> {analysis[2:].strip()}"
        elif analysis.upper().startswith("BEKLE"):
            analysis = f"🟡 <b>BEKLE</b> {analysis[4:].strip()}"
        results.append(f"\n<b>{symbol}</b> önerisi: {analysis}")
    return "\n".join(results)

def run_report_loop():
    while True:
        # -- Haberleri ve özetini sadece 1 KEZ çek --
        try:
            sources = [
                "https://cointelegraph.com/rss",
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "https://www.newsbtc.com/feed/",
                "https://cryptopanic.com/news/rss/"
            ]
            news = fetch_all_crypto_news(sources, limit=2)
            news_summary_tr = summarize_and_translate_news(news)
            news_summary_tr = news_summary_tr.strip()
            news_block = (
                "\n-------------------------------\n"
                "🌎 <b>Son Kripto Haberler</b>:\n"
                f"{news_summary_tr}"
            )
        except Exception as e:
            news_block = f"\n-------------------------------\nKripto haberleri alınırken hata: {e}"

        # ---- GRAFİK CACHE ----
        image_cache = {}
        all_symbols = set()
        user_symbols = {}
        for chat_id in get_all_user_ids():
            symbols = set(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            try:
                spot_balances = get_spot_balances(chat_id)
                if spot_balances:
                    symbols.update([a['asset']+"USDT" for a in spot_balances if a['asset'] != "USDT"])
            except Exception:
                pass
            try:
                fut_pos = get_futures_position_table(chat_id)
                if fut_pos:
                    symbols.update([p['symbol'] for p in fut_pos])
            except Exception:
                pass
            all_symbols.update(symbols)
            user_symbols[chat_id] = symbols

        for sym in all_symbols:
            try:
                img_path = plot_ohlc_ema_rsi_macd(list(get_all_user_ids())[0], sym)
                image_cache[sym] = img_path
            except Exception as e:
                print(f"Görsel oluşturma hatası {sym}: {e}")

        # --- TEKNİK ANALİZİ BİR KERE HESAPLA ---
        anyone = None
        for i in get_all_user_ids():
            anyone = i
            break
        tech_cache = technical_analysis_report_batch(anyone) if anyone else ""

        for chat_id in get_all_user_ids():
            try:
                report_to_telegram(chat_id, news_block=news_block, image_cache=image_cache, user_symbols=user_symbols, tech_cache=tech_cache)
            except Exception as e:
                print(f"{chat_id} için rapor hatası:", e)

        # -- Döngü sonunda temp dosyaları temizle --
        import os
        for fp in set(image_cache.values()):
            try:
                os.remove(fp)
            except Exception as ex:
                print(f"{fp} silinemedi: {ex}")
        print("Bir sonraki rapor için 1 saat bekleniyor...")
        time.sleep(3600)
async def send_portfolio_curve_telegram(context, chat_id, log_db="portfoy_logs.sqlite3"):
    conn = sqlite3.connect(log_db)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM pnl_log WHERE chat_id = ? ORDER BY timestamp",
            conn, params=(chat_id,)
        )
    finally:
        conn.close()
    if df.empty or len(df) < 2:
        await context.bot.send_message(chat_id=chat_id, text="Kullanıcıya ait yeterli log yok veya hiç yok!")
        return
    img_path = generate_portfolio_curve_figure(df)
    with open(img_path, "rb") as photo:
        await context.bot.send_photo(chat_id=chat_id, photo=photo, caption="Portföy zaman serisi (log)")
    try:
        os.remove(img_path)
    except: pass

if __name__ == "__main__":
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram import Update
    import threading

    async def handle_apikey(update: Update, context: ContextTypes.DEFAULT_TYPE):
        args = context.args
        if len(args) != 2:
            await update.message.reply_text("Kullanım: /apikey <API_KEY> <API_SECRET>")
            return
        key, secret = args[0], args[1]
        set_user_apikey(update.effective_chat.id, key, secret)
        await update.message.reply_text("API anahtarlarınız kaydedildi ✅\nArtık tüm analiz ve portföy işlemleri bu hesapla yapılacaktır.")
        try:
            report_to_telegram(update.effective_chat.id)
        except Exception as e:
            await update.message.reply_text(f"🚫 Rapor üretilemedi: {e}")

    async def handle_pnl(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await send_portfolio_curve_telegram(context, update.effective_chat.id)

    async def handle_hesap(update: Update, context: ContextTypes.DEFAULT_TYPE):
        import sqlite3
        from datetime import datetime

        chat_id = update.effective_chat.id
        limits = {"basic": 1, "normal": 3, "premium": 10}
        try:
            conn = sqlite3.connect('portfoy_logs.sqlite3')
            c = conn.cursor()
            c.execute('SELECT last_report_date, daily_report_count, user_type FROM user_api WHERE chat_id = ?', (chat_id,))
            row = c.fetchone()
            if row is None:
                await update.message.reply_text("Kayıtlı kullanıcı bulunamadı. Önce /apikey ile kayıt olun.")
                return
            last_date, count, user_type = row
            user_type = user_type or "normal"
            total_rights = limits.get(user_type, 3)
            today = datetime.now().strftime('%Y-%m-%d')
            # başka günden kalan sayaç varsa sıfır kabul et
            if last_date != today:
                used = 0
            else:
                used = count or 0
            kalan_hak = max(0, total_rights - used)
            msg = (f"👤 Kullanıcı Tipi: <b>{user_type.upper()}</b>\n"
                f"Günlük Rapor Hakkı: <b>{total_rights}</b>\n"
                f"Kalan Hak: <b>{kalan_hak}</b>")
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception as e:
            await update.message.reply_text(f"Hesap bilgisi alınamadı: {e}")

    async def handle_rapor(update: Update, context: ContextTypes.DEFAULT_TYPE):
        import sqlite3
        from datetime import datetime
        chat_id = update.effective_chat.id
        today = datetime.now().strftime('%Y-%m-%d')
        limits = {"basic": 1, "normal": 3, "premium": 10}
        try:
            conn = sqlite3.connect('portfoy_logs.sqlite3')
            c = conn.cursor()
            c.execute('SELECT last_report_date, daily_report_count, user_type FROM user_api WHERE chat_id = ?', (chat_id,))
            row = c.fetchone()
            if row is None:
                user_type = "normal"
                max_rights = limits[user_type]
                c.execute('INSERT INTO user_api (chat_id, last_report_date, daily_report_count, user_type) VALUES (?, ?, ?, ?)', (chat_id, today, 1, user_type))
                conn.commit()
                count = 1
                last_date = today
            else:
                last_date, count, user_type = row
                user_type = user_type or "normal"
                max_rights = limits.get(user_type, 3)  # default normal hak
                count = count or 0
                # Tarih farklıysa sıfırla
                if last_date != today:
                    count = 0
                if count >= max_rights:
                    await update.message.reply_text(f"🚫 Günlük rapor limitinize ulaştınız. ({user_type} için {max_rights} hak) Yarın tekrar deneyin.")
                    conn.close()
                    return
                # Kullanıcıya hak ver
                count += 1
                c.execute('UPDATE user_api SET last_report_date = ?, daily_report_count = ?, user_type = ? WHERE chat_id = ?', (today, count, user_type, chat_id))
                conn.commit()
            conn.close()
        except Exception as e:
            await update.message.reply_text(f"🚫 Rapor limiti kontrolünde hata: {e}")
            return
        await update.message.reply_text("Raporunuz hazırlanıyor, lütfen bekleyin...")
        try:
            # --- Haber, image ve symbol cache oluştur ---
            sources = [
                "https://cointelegraph.com/rss",
                "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "https://www.newsbtc.com/feed/",
                "https://cryptopanic.com/news/rss/"
            ]
            try:
                news = fetch_all_crypto_news(sources, limit=2)
                news_summary_tr = summarize_and_translate_news(news)
                news_block = (
                    "\n-------------------------------\n"
                    "🌎 <b>Son Kripto Haberler</b>:\n"
                    f"{news_summary_tr}"
                )
            except Exception as e:
                news_block = f"\n-------------------------------\nKripto haberleri alınırken hata: {e}"

            symbols = set(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            try:
                spot_balances = get_spot_balances(chat_id)
                if spot_balances:
                    symbols.update([a['asset']+"USDT" for a in spot_balances if a['asset'] != "USDT"])
            except Exception:
                pass
            try:
                fut_pos = get_futures_position_table(chat_id)
                if fut_pos:
                    symbols.update([p['symbol'] for p in fut_pos])
            except Exception:
                pass
            user_symbols = {chat_id: symbols}
            image_cache = {}
            for sym in symbols:
                try:
                    img_path = plot_ohlc_ema_rsi_macd(chat_id, sym)
                    image_cache[sym] = img_path
                except Exception as e:
                    print(f'Görsel oluşturma hatası {sym}: {e}')

            tech_cache = technical_analysis_report_batch(chat_id)

            import threading
            threading.Thread(target=report_to_telegram, args=(chat_id, news_block, image_cache, user_symbols, tech_cache)).start()
        except Exception as e:
            await update.message.reply_text(f"🚫 Rapor üretilemedi: {e}")

    t = threading.Thread(target=run_report_loop, daemon=True)
    t.start()
    
    app = Application.builder().token(config.TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("rapor", handle_rapor))
    app.add_handler(CommandHandler("apikey", handle_apikey))
    app.add_handler(CommandHandler("pnl", handle_pnl))
    app.add_handler(CommandHandler("hesap", handle_hesap))
    print("Telegram komut listener başlatıldı (örn. /pnl)")
    app.run_polling()
