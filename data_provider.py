import random
import statistics
from datetime import datetime, timedelta

# ------------------------------------------------
# SIMULADOR DE DATOS (para pruebas)
# Reemplaza esta función con tu conexión real a broker/API
# ------------------------------------------------
def get_current_vela():
    """
    Simula la obtención de la última vela de 1 minuto.
    Devuelve un diccionario con:
        - open: precio de apertura
        - high: máximo
        - low: mínimo
        - close: precio de cierre
        - volume: volumen (ticks)
        - timestamp: datetime de la vela (inicio del minuto)
    """
    now = datetime.now().replace(second=0, microsecond=0)
    # Generar datos aleatorios con cierta tendencia para pruebas
    base = 100 + random.uniform(-2, 2)
    open_price = base
    close_price = base + random.uniform(-1.5, 1.5)
    high_price = max(open_price, close_price) + random.uniform(0, 0.5)
    low_price = min(open_price, close_price) - random.uniform(0, 0.5)
    volume = random.randint(100, 1000)
    return {
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume,
        'timestamp': now
    }

def get_historical_velas(minutes=20):
    """
    Devuelve una lista de las últimas 'minutes' velas (simuladas).
    """
    velas = []
    now = datetime.now().replace(second=0, microsecond=0)
    for i in range(minutes, 0, -1):
        ts = now - timedelta(minutes=i)
        # Simular datos coherentes (con tendencia aleatoria)
        base = 100 + (i * 0.01) + random.uniform(-1, 1)
        open_price = base
        close_price = base + random.uniform(-1, 1)
        high_price = max(open_price, close_price) + random.uniform(0, 0.3)
        low_price = min(open_price, close_price) - random.uniform(0, 0.3)
        volume = random.randint(100, 1000)
        velas.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'timestamp': ts
        })
    return velas

# ------------------------------------------------
# CÁLCULOS TÉCNICOS
# ------------------------------------------------
def calcular_medias(velas):
    """
    Calcula medias de cuerpo y volumen de las últimas 5 velas.
    """
    ultimas5 = velas[-5:]
    cuerpos = [abs(v['close'] - v['open']) for v in ultimas5]
    volumenes = [v['volume'] for v in ultimas5]
    cuerpo_medio = statistics.mean(cuerpos) if cuerpos else 0
    volumen_medio = statistics.mean(volumenes) if volumenes else 0
    return cuerpo_medio, volumen_medio

def detectar_trampa(vela_actual, velas_anteriores):
    """
    Detecta trampas de liquidez (falsos rompimientos).
    Retorna un string con el tipo de trampa o None.
    """
    if len(velas_anteriores) < 10:
        return None
    maximos = [v['high'] for v in velas_anteriores[-10:]]
    minimos = [v['low'] for v in velas_anteriores[-10:]]
    max_reciente = max(maximos)
    min_reciente = min(minimos)

    alta = vela_actual['high']
    baja = vela_actual['low']
    cierre = vela_actual['close']

    # Trampa alcista: rompe máximo pero cierra por debajo
    if alta > max_reciente and cierre < max_reciente:
        return "ALCISTA (falso breakout) → señal de VENTA"
    # Trampa bajista: rompe mínimo pero cierra por encima
    if baja < min_reciente and cierre > min_reciente:
        return "BAJISTA (falso breakout) → señal de COMPRA"
    return None

def calcular_probabilidad_y_fuerza(vela_actual, velas_anteriores):
    """
    Calcula probabilidad de compra y venta, y la fuerza (0-1).
    Retorna (prob_compra, prob_venta, fuerza)
    """
    cuerpo_medio, volumen_medio = calcular_medias(velas_anteriores)

    cuerpo_actual = abs(vela_actual['close'] - vela_actual['open'])
    rango_actual = vela_actual['high'] - vela_actual['low']
    sombra_sup = vela_actual['high'] - max(vela_actual['open'], vela_actual['close'])
    sombra_inf = min(vela_actual['open'], vela_actual['close']) - vela_actual['low']

    # Inicializar probabilidades
    prob_compra = 0
    prob_venta = 0

    # 1. Impulso por cuerpo grande
    if cuerpo_actual > cuerpo_medio * 1.5:
        if vela_actual['close'] > vela_actual['open']:  # vela alcista
            prob_compra += 30
        else:
            prob_venta += 30

    # 2. Cierre vs apertura anterior
    if len(velas_anteriores) >= 1:
        vela_prev = velas_anteriores[-1]
        if vela_actual['close'] > vela_prev['open']:
            prob_compra += 20
        elif vela_actual['close'] < vela_prev['open']:
            prob_venta += 20

    # 3. Sombras
    if sombra_inf > sombra_sup:
        prob_compra += 20
    elif sombra_sup > sombra_inf:
        prob_venta += 20

    # 4. Volumen
    if vela_actual['volume'] > volumen_medio:
        if vela_actual['close'] > vela_actual['open']:
            prob_compra += 30
        else:
            prob_venta += 30
    else:
        # Volumen bajo, pero si el cuerpo es grande, puede ser manipulación
        if cuerpo_actual > cuerpo_medio * 2:
            # Podría ser manipulación, pero no aumentamos probabilidad
            pass

    # Normalizar a porcentajes (que sumen 100)
    total = prob_compra + prob_venta
    if total > 0:
        prob_compra = (prob_compra / total) * 100
        prob_venta = (prob_venta / total) * 100
    else:
        prob_compra = 50
        prob_venta = 50

    # Calcular fuerza (un valor entre 0 y 1 basado en la claridad de la señal)
    # Usamos la diferencia absoluta entre probabilidades normalizada
    fuerza = abs(prob_compra - prob_venta) / 100.0  # 0 si iguales, 1 si 100% vs 0%

    return prob_compra, prob_venta, fuerza
