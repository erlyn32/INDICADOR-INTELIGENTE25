# descargador_iq.py
import os
import time
import pandas as pd
from iqoptionapi.stable_api import IQ_Option

# Leer credenciales de variables de entorno (configuradas en Secrets de Streamlit)
IQ_EMAIL = os.environ.get('IQ_EMAIL')
IQ_PASSWORD = os.environ.get('IQ_PASSWORD')

if not IQ_EMAIL or not IQ_PASSWORD:
    raise ValueError("Faltan credenciales IQ_EMAIL o IQ_PASSWORD en las variables de entorno.")

# Configuración
ACTIVO = "EURUSD"
TIMEFRAME = 60  # segundos (1 minuto)
VELAS_POR_REQUEST = 1000
NUM_REQUESTS = 2000  # 1000 * 2000 = 2,000,000 velas
TIEMPO_ESPERA = 0.5

print("Conectando a IQ Option...")
iq = IQ_Option(IQ_EMAIL, IQ_PASSWORD)
check, reason = iq.connect()

if not check:
    print(f"Error de conexión: {reason}")
    exit(1)

print("Conexión exitosa. Usando cuenta demo.")
iq.change_balance('PRACTICE')  # Recomendado usar cuenta demo

# Obtener timestamp actual
end_from = int(time.time())

todas_las_velas = []
print("Descargando 2 millones de velas M1 de EURUSD...")

for i in range(NUM_REQUESTS):
    to_time = end_from - i * VELAS_POR_REQUEST * TIMEFRAME
    try:
        velas = iq.get_candles(ACTIVO, TIMEFRAME, VELAS_POR_REQUEST, to_time)
        if velas:
            df_batch = pd.DataFrame(velas)
            df_batch = df_batch[['from', 'open', 'max', 'min', 'close', 'volume']]
            df_batch.columns = ['from', 'open', 'high', 'low', 'close', 'volume']
            todas_las_velas.append(df_batch)

        if (i + 1) % 100 == 0:
            print(f"Progreso: {i+1}/{NUM_REQUESTS} lotes completados.")

        time.sleep(TIEMPO_ESPERA)
    except Exception as e:
        print(f"Error en lote {i+1}: {e}")
        time.sleep(TIEMPO_ESPERA * 2)

if todas_las_velas:
    df_final = pd.concat(todas_las_velas, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['from'])
    df_final = df_final.sort_values('from').reset_index(drop=True)
    csv_name = f"iqoption_data_{ACTIVO}_60.csv"
    df_final.to_csv(csv_name, index=False)
    print(f"✅ Datos guardados en {csv_name} con {len(df_final)} velas.")
else:
    print("❌ No se descargaron velas.")
