# aplicacion.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import os
from datetime import datetime
from iqoptionapi.stable_api import IQ_Option
from data_provider import EstrategiaAvanzada

# ========== CONFIGURACIÓN ==========
SEGUNDO_EJECUCION = 58
VELAS_HISTORICAS = 50
ACTIVO_DESCARGAR = "EURUSD-OTC"
TIMEFRAME = 60
VELAS_POR_REQUEST = 1000
LOTES_POR_DEFECTO = 500

# Archivos necesarios
MODELO_FILE = "modelo_xgb.pkl"
SCALER_FILE = "scaler.pkl"
CSV_FILE = "iqoption_data_EURUSD_60.csv"

# ========== FUNCIONES DE DESCARGA Y ENTRENAMIENTO (sin cambios) ==========
def descargar_datos(api, activo, total_requests):
    todas_las_velas = []
    end_from = int(time.time())
    progreso = st.progress(0, text="Iniciando descarga...")
    for i in range(total_requests):
        to_time = end_from - i * VELAS_POR_REQUEST * TIMEFRAME
        try:
            velas = api.get_candles(activo, TIMEFRAME, VELAS_POR_REQUEST, to_time)
            if velas:
                df_batch = pd.DataFrame(velas)
                df_batch = df_batch[['from', 'open', 'max', 'min', 'close', 'volume']]
                df_batch.columns = ['from', 'open', 'high', 'low', 'close', 'volume']
                todas_las_velas.append(df_batch)
            progreso.progress((i + 1) / total_requests, text=f"Descargando lote {i+1}/{total_requests}")
            time.sleep(0.3)
        except Exception as e:
            st.warning(f"Error en lote {i+1}: {e}")
            time.sleep(1)
    progreso.empty()
    if not todas_las_velas:
        return pd.DataFrame(columns=['from', 'open', 'high', 'low', 'close', 'volume'])
    df_final = pd.concat(todas_las_velas, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=['from'])
    df_final = df_final.sort_values('from').reset_index(drop=True)
    return df_final

def cargar_datos_existentes():
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            df['from'] = pd.to_numeric(df['from'])
            return df
        except:
            return pd.DataFrame(columns=['from', 'open', 'high', 'low', 'close', 'volume'])
    else:
        return pd.DataFrame(columns=['from', 'open', 'high', 'low', 'close', 'volume'])

def combinar_datos(df_existente, df_nuevo):
    if df_existente.empty:
        return df_nuevo
    if df_nuevo.empty:
        return df_existente
    df_combinado = pd.concat([df_existente, df_nuevo], ignore_index=True)
    df_combinado = df_combinado.drop_duplicates(subset=['from'])
    df_combinado = df_combinado.sort_values('from').reset_index(drop=True)
    return df_combinado

def entrenar_modelo_con_datos(df):
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    from sklearn.metrics import accuracy_score
    if len(df) < 100:
        st.warning("No hay suficientes datos para entrenar (mínimo 100 velas).")
        return None, None
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna(subset=['target']).reset_index(drop=True)
    if len(df) < 100:
        st.warning("Después de crear target, no hay suficientes muestras.")
        return None, None
    estrategia = EstrategiaAvanzada(modelo_path=None, scaler_path=None, ventana=20)
    features_list = []
    targets_list = []
    for i in range(20, len(df)):
        ventana_df = df.iloc[i-20:i][['open','high','low','close','volume']]
        feat = estrategia.calcular_features(ventana_df)
        if feat is not None:
            features_list.append(feat)
            targets_list.append(df.iloc[i]['target'])
    if len(features_list) < 100:
        st.warning("No se generaron suficientes features para entrenar.")
        return None, None
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    modelo = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    modelo.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"Precisión en validación: {acc:.4f}")
    with open(MODELO_FILE, 'wb') as f:
        pickle.dump(modelo, f)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    return modelo, scaler

# ========== NUEVA ESTRATEGIA INDEPENDIENTE (FUERZA Y VOLUMEN) ==========
def analizar_fuerza_volumen(velas_df, ventana=20):
    """
    Analiza la fuerza y volumen de un activo para predecir la próxima vela.
    Retorna un diccionario con la misma estructura que EstrategiaAvanzada.
    """
    df = velas_df.copy()
    if len(df) < ventana + 1:
        return None

    # Calcular indicadores
    df['precio_tipico'] = (df['high'] + df['low'] + df['close']) / 3
    df['cambio'] = df['close'].diff()
    df['volumen_medio'] = df['volume'].rolling(ventana).mean()
    df['volumen_relativo'] = df['volume'] / df['volumen_medio']

    # Money Flow Index (MFI)
    df['raw_money_flow'] = df['precio_tipico'] * df['volume']
    df['direccion'] = np.where(df['precio_tipico'] > df['precio_tipico'].shift(1), 1, -1)
    df['money_flow_positivo'] = np.where(df['direccion'] == 1, df['raw_money_flow'], 0)
    df['money_flow_negativo'] = np.where(df['direccion'] == -1, df['raw_money_flow'], 0)
    suma_positiva = df['money_flow_positivo'].rolling(ventana).sum()
    suma_negativa = df['money_flow_negativo'].rolling(ventana).sum()
    df['mfi'] = 100 - (100 / (1 + suma_positiva / (suma_negativa + 1e-10)))

    # Pendiente de la tendencia (regresión lineal simple)
    y = df['close'].values[-ventana:]
    x = np.arange(len(y))
    if len(y) > 1:
        pendiente = np.polyfit(x, y, 1)[0]
    else:
        pendiente = 0

    # Fuerza de volumen (basada en la última vela)
    ultima_vela = df.iloc[-1]
    penultima_vela = df.iloc[-2]
    fuerza_volumen = (ultima_vela['close'] - penultima_vela['close']) * ultima_vela['volume']
    # Normalizar fuerza (valor absoluto)
    fuerza_abs = abs(fuerza_volumen) / (df['volume'].mean() * df['close'].mean() + 1e-10)
    fuerza_abs = min(fuerza_abs, 1.0)  # acotar

    # Último valor de MFI
    mfi_actual = df['mfi'].iloc[-1] if not np.isnan(df['mfi'].iloc[-1]) else 50
    volumen_rel = df['volumen_relativo'].iloc[-1] if not np.isnan(df['volumen_relativo'].iloc[-1]) else 1.0

    # Determinar sentimiento
    if mfi_actual > 60 and pendiente > 0:
        sentimiento = "CALL"
        prob_CALL = min(0.5 + (mfi_actual - 50) / 100 + fuerza_abs * 0.2, 0.95)
        prob_PUT = 1 - prob_CALL
    elif mfi_actual < 40 and pendiente < 0:
        sentimiento = "PUT"
        prob_PUT = min(0.5 + (50 - mfi_actual) / 100 + fuerza_abs * 0.2, 0.95)
        prob_CALL = 1 - prob_PUT
    else:
        # Zona neutral, usar pendiente
        if pendiente > 0.0001:
            sentimiento = "CALL"
            prob_CALL = 0.55 + abs(pendiente) * 100
            prob_PUT = 1 - prob_CALL
        elif pendiente < -0.0001:
            sentimiento = "PUT"
            prob_PUT = 0.55 + abs(pendiente) * 100
            prob_CALL = 1 - prob_PUT
        else:
            sentimiento = "NEUTRAL"
            prob_CALL = 0.5
            prob_PUT = 0.5

    # Fuerza general (combinación de MFI y volumen relativo)
    fuerza = (abs(mfi_actual - 50) / 50) * 0.5 + (volumen_rel - 1) * 0.5
    fuerza = max(0, min(fuerza, 1))  # normalizar entre 0 y 1

    # Magnitud esperada (texto)
    if fuerza > 0.7:
        magnitud = "ALTA"
    elif fuerza > 0.4:
        magnitud = "MEDIA"
    else:
        magnitud = "BAJA"

    # Tiene tendencia clara?
    tiene_tendencia = abs(pendiente) > 0.0001 or abs(mfi_actual - 50) > 15

    # Es buena señal?
    es_bueno = fuerza > 0.5 and volumen_rel > 1.2

    # Volumen relativo actual
    volumen_rel_val = volumen_rel

    return {
        'sentimiento': sentimiento,
        'fuerza': fuerza,
        'prob_CALL': prob_CALL,
        'prob_PUT': prob_PUT,
        'magnitud_esperada': magnitud,
        'volumen': volumen_rel_val,
        'tiene_tendencia': tiene_tendencia,
        'es_bueno': es_bueno
    }

# ========== INTERFAZ DE LOGIN (sin cambios) ==========
st.set_page_config(layout="wide")
st.title("🤖 Indicador Inteligente IQ Option con IA")

with st.sidebar:
    st.header("🔐 Conexión a IQ Option")
    if 'iq_api' not in st.session_state:
        st.session_state.iq_api = None
        st.session_state.usuario_conectado = False

    if not st.session_state.usuario_conectado:
        email_user = st.text_input("Email", key="email_login")
        password_user = st.text_input("Contraseña", type="password", key="password_login")
        tipo_cuenta = st.selectbox("Tipo de cuenta", ["PRACTICE", "REAL"])
        if st.button("Conectar"):
            with st.spinner("Conectando..."):
                api = IQ_Option(email_user, password_user)
                try:
                    check, reason = api.connect()
                except Exception as e:
                    st.error(f"Excepción durante connect(): {e}")
                    st.stop()
                if check:
                    try:
                        api.change_balance(tipo_cuenta)
                        balance = api.get_balance()
                        st.session_state.iq_api = api
                        st.session_state.usuario_conectado = True
                        st.session_state.email_user = email_user
                        st.success(f"✅ Conectado. Balance: {balance}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conectado pero error al obtener balance: {e}")
                        st.session_state.iq_api = api
                        st.session_state.usuario_conectado = True
                        st.session_state.email_user = email_user
                        st.warning("Conectado, pero no se pudo verificar balance. Puede funcionar.")
                        st.rerun()
                else:
                    st.error(f"Error de conexión: {reason}")
                    st.info("Verifica credenciales y tipo de cuenta. Si usas cuenta demo, asegúrate de seleccionar PRACTICE.")
    else:
        st.success(f"Conectado como: {st.session_state.email_user}")
        if st.button("Desconectar"):
            if st.session_state.iq_api:
                st.session_state.iq_api.logout()
            st.session_state.iq_api = None
            st.session_state.usuario_conectado = False
            st.rerun()

# ========== CONTROLES DE ENTRENAMIENTO Y CONFIGURACIÓN ==========
if st.session_state.usuario_conectado:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Entrenamiento de IA")
    lotes_input = st.sidebar.number_input(
        "Número de lotes a descargar (cada lote ≈ 1000 velas)",
        min_value=1,
        max_value=2000,
        value=LOTES_POR_DEFECTO,
        step=100
    )
    if st.sidebar.button("📥 Descargar nuevos datos y reentrenar IA", type="primary"):
        with st.status("Iniciando proceso de entrenamiento...", expanded=True) as status:
            status.update(label="Cargando datos existentes...")
            df_existente = cargar_datos_existentes()
            st.write(f"Datos existentes: {len(df_existente)} velas")
            status.update(label="Descargando nuevos datos...")
            df_nuevo = descargar_datos(st.session_state.iq_api, ACTIVO_DESCARGAR, int(lotes_input))
            if df_nuevo.empty:
                status.update(label="No se descargaron velas nuevas.", state="error")
                st.warning("La descarga no devolvió velas. Puede que el mercado esté cerrado o el activo no esté disponible.")
                st.stop()
            st.write(f"Nuevas velas descargadas: {len(df_nuevo)}")
            status.update(label="Combinando datos...")
            df_combinado = combinar_datos(df_existente, df_nuevo)
            st.write(f"Total después de combinar: {len(df_combinado)} velas")
            df_combinado.to_csv(CSV_FILE, index=False)
            status.update(label="Datos guardados. Entrenando modelo...")
            modelo, scaler = entrenar_modelo_con_datos(df_combinado)
            if modelo is None:
                status.update(label="Entrenamiento fallido por datos insuficientes.", state="error")
                st.stop()
            else:
                status.update(label="✅ Entrenamiento completado con éxito", state="complete")
                st.success("Nuevo modelo guardado. La aplicación se recargará para usarlo.")
                time.sleep(2)
                st.rerun()

    # Configuración del umbral de confianza para la IA
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Configuración de señales")
    umbral_confianza = st.sidebar.slider(
        "Umbral de confianza para usar IA",
        min_value=0.5,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Si la probabilidad máxima de la IA es menor a este valor, se usará la estrategia de fuerza/volumen."
    )
    st.session_state.umbral_confianza = umbral_confianza

# ========== VERIFICACIÓN DE MODELOS PARA OPERAR ==========
modelos_existen = os.path.exists(MODELO_FILE) and os.path.exists(SCALER_FILE)

if not modelos_existen:
    if st.session_state.usuario_conectado:
        st.warning("⚠️ Los archivos del modelo de IA no existen. Usa el panel lateral para descargar datos y entrenar el modelo por primera vez.")
        # No detenemos la ejecución, el bot funcionará solo con estrategia de fuerza/volumen
    else:
        st.info("🔌 Conéctate a IQ Option para poder descargar datos y entrenar el modelo.")
        st.stop()
else:
    st.success("✅ Modelo de IA cargado y listo para operar.")

# ========== CLASES DEL BOT ==========

class DataManager:
    def __init__(self):
        self.historial = {}
        self.ultima_actualizacion_activos = 0
        self.activos_cache = []

    def obtener_activos_disponibles(self):
        ahora = time.time()
        if ahora - self.ultima_actualizacion_activos > 300:
            if st.session_state.usuario_conectado and st.session_state.iq_api:
                api = st.session_state.iq_api
                try:
                    api.update_ACTIVES_OPCODE()
                    activos_dict = api.get_all_ACTIVES_OPCODE()
                    self.activos_cache = [nombre for nombre in activos_dict.keys() if "-OTC" in nombre]
                except Exception as e:
                    st.warning(f"Error actualizando activos: {e}")
                    self.activos_cache = []
            else:
                self.activos_cache = []
            self.ultima_actualizacion_activos = ahora
        return self.activos_cache

    def obtener_velas(self, activo, count=VELAS_HISTORICAS):
        if st.session_state.usuario_conectado and st.session_state.iq_api:
            try:
                api = st.session_state.iq_api
                end_from = int(time.time())
                velas = api.get_candles(activo, 60, count, end_from)
                if velas:
                    df = pd.DataFrame(velas)
                    df = df[['from', 'open', 'max', 'min', 'close', 'volume']]
                    df.columns = ['from', 'open', 'high', 'low', 'close', 'volume']
                    self.historial[activo] = df
                    return df
            except Exception as e:
                st.warning(f"Error obteniendo velas de {activo}, usando simulación: {e}")
        # Fallback a simulación
        return self._simular_velas(activo, count)

    def _simular_velas(self, activo, count):
        if activo not in self.historial or len(self.historial[activo]) < count:
            precio = np.random.uniform(1.0, 1.2)
            velas = []
            for _ in range(count):
                open_p = precio
                close_p = open_p + np.random.normal(0, 0.001)
                high_p = max(open_p, close_p) + np.random.uniform(0, 0.0005)
                low_p = min(open_p, close_p) - np.random.uniform(0, 0.0005)
                volume = np.random.randint(100, 1000)
                velas.append([open_p, high_p, low_p, close_p, volume])
                precio = close_p
            df = pd.DataFrame(velas, columns=['open','high','low','close','volume'])
            self.historial[activo] = df
        else:
            df = self.historial[activo].tail(count).reset_index(drop=True)
        return df

class IQOptionBot:
    def __init__(self):
        self.data_manager = DataManager()
        # Si los modelos existen, cargar estrategia con IA; si no, usar solo fuerza/volumen
        if modelos_existen:
            self.estrategia = EstrategiaAvanzada(MODELO_FILE, SCALER_FILE, ventana=20)
        else:
            self.estrategia = None
        self.historial_analisis = []
        self.ultima_senal = None
        self.contador_ciclos = 0

    def ejecutar_ciclo(self):
        activos = self.data_manager.obtener_activos_disponibles()
        if not activos:
            return
        self.contador_ciclos += 1
        for activo in activos:
            try:
                velas = self.data_manager.obtener_velas(activo)
                if velas is None or len(velas) < 20:
                    continue

                # Obtener análisis de IA si está disponible
                if self.estrategia is not None:
                    analisis_ia = self.estrategia.analizar_activo(velas)
                    confianza_ia = max(analisis_ia.get('prob_CALL', 0), analisis_ia.get('prob_PUT', 0))
                else:
                    analisis_ia = None
                    confianza_ia = 0

                # Umbral de confianza (configurable por el usuario)
                umbral = st.session_state.get('umbral_confianza', 0.65)

                # Decidir qué análisis usar
                if analisis_ia is not None and confianza_ia >= umbral:
                    analisis = analisis_ia
                    analisis['fuente'] = 'IA'
                else:
                    analisis_fv = analizar_fuerza_volumen(velas)
                    if analisis_fv is not None:
                        analisis = analisis_fv
                        analisis['fuente'] = 'Fuerza/Volumen'
                    else:
                        # Si falla el análisis de fuerza/volumen, omitir
                        continue

                analisis['activo'] = activo
                analisis['timestamp'] = datetime.now().strftime('%H:%M:%S')
                self.historial_analisis.append(analisis)
                if len(self.historial_analisis) > 50:
                    self.historial_analisis.pop(0)

                if analisis.get('es_bueno', False):
                    self.ultima_senal = analisis
                    # Aquí podrías ejecutar la orden real
                    # self._ejecutar_orden(analisis)
            except Exception as e:
                print(f"Error en {activo}: {e}")

    # def _ejecutar_orden(self, analisis):
    #     pass

# ========== INICIALIZAR BOT ==========
if 'bot' not in st.session_state:
    st.session_state.bot = IQOptionBot()
    st.session_state.ultimo_segundo = -1

bot = st.session_state.bot

# ========== BUCLE PRINCIPAL (se ejecuta en cada refresh) ==========
now = datetime.now()
segundo_actual = now.second

if segundo_actual == SEGUNDO_EJECUCION and st.session_state.ultimo_segundo != segundo_actual:
    bot.ejecutar_ciclo()
    st.session_state.ultimo_segundo = segundo_actual

# ========== INTERFAZ DE USUARIO ==========
col1, col2 = st.columns([3,1])
with col1:
    if st.session_state.usuario_conectado:
        st.markdown(f"## ⚡ **BOT ACTIVO - CONECTADO**")
    else:
        st.markdown(f"## 🔍 **BOT EN MODO DEMO (sin conexión)**")
with col2:
    st.markdown(f"**{now.strftime('%Y-%m-%d %H:%M:%S')}**")
    st.markdown(f"**Ciclo #{bot.contador_ciclos}**")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Últimos análisis")
    activos = bot.data_manager.obtener_activos_disponibles()
    st.metric("Activos OTC", len(activos))
    if bot.historial_analisis:
        df_hist = pd.DataFrame(bot.historial_analisis[-10:])
        # Mostrar columnas relevantes
        columnas_mostrar = ['timestamp', 'activo', 'fuente', 'sentimiento', 'fuerza', 'prob_CALL', 'prob_PUT']
        # Asegurar que existan
        df_display = df_hist[[c for c in columnas_mostrar if c in df_hist.columns]].copy()
        if 'fuerza' in df_display.columns:
            df_display['fuerza'] = df_display['fuerza'].apply(lambda x: f"{x:.2%}")
        if 'prob_CALL' in df_display.columns:
            df_display['prob_CALL'] = df_display['prob_CALL'].apply(lambda x: f"{x:.2%}")
        if 'prob_PUT' in df_display.columns:
            df_display['prob_PUT'] = df_display['prob_PUT'].apply(lambda x: f"{x:.2%}")
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("Esperando análisis...")

with col_right:
    st.subheader("🔔 Última señal")
    if bot.ultima_senal:
        s = bot.ultima_senal
        fuente = s.get('fuente', 'IA')
        st.markdown(f"""
        - **Activo:** {s['activo']} - **{s['sentimiento']}** ({fuente})
        - **Probabilidad:** CALL {s['prob_CALL']:.2%} / PUT {s['prob_PUT']:.2%}
        - **Fuerza:** {s['fuerza']:.2%} - {s['magnitud_esperada']}
        - **Volumen:** {s['volumen']:.2f}
        - **Tendencia:** {'Sí' if s['tiene_tendencia'] else 'No'}
        """)
        if s['es_bueno']:
            st.success("✅ SEÑAL FAVORABLE")
    else:
        st.info("Esperando señal...")

with st.expander("📋 Activos OTC detectados"):
    if activos:
        st.write(", ".join(activos))
    else:
        st.write("No hay activos disponibles.")

# ========== REFRESCO AUTOMÁTICO (COMPATIBLE) ==========
if hasattr(st, 'autorefresh'):
    st.autorefresh(interval=1000, key="autorefresh")
elif hasattr(st, 'experimental_autorefresh'):
    st.experimental_autorefresh(interval=1000, key="autorefresh")
else:
    st.warning("""
        ⚠️ Tu versión de Streamlit no soporta autorefresh automático.
        La página no se actualizará cada segundo. 
        Considera actualizar Streamlit: `pip install --upgrade streamlit`
    """)
