# data_provider.py
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

class EstrategiaAvanzada:
    """
    Clase que implementa la estrategia de análisis de fuerza y volumen
    con Machine Learning para velas de 1 minuto.
    """
    def __init__(self, modelo_path=None, scaler_path=None, ventana=20):
        """
        Inicializa la estrategia. Si se proporcionan rutas válidas, intenta cargar el modelo.
        Si no, la estrategia opera sin modelo (útil durante entrenamiento).
        """
        self.ventana = ventana
        self.umbral_probabilidad = 0.65
        self.umbral_fuerza = 0.5
        self.umbral_tendencia = 0.6
        self.modelo = None
        self.scaler = None

        # Solo intenta cargar si las rutas no son None
        if modelo_path is not None and scaler_path is not None:
            try:
                with open(modelo_path, 'rb') as f:
                    self.modelo = pickle.load(f)
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Modelo y scaler cargados desde archivos.")
            except FileNotFoundError:
                print(f"Advertencia: Modelo o Scaler no encontrados en {modelo_path} y {scaler_path}. Operando sin IA en esta instancia.")
            except Exception as e:
                print(f"Error al cargar modelo/scaler: {e}. Operando sin IA en esta instancia.")
        else:
            # Si se pasan None, simplemente no se carga modelo (caso entrenamiento)
            print("Inicializando estrategia sin modelo (modo entrenamiento o sin IA).")

    def calcular_features(self, velas):
        """
        Calcula todos los features a partir de un DataFrame de velas.
        velas: DataFrame con columnas: ['open','high','low','close','volume']
        Debe tener al menos self.ventana filas.
        Retorna un diccionario con los features para la última vela.
        """
        if len(velas) < self.ventana:
            return None

        # Copia para no modificar original
        df = velas.copy().tail(self.ventana).reset_index(drop=True)

        # 1. Fuerza de vela (candle_strength)
        df['body'] = abs(df['close'] - df['open'])
        df['range_candle'] = df['high'] - df['low']
        df['candle_strength'] = df['body'] / (df['range_candle'] + 1e-6)

        # 2. Presión de mercado (pressure_ratio)
        df['buy_volume'] = df.apply(lambda row: row['volume'] if row['close'] > row['open'] else 0, axis=1)
        df['sell_volume'] = df.apply(lambda row: row['volume'] if row['close'] < row['open'] else 0, axis=1)
        # Acumulado en las últimas N velas
        buy_sum = df['buy_volume'].sum()
        sell_sum = df['sell_volume'].sum()
        pressure_ratio = buy_sum / (sell_sum + 1e-6)

        # 3. Velocidad del precio (price_speed): (close_actual - close_5_velas_atras)/5
        if len(df) >= 5:
            price_speed = (df['close'].iloc[-1] - df['close'].iloc[-5]) / 5.0
        else:
            price_speed = 0.0

        # 4. Eficiencia del movimiento (efficiency)
        price_move = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
        volume_last = df['volume'].iloc[-1]
        efficiency = price_move / (volume_last + 1e-6)

        # 5. Compresión de volatilidad (volatility_std) - std de cierres últimos 10
        if len(df) >= 10:
            volatility_std = df['close'].tail(10).std()
        else:
            volatility_std = 0.0

        # 6. Posición del precio en el rango (position_in_range)
        low_10 = df['low'].tail(10).min()
        high_10 = df['high'].tail(10).max()
        position_in_range = (df['close'].iloc[-1] - low_10) / (high_10 - low_10 + 1e-6)

        # 7. Trampas
        if len(df) >= 2:
            high_prev = df['high'].iloc[-2]
            low_prev = df['low'].iloc[-2]
            close_prev = df['close'].iloc[-2]
            high_curr = df['high'].iloc[-1]
            low_curr = df['low'].iloc[-1]
            close_curr = df['close'].iloc[-1]
            bull_trap = 1 if (high_curr > high_prev and close_curr < high_prev) else 0
            bear_trap = 1 if (low_curr < low_prev and close_curr > low_prev) else 0
        else:
            bull_trap = 0
            bear_trap = 0

        # 8. Absorción de volumen
        avg_volume = df['volume'].mean()
        range_last = df['high'].iloc[-1] - df['low'].iloc[-1]
        absorption = 1 if (volume_last > avg_volume * 1.5 and price_move < range_last * 0.2) else 0

        # 9. Indicadores de tendencia (para el nuevo control)
        # Pendiente de regresión lineal en los últimos 5 cierres
        if len(df) >= 5:
            y = df['close'].tail(5).values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            tendencia_direccion = 1 if slope > 0 else -1 if slope < 0 else 0
            tendencia_fuerza = abs(slope) / (df['close'].mean() + 1e-6)  # normalizado
        else:
            tendencia_direccion = 0
            tendencia_fuerza = 0.0

        # 10. SMA 5 y 10
        sma5 = df['close'].tail(5).mean()
        sma10 = df['close'].tail(10).mean() if len(df) >= 10 else sma5
        tendencia_sma = 1 if sma5 > sma10 else -1 if sma5 < sma10 else 0

        # Combinar tendencia (usamos pendiente y sma)
        tiene_tendencia = (tendencia_fuerza > self.umbral_tendencia) and (tendencia_direccion != 0)

        # Construir vector de features para el modelo (orden debe coincidir con entrenamiento)
        features_dict = {
            'candle_strength': df['candle_strength'].iloc[-1],
            'pressure_ratio': pressure_ratio,
            'price_speed': price_speed,
            'efficiency': efficiency,
            'volatility_std': volatility_std,
            'position_in_range': position_in_range,
            'bull_trap': bull_trap,
            'bear_trap': bear_trap,
            'absorption': absorption,
            'tendencia_fuerza': tendencia_fuerza,
            'tendencia_direccion': tendencia_direccion,
            'sma5': sma5,
            'sma10': sma10,
        }
        # También podemos incluir valores medios de algunas variables para dar más contexto
        features_dict['avg_candle_strength'] = df['candle_strength'].mean()
        features_dict['avg_pressure_ratio'] = pressure_ratio  # ya es acumulado
        features_dict['volatility_std_ratio'] = volatility_std / (df['close'].mean() + 1e-6)

        return features_dict

    def preparar_vector_modelo(self, features_dict):
        """
        Convierte el diccionario de features en un array 2D para el modelo,
        en el orden esperado por el scaler y el modelo.
        """
        # Definir el orden de las columnas (debe coincidir con el entrenamiento)
        column_order = [
            'candle_strength', 'pressure_ratio', 'price_speed', 'efficiency',
            'volatility_std', 'position_in_range', 'bull_trap', 'bear_trap',
            'absorption', 'tendencia_fuerza', 'tendencia_direccion', 'sma5',
            'sma10', 'avg_candle_strength', 'avg_pressure_ratio', 'volatility_std_ratio'
        ]
        vector = [features_dict[col] for col in column_order]
        return np.array(vector).reshape(1, -1)

    def predecir(self, velas):
        """
        Dado un DataFrame de velas, retorna:
        - probabilidad de CALL
        - probabilidad de PUT
        - dirección predicha (1 CALL, 0 PUT)
        - fuerza de la señal (signal_strength)
        - diccionario completo de features (para depuración)
        """
        if self.modelo is None or self.scaler is None:
            # Si no hay modelo, devolvemos valores por defecto
            return 0.5, 0.5, 0, 0.0, None

        features_dict = self.calcular_features(velas)
        if features_dict is None:
            return 0.5, 0.5, 0, 0.0, None

        X = self.preparar_vector_modelo(features_dict)
        X_scaled = self.scaler.transform(X)
        probas = self.modelo.predict_proba(X_scaled)[0]  # [prob_put, prob_call] según el modelo
        # Asumimos que el modelo devuelve [prob_0 (PUT), prob_1 (CALL)]
        prob_put, prob_call = probas[0], probas[1]
        direccion = 1 if prob_call > prob_put else 0

        # Calcular fuerza de señal (signal_strength) combinando probabilidad, presión y fuerza de vela
        peso_prob = 0.5
        peso_pressure = 0.3
        peso_strength = 0.2
        # Normalizar presión: convertir a ratio en [0,1] aproximadamente
        pressure_norm = min(features_dict['pressure_ratio'] / 5.0, 1.0)  # cap en 5
        strength_norm = features_dict['candle_strength']  # ya entre 0 y 1

        if direccion == 1:  # CALL
            signal_strength = (prob_call * peso_prob +
                               pressure_norm * peso_pressure +
                               strength_norm * peso_strength)
        else:  # PUT
            # Para PUT, usamos la presión inversa (venta) y la fuerza también es positiva
            pressure_inv = 1.0 - pressure_norm  # cuanto mayor presión compra, peor para PUT
            signal_strength = (prob_put * peso_prob +
                               pressure_inv * peso_pressure +
                               strength_norm * peso_strength)

        return prob_call, prob_put, direccion, signal_strength, features_dict

    def analizar_activo(self, velas):
        """
        Función principal que retorna el resultado del análisis.
        velas: DataFrame con las últimas velas del activo.
        Retorna un diccionario con:
            activo: str (debe ser pasado externamente)
            fuerza: float (signal_strength)
            volumen: float (presión de mercado)
            sentimiento: str ('CALL'/'PUT'/'SIN OPERACIÓN')
            es_bueno: bool
            prob_CALL: float
            prob_PUT: float
            tiene_tendencia: bool
            magnitud_esperada: str (descripción)
        """
        prob_call, prob_put, direccion, signal_strength, features = self.predecir(velas)

        # Determinar sentimiento
        if prob_call >= self.umbral_probabilidad and signal_strength >= self.umbral_fuerza:
            sentimiento = 'CALL'
            es_bueno = True
        elif prob_put >= self.umbral_probabilidad and signal_strength >= self.umbral_fuerza:
            sentimiento = 'PUT'
            es_bueno = True
        else:
            sentimiento = 'SIN OPERACIÓN'
            es_bueno = False

        # Determinar si hay tendencia clara (usando features calculados)
        # Podemos usar la combinación de tendencia_fuerza y dirección
        if features:
            tiene_tendencia = features['tendencia_fuerza'] > self.umbral_tendencia
        else:
            tiene_tendencia = False

        # Interpretación de la fuerza para la magnitud esperada
        if signal_strength > 0.9:
            magnitud = "¡Vela muy grande esperada!"
        elif signal_strength > 0.7:
            magnitud = "Vela grande esperada."
        elif signal_strength > 0.5:
            magnitud = "Vela mediana esperada."
        else:
            magnitud = "Vela pequeña esperada."

        # Volumen: usamos pressure_ratio como indicador de volumen
        volumen = features['pressure_ratio'] if features else 0.0

        return {
            'fuerza': signal_strength,
            'volumen': volumen,
            'sentimiento': sentimiento,
            'es_bueno': es_bueno,
            'prob_CALL': prob_call,
            'prob_PUT': prob_put,
            'tiene_tendencia': tiene_tendencia,
            'magnitud_esperada': magnitud,
            'features': features  # opcional, para debug
        }
