# entrenador_ia.py
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from data_provider import EstrategiaAvanzada  # Asegúrate de que data_provider.py exista

def entrenar_modelo(csv_path='iqoption_data_EURUSD_60.csv',
                    modelo_path='modelo_xgb.pkl',
                    scaler_path='scaler.pkl',
                    ventana=20,
                    test_split=0.2):
    """
    Carga el CSV, genera features, entrena XGBoost y guarda modelo + scaler.
    """
    print("Cargando datos...")
    df = pd.read_csv(csv_path)

    # Ordenar por tiempo si existe columna 'from'
    if 'from' in df.columns:
        df = df.sort_values('from').reset_index(drop=True)

    # Crear target: 1 si la siguiente vela cierra más arriba, 0 si más abajo
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna(subset=['target']).reset_index(drop=True)

    print(f"Total muestras con target: {len(df)}")

    # Instanciar estrategia para calcular features (sin modelo cargado)
    estrategia = EstrategiaAvanzada(modelo_path=None, scaler_path=None, ventana=ventana)

    features_list = []
    targets_list = []

    for i in range(ventana, len(df)):
        ventana_df = df.iloc[i-ventana:i][['open', 'high', 'low', 'close', 'volume']]
        feat = estrategia.calcular_features(ventana_df)
        if feat is not None:
            features_list.append(feat)
            targets_list.append(df.iloc[i]['target'])

    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)

    print(f"Muestras con features generadas: {len(X)}")

    if len(X) == 0:
        raise ValueError("No se generaron features. Revisa la longitud de los datos.")

    # División temporal (80% entrenamiento, 20% prueba)
    split = int(len(X) * (1 - test_split))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print(f"Entrenamiento: {len(X_train)} muestras | Prueba: {len(X_test)} muestras")

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar XGBoost
    modelo = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        use_label_encoder=False
    )
    modelo.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

    # Evaluación rápida
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Precisión en test: {acc:.4f}")

    # Guardar modelo y scaler
    with open(modelo_path, 'wb') as f:
        pickle.dump(modelo, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Modelo guardado en {modelo_path} y {scaler_path}")
    return modelo, scaler

if __name__ == "__main__":
    # Si se ejecuta directamente, realizar el entrenamiento
    entrenar_modelo()
