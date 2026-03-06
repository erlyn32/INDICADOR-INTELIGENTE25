import time
from datetime import datetime
from data_provider import (
    get_current_vela,
    get_historical_velas,
    detectar_trampa,
    calcular_probabilidad_y_fuerza
)

def formatear_senal(direccion, probabilidad, fuerza, con_advertencia=False):
    """
    Devuelve el string formateado según lo solicitado.
    """
    arriba_abajo = "MÁS ARRIBA" if direccion == "COMPRA" else "MÁS ABAJO"
    senal = f"{direccion} → Probabilidad {probabilidad:.0f}% / 100. Fuerza detectada: {fuerza:.2f}. Pronóstico: vela siguiente terminará {arriba_abajo}."
    if con_advertencia:
        senal += " ¡PRECAUCIÓN: vela siguiente podría ser de gran tamaño (alto potencial de movimiento fuerte)!"
    return senal

def main():
    print("Sistema de señales para velas de 1 minuto (OTC)")
    print("Esperando al segundo 58 para generar señal...\n")

    # Cargar velas históricas iniciales (para tener contexto)
    historico = get_historical_velas(minutes=20)

    while True:
        # Esperar hasta el segundo 58 del minuto actual
        now = datetime.now()
        # Si ya pasó el segundo 58, esperar al próximo minuto
        if now.second >= 58:
            # Esperar hasta el segundo 58 del próximo minuto
            seconds_to_wait = (60 - now.second) + 58
            time.sleep(seconds_to_wait)
        else:
            # Esperar hasta el segundo 58
            seconds_to_wait = 58 - now.second
            time.sleep(seconds_to_wait)

        # En este punto, es segundo 58 aproximadamente
        # Obtener la última vela (la que acaba de cerrar, porque estamos a 58-59)
        # Nota: Si tu API da la vela en tiempo real, asegúrate de tomar la vela completa del minuto anterior.
        # Aquí simulamos que obtenemos la vela que acaba de cerrar.
        vela_actual = get_current_vela()

        # Añadir al histórico y mantener solo las últimas 20
        historico.append(vela_actual)
        if len(historico) > 20:
            historico.pop(0)

        # Detectar trampas
        trampa = detectar_trampa(vela_actual, historico[:-1])  # excluimos la actual para no mirar futuro

        # Calcular probabilidades y fuerza
        prob_compra, prob_venta, fuerza = calcular_probabilidad_y_fuerza(vela_actual, historico[:-1])

        # Decidir dirección
        direccion = None
        probabilidad = 0
        umbral = 62  # mínimo para operar

        # Ajustar por trampa: si hay trampa, puede anular o reforzar
        if trampa:
            if "VENTA" in trampa:  # trampa alcista da señal de venta
                if prob_venta > umbral:
                    direccion = "VENTA"
                    probabilidad = prob_venta
                else:
                    # Si la trampa indica venta pero la probabilidad no alcanza, podemos igual mostrar venta pero con menor confianza
                    direccion = "VENTA"
                    probabilidad = prob_venta
            elif "COMPRA" in trampa:  # trampa bajista da señal de compra
                if prob_compra > umbral:
                    direccion = "COMPRA"
                    probabilidad = prob_compra
                else:
                    direccion = "COMPRA"
                    probabilidad = prob_compra
        else:
            # Sin trampa, elegir la de mayor probabilidad si supera umbral
            if prob_compra > umbral and prob_compra > prob_venta:
                direccion = "COMPRA"
                probabilidad = prob_compra
            elif prob_venta > umbral and prob_venta > prob_compra:
                direccion = "VENTA"
                probabilidad = prob_venta

        # Si no hay dirección clara, podemos optar por no operar o elegir la mayor aunque no supere umbral
        if direccion is None:
            # Por defecto, elegir la de mayor probabilidad (aunque sea baja)
            if prob_compra > prob_venta:
                direccion = "COMPRA"
                probabilidad = prob_compra
            else:
                direccion = "VENTA"
                probabilidad = prob_venta

        # Advertencia de vela grande si probabilidad > 75%
        advertencia = probabilidad > 75

        # Mostrar señal
        print(formatear_senal(direccion, probabilidad, fuerza, advertencia))
        if trampa:
            print(f"   (Trampa detectada: {trampa})")
        print("-" * 50)

        # Pequeña pausa para no repetir en el mismo segundo 58 (opcional)
        time.sleep(2)

if __name__ == "__main__":
    main()
