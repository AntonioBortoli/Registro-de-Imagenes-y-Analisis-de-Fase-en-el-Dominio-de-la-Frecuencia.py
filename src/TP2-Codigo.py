

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v3 as iio

def leer_imagen_segura(ruta, flag_cv2):

    try:
        array_bytes = np.fromfile(ruta, dtype=np.uint8)
        img = cv2.imdecode(array_bytes, flag_cv2)
        if img is None:
            raise ValueError("cv2.imdecode retornó None")
        return img
    except Exception as e:
        print(f"  [ERROR] No se pudo leer '{ruta}': {e}")
        return None

# CONSIGNA 1 — TRASLACIÓN
def correlacion_fase(img1, img2):

    F1 = fft2(img1)
    F2 = fft2(img2)

    # Espectro cruzado normalizado: solo conserva información de FASE
    # El +1e-8 evita división por cero en regiones de baja energía
    espectro_cruzado = (F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-8)

    # IDFT → delta de Dirac en la posición del desplazamiento
    correlacion = np.real(ifft2(espectro_cruzado))

    # Localizar el pico (posición del máximo)
    y0, x0 = np.unravel_index(np.argmax(correlacion), correlacion.shape)

    # Corrección de wrap-around: si el pico está en la segunda mitad del array,
    # corresponde a un desplazamiento negativo (el espectro es periódico)
    if x0 > img1.shape[1] // 2:
        x0 -= img1.shape[1]
    if y0 > img1.shape[0] // 2:
        y0 -= img1.shape[0]

    return x0, y0, correlacion  # también devolvemos la correlación para graficar


def traslacion_espacial(img1, img2, rango=50):

    H, W = img1.shape
    mejor_dx, mejor_dy = 0, 0
    min_sad = np.inf

    t_inicio = time.time()

    for dy in range(-rango, rango + 1):
        for dx in range(-rango, rango + 1):
            # Construimos la región de solapamiento entre img1 e img2 desplazada
            y1_ini = max(0, dy);   y1_fin = min(H, H + dy)
            x1_ini = max(0, dx);   x1_fin = min(W, W + dx)
            y2_ini = max(0, -dy);  y2_fin = min(H, H - dy)
            x2_ini = max(0, -dx);  x2_fin = min(W, W - dx)

            region1 = img1[y1_ini:y1_fin, x1_ini:x1_fin]
            region2 = img2[y2_ini:y2_fin, x2_ini:x2_fin]

            if region1.size == 0:
                continue

            sad = np.sum(np.abs(region1.astype(float) - region2.astype(float)))

            if sad < min_sad:
                min_sad = sad
                mejor_dx, mejor_dy = dx, dy

    tiempo_total = time.time() - t_inicio
    return mejor_dx, mejor_dy, tiempo_total

# CONSIGNA 2 — ROTACIÓN

def registrar_rotacion(img1, img2):

    # Magnitud del espectro centrado en la frecuencia cero
    mag1 = np.log1p(np.abs(fftshift(fft2(img1))))
    mag2 = np.log1p(np.abs(fftshift(fft2(img2))))
    # Nota: usamos log1p para comprimir el rango dinámico y mejorar la
    # detección del pico en la correlación de fase.

    centro = (img1.shape[1] // 2, img1.shape[0] // 2)
    max_radio = np.sqrt(centro[0]**2 + centro[1]**2)

    # Mapeo a coordenadas polares: la dimensión vertical → θ, horizontal → ρ
    polar1 = cv2.linearPolar(mag1.astype(np.float32), centro, max_radio,
                             cv2.WARP_FILL_OUTLIERS)
    polar2 = cv2.linearPolar(mag2.astype(np.float32), centro, max_radio,
                             cv2.WARP_FILL_OUTLIERS)

    # La correlación de fase detecta el desplazamiento en θ
    _, desp_theta, _ = correlacion_fase(polar1.astype(float),
                                        polar2.astype(float))

    # Conversión: desplazamiento en píxeles → ángulo en grados
    # El eje θ del mapa polar tiene shape[0] píxeles para 360°
    angulo = (desp_theta * 360.0) / polar1.shape[0]

    return angulo, mag1, mag2, polar1, polar2


def registrar_escala(img1, img2):

    mag1 = np.log1p(np.abs(fftshift(fft2(img1))))
    mag2 = np.log1p(np.abs(fftshift(fft2(img2))))

    centro = (img1.shape[1] // 2, img1.shape[0] // 2)
    max_radio = np.sqrt(centro[0]**2 + centro[1]**2)

    # logPolar: eje horizontal → log(ρ), eje vertical → θ
    logpolar1 = cv2.logPolar(mag1.astype(np.float32), centro, max_radio,
                             cv2.WARP_FILL_OUTLIERS)
    logpolar2 = cv2.logPolar(mag2.astype(np.float32), centro, max_radio,
                             cv2.WARP_FILL_OUTLIERS)

    # El desplazamiento en ρ contiene la información de escala
    desp_rho, _, _ = correlacion_fase(logpolar1.astype(float),
                                      logpolar2.astype(float))

    # La escala M usada por logPolar mapea log(ρ) al ancho de la imagen.
    # M = ancho / log(max_radio)  →  desp_rho corresponde a log(s) · M
    M = logpolar1.shape[1] / np.log(max_radio)
    factor_escala = np.exp(desp_rho / M)

    return factor_escala, mag1, mag2, logpolar1, logpolar2



# CONSIGNA 4 — COHERENCIA DE FASE


def busqueda_k_optimo(img6, img7, n_valores=200, k_min=0.1, k_max=10.0):

    # Aplicar FFT a las imágenes originales y extraer magnitud y fase espectrales
    F6 = fftshift(fft2(img6.astype(np.float64)))
    F7 = fftshift(fft2(img7.astype(np.float64)))
    magnitud_espectral = np.abs(F6)
    fase_espectral     = np.angle(F7)

    mejor_k = k_min
    max_nitidez = 0.0
    mejor_imagen = None
    valores_k = np.linspace(k_min, k_max, n_valores)
    curva_nitidez = np.zeros(n_valores)

    for i, k in enumerate(valores_k):
        # Reconstruir el espectro: magnitud de f6, fase de f7 escalada por k
        espectro_reconstruido = magnitud_espectral * np.exp(1j * k * fase_espectral)

        # Transformada inversa → imagen reconstruida (parte real)
        img_recuperada = np.real(ifft2(ifftshift(espectro_reconstruido)))

        # Normalizar a [0, 255] para que Sobel opere en rango estable
        img_norm = cv2.normalize(img_recuperada, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.float32)

        # Métrica de nitidez Tenengrad: energía media del gradiente Sobel
        sobel_x = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
        nitidez = float((sobel_x**2 + sobel_y**2).mean())

        curva_nitidez[i] = nitidez

        if nitidez > max_nitidez:
            max_nitidez = nitidez
            mejor_k = k
            mejor_imagen = img_recuperada.copy()

    return mejor_k, mejor_imagen, curva_nitidez, valores_k



# FUNCIONES DE VISUALIZACIÓN


def graficar_traslacion(img1, img2, correlacion, dx, dy):
    """Visualiza las imágenes y el mapa de correlación de fase."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Consigna 1 — Estimación de Traslación por Correlación de Fase",
                 fontsize=13, fontweight='bold')

    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Imagen 1 (referencia)")
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Imagen 2 (desplazada)")
    axes[1].axis('off')

    # La correlación se muestra centrada para que el pico sea visible
    corr_shift = np.fft.fftshift(correlacion)
    axes[2].imshow(corr_shift, cmap='hot', interpolation='nearest')
    axes[2].set_title(f"Correlación de Fase\nPico en (dx={dx}, dy={dy})")
    # Marcamos el pico en el centro del gráfico (que es donde queda tras fftshift)
    cy, cx = np.array(corr_shift.shape) // 2
    axes[2].plot(cx + dx, cy + dy, 'b+', markersize=15, markeredgewidth=2,
                 label=f"Pico ({dx},{dy})")
    axes[2].legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'resultado_traslacion.png'), dpi=150)
    plt.show()


def graficar_comparacion_algoritmos(tiempo_fase, tiempo_espacial,
                                    dx_fase, dy_fase, dx_esp, dy_esp):
    """Compara tiempos y resultados de ambos algoritmos de traslación."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Consigna 1 — Comparación de Costos Computacionales",
                 fontsize=13, fontweight='bold')

    metodos = ['Correlación de Fase\n(dominio frecuencia)', 'Diferencias Absolutas\n(dominio espacial)']
    tiempos = [tiempo_fase, tiempo_espacial]
    colores = ['steelblue', 'tomato']

    bars = ax.bar(metodos, tiempos, color=colores, width=0.4)
    ax.set_ylabel("Tiempo de ejecución (segundos)", fontsize=11)
    ax.set_title("O(N² log N)  vs  O(rango² · N²)", fontsize=11)

    for bar, t, res in zip(bars, tiempos,
                           [f"dx={dx_fase}, dy={dy_fase}",
                            f"dx={dx_esp}, dy={dy_esp}"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(tiempos)*0.02,
                f"{t:.4f}s\n{res}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'resultado_comparacion_algoritmos.png'), dpi=150)
    plt.show()


def graficar_rotacion(img3, img4, mag1, mag2, polar1, polar2, angulo):
    """Visualiza el proceso de detección de rotación."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Consigna 2 — Registro de Rotación", fontsize=13, fontweight='bold')

    axes[0, 0].imshow(img3, cmap='gray')
    axes[0, 0].set_title("Imagen 3 (referencia)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img4, cmap='gray')
    axes[0, 1].set_title("Imagen 4 (rotada)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mag1, cmap='hot')
    axes[0, 2].set_title("|FFT| centrada — Imagen 3\n(invariante a traslación)")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(mag2, cmap='hot')
    axes[1, 0].set_title("|FFT| centrada — Imagen 4")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(polar1, cmap='hot', aspect='auto')
    axes[1, 1].set_title("Mapa Polar — Imagen 3\n(eje Y = θ, eje X = ρ)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(polar2, cmap='hot', aspect='auto')
    axes[1, 2].set_title(f"Mapa Polar — Imagen 4\n→ Rotación estimada: {angulo:.2f}°")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'resultado_rotacion.png'), dpi=150)
    plt.show()


def graficar_escala(img1, img5, mag1, mag2, logpolar1, logpolar2, factor):
    """Visualiza el proceso de detección de escala."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Consigna 3 — Registro de Escalamiento", fontsize=13, fontweight='bold')

    axes[0, 0].imshow(img1, cmap='gray')
    axes[0, 0].set_title("Imagen 1 (referencia)")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img5, cmap='gray')
    axes[0, 1].set_title("Imagen 5 (reescalada)")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(mag1, cmap='hot')
    axes[0, 2].set_title("|FFT| centrada — Imagen 1")
    axes[0, 2].axis('off')

    axes[1, 0].imshow(mag2, cmap='hot')
    axes[1, 0].set_title("|FFT| centrada — Imagen 5")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(logpolar1, cmap='hot', aspect='auto')
    axes[1, 1].set_title("Mapa Log-Polar — Imagen 1\n(eje X = log(ρ), eje Y = θ)")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(logpolar2, cmap='hot', aspect='auto')
    axes[1, 2].set_title(f"Mapa Log-Polar — Imagen 5\n→ Factor escala: {factor:.4f}")
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'resultado_escala.png'), dpi=150)
    plt.show()


def graficar_coherencia_fase(img_recuperada, k_optimo, curva, valores_k,
                             titulo_extra=""):
    """Visualiza la imagen recuperada y la curva de nitidez vs k."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Consigna 4 — Coherencia de Fase {titulo_extra}",
                 fontsize=13, fontweight='bold')

    # Imagen recuperada normalizada para display
    img_display = cv2.normalize(img_recuperada, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title(f"Imagen recuperada\n(k óptimo = {k_optimo:.2f})")
    axes[0].axis('off')

    axes[1].plot(valores_k, curva, color='steelblue', linewidth=1.5)
    axes[1].axvline(x=k_optimo, color='red', linestyle='--',
                    label=f'k óptimo = {k_optimo:.2f}')
    axes[1].set_xlabel("Factor k")
    axes[1].set_ylabel("Nitidez (suma gradiente Sobel)")
    axes[1].set_title("Curva de nitidez vs factor k")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    nombre = f"resultado_fase{'_tif' if 'TIF' in titulo_extra else '_jpg'}.png"
    plt.savefig(os.path.join(BASE_DIR, nombre), dpi=150)
    plt.show()

# BLOQUE PRINCIPAL
if __name__ == "__main__":

    # Directorio donde están las imágenes (mismo directorio que este script)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  TP2 — Registro de Imágenes y Análisis de Fase")
    print("=" * 60)

    # CONSIGNA 1 — TRASLACIÓN

    print("\n--- CONSIGNA 1: Traslación ---")

    img1 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen1.jpg'), cv2.IMREAD_GRAYSCALE)
    img2 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen2.jpg'), cv2.IMREAD_GRAYSCALE)

    if img1 is not None and img2 is not None:

        # Aseguramos que ambas imágenes tengan el mismo tamaño para la FFT
        H = min(img1.shape[0], img2.shape[0])
        W = min(img1.shape[1], img2.shape[1])
        img1c = img1[:H, :W].astype(float)
        img2c = img2[:H, :W].astype(float)

        # --- Algoritmo de correlación de fase (dominio frecuencia) ---
        t0 = time.time()
        dx, dy, corr = correlacion_fase(img1c, img2c)
        tiempo_fase = time.time() - t0

        print(f"  [Correlación de Fase]")
        print(f"    Desplazamiento detectado: X={dx} px,  Y={dy} px")
        print(f"    Tiempo de ejecución:      {tiempo_fase:.6f} s")
        print(f"    Complejidad:              O(N² log N)")

        # --- Algoritmo espacial (SAD - Suma de Diferencias Absolutas) ---
        print(f"\n  [Algoritmo Espacial — SAD, rango ±50 px]")
        # Usamos imágenes reducidas para que el algoritmo O(N²·rango²) sea viable
        escala_reduccion = 0.25
        img1_peq = cv2.resize(img1c, None, fx=escala_reduccion, fy=escala_reduccion)
        img2_peq = cv2.resize(img2c, None, fx=escala_reduccion, fy=escala_reduccion)
        rango_px = 30

        dx_esp, dy_esp, tiempo_esp = traslacion_espacial(img1_peq, img2_peq, rango=rango_px)
        # Reescalar el desplazamiento al tamaño original
        dx_esp_real = int(round(dx_esp / escala_reduccion))
        dy_esp_real = int(round(dy_esp / escala_reduccion))

        print(f"    Desplazamiento detectado: X={dx_esp_real} px,  Y={dy_esp_real} px")
        print(f"    Tiempo de ejecución:      {tiempo_esp:.6f} s  (imagen al 25%)")
        print(f"    Complejidad:              O(rango² · N²)")

        print(f"\n  [Comparación]")
        if tiempo_esp > 0:
            print(f"    La correlación de fase fue {tiempo_esp/tiempo_fase:.1f}x más rápida.")
        print(f"    Ambos métodos coinciden en el resultado: "
              f"{'✓' if (dx == dx_esp_real and dy == dy_esp_real) else '✗ (diferencia de sub-píxel esperada)'}")

        graficar_traslacion(img1c, img2c, corr, dx, dy)
        graficar_comparacion_algoritmos(tiempo_fase, tiempo_esp,
                                        dx, dy, dx_esp_real, dy_esp_real)
    else:
        print("  ERROR: No se pudieron cargar imagen1.jpg o imagen2.jpg")

    # CONSIGNA 2 — ROTACIÓN

    print("\n--- CONSIGNA 2: Rotación ---")

    img3 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen3.jpg'), cv2.IMREAD_GRAYSCALE)
    img4 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen4.jpg'), cv2.IMREAD_GRAYSCALE)

    if img3 is not None and img4 is not None:

        H = min(img3.shape[0], img4.shape[0])
        W = min(img3.shape[1], img4.shape[1])
        img3c = img3[:H, :W].astype(float)
        img4c = img4[:H, :W].astype(float)

        angulo, mag3, mag4, polar3, polar4 = registrar_rotacion(img3c, img4c)

        print(f"  Ángulo de rotación estimado: {angulo:.2f}°")
        print(f"  (Positivo = sentido antihorario, negativo = horario)")

        graficar_rotacion(img3c, img4c, mag3, mag4, polar3, polar4, angulo)
    else:
        print("  ERROR: No se pudieron cargar imagen3.jpg o imagen4.jpg")


    # CONSIGNA 3 — ESCALAMIENTO

    print("\n--- CONSIGNA 3: Escalamiento ---")

    img5 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen5.jpg'), cv2.IMREAD_GRAYSCALE)

    if img5 is not None and img1 is not None:

        H = min(img1.shape[0], img5.shape[0])
        W = min(img1.shape[1], img5.shape[1])
        img1s = img1[:H, :W].astype(float)
        img5s = img5[:H, :W].astype(float)

        factor, mag1s, mag5, lp1, lp5 = registrar_escala(img5s, img1s)

        print(f"  Factor de escala estimado (img5 respecto a img1): {factor:.4f}")
        if factor > 1:
            print(f"  → img5 está ampliada {factor:.2f}x respecto a img1")
        else:
            print(f"  → img5 está reducida a {factor:.2f}x respecto a img1")

        graficar_escala(img1s, img5s, mag1s, mag5, lp1, lp5, factor)
    else:
        print("  ERROR: No se pudieron cargar imagen5.jpg o imagen1.jpg")


    # CONSIGNA 4 — COHERENCIA DE FASE

    print("\n--- CONSIGNA 4: Coherencia de Fase ---")

    # --- 4a. Con archivos TIF (sin pérdidas) ---
    print("\n  [4a. Archivos TIF — sin pérdidas]")
    try:
        img6_tif = iio.imread(os.path.join(BASE_DIR, 'imagen6.tif')).astype(np.float64)
        img7_tif = iio.imread(os.path.join(BASE_DIR, 'imagen7.tif')).astype(np.float64)

        # Si las imágenes TIF son color, convertir a escala de grises
        if img6_tif.ndim == 3:
            img6_tif = cv2.cvtColor(img6_tif.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
        if img7_tif.ndim == 3:
            img7_tif = cv2.cvtColor(img7_tif.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Pasar las imágenes originales: busqueda_k_optimo aplica FFT internamente
        k_tif, img_tif, curva_tif, vals_tif = busqueda_k_optimo(
            img6_tif, img7_tif, n_valores=200, k_min=0.1, k_max=10.0)

        print(f"    Factor k óptimo (TIF): {k_tif:.2f}")
        graficar_coherencia_fase(img_tif, k_tif, curva_tif, vals_tif, "(TIF — sin pérdidas)")

    except FileNotFoundError:
        print("  ERROR: No se encontraron imagen6.tif o imagen7.tif")
    except Exception as e:
        print(f"  ERROR procesando TIFs: {e}")

    # ---  Con archivos JPG (con pérdidas) ---
    print("\n  [4b. Archivos JPG — con pérdidas de compresión]")
    img6_jpg = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen6.jpg'), cv2.IMREAD_GRAYSCALE)
    img7_jpg = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen7.jpg'), cv2.IMREAD_GRAYSCALE)

    if img6_jpg is not None and img7_jpg is not None:

        # Asegurar mismo tamaño y pasar las imágenes originales
        H = min(img6_jpg.shape[0], img7_jpg.shape[0])
        W = min(img6_jpg.shape[1], img7_jpg.shape[1])
        img6_jpg_c = img6_jpg[:H, :W].astype(np.float64)
        img7_jpg_c = img7_jpg[:H, :W].astype(np.float64)

        # Pasar las imágenes originales: busqueda_k_optimo aplica FFT internamente
        k_jpg, img_jpg, curva_jpg, vals_jpg = busqueda_k_optimo(
            img6_jpg_c, img7_jpg_c, n_valores=200, k_min=0.1, k_max=10.0)

        print(f"    Factor k óptimo (JPG): {k_jpg:.2f}")
        graficar_coherencia_fase(img_jpg, k_jpg, curva_jpg, vals_jpg, "(JPG — con pérdidas)")



    else:
        print("  ERROR: No se pudieron cargar imagen6.jpg o imagen7.jpg")

    print("\n" + "=" * 60)
    print("  Análisis completado. Resultados guardados como PNG.")
    print("=" * 60)
