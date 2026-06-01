import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import imageio.v3 as iio

# ==========================================
# FUNCIÓN AUXILIAR (Para rutas con tildes)
# ==========================================
def leer_imagen_segura(ruta, flag_cv2):
    try:
        array_bytes = np.fromfile(ruta, dtype=np.uint8)
        return cv2.imdecode(array_bytes, flag_cv2)
    except Exception:
        return None

# ==========================================
# 1. TRASLACIÓN
# ==========================================
def correlacion_fase(img1, img2):
    F1 = fft2(img1)
    F2 = fft2(img2)
    espectro_cruzado = (F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-8)
    correlacion = np.real(ifft2(espectro_cruzado))
    y0, x0 = np.unravel_index(np.argmax(correlacion), correlacion.shape)

    # Ajuste para desplazamientos negativos
    if x0 > img1.shape[1] // 2: x0 -= img1.shape[1]
    if y0 > img1.shape[0] // 2: y0 -= img1.shape[0]
    return x0, y0

# ==========================================
# 2. ROTACIÓN
# ==========================================
def registrar_rotacion(img1, img2):
    mag1 = np.abs(fftshift(fft2(img1)))
    mag2 = np.abs(fftshift(fft2(img2)))

    centro = (img1.shape[1]//2, img1.shape[0]//2)
    max_radio = np.sqrt(centro[0]**2 + centro[1]**2)

    polar1 = cv2.linearPolar(mag1, centro, max_radio, cv2.WARP_FILL_OUTLIERS)
    polar2 = cv2.linearPolar(mag2, centro, max_radio, cv2.WARP_FILL_OUTLIERS)

    _, desp_theta = correlacion_fase(polar1, polar2)
    angulo = (desp_theta * 360.0) / polar1.shape[0]
    return angulo

# ==========================================
# 3. ESCALAMIENTO
# ==========================================
def registrar_escala(img1, img2):
    mag1 = np.abs(fftshift(fft2(img1)))
    mag2 = np.abs(fftshift(fft2(img2)))

    centro = (img1.shape[1]//2, img1.shape[0]//2)
    max_radio = np.sqrt(centro[0]**2 + centro[1]**2)

    logpolar1 = cv2.logPolar(mag1, centro, max_radio, cv2.WARP_FILL_OUTLIERS)
    logpolar2 = cv2.logPolar(mag2, centro, max_radio, cv2.WARP_FILL_OUTLIERS)

    desp_rho, _ = correlacion_fase(logpolar1, logpolar2)
    # Factor de escala ajustado a la constante de OpenCV
    factor_escala = np.exp(desp_rho / (max_radio / np.log(max_radio)))
    return factor_escala

# ==========================================
# 4. COHERENCIA DE FASE
# ==========================================
def busqueda_k_optimo(magnitud, fase_alterada):
    mejor_k = 0
    max_nitidez = 0
    mejor_imagen = None

    valores_k = np.linspace(0.1, 5.0, 50)
    for k in valores_k:
        fase_restaurada = fase_alterada / k
        espectro_reconstruido = magnitud * np.exp(1j * fase_restaurada)

        # Deshacemos el centrado espacial del espectro antes de la inversa
        espectro_reconstruido = ifftshift(espectro_reconstruido)
        img_recuperada = np.real(ifft2(espectro_reconstruido))

        # Filtro pasa-bajos para eliminar el ruido blanco de los desfasajes
        img_suavizada = cv2.GaussianBlur(img_recuperada, (5, 5), 0)

        # Métrica robusta: Magnitud del gradiente espacial (Sobel)
        sobel_x = cv2.Sobel(img_suavizada, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_suavizada, cv2.CV_64F, 0, 1, ksize=3)
        magnitud_sobel = cv2.magnitude(sobel_x, sobel_y)
        nitidez = np.sum(magnitud_sobel)

        if nitidez > max_nitidez:
            max_nitidez = nitidez
            mejor_k = k
            mejor_imagen = img_recuperada

    return mejor_k, mejor_imagen


# ==========================================
# BLOQUE PRINCIPAL (EJECUCIÓN)
# ==========================================
if __name__ == "__main__":
    # Resuelve dinámicamente la ruta donde está guardado este script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    print("Iniciando análisis de TP2...\n")

    # --- 1. Traslación ---
    print("--- 1. Analizando Traslación ---")
    img1 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen1.jpg'), cv2.IMREAD_GRAYSCALE)
    img2 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen2.jpg'), cv2.IMREAD_GRAYSCALE)

    if img1 is not None and img2 is not None:
        dx, dy = correlacion_fase(img1.astype(float), img2.astype(float))
        print(f"Desplazamiento detectado: X={dx}, Y={dy}")
    else:
        print("Error al cargar imagen1.jpg o imagen2.jpg")

    # --- 2. Rotación ---
    print("\n--- 2. Analizando Rotación ---")
    img3 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen3.jpg'), cv2.IMREAD_GRAYSCALE)
    img4 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen4.jpg'), cv2.IMREAD_GRAYSCALE)

    if img3 is not None and img4 is not None:
        angulo = registrar_rotacion(img3.astype(float), img4.astype(float))
        print(f"Ángulo de rotación estimado: {angulo:.2f} grados")
    else:
        print("Error al cargar imagen3.jpg o imagen4.jpg")

    # --- 3. Escalamiento ---
    print("\n--- 3. Analizando Escalamiento ---")
    img5 = leer_imagen_segura(os.path.join(BASE_DIR, 'imagen5.jpg'), cv2.IMREAD_GRAYSCALE)
    # Reutilizamos img1 según el PDF
    if img5 is not None and img1 is not None:
        escala = registrar_escala(img5.astype(float), img1.astype(float))
        print(f"Factor de escala estimado: {escala:.4f}")
    else:
        print("Error al cargar imagen5.jpg o imagen1.jpg")

    # --- 4. Coherencia de Fase (TIF) ---
    print("\n--- 4. Buscando K óptimo en TIF ---")
    try:
        # Lectura cruda de punto flotante para espectros
        mag_tif = iio.imread(os.path.join(BASE_DIR, 'imagen6.tif'))
        fase_tif = iio.imread(os.path.join(BASE_DIR, 'imagen7.tif'))

        k_optimo, img_recuperada = busqueda_k_optimo(mag_tif, fase_tif)
        print(f"El factor K óptimo encontrado es: {k_optimo:.2f}")

        plt.figure(figsize=(6,6))
        plt.imshow(img_recuperada, cmap='gray')
        plt.title(f"Imagen recuperada (K = {k_optimo:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: No se encontraron los archivos .tif.")
    except Exception as e:
        print(f"Error procesando TIFs: {e}")