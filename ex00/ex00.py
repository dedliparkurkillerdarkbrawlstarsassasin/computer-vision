import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# 1. Загрузка изображения в оттенках серого

image = cv2.imread('sar_1_gray.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise SystemExit("Не удалось прочитать 'sar_1_gray.jpg'")
print(f"[1] Загружено: shape={image.shape}, dtype={image.dtype}, "
      f"min={image.min()}, max={image.max()}, mean={image.mean():.2f}")

 
# 2. Гистограмма

plt.figure(figsize=(8, 4))
plt.hist(image.ravel(), bins=256, range=(0, 256), color='steelblue')
plt.title('Гистограмма sar_1_gray')
plt.xlabel('Яркость'); plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('01_histogram.png', dpi=120)
plt.close()
print("[2] Гистограмма сохранена: 01_histogram.png")


# 3. Гамма-коррекция

def gamma_correction(img, gamma):
    """img: uint8 [0..255]. gamma<1 — осветляет, gamma>1 — затемняет."""
    img_f = img.astype(np.float32) / 255.0
    corrected = np.power(img_f, gamma)
    return np.clip(corrected * 255, 0, 255).astype(np.uint8)

gamma_dark  = 2.2   # >1 — затемнение
gamma_light = 0.45  # <1 — осветление

img_gamma_dark  = gamma_correction(image, gamma_dark)
img_gamma_light = gamma_correction(image, gamma_light)


# 4. Сравнение: MSE, SSIM
def mse(a, b):
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def ssim_metric(a, b):
    return float(ssim(a, b, data_range=255))

metrics = [
    ("gamma > 1 (затемнение)", img_gamma_dark),
    ("gamma < 1 (осветление)", img_gamma_light),
]
print("[4] Сравнение с исходным:")
print(f"    {'вариант':<25} {'MSE':>12} {'SSIM':>8}")
for name, img in metrics:
    print(f"    {name:<25} {mse(image, img):>12.2f} {ssim_metric(image, img):>8.4f}")

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, t in zip(ax,
                    [image, img_gamma_dark, img_gamma_light],
                    ['original', f'gamma={gamma_dark}', f'gamma={gamma_light}']):
    a.imshow(im, cmap='gray', vmin=0, vmax=255)
    a.set_title(t); a.axis('off')
plt.tight_layout()
plt.savefig('04_gamma_compare.png', dpi=120)
plt.close()
print("    Картинка сравнения: 04_gamma_compare.png")

# 5. Статистическая цветокоррекция (по eq_gray)
def stats_correction(img, mean_target=128.0, std_target=64.0):
    """Приводит mean/std изображения к заданным."""
    img_f = img.astype(np.float32)
    m, s = img_f.mean(), img_f.std() + 1e-8
    out = (img_f - m) / s * std_target + mean_target
    return np.clip(out, 0, 255).astype(np.uint8)

eq_gray = stats_correction(image, mean_target=128.0, std_target=64.0)
print(f"[5] Статистическая коррекция:")
print(f"    до:    mean={image.mean():.2f}, std={image.std():.2f}")
print(f"    после: mean={eq_gray.mean():.2f}, std={eq_gray.std():.2f}")
cv2.imwrite('05_eq_gray.png', eq_gray)
print("    Сохранено: 05_eq_gray.png")
# 6. Пороговая фильтрация с разными параметрами
thresholds = [60, 100, 140, 180]
fig, ax = plt.subplots(1, len(thresholds) + 1, figsize=(4 * (len(thresholds) + 1), 4))
ax[0].imshow(image, cmap='gray', vmin=0, vmax=255)
ax[0].set_title('original'); ax[0].axis('off')

print("[6] Пороговая фильтрация (binary, THRESH_BINARY):")
for i, t in enumerate(thresholds, start=1):
    _, bw = cv2.threshold(image, t, 255, cv2.THRESH_BINARY)
    fg = (bw > 0).mean() * 100
    print(f"    порог={t:>3}  доля белых пикселей={fg:5.2f}%")
    ax[i].imshow(bw, cmap='gray', vmin=0, vmax=255)
    ax[i].set_title(f'thr={t}'); ax[i].axis('off')

plt.tight_layout()
plt.savefig('06_thresholds.png', dpi=120)
plt.close()
print("    Сохранено: 06_thresholds.png")

# дополнительно — Otsu и адаптивный порог
t_otsu, bw_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bw_adapt = cv2.adaptiveThreshold(image, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, blockSize=31, C=5)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image, cmap='gray'); ax[0].set_title('original'); ax[0].axis('off')
ax[1].imshow(bw_otsu, cmap='gray'); ax[1].set_title(f'Otsu (t={t_otsu:.0f})'); ax[1].axis('off')
ax[2].imshow(bw_adapt, cmap='gray'); ax[2].set_title('Adaptive Gaussian'); ax[2].axis('off')
plt.tight_layout()
plt.savefig('06_thresholds_extra.png', dpi=120)
plt.close()

# Финальная сводка
print("\n=== ИТОГ ===")
print(f"mean/std исходного:       {image.mean():.2f} / {image.std():.2f}")
print(f"mean/std после gamma>1:   {img_gamma_dark.mean():.2f} / {img_gamma_dark.std():.2f}")
print(f"mean/std после gamma<1:   {img_gamma_light.mean():.2f} / {img_gamma_light.std():.2f}")
print(f"mean/std после stats eq:  {eq_gray.mean():.2f} / {eq_gray.std():.2f}")
print(f"Otsu порог: {t_otsu:.0f}")