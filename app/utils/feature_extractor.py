import numpy as np
import cv2
from skimage.feature import local_binary_pattern

IMG_SIZE = 224


def _load_rgb(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = _crop_white_margins(img)
    img = _apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _crop_white_margins(img_bgr: np.ndarray, threshold: int = 240) -> np.ndarray:
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    coords = np.argwhere(gray < threshold)
    if coords.size == 0:
        return img_bgr
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 5; h, w = img_bgr.shape[:2]
    return img_bgr[max(0, y0-pad):min(h, y1+pad), max(0, x0-pad):min(w, x1+pad)]


def _apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    lab     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def feat_hsv(img: np.ndarray) -> np.ndarray:
    hsv     = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    H, S, V = hsv[:,:,0]/180., hsv[:,:,1]/255., hsv[:,:,2]/255.
    H_raw   = hsv[:,:,0]
    stats   = [H.mean(), H.var(), S.mean(), S.var(), V.mean(), V.var()]
    warm    = float(((H_raw <= 35) | (H_raw >= 155)).mean())
    cool    = float(((H_raw > 85)  & (H_raw < 155)).mean())
    dark    = float((hsv[:,:,2] < 60).mean())
    hist, _ = np.histogram(H_raw, bins=6, range=(0, 180))
    return np.array(stats + [warm, cool, dark] + (hist/(hist.sum()+1e-8)).tolist(), dtype=np.float32)


def feat_composition(img: np.ndarray) -> np.ndarray:
    gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w      = gray.shape
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    M  = cv2.moments(binary)
    cx = (M['m10']/(M['m00']+1e-6))/w if M['m00'] > 0 else 0.5
    cy = (M['m01']/(M['m00']+1e-6))/h if M['m00'] > 0 else 0.5
    coords = np.argwhere(binary > 0)
    if len(coords) > 0:
        y0, x0 = coords.min(axis=0); y1, x1 = coords.max(axis=0)
        bbox_ar = ((y1-y0+1)*(x1-x0+1))/(h*w); asp = (x1-x0+1)/(y1-y0+2)
    else:
        bbox_ar, asp = 0., 1.
    edges    = cv2.Canny(gray, 50, 150)
    chaos    = float(edges.astype(np.float32).std()/255.)
    conts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_norm   = min(len(conts)/100., 1.)
    fill     = float((binary > 0).mean())
    flip     = cv2.flip(binary, 1)
    sym      = float((binary.astype(np.float32)*flip.astype(np.float32)).sum()) / \
               float(binary.astype(np.float32).sum()+flip.astype(np.float32).sum()+1e-6)
    sd       = min(float(edges.sum())/(255.*(binary.sum()/255.+1e-6)), 1.)
    return np.array([cx, cy, bbox_ar, asp, chaos, c_norm, fill, sym, sd], dtype=np.float32)


def feat_lbp(img: np.ndarray, radius=3, n_points=24, n_bins=32) -> np.ndarray:
    gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp     = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_points+2), density=True)
    return hist.astype(np.float32)


def feat_spatial(img: np.ndarray) -> np.ndarray:
    gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    total     = binary.sum() + 1e-6
    h, w      = binary.shape
    M  = cv2.moments(binary)
    cy = (M['m01']/(M['m00']+1e-6))/h
    return np.array([
        binary[:h//2,:].sum()/total, binary[h//2:,:].sum()/total,
        binary[:,:w//2].sum()/total, binary[:,w//2:].sum()/total, cy
    ], dtype=np.float32)


def feat_complexity(img: np.ndarray) -> np.ndarray:
    gray      = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges     = cv2.Canny(gray, 50, 150)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    conts, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return np.array([
        edges.sum()/(255.*edges.size),
        min(len(conts)/100., 1.),
        float((thresh > 0).mean())
    ], dtype=np.float32)


def feat_emotional_gradient_flow(img: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w    = gray.shape
    Gx      = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Gy      = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag     = np.sqrt(Gx**2 + Gy**2)
    abs_ang = np.abs(np.degrees(np.arctan2(Gy, Gx)))
    bin_map = np.zeros_like(abs_ang, dtype=np.int32)
    bin_map[abs_ang >= 45] = 1
    bin_map[abs_ang >= 90] = 2
    zones = [
        (0,      h//2,  0,      w   ),
        (h//2,   h,     0,      w   ),
        (0,      h,     0,      w//2),
        (0,      h,     w//2,   w   ),
        (h//4, 3*h//4,  w//4, 3*w//4),
    ]
    zone_hists = []
    for (r0, r1, c0, c1) in zones:
        z_mag = mag[r0:r1, c0:c1].ravel()
        z_bin = bin_map[r0:r1, c0:c1].ravel()
        hist  = np.array([z_mag[z_bin == b].sum() for b in range(3)], dtype=np.float32)
        zone_hists.extend((hist / (hist.sum()+1e-8)).tolist())
    return np.array(zone_hists, dtype=np.float32)


def extract_all(img_path: str) -> np.ndarray:
    img = _load_rgb(img_path)
    return np.concatenate([
        feat_hsv(img),                      # 15
        feat_composition(img),              #  9
        feat_lbp(img),                      # 32
        feat_spatial(img),                  #  5
        feat_complexity(img),               #  3
        feat_emotional_gradient_flow(img),  # 15
    ]).astype(np.float32)
