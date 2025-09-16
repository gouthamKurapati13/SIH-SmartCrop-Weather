# pH Analysis Module
# pip install opencv-python scikit-image numpy
import cv2 as cv
import numpy as np
from skimage import color
import logging

logger = logging.getLogger(__name__)

# ---------------------------
# 1) Imaging corrections
# ---------------------------

def white_balance_gray(img_bgr, gray_roi):
    """
    Simple 1-point gray balance. gray_roi = (x,y,w,h) covering a gray card patch.
    """
    x, y, w, h = gray_roi
    patch = img_bgr[y:y+h, x:x+w].astype(np.float32) + 1e-6
    mean = patch.reshape(-1, 3).mean(axis=0)
    scale = mean.mean() / mean
    wb = np.clip(img_bgr.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return wb

def fit_ccm_rgb_to_lab(cam_rgbs, ref_labs):
    """
    Fit a linear RGB->Lab mapping: [R,G,B,1] @ M -> [L*,a*,b*]
    cam_rgbs: Nx3 floats in [0,1] from your measured color patches after WB
    ref_labs: Nx3 target Lab under D65 for those patches
    """
    X = np.hstack([cam_rgbs, np.ones((cam_rgbs.shape[0], 1))])  # N x 4
    Y = ref_labs  # N x 3
    M, *_ = np.linalg.lstsq(X, Y, rcond=None)  # 4 x 3
    return M  # apply with apply_ccm_to_img()

def apply_ccm_to_img(img_bgr, M):
    """
    Apply linear RGB->Lab mapping to full image, returns approximate Lab image.
    If you don't have CCM, skip and use skimage.color.rgb2lab instead on WB image.
    """
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    X = np.hstack([flat, np.ones((flat.shape[0], 1))])  # N x 4
    lab = X @ M  # N x 3
    return lab.reshape(H, W, 3)

def to_lab(img_bgr, M=None):
    """
    Convert to Lab either via CCM (preferred across devices) or rgb2lab fallback.
    """
    if M is not None:
        return apply_ccm_to_img(img_bgr, M)
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return color.rgb2lab(rgb)

# ---------------------------
# 2) Strip / pads extraction
# ---------------------------

def find_strip_roi(img_bgr, expected_aspect_range=(3.0, 12.0), min_area=5_000):
    """
    Naive geometry-based detector: returns a tight rect around the largest
    long-ish contour. If your setup is fixed, you can skip this and pass a manual ROI.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        area = w * h
        if area < min_area: 
            continue
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if expected_aspect_range[0] <= aspect <= expected_aspect_range[1]:
            score = area * aspect
            if score > best_score:
                best_score = score
                best = (x, y, w, h)
    return best  # or None

def crop_and_rectify(img_bgr, roi):
    """
    For now, just crop. If your strip is skewed, add a 4-point warp here.
    """
    x, y, w, h = roi
    return img_bgr[y:y+h, x:x+w]

def extract_pads(roi_bgr, n_pads=4, axis='long'):
    """
    Split the strip ROI into N equal pads along its long axis.
    For universal pH paper with 3–4 pads in a row.
    Returns list of BGR subimages for each pad (ordered left->right or top->bottom).
    """
    H, W = roi_bgr.shape[:2]
    if (axis == 'long' and W >= H) or (axis == 'short' and H > W):
        # split horizontally (left->right)
        splits = np.array_split(np.arange(W), n_pads)
        pads = [roi_bgr[:, s[0]:s[-1]+1] for s in splits]
    else:
        # split vertically (top->bottom)
        splits = np.array_split(np.arange(H), n_pads)
        pads = [roi_bgr[s[0]:s[-1]+1, :] for s in splits]
    return pads

def pad_mean_lab(pad_bgr, lab_img=None):
    """
    Compute L*a*b* mean for one pad. If lab_img provided (same crop), average from it;
    else convert the pad on the fly.
    """
    if lab_img is None:
        lab = to_lab(pad_bgr, M=None)
    else:
        lab = lab_img
    mean = lab.reshape(-1, 3).mean(axis=0)
    return mean

# ---------------------------
# 3) ΔE matching
# ---------------------------

def ciede2000(lab1, lab2):
    return color.deltaE_ciede2000(np.atleast_2d(lab1), np.atleast_2d(lab2)).ravel()[0]

def predict_single_pad(lab_patch, chart_labs, chart_phs):
    """
    chart_labs: list of Lab triplets, one per pH step
    chart_phs:  list of corresponding pH values (floats)
    """
    dists = [ciede2000(lab_patch, lab_ref) for lab_ref in chart_labs]
    idx = int(np.argmin(dists))
    return dict(pred_ph=chart_phs[idx], deltaE=float(dists[idx]), match_index=idx)

def predict_multi_pad(lab_pads, chart_lab_rows, chart_phs, agg="sum"):
    """
    lab_pads: list of Lab means for each detected pad (length N)
    chart_lab_rows: list of rows; each row is a list of Lab triplets (length N) for a pH on the chart
        Example: for N=4 pads, chart_lab_rows[k] = [Lab_pad1@pHk, Lab_pad2@pHk, Lab_pad3@pHk, Lab_pad4@pHk]
    chart_phs: list of pH values, one per row
    agg: "sum" or "mean" of per-pad ΔE
    """
    scores = []
    for row in chart_lab_rows:
        de = [ciede2000(lp, lr) for lp, lr in zip(lab_pads, row)]
        score = np.sum(de) if agg == "sum" else np.mean(de)
        scores.append(score)
    idx = int(np.argmin(scores))
    return dict(pred_ph=chart_phs[idx], distance=float(scores[idx]), match_index=idx, per_row_scores=scores)

# ---------------------------
# 4) Helpers for chart capture
# ---------------------------

def measure_chart_from_reference(chart_bgr, chart_rois):
    """
    Build chart Lab(s) by sampling fixed rectangles on a high-quality photo/scan
    of the printed reference chart under the SAME lighting.
    chart_rois:
       - single-pad mode: [(x,y,w,h), ...] one per pH step
       - multi-pad mode:  [[(x,y,w,h),... (N pads)],  ...] one row per pH step
    Returns Lab arrays mirroring chart_rois shape.
    """
    lab_chart = to_lab(chart_bgr, M=None)
    result = []
    for item in chart_rois:
        if isinstance(item[0], tuple) or isinstance(item[0], list):  # multi-pad row
            row_labs = []
            for (x,y,w,h) in item:
                patch = lab_chart[y:y+h, x:x+w]
                row_labs.append(patch.reshape(-1,3).mean(axis=0))
            result.append(row_labs)
        else:  # single swatch
            x,y,w,h = item
            patch = lab_chart[y:y+h, x:x+w]
            result.append(patch.reshape(-1,3).mean(axis=0))
    return result

# ---------------------------
# 5) End-to-end examples
# ---------------------------

def predict_ph_singlepad_image(
    img_bgr,
    gray_roi,
    strip_roi=None,
    chart_labs=None,
    chart_phs=None,
    ccm_M=None
):
    """
    Single reactive area pH paper.
    - gray_roi: tuple for gray card in the frame
    - strip_roi: if None, auto-detect; else (x,y,w,h)
    - chart_labs, chart_phs: provide from prior measurement
    - ccm_M: optional RGB->Lab CCM matrix
    """
    wb = white_balance_gray(img_bgr, gray_roi)
    if strip_roi is None:
        r = find_strip_roi(wb)
        if r is None:
            raise ValueError("Strip ROI not found; provide strip_roi manually.")
    else:
        r = strip_roi

    crop = crop_and_rectify(wb, r)
    lab = to_lab(crop, M=ccm_M)
    lab_mean = lab.reshape(-1, 3).mean(axis=0)

    return predict_single_pad(lab_mean, chart_labs, chart_phs)

def predict_ph_multipad_image(
    img_bgr,
    gray_roi,
    n_pads,
    strip_roi=None,
    chart_lab_rows=None,
    chart_phs=None,
    ccm_M=None,
    pad_axis='long'
):
    """
    Universal pH paper with N pads.
    - chart_lab_rows: list of rows, each row is length N list of Lab triplets.
    """
    wb = white_balance_gray(img_bgr, gray_roi)
    if strip_roi is None:
        r = find_strip_roi(wb)
        if r is None:
            raise ValueError("Strip ROI not found; provide strip_roi manually.")
    else:
        r = strip_roi

    crop = crop_and_rectify(wb, r)
    lab_crop = to_lab(crop, M=ccm_M)

    pads_bgr = extract_pads(crop, n_pads=n_pads, axis=pad_axis)

    # If you didn't use CCM, recompute Lab pad-wise; else map indices carefully.
    lab_pads = [color.rgb2lab(cv.cvtColor(p, cv.COLOR_BGR2RGB).astype(np.float32)/255.0) for p in pads_bgr]
    lab_means = [p.reshape(-1,3).mean(axis=0) for p in lab_pads]

    return predict_multi_pad(lab_means, chart_lab_rows, chart_phs, agg="sum")

# ---------------------------
# 6) Main analysis function
# ---------------------------

def analyze_ph_from_image(img_bgr, gray_roi=None):
    """
    Analyze pH from an image containing pH strips.
    Returns the median of single-pad and multi-pad results.
    """
    try:
        # Default gray card ROI if not provided (top-left corner)
        if gray_roi is None:
            gray_roi = (20, 20, 60, 60)

        # Default chart values for single-pad paper
        chart_phs_single = [4.0, 5.0, 6.0, 7.0, 8.0]
        chart_labs_single = [
            np.array([55.0, 75.0, 50.0]),  # swatch Lab at pH 4.0
            np.array([60.0, 40.0, 50.0]),  # pH 5.0
            np.array([65.0, 10.0, 10.0]),  # pH 6.0
            np.array([70.0,  0.0,  0.0]),  # pH 7.0
            np.array([65.0,-15.0,-20.0]),  # pH 8.0
        ]

        # Default chart values for multi-pad paper (4 pads)
        n_pads = 4
        chart_phs_multi = [5.0, 5.5, 6.0, 6.5, 7.0]
        chart_lab_rows = [
            [np.array([55, 70, 50]), np.array([65, 50, 40]), np.array([60, 20, 30]), np.array([55, 10,  5])],  # pH 5.0 row
            [np.array([58, 65, 45]), np.array([66, 45, 35]), np.array([62, 15, 25]), np.array([57,  8,  4])],  # pH 5.5
            [np.array([60, 60, 40]), np.array([68, 40, 30]), np.array([64, 10, 18]), np.array([59,  5,  3])],  # pH 6.0
            [np.array([63, 50, 30]), np.array([70, 25, 20]), np.array([66,  5, 10]), np.array([60,  3,  2])],  # pH 6.5
            [np.array([65, 40, 20]), np.array([72,  5,  5]), np.array([68, -2,  0]), np.array([62,  2,  2])],  # pH 7.0
        ]

        results = {}
        
        # Try single-pad analysis
        try:
            res_single = predict_ph_singlepad_image(
                img_bgr=img_bgr,
                gray_roi=gray_roi,
                strip_roi=None,
                chart_labs=chart_labs_single,
                chart_phs=chart_phs_single,
                ccm_M=None
            )
            results['single_pad'] = res_single
            logger.info(f"Single-pad result: {res_single}")
        except Exception as e:
            logger.warning(f"Single-pad analysis failed: {e}")
            results['single_pad'] = None

        # Try multi-pad analysis
        try:
            res_multi = predict_ph_multipad_image(
                img_bgr=img_bgr,
                gray_roi=gray_roi,
                n_pads=n_pads,
                strip_roi=None,
                chart_lab_rows=chart_lab_rows,
                chart_phs=chart_phs_multi,
                ccm_M=None,
                pad_axis='long'
            )
            results['multi_pad'] = res_multi
            logger.info(f"Multi-pad result: {res_multi}")
        except Exception as e:
            logger.warning(f"Multi-pad analysis failed: {e}")
            results['multi_pad'] = None

        # Calculate median pH
        ph_values = []
        if results['single_pad']:
            ph_values.append(results['single_pad']['pred_ph'])
        if results['multi_pad']:
            ph_values.append(results['multi_pad']['pred_ph'])

        if ph_values:
            median_ph = float(np.median(ph_values))
            results['median_ph'] = median_ph
            results['confidence'] = 'high' if len(ph_values) == 2 else 'medium'
        else:
            results['median_ph'] = None
            results['confidence'] = 'low'
            results['error'] = 'Both single-pad and multi-pad analysis failed'

        return results

    except Exception as e:
        logger.error(f"pH analysis failed: {e}")
        return {
            'single_pad': None,
            'multi_pad': None,
            'median_ph': None,
            'confidence': 'low',
            'error': str(e)
        }
