import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = bool(cp.cuda.runtime.runtimeGetVersion())
except Exception:
    cp = np
    _HAS_CUPY = False


def precompute_swing_points(h, l, min_dist=4, prominence_pct=0.002):
    n = len(h)
    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)

    raw_sh = np.zeros(n, dtype=bool)
    raw_sl = np.zeros(n, dtype=bool)
    raw_sh[1:-1] = (h[1:-1] > h[:-2]) & (h[1:-1] > h[2:])
    raw_sl[1:-1] = (l[1:-1] < l[:-2]) & (l[1:-1] < l[2:])
    # Shift labels forward by 1 bar: bar i sees swing points from bar i-1
    shifted_sh = np.zeros(n, dtype=bool)
    shifted_sl = np.zeros(n, dtype=bool)
    shifted_sh[2:] = raw_sh[1:-1]
    shifted_sl[2:] = raw_sl[1:-1]
    raw_sh, raw_sl = shifted_sh, shifted_sl

    last_hi = -min_dist
    last_li = -min_dist
    for i in range(1, n - 1):
        if raw_sh[i]:
            if i - last_hi >= min_dist:
                if h[i] >= np.max(h[max(0, i-3):i]) * (1 + prominence_pct):
                    is_swing_high[i] = True
                    last_hi = i
        if raw_sl[i]:
            if i - last_li >= min_dist:
                if l[i] <= np.min(l[max(0, i-3):i]) * (1 - prominence_pct):
                    is_swing_low[i] = True
                    last_li = i

    return is_swing_high, is_swing_low


def compute_all_signals(df, engulf_body=0.6, swing_dist=4, inside_vol=1.3):
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    o = df['open'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    vol_sma = np.where(df['volume_sma_24'].values != 0, df['volume_sma_24'].values, 1.0).astype(np.float64)
    rsi = df['rsi_14'].values.astype(np.float64)
    ema_50 = df['ema_50'].values.astype(np.float64)
    ema_200 = df['ema_200'].values.astype(np.float64)
    bb_lower = df['bb_lower'].values.astype(np.float64)
    bb_upper = df['bb_upper'].values.astype(np.float64)
    atr_14 = df['atr_14'].values.astype(np.float64)

    h_np, l_np, c_np, o_np = h, l, c, o
    v_np, vol_sma_np = v, vol_sma
    rsi_np = rsi
    ema_50_np, ema_200_np = ema_50, ema_200
    atr_np = atr_14
    bb_lower_np, bb_upper_np = bb_lower, bb_upper
    vol_ratio_np = v_np / vol_sma_np

    if _HAS_CUPY:
        vol_ratio_np = cp.asnumpy(cp.divide(cp.asarray(v_np), cp.asarray(vol_sma_np)))

    is_sh, is_sl = precompute_swing_points(h_np, l_np, min_dist=swing_dist)

    sh_idx_list = np.where(is_sh)[0].tolist()
    sl_idx_list = np.where(is_sl)[0].tolist()

    last_sl_val = np.zeros(len(h_np))
    last_sh_val = np.zeros(len(h_np))
    last_sl_val[:] = np.nan
    last_sh_val[:] = np.nan
    cur_sl = np.inf
    cur_sh = 0.0
    for i in range(len(h_np)):
        if is_sl[i]:
            cur_sl = l_np[i]
        if is_sh[i]:
            cur_sh = h_np[i]
        last_sl_val[i] = cur_sl if cur_sl != np.inf else np.nan
        last_sh_val[i] = cur_sh if cur_sh > 0 else np.nan

    min_low_since_sl = np.full(len(h_np), np.inf)
    max_high_since_sh = np.full(len(h_np), -np.inf)
    cur_min_sl = np.inf
    cur_max_sh = -np.inf
    for i in range(len(h_np)):
        if is_sl[i]:
            cur_min_sl = np.inf
        if is_sh[i]:
            cur_max_sh = -np.inf
        cur_min_sl = min(cur_min_sl, l_np[i])
        cur_max_sh = max(cur_max_sh, h_np[i])
        min_low_since_sl[i] = cur_min_sl
        max_high_since_sh[i] = cur_max_sh

    sig_long = np.zeros(len(h_np), dtype=np.float32)
    sig_short = np.zeros(len(h_np), dtype=np.float32)
    sig_conf = np.zeros(len(h_np), dtype=np.float32)

    for i in range(20, len(h_np)):
        best_dir = 0
        best_sig = 0.0
        best_conf = 0.0

        # 1. Engulfing at key level
        if i >= 1:
            pb = abs(c_np[i-1] - o_np[i-1])
            if pb > 0:
                eng_bull = c_np[i] > o_np[i] and o_np[i] < c_np[i-1] and c_np[i] > o_np[i-1]
                eng_bear = c_np[i] < o_np[i] and o_np[i] > c_np[i-1] and c_np[i] < o_np[i-1]
                if eng_bull:
                    br = abs(c_np[i] - o_np[i]) / pb
                    if br > engulf_body:
                        bonus = 0.2 if abs(l_np[i] / ema_50_np[i] - 1) < 0.005 else 0
                        bonus = max(bonus, 0.3 if abs(l_np[i] / ema_200_np[i] - 1) < 0.005 else 0)
                        sig = 0.5 + min(0.3, br * 0.2) + bonus
                        if sig > best_sig:
                            best_sig, best_conf, best_dir = sig, 0.5, 1
                if eng_bear:
                    br = abs(c_np[i] - o_np[i]) / pb
                    if br > engulf_body:
                        bonus = 0.2 if abs(h_np[i] / ema_50_np[i] - 1) < 0.005 else 0
                        bonus = max(bonus, 0.3 if abs(h_np[i] / ema_200_np[i] - 1) < 0.005 else 0)
                        sig = 0.5 + min(0.3, br * 0.2) + bonus
                        if sig > abs(best_sig):
                            best_sig, best_conf, best_dir = sig, 0.5, -1

        # 2. Inside bar breakout
        if i >= 1:
            inside = h_np[i] <= h_np[i-1] * 1.001 and l_np[i] >= l_np[i-1] * 0.999
            ib_body = abs(c_np[i] - o_np[i]) < abs(c_np[i-1] - o_np[i-1]) * 1.1
            if inside or ib_body:
                vr = vol_ratio_np[i]
                if c_np[i] > h_np[i-1] and vr > inside_vol:
                    sig = 0.5 + min(0.3, (vr - 1) * 0.3)
                    if sig > best_sig:
                        best_sig, best_conf, best_dir = sig, 0.5, 1
                elif c_np[i] < l_np[i-1] and vr > inside_vol:
                    sig = 0.5 + min(0.3, (vr - 1) * 0.3)
                    if sig > abs(best_sig):
                        best_sig, best_conf, best_dir = sig, 0.5, -1

        # 3. SFP (swing failure)
        sl_val = last_sl_val[i] if not np.isnan(last_sl_val[i]) else 0
        if sl_val > 0:
            lowest = np.min(l_np[max(0, i-7):i+1])
            if lowest < sl_val * 0.998 and c_np[i] > sl_val and c_np[i] > o_np[i]:
                pen = (sl_val - lowest) / max(sl_val * 0.01, 1)
                sig = 0.55
                conf = min(0.8, 0.5 + min(0.3, pen * 0.5))
                if sig > best_sig:
                    best_sig, best_conf, best_dir = sig, conf, 1

        sh_val = last_sh_val[i] if not np.isnan(last_sh_val[i]) else 0
        if sh_val > 0:
            highest = np.max(h_np[max(0, i-7):i+1])
            if highest > sh_val * 1.002 and c_np[i] < sh_val and c_np[i] < o_np[i]:
                pen = (highest - sh_val) / max(sh_val * 0.01, 1)
                sig = 0.55
                conf = min(0.8, 0.5 + min(0.3, pen * 0.5))
                if sig > abs(best_sig):
                    best_sig, best_conf, best_dir = sig, conf, -1

        # 4. RSI Divergence
        if len(sl_idx_list) >= 2:
            prev_sl = None
            for idx in reversed(sl_idx_list):
                if idx < i:
                    if prev_sl is not None:
                        if rsi_np[idx] > rsi_np[prev_sl] and l_np[idx] < l_np[prev_sl] and idx > prev_sl + 3:
                            if c_np[i] > l_np[idx] * 1.002:
                                ds = min(1.0, (rsi_np[idx] - rsi_np[prev_sl]) * 3)
                                sig = 0.5 + ds * 0.3
                                if sig > best_sig:
                                    best_sig, best_conf, best_dir = sig, 0.6, 1
                        break
                    prev_sl = idx

        if len(sh_idx_list) >= 2:
            prev_sh = None
            for idx in reversed(sh_idx_list):
                if idx < i:
                    if prev_sh is not None:
                        if rsi_np[idx] < rsi_np[prev_sh] and h_np[idx] > h_np[prev_sh] and idx > prev_sh + 3:
                            if c_np[i] < h_np[idx] * 0.998:
                                ds = min(1.0, (rsi_np[prev_sh] - rsi_np[idx]) * 3)
                                sig = 0.5 + ds * 0.3
                                if sig > abs(best_sig):
                                    best_sig, best_conf, best_dir = sig, 0.6, -1
                        break
                    prev_sh = idx

        if best_dir > 0 and best_sig >= 0.35:
            sig_long[i] = best_sig
            sig_conf[i] = best_conf
        elif best_dir < 0 and best_sig >= 0.35:
            sig_short[i] = best_sig
            sig_conf[i] = best_conf

    return sig_long, sig_short, sig_conf
