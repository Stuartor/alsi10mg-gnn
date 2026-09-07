# -*- coding: utf-8 -*-
"""
Abaqus ODB extractor: converts FE simulation outputs to graph-format NPZ files.
Run inside the ODB directory:  abaqus python extract_odb_to_npz.py
"""
import os
import glob
import csv
import numpy as np
from odbAccess import openOdb
from abaqusConstants import *

# ================================================================
# ★★★ CONFIG ★★★
# ================================================================
CONFIG = {
    'ODB_DIR':    os.getcwd(),
    'OUTPUT_DIR': os.path.join(os.getcwd(), 'graph_dataset'),
    'META_CSV':   os.path.join(os.getcwd(), 'samples_meta.csv'),


    'ODB_NAME_PATTERN': 'Job_{:04d}.odb',
    'ODB_INDEX_RANGE':  (0, 81),

    'AUTO_DISCOVER_ODB': True,


    'INSTANCE_NAME_HINTS': ['PART-1-1', 'PART-1'],

    'LOAD_AXIS': 1,

    'MESH_SIZE': 0.5,
    'BC_TOL_FACTOR': 1.2,

    'TARGET_DISPLACEMENTS': [
        0.00,
        0.04, 0.09, 0.14,
        0.25, 0.42, 0.62, 0.85,
        1.25, 1.75, 2.30, 2.85, 3.40,
        3.75, 4.05, 4.32, 4.56, 4.78,
        4.96, 5.12, 5.26, 5.38, 5.50, 5.62, 5.73, 5.83, 5.92, 5.99,
    ],
    'MIN_COMPLETION_RATIO': 0.95,


    'HISTORY_RP_FIXED_HINT':  'RP2',

    'HISTORY_RP_LOADED_HINT': 'RP1',

    'AUTO_DETECT_HISTORY_REGIONS': True,

    'FORCE_SMOOTH_WINDOW': 21,

    'FORCE_SMOOTH_SIGNED_FIRST': True,

    'FRACTURE_LOW_FRAC': 0.05,

    'FRACTURE_PERSIST_POINTS': 3,


    'USE_STATUS_FRACTURE': True,
    'STATUS_MIN_DELETED_ELEMENTS': 1,
    'STATUS_COMPLETE_FRACTION': 0.90,

    'ZERO_FORCE_AFTER_FRACTURE': True,

    'DENSE_CURVE_N': 501,

    'DENSE_CURVE_VALIDATE_ABS_N': 120.0,

    'DENSE_CURVE_VALIDATE_REL': 0.06,


    'USE_ELEMENT_NODAL_STRESS': True,
    'USE_ELEMENT_NODAL_STRAIN': True,

    'REQUIRE_COMPLETE_FRACTURE': False,
    'MAX_FRACTURE_DISP_RATIO': 0.999,

    'EXTRACT_DIC_EYY': True,
    'DIC_STRAIN_FIELD': 'LE',
    'DIC_STRAIN_COMPONENT': 1,          # LE22，0-based
    'DIC_STRAIN_SCALE': 100.0,

    'DIC_EYY_RANGE': (-1.0, 1.0),
    'DIC_EYY_TICKS': [-1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25,
                      -0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75,
                      0.875, 1.0],

    'NODE_FEATURE_DIM': 8,   # [x,y,z, bc_free, bc_fixed, bc_loaded, sigma_prev, degree]
    'EDGE_FEATURE_DIM': 4,


    'USE_BULK_DATA': True,

}

# ================================================================
# ================================================================
ELEMENT_EDGE_TABLE = {
    'C3D8': [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)],
    'C3D4': [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    'C3D6': [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
             (0, 3), (1, 4), (2, 5)],
}

ELEMENT_FACE_TABLE = {
    'C3D8': [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)],
    'C3D6': [(0, 1, 2), (3, 5, 4), (0, 3, 4, 1), (1, 4, 5, 2), (2, 5, 3, 0)],
    'C3D4': [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],
}

MATERIAL_KEYS = ['E_ratio', 'nu', 'sy_ratio', 'n_ratio', 'ef_ratio']

def get_table_for_type(elem_type, table_dict):
    et = elem_type.upper()
    for prefix, table in table_dict.items():
        if et.startswith(prefix):
            return table
    return None

# ================================================================
# ================================================================
def moving_average(arr, window):
    n = len(arr)
    if window <= 1 or n < window:
        return np.asarray(arr, dtype=np.float64)
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.mean(arr[max(0, i - half):min(n, i + half + 1)])
    return out

def mises_from_components(s):
    """s: (n,6) = [S11,S22,S33,S12,S13,S23] -> (n,) von Mises"""
    s11, s22, s33 = s[:, 0], s[:, 1], s[:, 2]
    s12, s13, s23 = s[:, 3], s[:, 4], s[:, 5]
    j2 = 0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) \
         + 3.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)
    return np.sqrt(np.maximum(j2, 0.0))

class LabelMapper(object):

    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.order = np.argsort(self.labels)
        self.sorted = self.labels[self.order]

    def map(self, query):
        q = np.asarray(query, dtype=np.int64)
        pos = np.searchsorted(self.sorted, q)
        pos_c = np.clip(pos, 0, len(self.sorted) - 1)
        valid = self.sorted[pos_c] == q
        return self.order[pos_c], valid

def get_target_instance(odb, hints):
    insts = odb.rootAssembly.instances
    names = list(insts.keys())
    for h in hints:
        for name in names:
            if h.upper() in name.upper():
                return name, insts[name]
    best_name, best_n = None, -1
    for name in names:
        n = len(insts[name].nodes)
        if n > best_n:
            best_name, best_n = name, n
    return best_name, insts[best_name]

# ================================================================
# ================================================================
def load_material_meta(meta_csv):
    table = {}
    if not os.path.exists(meta_csv):
        print('[Meta][WARNING] Missing {}'.format(meta_csv))
        return table
    f = open(meta_csv, 'r')
    reader = csv.reader(f)
    header = next(reader)
    idx = dict((k, header.index(k)) for k in MATERIAL_KEYS if k in header)
    for k in MATERIAL_KEYS:
        if k not in idx:
            print('[Meta][WARNING] meta missing column {}'.format(k))
    jn = header.index('job_name') if 'job_name' in header else 2
    for row in reader:
        if not row:
            continue
        table[row[jn]] = np.array(
            [float(row[idx[k]]) if k in idx else 0.0 for k in MATERIAL_KEYS],
            dtype=np.float32)
    f.close()
    return table

# ================================================================
# ================================================================
def build_static_topology(elements, node_labels, node_index_map, node_coords):
    edges = set()
    unsupported = set()
    for elem in elements:
        conn = list(elem.connectivity)
        table = get_table_for_type(elem.type, ELEMENT_EDGE_TABLE)
        if table is None:
            unsupported.add(elem.type)
            n = min(len(conn), 8)
            table = [(i, j) for i in range(n) for j in range(i + 1, n)]
        for a, b in table:
            if a >= len(conn) or b >= len(conn):
                continue
            ni, nj = conn[a], conn[b]
            if ni not in node_index_map or nj not in node_index_map:
                continue
            edges.add((ni, nj) if ni < nj else (nj, ni))

    if unsupported:

    edges = sorted(edges)
    ei = np.array([[node_index_map[a], node_index_map[b]] for a, b in edges],
                  dtype=np.int64).T

    coords_arr = np.array([node_coords[nl][:3] for nl in node_labels], dtype=np.float64)
    diff = coords_arr[ei[1]] - coords_arr[ei[0]]
    ef = np.zeros((ei.shape[1], 4), dtype=np.float32)
    ef[:, 0:3] = diff
    ef[:, 3] = np.linalg.norm(diff, axis=1)

    degree = (np.bincount(ei[0], minlength=len(node_labels)) +
              np.bincount(ei[1], minlength=len(node_labels))).astype(np.float32)

    print('  [Topology] nodes {} | edges {} | elements {}'.format(
        len(node_labels), ei.shape[1], len(elements)))
    return ei, ef, degree, coords_arr

def build_surface_tris(elements, node_index_map):

    """
    face_count, face_oriented = {}, {}
    for e_i, elem in enumerate(elements):
        table = get_table_for_type(elem.type, ELEMENT_FACE_TABLE)
        if table is None:
            continue
        conn = list(elem.connectivity)
        for face in table:
            if max(face) >= len(conn):
                continue
            labs = tuple(conn[i] for i in face)
            if not all(nl in node_index_map for nl in labs):
                continue
            key = tuple(sorted(labs))
            face_count[key] = face_count.get(key, 0) + 1
            face_oriented[key] = (labs, e_i)
    tris, owners = [], []
    for key, item in face_oriented.items():
        if face_count[key] != 1:
            continue
        labs, e_i = item
        ids = [node_index_map[nl] for nl in labs]
        if len(ids) == 4:
            tris.append([ids[0], ids[1], ids[2]])
            owners.append(e_i)
            tris.append([ids[0], ids[2], ids[3]])
            owners.append(e_i)
        elif len(ids) == 3:
            tris.append(ids)
            owners.append(e_i)
    tris = np.array(tris, dtype=np.int64)
    owners = np.array(owners, dtype=np.int64)
    return tris, owners

# ================================================================
# ================================================================
def read_element_field(frame, name, instance, inst_name, ncomp, use_bulk):

    """
    if name not in frame.fieldOutputs.keys():
        return None, None
    fo = frame.fieldOutputs[name]

    if use_bulk:
        try:
            labs, dats = [], []
            for b in fo.bulkDataBlocks:
                if b.instance is not None and b.instance.name != inst_name:
                    continue
                el = b.elementLabels
                if el is None:
                    continue
                d = np.asarray(b.data, dtype=np.float32)
                if d.ndim == 1:
                    d = d.reshape(-1, 1)
                labs.append(np.asarray(el, dtype=np.int64))
                dats.append(d)
            if labs:
                D = np.concatenate(dats, axis=0)
                if D.shape[1] >= ncomp:
                    return np.concatenate(labs), D[:, :ncomp]
        except Exception:
            pass

    try:
        sub = fo.getSubset(region=instance, position=CENTROID)
    except Exception:
        try:
            sub = fo.getSubset(region=instance)
        except Exception:
            return None, None
    labs, dats = [], []
    for v in sub.values:
        if not hasattr(v, 'elementLabel'):
            continue
        labs.append(v.elementLabel)
        d = v.data
        dats.append(list(d) if hasattr(d, '__len__') else [d])
    if not labs:
        return None, None
    D = np.array(dats, dtype=np.float32)
    if D.shape[1] < ncomp:
        D = np.concatenate(
            [D, np.zeros((D.shape[0], ncomp - D.shape[1]), dtype=np.float32)], axis=1)
    return np.array(labs, dtype=np.int64), D[:, :ncomp]

def read_nodal_U(frame, instance, inst_name, node_mapper, n_nodes, use_bulk):
    out = np.zeros((n_nodes, 3), dtype=np.float32)
    if 'U' not in frame.fieldOutputs.keys():
        return out
    fo = frame.fieldOutputs['U']

    if use_bulk:
        try:
            got = False
            for b in fo.bulkDataBlocks:
                if b.instance is not None and b.instance.name != inst_name:
                    continue
                nl = b.nodeLabels
                if nl is None:
                    continue
                d = np.asarray(b.data, dtype=np.float32)
                rows, valid = node_mapper.map(nl)
                out[rows[valid], :] = d[valid, :3]
                got = True
            if got:
                return out
        except Exception:
            pass

    try:
        sub = fo.getSubset(region=instance, position=NODAL)
        labs = np.array([v.nodeLabel for v in sub.values], dtype=np.int64)
        vals = np.array([v.data[:3] for v in sub.values], dtype=np.float32)
        rows, valid = node_mapper.map(labs)
        out[rows[valid], :] = vals[valid]
    except Exception as e:
        print('  [WARN] Read U failed: {}'.format(e))
    return out

def elem_field_to_rows(labs, data, elem_mapper, n_elem, ncomp, default=0.0):
    out = np.full((n_elem, ncomp), default, dtype=np.float32)
    if labs is None or data is None or len(labs) == 0:
        return out
    rows, valid = elem_mapper.map(labs)
    rows = rows[valid]
    dat = np.asarray(data[valid, :ncomp], dtype=np.float64)
    if len(rows) == 0:
        return out
    cnt = np.bincount(rows, minlength=n_elem).astype(np.float64)
    good = cnt > 0
    for j in range(ncomp):
        sm = np.bincount(rows, weights=dat[:, j], minlength=n_elem)
        out[good, j] = (sm[good] / cnt[good]).astype(np.float32)
    return out

def read_element_nodal_to_nodes(frame, name, instance, inst_name,
                                node_mapper, elem_mapper, alive_e,
                                n_nodes, ncomp, use_bulk):

    """
    if name not in frame.fieldOutputs.keys():
        return None
    fo = frame.fieldOutputs[name]
    try:
        sub = fo.getSubset(region=instance, position=ELEMENT_NODAL)
    except Exception:
        return None

    nl_all, el_all, d_all = [], [], []
    if use_bulk:
        try:
            for b in sub.bulkDataBlocks:
                if b.instance is not None and b.instance.name != inst_name:
                    continue
                nl = b.nodeLabels
                el = b.elementLabels
                if nl is None or el is None:
                    continue
                d = np.asarray(b.data, dtype=np.float32)
                if d.ndim == 1:
                    d = d.reshape(-1, 1)
                if d.shape[1] < ncomp:
                    continue
                nl_all.append(np.asarray(nl, dtype=np.int64))
                el_all.append(np.asarray(el, dtype=np.int64))
                d_all.append(d[:, :ncomp])
        except Exception:
            nl_all, el_all, d_all = [], [], []

    if not nl_all:
        try:
            nls, els, ds = [], [], []
            for v in sub.values:
                if not hasattr(v, 'nodeLabel') or not hasattr(v, 'elementLabel'):
                    continue
                nls.append(v.nodeLabel)
                els.append(v.elementLabel)
                d = v.data
                row = list(d) if hasattr(d, '__len__') else [d]
                if len(row) < ncomp:
                    row += [0.0] * (ncomp - len(row))
                ds.append(row[:ncomp])
            if not nls:
                return None
            nl = np.asarray(nls, dtype=np.int64)
            el = np.asarray(els, dtype=np.int64)
            dat = np.asarray(ds, dtype=np.float32)
        except Exception:
            return None
    else:
        nl = np.concatenate(nl_all)
        el = np.concatenate(el_all)
        dat = np.concatenate(d_all, axis=0)

    nrows, nv = node_mapper.map(nl)
    erows, ev = elem_mapper.map(el)
    valid = nv & ev
    if np.any(valid):
        valid_idx = np.where(valid)[0]
        valid_idx = valid_idx[alive_e[erows[valid_idx]] >= 0.5]
    else:
        valid_idx = np.array([], dtype=np.int64)
    if len(valid_idx) == 0:
        return np.zeros((n_nodes, ncomp), dtype=np.float32)

    rr = nrows[valid_idx]
    dd = dat[valid_idx, :ncomp].astype(np.float64)
    cnt = np.bincount(rr, minlength=n_nodes).astype(np.float64)
    safe = np.maximum(cnt, 1.0)
    out = np.zeros((n_nodes, ncomp), dtype=np.float32)
    for j in range(ncomp):
        out[:, j] = (np.bincount(rr, weights=dd[:, j], minlength=n_nodes) /
                     safe).astype(np.float32)
    return out

# ================================================================
# ================================================================
def identify_bc_nodes(coords_arr, cfg, u_last):
    axis = cfg['LOAD_AXIS']
    tol = cfg['MESH_SIZE'] * cfg['BC_TOL_FACTOR']
    ca = coords_arr[:, axis]
    vmin, vmax = ca.min(), ca.max()

    low_mask = ca <= vmin + tol
    high_mask = ca >= vmax - tol

    u_low = float(np.mean(np.abs(u_last[low_mask, axis]))) if low_mask.any() else 0.0
    u_high = float(np.mean(np.abs(u_last[high_mask, axis]))) if high_mask.any() else 0.0

    if u_high >= u_low:
        loaded_mask, fixed_mask, which = high_mask, low_mask, 'y_max'
    else:
        loaded_mask, fixed_mask, which = low_mask, high_mask, 'y_min'

    print('  [BC] loaded={} |U|={:.4f} / fixed |U|={:.4f} | loaded {} nodes, fixed {} nodes'
          .format(which, max(u_low, u_high), min(u_low, u_high),
                  int(loaded_mask.sum()), int(fixed_mask.sum())))
    return fixed_mask, loaded_mask

# ================================================================
# ================================================================
def _sorted_history(data):
    if not data:
        return None, None
    t = np.asarray([a for a, _ in data], dtype=np.float64)
    v = np.asarray([b for _, b in data], dtype=np.float64)
    good = np.isfinite(t) & np.isfinite(v)
    t, v = t[good], v[good]
    if len(t) == 0:
        return None, None
    order = np.argsort(t)
    t, v = t[order], v[order]
    tu, inv = np.unique(t, return_inverse=True)
    if len(tu) != len(t):
        cnt = np.bincount(inv).astype(np.float64)
        vv = np.bincount(inv, weights=v) / np.maximum(cnt, 1.0)
        t, v = tu, vv
    return t, v

def read_history_force(step, cfg):

    """
    axis_force = 'RF%d' % (cfg['LOAD_AXIS'] + 1)
    axis_disp = 'U%d' % (cfg['LOAD_AXIS'] + 1)

    candidates = []
    for key in step.historyRegions.keys():
        hr = step.historyRegions[key]
        tu, uv = (None, None)
        tf, rf = (None, None)
        if axis_disp in hr.historyOutputs.keys():
            tu, uv = _sorted_history(list(hr.historyOutputs[axis_disp].data))
        if axis_force in hr.historyOutputs.keys():
            tf, rf = _sorted_history(list(hr.historyOutputs[axis_force].data))
        if tu is None and tf is None:
            continue
        umax = float(np.max(np.abs(uv))) if uv is not None and len(uv) else 0.0
        urange = float(np.ptp(uv)) if uv is not None and len(uv) else 0.0
        if rf is not None and len(rf):
            rs = moving_average(rf, cfg['FORCE_SMOOTH_WINDOW'])
            rpeak = float(np.max(np.abs(rs)))
        else:
            rpeak = 0.0
        candidates.append({'key': key, 'tu': tu, 'u': uv, 'tf': tf, 'rf': rf,
                           'umax': umax, 'urange': abs(urange), 'rpeak': rpeak})

    if not candidates:
        return None, None, None, 0.0, None

    print('  [History] candidate regions:')
    for c in candidates:
        print('    {} | max|U|={:.4f} | spanU={:.4f} | peak|RF|={:.1f}'.format(
            c['key'], c['umax'], c['urange'], c['rpeak']))

    ucan = [c for c in candidates if c['u'] is not None]
    rcan = [c for c in candidates if c['rf'] is not None]
    if not ucan or not rcan:
        return None, None, None, 0.0, None

    if cfg.get('AUTO_DETECT_HISTORY_REGIONS', True):
        disp_c = max(ucan, key=lambda c: (c['umax'], c['urange']))
        max_rpeak = max(c['rpeak'] for c in rcan)
        strong = [c for c in rcan if c['rpeak'] >= 0.70 * max_rpeak]
        force_c = min(strong, key=lambda c: (c['umax'], -c['rpeak']))
    else:
        def hinted(arr, hint):
            for c in arr:
                if hint and hint.upper() in c['key'].upper():
                    return c
            return arr[0]
        disp_c = hinted(ucan, cfg.get('HISTORY_RP_LOADED_HINT'))
        force_c = hinted(rcan, cfg.get('HISTORY_RP_FIXED_HINT'))

    times = force_c['tf']
    force_signed = force_c['rf']
    raw_smooth = moving_average(force_signed, cfg['FORCE_SMOOTH_WINDOW'])
    peak_abs_idx = int(np.argmax(np.abs(raw_smooth)))
    orient = 1.0 if raw_smooth[peak_abs_idx] >= 0.0 else -1.0
    f_smooth = moving_average(orient * force_signed, cfg['FORCE_SMOOTH_WINDOW'])
    f_smooth = np.maximum(f_smooth, 0.0)

    disp_hist = (disp_c['tu'], np.abs(disp_c['u']))
    peak_idx = int(np.argmax(f_smooth))
    peak = float(f_smooth[peak_idx])
    thr = peak * cfg['FRACTURE_LOW_FRAC']

    t_fracture = None
    n_persist = max(1, int(cfg.get('FRACTURE_PERSIST_POINTS', 1)))
    below = f_smooth < thr
    for i in range(peak_idx, len(f_smooth)):
        j = min(len(f_smooth), i + n_persist)
        if j - i >= n_persist and np.all(below[i:j]):
            t_fracture = float(times[i])
            break

    tail_n = min(10, len(f_smooth))
    tail_mean = float(np.mean(f_smooth[-tail_n:])) if tail_n else 0.0
    tail_min = float(np.min(f_smooth[peak_idx:])) if peak_idx < len(f_smooth) else 0.0
    print('  [Force] peak {:.1f} N @ t={:.5f} | post-peak min {:.1f} N | tail mean {:.1f} N'.format(
        peak, times[peak_idx], tail_min, tail_mean))

    if t_fracture is None:
            thr, 100.0 * cfg['FRACTURE_LOW_FRAC']))
    else:
        d_at = float(np.interp(t_fracture, disp_hist[0], disp_hist[1]))

    return times, f_smooth, disp_hist, peak, t_fracture

def read_status_alive(frame, instance, inst_name, elem_mapper, n_elem, use_bulk):
    """Read one frame STATUS and align to (n_elem,) 0/1 alive array.

    Abaqus/Explicit may omit deleted elements rather than return STATUS=0.
    Missing element labels must be treated as 0 (deleted).
    """
    labs, data = read_element_field(
        frame, 'STATUS', instance, inst_name, 1, use_bulk)
    if labs is None or data is None:
        return np.ones(n_elem, dtype=np.float32), False

    alive = np.zeros(n_elem, dtype=np.float32)

    rows, valid = elem_mapper.map(labs)
    if np.any(valid):
        vals = np.asarray(data[valid, 0], dtype=np.float32)
        alive[rows[valid]] = (vals >= 0.5).astype(np.float32)
    return alive, True

def scan_status_fracture(frames, instance, inst_name, elem_mapper, n_elem, cfg):

    """
    cache = []
    available_any = False
    last_alive = np.ones(n_elem, dtype=np.float32)

    for frame in frames:
        alive, available = read_status_alive(
            frame, instance, inst_name, elem_mapper, n_elem, cfg['USE_BULK_DATA'])
        if available:
            available_any = True
            last_alive = alive
        else:
            alive = last_alive.copy()
        cache.append(alive)

    if not available_any:
        return None, cache, np.zeros(len(frames), dtype=np.int32), False

    alive_counts = np.array([int(np.sum(a >= 0.5)) for a in cache], dtype=np.int32)
    n_head = min(5, len(alive_counts))
    baseline = int(np.max(alive_counts[:n_head])) if n_head else n_elem
    baseline = max(baseline, int(np.max(alive_counts)))
    deleted_counts = np.maximum(baseline - alive_counts, 0).astype(np.int32)
    max_deleted = int(np.max(deleted_counts)) if len(deleted_counts) else 0

    min_deleted = max(1, int(cfg.get('STATUS_MIN_DELETED_ELEMENTS', 1)))
    if max_deleted < min_deleted:
            int(alive_counts[-1]) if len(alive_counts) else n_elem, baseline))
        return None, cache, deleted_counts, True

    frac = float(cfg.get('STATUS_COMPLETE_FRACTION', 0.90))
    frac = min(max(frac, 0.0), 1.0)
    target_deleted = max(min_deleted, int(np.ceil(max_deleted * frac)))
    idx = int(np.where(deleted_counts >= target_deleted)[0][0])
    t_status = float(frames[idx].frameValue)

          'odb#{} t={:.5f}'.format(
              max_deleted, target_deleted, 100.0 * frac, idx, t_status))
    return t_status, cache, deleted_counts, True

def _curve_from_disp_force(disp, force, t_fracture_disp, cfg):
    n = int(cfg['DENSE_CURVE_N'])
    d_max = float(cfg['TARGET_DISPLACEMENTS'][-1])
    grid = np.linspace(0.0, d_max, n)

    d = np.asarray(disp, dtype=np.float64)
    f = np.asarray(force, dtype=np.float64)
    good = np.isfinite(d) & np.isfinite(f)
    d, f = np.abs(d[good]), f[good]
    if len(d) < 2:
        return grid.astype(np.float32), np.zeros(n, np.float32)

    order = np.argsort(d)
    d, f = d[order], f[order]
    dr = np.round(d, 8)
    du, inv = np.unique(dr, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    fu = np.bincount(inv, weights=f) / np.maximum(cnt, 1.0)
    if len(du) < 2:
        return grid.astype(np.float32), np.zeros(n, np.float32)

    out = np.interp(grid, du, fu)
    out[grid > float(np.max(du))] = 0.0
    if t_fracture_disp is not None and t_fracture_disp >= 0.0 and \
            cfg['ZERO_FORCE_AFTER_FRACTURE']:
        out[grid >= float(t_fracture_disp)] = 0.0
    out = np.maximum(out, 0.0)
    return grid.astype(np.float32), out.astype(np.float32)

def build_dense_curve(times, f_smooth, disp_hist, t_fracture, cfg):
    n = int(cfg['DENSE_CURVE_N'])
    d_max = float(cfg['TARGET_DISPLACEMENTS'][-1])
    grid = np.linspace(0.0, d_max, n).astype(np.float32)
    if times is None or disp_hist is None:
        return grid, np.zeros(n, dtype=np.float32), np.float32(-1.0)

    dt, dv = disp_hist
    order = np.argsort(dt)
    dt = np.asarray(dt, dtype=np.float64)[order]
    dv = np.abs(np.asarray(dv, dtype=np.float64)[order])
    dv = np.maximum.accumulate(dv)
    f_at_dt = np.interp(dt, times, f_smooth)

    d_fracture = -1.0
    if t_fracture is not None:
        d_fracture = float(np.interp(t_fracture, dt, dv))

    gd, gf = _curve_from_disp_force(dv, f_at_dt, d_fracture, cfg)
    return gd, gf, np.float32(d_fracture)

# ================================================================
# ================================================================
def process_odb(odb_path, output_dir, cfg, material_table):
    odb_name = os.path.basename(odb_path).replace('.odb', '')
    print('\n=== {} ==='.format(odb_name))

    try:
        odb = openOdb(path=odb_path, readOnly=True)
    except Exception as e:
        print('  Open failed: {}'.format(e))
        return 0

    try:
        inst_name, instance = get_target_instance(odb, cfg['INSTANCE_NAME_HINTS'])

        node_list = list(instance.nodes)
        node_labels = sorted([n.label for n in node_list])
        node_coords = dict((n.label, n.coordinates) for n in node_list)
        node_index_map = dict((nl, i) for i, nl in enumerate(node_labels))
        node_mapper = LabelMapper(node_labels)
        n_nodes = len(node_labels)

        elements = list(instance.elements)
        n_elem = len(elements)
        elem_mapper = LabelMapper([e.label for e in elements])

        step = list(odb.steps.values())[0]
        frames = step.frames
        n_frames = len(frames)
        print('  nodes {} | elements {} | ODB frames {}'.format(n_nodes, n_elem, n_frames))
        if n_frames < 3:
            odb.close()
            return 0

        edge_index, edge_features, degree, coords_arr = build_static_topology(
            elements, node_labels, node_index_map, node_coords)
        surface_tris, surface_tri_owner = build_surface_tris(elements, node_index_map)

        conn_rows, conn_cols = [], []
        for e_i, elem in enumerate(elements):
            for nl in elem.connectivity:
                j = node_index_map.get(nl)
                if j is not None:
                    conn_rows.append(e_i)
                    conn_cols.append(j)
        conn_rows = np.array(conn_rows, dtype=np.int64)
        conn_cols = np.array(conn_cols, dtype=np.int64)

        U_cache = [read_nodal_U(frames[fi], instance, inst_name, node_mapper,
                                n_nodes, cfg['USE_BULK_DATA'])
                   for fi in range(n_frames)]

        fixed_mask, loaded_mask = identify_bc_nodes(coords_arr, cfg, U_cache[-1])
        bc_arr = np.zeros((n_nodes, 3), dtype=np.float32)
        bc_arr[:, 0] = (~(fixed_mask | loaded_mask)).astype(np.float32)   # free
        bc_arr[:, 1] = fixed_mask.astype(np.float32)
        bc_arr[:, 2] = loaded_mask.astype(np.float32)

        axis = cfg['LOAD_AXIS']
        li = np.where(loaded_mask)[0]
        frame_disp = np.array(
            [abs(np.mean(U_cache[fi][li, axis])) if li.size else
             abs(np.max(np.abs(U_cache[fi][:, axis]))) for fi in range(n_frames)])

        if cfg.get('USE_STATUS_FRACTURE', True):
            t_status, status_cache, deleted_counts, status_available = \
                scan_status_fracture(
                    frames, instance, inst_name, elem_mapper, n_elem, cfg)
        else:
            t_status = None
            status_cache = [np.ones(n_elem, dtype=np.float32) for _ in frames]
            deleted_counts = np.zeros(n_frames, dtype=np.int32)
            status_available = False

        times, f_smooth, disp_hist, peak_force, t_force = read_history_force(step, cfg)

        if t_status is not None:
            t_fracture = t_status
            fracture_source = 2       # 2=STATUS
            source_name = 'STATUS'
        elif t_force is not None:
            t_fracture = t_force
            fracture_source = 1       # 1=force
            source_name = 'Force'
        else:
            t_fracture = None
            fracture_source = 0       # 0=not found
            source_name = 'None'

        if t_status is not None and t_force is not None:
            print('  [Fracture] STATUS t={:.5f} / Force t={:.5f}; using STATUS'.format(
                t_status, t_force))
        elif t_fracture is not None:
            print('  [Fracture] using {} criterion, t={:.5f}'.format(source_name, t_fracture))
        else:

        curve_d, curve_f, d_fracture = build_dense_curve(
            times, f_smooth, disp_hist, t_fracture, cfg)
        print('  [DenseCurve] {} pts, disp 0~{:.2f}mm, fracture disp {:.3f}mm'.format(
            len(curve_d), curve_d[-1], float(d_fracture)))

        if cfg['REQUIRE_COMPLETE_FRACTURE']:
            if t_fracture is None:
                odb.close()
                return 0
            if float(d_fracture) >= cfg['MAX_FRACTURE_DISP_RATIO'] * cfg['TARGET_DISPLACEMENTS'][-1]:
                    float(d_fracture)))
                odb.close()
                return 0

        targets = cfg['TARGET_DISPLACEMENTS']
        max_disp = float(frame_disp.max())
        if max_disp < cfg['MIN_COMPLETION_RATIO'] * targets[-1]:
                  .format(max_disp, targets[-1], 100 * cfg['MIN_COMPLETION_RATIO']))
            odb.close()
            return 0

        sel = [int(np.argmin(np.abs(frame_disp - d))) for d in targets]
        n_dup = len(sel) - len(set(sel))
        print('  [FrameSelect] {} frames (dup {}) | actual disp {:.3f} ~ {:.3f} mm'
              .format(len(sel), n_dup, frame_disp[sel[0]], frame_disp[sel[-1]]))
        if n_dup > 3:

        frame_times_all = np.array([float(fr.frameValue) for fr in frames], dtype=np.float64)
        if times is not None:
            force_all = np.interp(frame_times_all, times, f_smooth)
            if t_fracture is not None and cfg['ZERO_FORCE_AFTER_FRACTURE']:
                force_all[frame_times_all >= t_fracture - 1e-12] = 0.0
            dense_sel = np.interp(frame_disp[sel], curve_d, curve_f)
            force_sel = force_all[sel]
            dense_mae = float(np.mean(np.abs(dense_sel - force_sel)))
            dense_max = float(np.max(np.abs(dense_sel - force_sel)))
            tol = max(float(cfg['DENSE_CURVE_VALIDATE_ABS_N']),
                      float(cfg['DENSE_CURVE_VALIDATE_REL']) * max(peak_force, 1.0))
                dense_mae, dense_max, tol))
            if dense_mae > tol:
                curve_d, curve_f = _curve_from_disp_force(
                    frame_disp, force_all, float(d_fracture), cfg)
                dense_sel = np.interp(frame_disp[sel], curve_d, curve_f)
                dense_mae = float(np.mean(np.abs(dense_sel - force_sel)))
                print('  [DenseCurve][Repaired] MAE={:.1f}N'.format(dense_mae))

        material_vec = material_table.get(
            odb_name, np.zeros(len(MATERIAL_KEYS), dtype=np.float32))

        ok = 0
        sigma_prev = np.zeros(n_nodes, dtype=np.float32)
        for k, fi in enumerate(sel):
            try:
                frame = frames[fi]
                t = float(frame.frameValue)

                post_frac = (t_fracture is not None) and (t >= t_fracture - 1e-12)
                fv = float(np.interp(t, times, f_smooth)) if times is not None else 0.0
                if post_frac and cfg['ZERO_FORCE_AFTER_FRACTURE']:
                    fv = 0.0

                alive_e = status_cache[fi].copy()

                stress_labels = np.zeros((n_nodes, 7), dtype=np.float32)
                stress_nodal = None
                if cfg.get('USE_ELEMENT_NODAL_STRESS', True):
                    stress_nodal = read_element_nodal_to_nodes(
                        frame, 'S', instance, inst_name, node_mapper, elem_mapper,
                        alive_e, n_nodes, 6, cfg['USE_BULK_DATA'])
                if stress_nodal is not None:
                    stress_labels[:, 0:6] = stress_nodal
                    stress_labels[:, 6] = mises_from_components(stress_nodal)
                else:
                    s_l, s_d = read_element_field(frame, 'S', instance, inst_name,
                                                  6, cfg['USE_BULK_DATA'])
                    stress_e = np.zeros((n_elem, 7), dtype=np.float32)
                    if s_l is not None:
                        stress_e[:, 0:6] = elem_field_to_rows(
                            s_l, s_d, elem_mapper, n_elem, 6)
                        stress_e[:, 6] = mises_from_components(stress_e[:, 0:6])

                strain_labels = np.zeros((n_nodes, 6), dtype=np.float32)
                strain_nodal = None
                if cfg['EXTRACT_DIC_EYY'] and cfg.get('USE_ELEMENT_NODAL_STRAIN', True):
                    strain_nodal = read_element_nodal_to_nodes(
                        frame, cfg['DIC_STRAIN_FIELD'], instance, inst_name,
                        node_mapper, elem_mapper, alive_e, n_nodes, 6,
                        cfg['USE_BULK_DATA'])
                if strain_nodal is not None:
                    strain_labels[:, :] = strain_nodal
                    strain_e = None
                else:
                    strain_e = np.zeros((n_elem, 6), dtype=np.float32)
                    if cfg['EXTRACT_DIC_EYY']:
                        le_l, le_d = read_element_field(
                            frame, cfg['DIC_STRAIN_FIELD'], instance, inst_name,
                            6, cfg['USE_BULK_DATA'])
                        if le_l is not None:
                            strain_e[:, 0:6] = elem_field_to_rows(
                                le_l, le_d, elem_mapper, n_elem, 6)
                        elif k == 0:
                                  .format(cfg['DIC_STRAIN_FIELD']))

                p_l, p_d = read_element_field(frame, 'PEEQ', instance, inst_name,
                                              1, cfg['USE_BULK_DATA'])
                peeq_e = elem_field_to_rows(p_l, p_d, elem_mapper, n_elem, 1)[:, 0]

                d_l, d_d = read_element_field(frame, 'SDEG', instance, inst_name,
                                              1, cfg['USE_BULK_DATA'])
                sdeg_e = elem_field_to_rows(d_l, d_d, elem_mapper, n_elem, 1)[:, 0]

                keep = alive_e[conn_rows] >= 0.5
                r = conn_rows[keep]
                c = conn_cols[keep]
                cnt = np.bincount(c, minlength=n_nodes).astype(np.float64)
                safe = np.maximum(cnt, 1.0)

                if stress_nodal is None:
                    for comp in range(7):
                        stress_labels[:, comp] = (
                            np.bincount(c, weights=stress_e[r, comp].astype(np.float64),
                                        minlength=n_nodes) / safe).astype(np.float32)

                if strain_nodal is None:
                    for comp in range(6):
                        strain_labels[:, comp] = (
                            np.bincount(c, weights=strain_e[r, comp].astype(np.float64),
                                        minlength=n_nodes) / safe).astype(np.float32)
                dic_eyy_percent = (
                    strain_labels[:, cfg['DIC_STRAIN_COMPONENT']] *
                    cfg['DIC_STRAIN_SCALE']).astype(np.float32)[:, None]

                peeq_labels = (np.bincount(c, weights=peeq_e[r].astype(np.float64),
                                           minlength=n_nodes) / safe
                               ).astype(np.float32)[:, None]
                sdeg_labels = (np.bincount(c, weights=sdeg_e[r].astype(np.float64),
                                           minlength=n_nodes) / safe
                               ).astype(np.float32)[:, None]
                node_alive = (cnt > 0).astype(np.float32)[:, None]

                nf = np.concatenate([
                    coords_arr.astype(np.float32),
                    bc_arr,
                    sigma_prev[:, None],
                    degree[:, None],
                ], axis=1).astype(np.float32)

                disp = U_cache[fi]
                load_disp = float(np.mean(disp[li, axis])) if li.size else \
                    float(np.max(np.abs(disp[:, axis])))

                global_force = np.zeros(3, dtype=np.float32)
                global_force[axis] = fv

                data = {
                    'node_features':       nf,
                    'edge_index':          edge_index,
                    'edge_features':       edge_features,
                    'stress_labels':       stress_labels,   # S11,S22,S33,S12,S13,S23,Mises
                    'strain_labels':       strain_labels,   # LE11,LE22,LE33,LE12,LE13,LE23
                    'dic_eyy_percent':     dic_eyy_percent,

                    'displacement_labels': disp,
                    'peeq_labels':         peeq_labels,
                    'sdeg_labels':         sdeg_labels,
                    'node_alive':          node_alive,
                    'global_force':        global_force,
                    'load_displacement':   np.array([load_disp], dtype=np.float32),
                    'material_params':     material_vec,
                    'post_fracture':       np.array([1.0 if post_frac else 0.0],
                                                    dtype=np.float32),
                    'frame_time':          np.array([t], dtype=np.float32),
                    'peak_force':          np.array([peak_force], dtype=np.float32),
                    'node_labels':         np.array(node_labels, dtype=np.int32),
                    'surface_tris':        surface_tris,
                    'surface_tri_owner':   surface_tri_owner.astype(np.int32),
                    'surface_tri_alive':   (alive_e[surface_tri_owner] >= 0.5).astype(np.float32),
                    'dic_eyy_range':       np.array(cfg['DIC_EYY_RANGE'], dtype=np.float32),
                    'dic_eyy_ticks':       np.array(cfg['DIC_EYY_TICKS'], dtype=np.float32),
                    'curve_disp':          curve_d,
                    'curve_force':         curve_f,
                    'fracture_disp':       np.array([d_fracture], dtype=np.float32),
                    'fracture_source':     np.array([fracture_source], dtype=np.int8),
                    'force_fracture_time': np.array([
                        -1.0 if t_force is None else t_force], dtype=np.float32),
                    'status_fracture_time': np.array([
                        -1.0 if t_status is None else t_status], dtype=np.float32),
                    'deleted_element_count': np.array([
                        int(deleted_counts[fi])], dtype=np.int32),
                    'max_deleted_elements': np.array([
                        int(np.max(deleted_counts)) if len(deleted_counts) else 0],
                        dtype=np.int32),
                    'status_available':    np.array([
                        1 if status_available else 0], dtype=np.int8),
                }

                fn = '{:s}_frame{:04d}.npz'.format(odb_name, k)
                np.savez_compressed(os.path.join(output_dir, fn), **data)

                print('   f{:02d} odb#{:<4d} disp={:6.3f}mm  F={:8.1f}N  maxMises={:7.2f}'
                      '  eyy=[{:6.3f},{:6.3f}]%  alive_elems={:5d}/{:d}{}'.format(
                          k, fi, load_disp, fv, float(stress_labels[:, 6].max()),
                          float(dic_eyy_percent.min()), float(dic_eyy_percent.max()),
                          int((alive_e >= 0.5).sum()), n_elem,
                          '  [post-frac,del{}]'.format(int(deleted_counts[fi])) if post_frac
                          else '  [del{}]'.format(int(deleted_counts[fi]))))

                sigma_prev = stress_labels[:, 6].copy()

                ok += 1
            except Exception as e:
                print('   [ERR] frame {} (odb#{}): {}'.format(k, fi, e))
                import traceback
                traceback.print_exc()

        odb.close()
        print('  Completed {}/{} frames'.format(ok, len(sel)))
        return ok

    except Exception as e:
        print('  [Exception] {}'.format(e))
        import traceback
        traceback.print_exc()
        try:
            odb.close()
        except Exception:
            pass
        return 0

# ================================================================
# ================================================================
def main(cfg):
    out = cfg['OUTPUT_DIR']
    if not os.path.exists(out):
        os.makedirs(out)
    else:
        old = glob.glob(os.path.join(out, '*.npz'))
        if old:

    material_table = load_material_meta(cfg['META_CSV'])

    n_job, n_frame, skipped = 0, 0, []

    if cfg.get('AUTO_DISCOVER_ODB', True):
        odb_paths = sorted(glob.glob(os.path.join(cfg['ODB_DIR'], 'Job_*.odb')))
        print('[ODB] Auto-discovered {} Job_*.odb files'.format(len(odb_paths)))
    else:
        start, end = cfg['ODB_INDEX_RANGE']
        odb_paths = []
        for i in range(start, end):
            fname = cfg['ODB_NAME_PATTERN'].format(i)
            path = os.path.join(cfg['ODB_DIR'], fname)
            if os.path.exists(path):
                odb_paths.append(path)
            else:
                skipped.append(fname + '(missing)')

    if not odb_paths:
        return

    for path in odb_paths:
        fname = os.path.basename(path)
        got = process_odb(path, out, cfg, material_table)
        if got > 0:
            n_job += 1
            n_frame += got
        else:
            skipped.append(fname)

    n_expect = len(cfg['TARGET_DISPLACEMENTS'])
    print('\n' + '=' * 62)
    print('Extraction complete: {} cases, {} samples (frames)'.format(n_job, n_frame))
    print('Output dir: {}'.format(out))
    if skipped:
        print('Failed {}: {}{}'.format(
            len(skipped), ', '.join(skipped[:20]), ' ...' if len(skipped) > 20 else ''))
    print('=' * 62)

if __name__ == '__main__':
    main(CONFIG)
