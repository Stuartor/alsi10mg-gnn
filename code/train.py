# -*- coding: utf-8 -*-
"""
GNN surrogate model for FE stress field prediction with independent force branch.
Query-mode: f(mesh, displacement, material_params) -> stress + force.
Supports displacement-query, two-stage training (field + force finetune),
and fracture visualization.
"""
import os
import re
import glob
import json
import math
import random
import warnings
import sys
import csv
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import tri as mtri
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, PowerNorm

warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
print('  PyTorch version: {}'.format(torch.__version__))

def set_global_seed(seed, deterministic=False):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

# ============================== CONFIG ==============================
CFG = {
    'DATA_DIR': r"./graph_dataset",

    'CKPT_DIR': r"./checkpoints",

    'FRAMES_PER_JOB': 28,

    'LOAD_AXIS': 1,

    'STRESS_TARGET_INDEX': 6,

    'DIC_STRAIN_INDEX': 1,

    'DIC_STRAIN_MEASURE': 'engineering',

    'DIC_OUTPUT_UNIT': 'strain',

    'USE_MATERIAL_COND': True,

    'N_MATERIAL_PARAMS': 5,

    'QUERY_MODE': True,

    'DISP_FOURIER_K': 4,

    'N_DENSE_DISP': 2,

    'MAX_DENSE_GRAPHS': 4,

    'W_DENSE_FORCE': 2.0,
    'W_FORCE_SLOPE': 0.0,
    'W_FORCE_CURVATURE': 0.0,
    'W_POST_FRACTURE_ZERO_FORCE': 0.0,
    'POST_FORCE_FRAC': 0.02,

    'DENSE_STRATIFIED_SAMPLING': False,

    'FORCE_LOSS_HUBER_BETA': 0.05,

    'TRAIN_LEARNED_CRACK_HEAD': False,
    'W_ALIVE': 0.0,
    'W_DAMAGE': 0.0,
    'W_FRACTURE_STATE': 0.0,
    'W_CRACK_CONSISTENCY': 0.0,
    'DEAD_FOCAL_GAMMA': 2.0,
    'DEAD_POS_WEIGHT_MIN': 1.5,
    'DEAD_POS_WEIGHT_MAX': 20.0,
    'W_DEAD_TVERSKY': 0.0,
    'DEAD_TVERSKY_ALPHA': 0.70,

    'DEAD_TVERSKY_BETA': 0.30,

    'W_CRACK_EDGE': 0.0,
    'DEAD_PROB_THRESHOLD': 0.55,

    'DAMAGE_CRACK_THRESHOLD': 0.78,

    'FRACTURE_PROB_THRESHOLD': 0.45,

    'DEAD_TRI_MIN_NODES': 1,

    'CRACK_BAND_FILTER': True,
    'CRACK_BAND_MIN_TRIANGLES': 4,
    'CRACK_DISPLAY_MODE': 'terminal_overlay',
    'CRACK_BAND_HALF_WIDTH': 0.82,
    'PRED_KEEP_COMPONENTS_AFTER_FRACTURE': 2,
    'CRACK_REMOVE_SMALL_MASK_ISLANDS': 2,
    'DISPLAY_DAMAGE_THRESHOLD': 0.84,
    'DISPLAY_DAMAGE_PERCENTILE': 0.55,
    'CRACK_EDGE_SHAVE_PASSES': 0,
    'CRACK_EDGE_SHAVE_MIN_NEIGHBORS': 2,
    'CRACK_MASK_DILATE_RINGS': 0,

    'SMOOTH_CRACK_GAP_FACTOR': 0.62,
    'SMOOTH_CRACK_GAP_MIN': 0.060,
    'SMOOTH_CRACK_GAP_MAX': 0.220,
    'SMOOTH_CRACK_EXTRA_BUFFER': 0.040,
    'SMOOTH_CRACK_WAVINESS': 0.0,
    'SMOOTH_CRACK_EDGE_LW': 3.2,
    'SMOOTH_CRACK_STYLE': 'flat',         # flat / mild_wave
    'SMOOTH_CRACK_COVERAGE_Q': (0.5, 99.5),
    'SMOOTH_CRACK_BAND_ROWS': 3.0,
    'SMOOTH_CRACK_FORCE_FULL_OPEN': True,
    'SMOOTH_CRACK_FORCE_ZERO_RATIO': 0.10,
    'SMOOTH_CRACK_FORCE_ZERO_ABS': 240.0,
    'SMOOTH_CRACK_MIN_COMPONENT_TRIANGLES': 6,
    'SHOW_JOB_NAME_IN_TITLES': False,

    'CRACK_DISPLAY_TERMINAL_ONLY': True,
    'CRACK_DISPLAY_MIN_DISP_RATIO': 0.94,
    'CRACK_DISPLAY_FROM_STRESS_ONLY': True,
    'CRACK_LOCATOR_FORCE_RATIO': 0.30,
    'CRACK_LOCATOR_FORCE_ABS_N': 700.0,
    'CRACK_LOCATOR_DISP_MIN_RATIO': 0.55,
    'CRACK_LOCATOR_SEARCH_QUANTILES': (0.25, 0.75),
    'CRACK_LOCATOR_BAND_ROWS': 1.10,
    'CRACK_LOCATOR_TOP_PERCENTILE': 75.0,
    'FIELD_SELECTION_USE_CRACK_METRICS': False,
    'CRACK_DISPLAY_FORCE_RATIO': 0.06,
    'CRACK_DISPLAY_FORCE_ABS_N': 150.0,
    'CRACK_DISPLAY_REQUIRE_FRAC_PROB': False,
    'CRACK_TERMINAL_FALLBACK_FROM_DAMAGE': True,
    'CRACK_TERMINAL_FALLBACK_HALF_ROWS': 1.10,
    'CRACK_TERMINAL_SEARCH_QUANTILES': (0.25, 0.75),

    'CRACK_ENABLE_DISP': 4.0,

    'CRACK_FULL_DISP': 5.9,

    'CRACK_THRESHOLD_CALIBRATE': False,
    'CRACK_CALIBRATE_MIN_RECALL': 0.90,
    'CRACK_THRESHOLD_FILE': 'crack_thresholds.json',
    'PRED_MASK_CLEANUP': True,
    'PRED_KEEP_COMPONENTS_AFTER_FRACTURE': 2,
    'QUERY_CHUNK': 2,


    'FORCE_DISPLAY_SMOOTH': True,
    'FORCE_DISPLAY_SMOOTH_PASSES': 1,
    'FORCE_DISPLAY_KERNEL': 7,
    'FORCE_PLOT_RAW_TOO': True,

    'FORCE_DISPLAY_VERTICAL_DROP': True,
    'FORCE_DISPLAY_DROP_MODE': 'manual',       # manual / auto / off
    'FORCE_DISPLAY_MANUAL_DROP_DISP': None,

    'ASK_MANUAL_FRACTURE_DISP_AFTER_TRAIN': True,
    'MANUAL_FRACTURE_DISP_FILE': 'manual_fracture_drop.json',
    'FORCE_DISPLAY_DROP_USE_FRAC_PROB': True,
    'FORCE_DISPLAY_DROP_FRAC_PROB_TH': 0.55,
    'FORCE_DISPLAY_DROP_FORCE_RATIO': 0.05,
    'FORCE_DISPLAY_DROP_FORCE_ABS_N': 80.0,

    'DENSE_SELECT_EVERY': 5,
    'DENSE_SELECT_POINTS': 201,

    'DENSE_SELECT_JOBS': 0,

    'DENSE_FINAL_POINTS': 301,

    'DENSE_TOPK': 6,

    'DENSE_WORST_QUANTILE': 0.90,

    'BEST_MIN_STRESS_R2': 0.965,
    'BEST_MIN_DEAD_IOU': 0.0,
    'ENABLE_EYY': False,
    'FIELD_SELECTION_EYY_WEIGHT': 0.0,
    'FIELD_SELECTION_HOT_WEIGHT': 0.35,
    'FIELD_SELECTION_CRACK_WEIGHT': 0.0,
    'FIELD_SELECTION_LATE_WEIGHT': 0.0,
    'BEST_CURVE_MIN_EPOCH': 10,
    'FINAL_TOPK_AVERAGE': 3,

    'VIZ_JOB_NAME': 'Job_0057',


    'N_VAL_JOBS': 8,
    'N_TEST_JOBS': 9,
    'SMALL_DATA_VAL_RATIO': 0.20,
    'SMALL_DATA_TEST_RATIO': 0.20,
    'MIN_TRAIN_JOBS': 2,
    'SPLIT_SEED': 42,
    'TRAIN_SEED': 42,
    'DETERMINISTIC_TRAINING': False,


    'HIDDEN': 128,
    'COORD_FOURIER_K': 4,
    'LAYERS': 8,
    'COND_DIM': 96,
    'PRIMARY_V41_EXACT': True,
    'EYY_DETACHED_BRANCH': False,
    'EYY_AUX_LAYERS': 0,
    'EYY_AUX_HIDDEN': 0,
    'V41_PRIMARY_WARMSTART_PATH': '',
    'TF_LAYERS': 2,
    'TF_HEADS': 4,
    'TF_DROPOUT': 0.1,

    'BATCH_JOBS': 4,
    'GRAD_ACCUM': 2,
    'EPOCHS': 220,
    'LR': 1e-4,
    'PATIENCE': 40,
    'MIN_EPOCHS': 60,
    'USE_AMP': True,
    'USE_COMPILE': False,


    'PHASE1_EPOCHS': 10,
    'PHASE2_SCHED_START': 10,
    'PHASE2_SCHED_END': 80,
    'PHASE2_LR_DROP': 40,
    'PHASE2_LR': 2e-5,

    'W_STRESS': 2.0,
    'W_EYY': 0.0,

    'W_DISP': 0.1,
    'W_FORCE': 0.20,
    'W_V41_GUIDE_FORCE': 0.50,
    'W_V41_GUIDE_DENSE_FORCE': 2.00,
    'W_SMOOTH': 0.1,
    'W_STRESS_SMOOTH': 0.05,
    'W_VAR0': 5.0,
    'T0_STRESS_WEIGHT': 15.0,
    'POST_FRACTURE_STRESS_W': 0.8,

    'W_DAMAGE_REGION': 4.0,

    'W_HIGH_STRESS_REGION': 1.5,

    'DENSE_FORCE_ONLY_LAST_FRAME': True,
    'DENSE_FORCE_DETACH_TRUNK': True,

    'FORCE_BRANCH_SEPARATE': True,
    'FORCE_RESIDUAL_LIMIT': 0.08,

    'FORCE_FAST_RATE_MIN': 18.0,
    'FORCE_FAST_RATE_MAX': 140.0,
    'FORCE_SLOW_RATE_MIN': 1.5,
    'FORCE_SLOW_RATE_MAX': 28.0,
    'FORCE_DCRIT_MIN': 0.82,

    'FORCE_DCRIT_MAX': 0.98,
    'FORCE_SHARPNESS_MIN': 25.0,
    'FORCE_SHARPNESS_MAX': 140.0,

    'FIELD_LATE_FRAME_START': 21,
    'FIELD_LATE_FRAME_WEIGHT': 1.0,
    'FIELD_FINAL_FRAME_WEIGHT': 1.0,
    'FIELD_CRACK_NEIGHBOR_RINGS': 2,
    'FIELD_CRACK_STRESS_WEIGHT': 0.0,
    'FIELD_CRACK_EYY_WEIGHT': 0.0,
    'FIELD_LOCAL_GRAD_WEIGHT': 0.0,
    'FIELD_GT_GRAD_QUANTILE': 0.88,
    'INTACT_FIELD_V41_PROFILE': True,
    'EYY_DAMAGE_REGION_WEIGHT': 0.0,

    'FORCE_FINETUNE': True,
    'FORCE_FINETUNE_EPOCHS': 160,
    'FORCE_FINETUNE_LR': 2.5e-4,
    'FORCE_FINETUNE_WEIGHT_DECAY': 1e-6,
    'FORCE_FINETUNE_BATCH_JOBS': 8,
    'FORCE_FINETUNE_PATIENCE': 28,
    'FORCE_FINETUNE_D1_WEIGHT': 0.16,
    'FORCE_FINETUNE_D2_WEIGHT': 0.06,
    'FORCE_FINETUNE_RIPPLE_WEIGHT': 0.10,
    'FORCE_FINETUNE_POST_WEIGHT': 0.50,
    'FORCE_FINETUNE_EVAL_POINTS': 501,
    'FORCE_QUERY_CHUNK': 4096,
    'FORCE_FINETUNE_REGION_BALANCE': True,
    'FORCE_FINETUNE_EARLY_RATIO': 0.16,
    'FORCE_FINETUNE_EARLY_WEIGHT': 2.80,
    'FORCE_FINETUNE_ELASTIC_SLOPE_WEIGHT': 0.45,
    'FORCE_FINETUNE_ELASTIC_CURV_WEIGHT': 0.10,
    'FORCE_FINETUNE_ELASTIC_ANCHOR_WEIGHT': 0.25,
    'FORCE_FINETUNE_DROP_WEIGHT': 2.50,
    'FORCE_FINETUNE_POST_VALUE_WEIGHT': 2.00,
    'FORCE_FINETUNE_PEAK_WEIGHT': 0.10,
    'FORCE_FINETUNE_DROP_D1_WEIGHT': 0.25,
    'FORCE_FINETUNE_LR_PATIENCE': 8,
    'FORCE_FINETUNE_MIN_LR': 2e-5,
    'DENSE_ELASTIC_RATIO': 0.16,
    'DENSE_ELASTIC_RMSE_WEIGHT': 0.10,
    'DENSE_ELASTIC_SLOPE_WEIGHT': 0.10,

    'CLEAN_FIELD_CKPT_PATH': '',
    'RESUME_FORCE_STAGE_ONLY': False,
    'RESUME_FIELD_EPOCH_HINT': 50,

    'FAIL_ON_EMPTY_TRAIN': True,
    'PRINT_GRAD_NORM': True,
    'RECOMPUTE_STATS_IF_DATA_CHANGED': True,
    'STATS_VERSION': 6,
    'CURVE_AUTO_REPAIR': True,
    'CURVE_REPAIR_ABS_N': 120.0,
    'CURVE_REPAIR_REL': 0.06,
    'TRAIN_DIAG_EVERY': 5,

    'TRAIN_DIAG_JOBS': 8,
    'OVERFIT_WARN_RATIO': 1.15,


    'SURFACE_AXIS': 2,

    'SURFACE_SIDE': 'min',

    'VIZ_CROP': (-36.0, 6.0),

    'VIZ_FRAMES': 6,
    'STRESS_PLOT_RANGE': (0.0, 330.0),
    'STRESS_PLOT_PERCENTILES': (1.0, 99.5),
    'STRESS_PLOT_LEVELS': 41,

    'STRESS_PLOT_TICKS': [0, 55, 110, 165, 220, 275, 330],
    'STRESS_DISPLAY_MODE': 'balanced_contour',  # balanced_contour / contourf / gouraud
    'STRESS_DISPLAY_SMOOTH_PASSES': 1,
    'STRESS_DISPLAY_SMOOTH_ALPHA': 0.14,
    'PRED_STRESS_DISPLAY_EXTRA_SMOOTH_PASSES': 1,
    'PRED_STRESS_DISPLAY_EXTRA_SMOOTH_ALPHA': 0.10,
    'PRED_STRESS_DISPLAY_DEJAG': True,
    'PRED_STRESS_DISPLAY_DEJAG_BLEND': 0.65,
    'PRED_STRESS_DISPLAY_DEJAG_CAP_MPA': 4.5,
    'PRED_STRESS_DISPLAY_DEJAG_REL': 1.35,
    'STRESS_CONTOUR_LINE_EVERY': 4,
    'STRESS_CONTOUR_LINE_ALPHA': 0.22,
    'STRESS_CONTOUR_LINE_WIDTH': 0.22,
    'STRESS_CONTOUR_LINE_COLOR': 'white',
    'PRED_GEOMETRY_DISPLAY_SMOOTH_PASSES': 2,
    'PRED_GEOMETRY_DISPLAY_SMOOTH_ALPHA': 0.18,
    'ERROR_PLOT_FIXED_MAX_MPA': 30.0,
    'ERROR_PLOT_GAMMA': 1.08,
    'ERROR_PLOT_TICKS_MPA': [0, 5, 10, 15, 20, 25, 30],
    'STRESS_PLOT_LABEL': 'Stress (MPa)',
    'SAVE_EVERY_EPOCHS': 10,
    'VIZ_DPI': 280,
    'DIC_EYY_RANGE': (-0.0076, 0.0254),
    'DIC_EYY_TICKS': [-0.0076, -0.0055375, -0.003475, -0.0014125,
                      0.00065, 0.0027125, 0.004775, 0.0068375,
                      0.0089, 0.0109625, 0.013025, 0.0150875,
                      0.01715, 0.0192125, 0.021275, 0.0233375, 0.0254],
    'DIC_PLOT_LEVELS': 97,
    'DIC_FIELD_CMAP': 'turbo',

    'STRESS_FIELD_CMAP': 'jet',
    'DAMAGE_CMAP': 'YlOrRd',

    'ERROR_MODE': 'absolute',

    'ERROR_PERCENTILE': 99.0,
    'SAVE_DIC_VIZ': False,

    'QUERY_DEMO': True,
    'QUERY_CURVE_N': 201,

    'QUERY_DEMO_DISPS': [1.0, 3.0, 4.8, 5.4],

    'INTERACTIVE_QUERY_AFTER_TRAIN': True,

    'QUERY_REPORT_DPI': 220,

    'QUERY_COMPARE_WITH_GT': True,
    'QUERY_SAVE_CRACK_REPORT': True,
    'QUERY_GT_INTERPOLATE': True,

    'QUERY_ERROR_PERCENTILE': 99.0,

    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
}
os.makedirs(CFG['CKPT_DIR'], exist_ok=True)

try:
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW = False

def amp_ctx(enabled):
    if _AMP_NEW:
        return autocast('cuda', enabled=enabled)
    return autocast(enabled=enabled)

def make_scaler(enabled):
    if _AMP_NEW:
        return GradScaler('cuda', enabled=enabled)
    return GradScaler(enabled=enabled)

def dic_from_log_strain(le, cfg):
    arr = np.asarray(le, dtype=np.float32)
    if cfg.get('DIC_STRAIN_MEASURE', 'engineering') == 'engineering':
        out = np.expm1(arr)
    else:
        out = arr
    if cfg.get('DIC_OUTPUT_UNIT', 'strain') == 'percent':
        out = out * 100.0
    return out.astype(np.float32)

def dic_from_percent_array(arr, cfg):
    out = np.asarray(arr, dtype=np.float32).reshape(-1)
    if cfg.get('DIC_OUTPUT_UNIT', 'strain') == 'strain':
        out = out / 100.0
    return out.astype(np.float32)

def dic_unit_label(cfg):
    return 'eyy [%] - engr.' if cfg.get('DIC_OUTPUT_UNIT') == 'percent' else 'eyy [1] - engr.'

def make_abs_error_cmap():

    Purpose:
    """
    cmap = LinearSegmentedColormap.from_list(
        'abs_error_balanced_yellow_orange_red',
        [
            (0.00, '#fff3c4'),
            (0.10, '#fee8a6'),
            (0.25, '#fdd17d'),
            (0.42, '#fdbb63'),
            (0.60, '#fc8d59'),
            (0.78, '#ef6548'),
            (1.00, '#d9482f'),
        ],
        N=256)
    cmap.set_over('#c83e2c')
    cmap.set_under('#fff3c4')
    return cmap

ABS_ERROR_CMAP = make_abs_error_cmap()

def make_error_norm(vmax, cfg):

    """
    return PowerNorm(
        gamma=max(float(cfg.get('ERROR_PLOT_GAMMA', 1.08)), 1e-6),
        vmin=0.0,
        vmax=max(float(vmax), 1e-6),
        clip=True,
    )

def scatter_sum(src, index, dim_size):
    size = list(src.shape)
    size[0] = dim_size
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    idx = index.long().view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_add_(0, idx, src)
    return out

def scatter_mean(src, index, dim_size):
    s = scatter_sum(src, index, dim_size)
    cnt = scatter_sum(torch.ones_like(src[:, :1]), index, dim_size).clamp(min=1)
    return s / cnt

def get_axis_step_np(coords, axis):
    u = np.unique(np.round(coords[:, axis], 6))
    d = np.diff(np.sort(u))
    d = d[d > 1e-8]
    return float(np.median(d)) if len(d) else 1e-6

def cond_dim(cfg):
    d = 1 + 2 * cfg['DISP_FOURIER_K'] if cfg['QUERY_MODE'] else 1
    if cfg['USE_MATERIAL_COND']:
        d += cfg['N_MATERIAL_PARAMS']
    return d

def make_cond(load_disp, mat_params_n, stats, cfg):
    d = np.atleast_1d(np.asarray(load_disp, dtype=np.float64))
    dn = d / float(stats['max_load_disp'])
    feats = [dn[:, None]]
    if cfg['QUERY_MODE']:
        for k in range(cfg['DISP_FOURIER_K']):
            w = np.pi * (2.0 ** k)
            feats += [np.sin(w * dn)[:, None], np.cos(w * dn)[:, None]]
    if cfg['USE_MATERIAL_COND']:
        mp = np.asarray(mat_params_n, dtype=np.float64).reshape(1, -1)
        feats.append(np.repeat(mp, len(dn), axis=0))
    return np.concatenate(feats, axis=1).astype(np.float32)

def group_files_by_job(data_dir):
    files = glob.glob(os.path.join(data_dir, '*_frame*.npz'))
    jobs = {}
    for f in files:
        job = re.sub(r'_frame\d+\.npz$', '', os.path.basename(f))
        jobs.setdefault(job, []).append(f)
    return jobs

def dataset_signature(data_dir):
    files = glob.glob(os.path.join(data_dir, '*_frame*.npz'))
    jobs = group_files_by_job(data_dir)
    newest = max([os.path.getmtime(f) for f in files], default=0.0)
    return '{}jobs_{}files_{:.0f}'.format(len(jobs), len(files), newest)

def compute_stats(data_dir, cfg, max_files=1200):
    files = glob.glob(os.path.join(data_dir, '*_frame*.npz'))
    if not files:
        raise RuntimeError('DATA_DIR does not contain npz：{}'.format(data_dir))
    random.Random(0).shuffle(files)
    files = files[:max_files]
    si = cfg['STRESS_TARGET_INDEX']
    ax = cfg['LOAD_AXIS']

    coords, stress, eyy, disp, degree = [], [], [], [], []
    use_eyy = bool(cfg.get('ENABLE_EYY', False))
    force, loaddisp, matp = [], [], []
    for f in tqdm(files, desc='[Stats]'):
        d = np.load(f, allow_pickle=True)
        nf = d['node_features']
        coords.append(nf[:, 0:3])
        degree.append(nf[:, 7])
        stress.append(d['stress_labels'][:, si])
        if use_eyy:
            if 'strain_labels' in d:
                le = d['strain_labels'][:, cfg['DIC_STRAIN_INDEX']]
                eyy.append(dic_from_log_strain(le, cfg))
            elif 'dic_eyy_percent' in d:
                eyy.append(dic_from_percent_array(d['dic_eyy_percent'], cfg))
            else:
        disp.append(d['displacement_labels'])
        force.append(float(d['global_force'][ax]))
        loaddisp.append(float(d['load_displacement'][0]))
        if 'material_params' in d:
            matp.append(d['material_params'])

    coords = np.concatenate(coords, 0)
    stress = np.concatenate(stress, 0)
    eyy = np.concatenate(eyy, 0) if use_eyy and eyy else np.zeros(1, dtype=np.float32)
    disp = np.concatenate(disp, 0)
    degree = np.concatenate(degree, 0)
    matp = np.array(matp, dtype=np.float64) if matp else np.zeros((1, cfg['N_MATERIAL_PARAMS']))

    return {
        'coord_mean': coords.mean(0).astype(np.float32),
        'coord_std': (coords.std(0) + 1e-6).astype(np.float32),
        'stress_mean': float(stress.mean()),
        'stress_std': float(stress.std() + 1e-6),
        'eyy_mean': float(eyy.mean()),
        'eyy_std': float(eyy.std() + 1e-6),
        'disp_mean': disp.mean(0).astype(np.float32),
        'disp_std': (disp.std(0) + 1e-6).astype(np.float32),
        'degree_mean': float(degree.mean()),
        'degree_std': float(degree.std() + 1e-6),
        'force_mean': 0.0,
        'force_std': float(np.percentile(np.abs(force), 99.5) + 1e-6),
        'max_load_disp': float(np.max(np.abs(loaddisp)) + 1e-8),
        'mat_mean': matp.mean(0).astype(np.float32),
        'mat_std': (matp.std(0) + 1e-6).astype(np.float32),
        'dataset_signature': dataset_signature(data_dir),
        'stats_version': int(cfg.get('STATS_VERSION', 5)),
    }

def save_stats(stats, path):
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in stats.items()}, open(path, 'w'))

def load_stats(path):
    s = json.load(open(path))
    for k in ['coord_mean', 'coord_std', 'disp_mean', 'disp_std', 'mat_mean', 'mat_std']:
        if k in s:
            s[k] = np.array(s[k], dtype=np.float32)
    return s

class SharedTopology(object):
    """
    """

    def __init__(self, sample_file, stats):
        d = np.load(sample_file, allow_pickle=True)
        nf = d['node_features'].astype(np.float32)
        coords = nf[:, 0:3]
        bc = nf[:, 3:6]
        degree = nf[:, 7]

        self.n_nodes = coords.shape[0]
        self.raw_coords = coords.copy()

        coords_n = (coords - stats['coord_mean']) / stats['coord_std']
        degree_n = (degree - stats['degree_mean']) / stats['degree_std']
        pos_feats = []
        for k in range(int(CFG.get('COORD_FOURIER_K', 0))):
            w = np.pi * (2.0 ** k)
            pos_feats += [np.sin(w * coords_n), np.cos(w * coords_n)]
        pos_fourier = np.concatenate(pos_feats, 1).astype(np.float32) \
            if pos_feats else np.zeros((len(coords_n), 0), dtype=np.float32)
        self.x_static = torch.tensor(
            np.concatenate([coords_n, bc, degree_n[:, None], pos_fourier], 1),
            dtype=torch.float32)
        self.in_node_dim = int(self.x_static.shape[1] + 1)  # + sigma_prev

        ea = d['edge_features'].astype(np.float32).copy()
        ea[:, :3] /= stats['coord_std']
        ea[:, 3] /= float(stats['coord_std'].mean())
        self.edge_index = torch.tensor(d['edge_index'].astype(np.int64))
        self.edge_attr = torch.tensor(ea)
        self.surface_tris = d['surface_tris'].astype(np.int64)
        self.surface_tri_owner = d['surface_tri_owner'].astype(np.int64)             if 'surface_tri_owner' in d else None
        self.bc = bc
        self.zero_prev = torch.zeros(self.n_nodes, dtype=torch.float32)

            self.n_nodes, self.edge_index.shape[1], len(self.surface_tris)))

    def build_x(self, sigma_prev_n):
        """Insert dynamic sigma_prev after coordinates+BC."""
        return torch.cat([self.x_static[:, 0:6],
                          sigma_prev_n.view(-1, 1),
                          self.x_static[:, 6:]], dim=1)

    def batched_topology(self, B, device, _cache={}):
        key = (B, str(device))
        if key not in _cache:
            n = self.n_nodes
            ei = torch.cat([self.edge_index + k * n for k in range(B)], 1).to(device)
            ea = self.edge_attr.repeat(B, 1).to(device)
            bi = torch.arange(B, device=device).repeat_interleave(n)
            _cache[key] = (ei, ea, bi)
        return _cache[key]

class JobGroup(object):

    def __init__(self, files, stats, cfg, shared):
        files = sorted(files, key=lambda f: int(re.search(r'frame(\d+)', f).group(1)))
        si = cfg['STRESS_TARGET_INDEX']
        ax = cfg['LOAD_AXIS']
        n_keep = cfg['FRAMES_PER_JOB']
        self.frames = []
        self.job_name = re.sub(r'_frame\d+\.npz$', '', os.path.basename(files[0]))

        self.curve_disp = None
        self.curve_force = None
        self.mat_n = np.zeros(cfg['N_MATERIAL_PARAMS'], np.float32)
        self.fracture_disp = -1.0

        for fp in files[:n_keep]:
            d = np.load(fp, allow_pickle=True)
            nf = d['node_features'].astype(np.float32)

            if self.curve_disp is None:
                if 'curve_disp' in d:
                    self.curve_disp = d['curve_disp'].astype(np.float32)
                    self.curve_force = d['curve_force'].astype(np.float32)
                if 'fracture_disp' in d:
                    self.fracture_disp = float(d['fracture_disp'][0])

            sigma_prev_n = (nf[:, 6] - stats['stress_mean']) / stats['stress_std']
            y_raw = d['stress_labels'][:, si].astype(np.float32)
            y_stress_n = (y_raw - stats['stress_mean']) / stats['stress_std']

            if cfg.get('ENABLE_EYY', False):
                if 'strain_labels' in d:
                    le = d['strain_labels'][:, cfg['DIC_STRAIN_INDEX']].astype(np.float32)
                    eyy_raw = dic_from_log_strain(le, cfg)
                elif 'dic_eyy_percent' in d:
                    eyy_raw = dic_from_percent_array(d['dic_eyy_percent'], cfg)
                else:
                y_eyy_n = (eyy_raw - stats['eyy_mean']) / stats['eyy_std']
            else:
                eyy_raw = np.zeros_like(y_raw, dtype=np.float32)
                y_eyy_n = np.zeros_like(y_raw, dtype=np.float32)

            disp = d['displacement_labels'].astype(np.float32)
            y_disp_n = (disp - stats['disp_mean']) / stats['disp_std']
            force = float(d['global_force'][ax])
            force_n = (force - stats['force_mean']) / stats['force_std']
            ld = float(d['load_displacement'][0])

            mp = d['material_params'].astype(np.float32) if 'material_params' in d \
                else np.zeros(cfg['N_MATERIAL_PARAMS'], np.float32)
            self.mat_n = ((mp - stats['mat_mean']) / stats['mat_std']).astype(np.float32)
            cond = make_cond(ld, self.mat_n, stats, cfg)[0]

            alive = d['node_alive'][:, 0].astype(np.float32)
            tri_alive = d['surface_tri_alive'].astype(np.float32) \
                if 'surface_tri_alive' in d else None
            if tri_alive is not None and len(tri_alive) == len(shared.surface_tris):
                surface_alive = np.ones(shared.n_nodes, dtype=np.float32)
                dead_tris = shared.surface_tris[tri_alive < 0.5]
                if dead_tris.size:
                    surface_alive[np.unique(dead_tris.reshape(-1))] = 0.0
                alive = np.minimum(alive, surface_alive)
            damage = d['sdeg_labels'][:, 0].astype(np.float32) \
                if 'sdeg_labels' in d else np.zeros(shared.n_nodes, np.float32)
            self.frames.append({
                'sigma_prev': torch.tensor(sigma_prev_n),
                'y_stress': torch.tensor(y_stress_n),
                'y_eyy': torch.tensor(y_eyy_n),
                'raw_eyy': eyy_raw,
                'y_disp': torch.tensor(y_disp_n),
                'y_force': torch.tensor([force_n], dtype=torch.float32),
                'cond': torch.tensor(cond, dtype=torch.float32),
                'alive': torch.tensor(alive),
                'damage': torch.tensor(damage),
                'tri_alive': tri_alive,
                'raw_load_disp': ld,
                'raw_force': force,
                'post_frac': float(d['post_fracture'][0]) if 'post_fracture' in d else 0.0,
            })

        if len(self.frames) < n_keep:
            raise RuntimeError('{} has only {} frames, fewer than FRAMES_PER_JOB={}'.format(
                self.job_name, len(self.frames), n_keep))

        if self.curve_disp is not None:
            fd = np.array([fr['raw_load_disp'] for fr in self.frames], dtype=np.float64)
            ff = np.array([fr['raw_force'] for fr in self.frames], dtype=np.float64)
            cf = np.interp(fd, self.curve_disp, self.curve_force)
            mae = float(np.mean(np.abs(cf - ff)))
            tol = max(float(cfg.get('CURVE_REPAIR_ABS_N', 120.0)),
                      float(cfg.get('CURVE_REPAIR_REL', 0.06)) * max(float(ff.max()), 1.0))
            if cfg.get('CURVE_AUTO_REPAIR', True) and mae > tol:
                order = np.argsort(fd)
                du, idx = np.unique(np.round(fd[order], 8), return_index=True)
                fu = ff[order][idx]
                self.curve_force = np.interp(self.curve_disp, du, fu).astype(np.float32)
                self.curve_force[self.curve_disp > du.max()] = 0.0
                print('  [CurveRepair] {}: dense/frame MAE {:.1f}N > {:.1f}N, rebuilt with 28 frames'.format(
                    self.job_name, mae, tol))
            self.curve_cond = torch.tensor(make_cond(self.curve_disp, self.mat_n, stats, cfg))
            self.curve_force_n = torch.tensor(
                (self.curve_force - stats['force_mean']) / stats['force_std'], dtype=torch.float32)
        else:
            self.curve_cond, self.curve_force_n = None, None

    def __getitem__(self, t):
        return self.frames[t]

    def __len__(self):
        return len(self.frames)

class FiLM(nn.Module):
    def __init__(self, c, cd):
        super().__init__()
        self.g = nn.Linear(cd, c)
        self.b = nn.Linear(cd, c)
        nn.init.xavier_normal_(self.g.weight, gain=0.01)
        nn.init.zeros_(self.g.bias)
        nn.init.xavier_normal_(self.b.weight, gain=0.01)
        nn.init.zeros_(self.b.bias)

    def forward(self, x, c):
        return (1 + self.g(c)) * x + self.b(c)

class ProcLayer(nn.Module):
    def __init__(self, h, cd):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(3 * h, h), nn.LayerNorm(h), nn.SiLU(),
                                 nn.Linear(h, h))
        self.upd = nn.Sequential(nn.Linear(2 * h, h), nn.LayerNorm(h), nn.SiLU())
        self.film = FiLM(h, cd)

    def forward(self, h_n, h_e, ei, cond_per_node):
        r, c = ei
        m = self.msg(torch.cat([h_n[r], h_n[c], h_e], -1))
        agg = scatter_sum(m, c, h_n.size(0))
        h_new = self.upd(torch.cat([h_n, agg], -1))
        h_new = self.film(h_new, cond_per_node)
        return h_n + h_new, h_e

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x):
        pos = torch.arange(x.size(0), device=x.device)
        return x + self.pos_embed(pos).unsqueeze(1)

class TransformerForceHead(nn.Module):

    def __init__(self, d_model, nhead=4, num_layers=2, dropout=0.1, max_len=64):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=False)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, pooled_seq):
        L = pooled_seq.size(0)
        mask = torch.triu(torch.full((L, L), float('-inf'),
                                     device=pooled_seq.device), diagonal=1)
        x = self.pos_encoder(pooled_seq)
        x = self.transformer(x, mask=mask)
        return self.head(x[-1]).squeeze(-1)

class GNNModel(nn.Module):
    def __init__(self, in_n=8, in_e=4, in_cond=6, h=192, layers=12, cd=128,
                 tf_layers=2, tf_heads=4, tf_dropout=0.1, max_len=64,
                 query_mode=True, force_fourier_k=4, force_mat_dim=5,
                 force_residual_limit=0.08,
                 force_dcrit_min=0.82, force_dcrit_max=0.98,
                 force_sharp_min=25.0, force_sharp_max=140.0,
                 eyy_aux_layers=2, enable_eyy=False,
                 force_fast_rate_min=18.0, force_fast_rate_max=140.0,
                 force_slow_rate_min=1.5, force_slow_rate_max=28.0):
        super().__init__()
        self.query_mode = query_mode
        self.node_enc = nn.Sequential(nn.Linear(in_n, h), nn.LayerNorm(h), nn.SiLU(),
                                      nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU())
        self.edge_enc = nn.Sequential(nn.Linear(in_e, h), nn.LayerNorm(h), nn.SiLU(),
                                      nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU())
        self.cond_enc = nn.Sequential(nn.Linear(in_cond, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.layers = nn.ModuleList([ProcLayer(h, cd) for _ in range(layers)])
        self.stress_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        self.disp_head = nn.Sequential(nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 3))
        self.v41_guide_force_q = nn.Sequential(
            nn.Linear(h + cd, h), nn.GELU(),
            nn.Linear(h, h // 2), nn.GELU(),
            nn.Linear(h // 2, 1))
        self._last_v41_guide_force = None

        self.enable_eyy = bool(enable_eyy)
        if self.enable_eyy:
            self.eyy_edge_enc = nn.Sequential(
                nn.Linear(in_e, h), nn.LayerNorm(h), nn.SiLU(),
                nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU())
            self.eyy_layers = nn.ModuleList(
                [ProcLayer(h, cd) for _ in range(max(1, int(eyy_aux_layers)))])
            self.eyy_norm = nn.LayerNorm(h)
            self.eyy_head = nn.Sequential(
                nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 1))
        else:
            self.eyy_edge_enc = None
            self.eyy_layers = nn.ModuleList()
            self.eyy_norm = None
            self.eyy_head = None
        self.dead_head = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))
        self.damage_head = nn.Sequential(nn.Linear(h, h // 2), nn.SiLU(), nn.Linear(h // 2, 1))
        nn.init.constant_(self.dead_head[-1].bias, -4.0)
        nn.init.constant_(self.damage_head[-1].bias, -4.0)

        if query_mode:
            self.force_fourier_k = int(force_fourier_k)
            self.force_mat_dim = int(force_mat_dim)
            self.force_residual_limit = float(force_residual_limit)
            self.force_dcrit_min = float(force_dcrit_min)
            self.force_dcrit_max = float(force_dcrit_max)
            self.force_sharp_min = float(force_sharp_min)
            self.force_sharp_max = float(force_sharp_max)

            mat_in = max(1, self.force_mat_dim)
            self.force_mat_enc = nn.Sequential(
                nn.Linear(mat_in, 64), nn.SiLU(),
                nn.Linear(64, 64), nn.SiLU())
            # fast_amp/fast_rate/slow_amp/slow_rate/hardening/dcrit/sharp
            self.force_param_head = nn.Linear(64, 7)
            self.force_fast_rate_min = float(force_fast_rate_min)
            self.force_fast_rate_max = float(force_fast_rate_max)
            self.force_slow_rate_min = float(force_slow_rate_min)
            self.force_slow_rate_max = float(force_slow_rate_max)
            force_feat_dim = 5 + self.force_mat_dim
            self.force_residual = nn.Sequential(
                nn.Linear(force_feat_dim, 64), nn.SiLU(),
                nn.Linear(64, 32), nn.SiLU(),
                nn.Linear(32, 1))
            nn.init.zeros_(self.force_residual[-1].weight)
            nn.init.zeros_(self.force_residual[-1].bias)
        else:
            self.tf_force = TransformerForceHead(h, tf_heads, tf_layers, tf_dropout, max_len)
            self.force_first = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
        self.fracture_head = nn.Sequential(nn.Linear(h + cd, h // 2), nn.SiLU(),
                                           nn.Linear(h // 2, 1))
        nn.init.constant_(self.fracture_head[-1].bias, -4.0)

    def encode_cond(self, cond, num_graphs):
        return self.cond_enc(cond.view(num_graphs, -1).float())

    def trunk(self, x, ei, ea, cond_emb, batch_idx):
        h_n = self.node_enc(x)
        h_e = self.edge_enc(ea)
        cond_per_node = cond_emb[batch_idx]
        for layer in self.layers:
            h_n, h_e = layer(h_n, h_e, ei, cond_per_node)
        return h_n

    def eyy_branch(self, h_primary, ei, ea, cond_emb, batch_idx):
        if not self.enable_eyy:
            return h_primary.new_zeros(h_primary.size(0))
        h_n = h_primary.detach()
        h_e = self.eyy_edge_enc(ea)
        cond_per_node = cond_emb.detach()[batch_idx]
        for layer in self.eyy_layers:
            h_n, h_e = layer(h_n, h_e, ei, cond_per_node)
        return self.eyy_head(self.eyy_norm(h_n)).squeeze(-1)

    def guide_force_from_features(self, pooled, cond_emb, cond):
        raw = self.v41_guide_force_q(
            torch.cat([pooled, cond_emb], dim=-1)).squeeze(-1)
        dn = cond[:, 0].view(-1).float().clamp(min=0.0)
        return dn * nn.functional.softplus(raw)

    def guide_force_forward(self, x, ei, ea, cond, batch_idx, num_graphs):
        cond_emb = self.encode_cond(cond, num_graphs)
        h_n = self.trunk(x, ei, ea, cond_emb, batch_idx)
        pooled = scatter_mean(h_n, batch_idx, num_graphs)
        return self.guide_force_from_features(pooled, cond_emb, cond)

    def force_features(self, cond):
        dn = cond[:, 0:1].float().clamp(min=0.0, max=1.15)
        mat_start = 1 + 2 * self.force_fourier_k
        if self.force_mat_dim > 0 and cond.size(1) >= mat_start + self.force_mat_dim:
            mat = cond[:, mat_start:mat_start + self.force_mat_dim].float()
        else:
            mat = cond.new_zeros((cond.size(0), max(1, self.force_mat_dim))).float()

        feat = torch.cat([
            dn,
            dn * dn,
            dn * dn * dn,
            torch.sin(np.pi * dn),
            torch.cos(np.pi * dn),
            mat[:, :self.force_mat_dim] if self.force_mat_dim > 0
            else mat[:, :0],
        ], dim=1)
        return dn, mat, feat

    def force_from_cond(self, cond):
        dn, mat, feat = self.force_features(cond)
        if self.force_mat_dim <= 0:
            mat_ctx = mat[:, :1]
        else:
            mat_ctx = mat
        ctx = self.force_mat_enc(mat_ctx)
        pars = self.force_param_head(ctx)

        amp_fast = nn.functional.softplus(pars[:, 0:1]) + 0.02
        rate_fast = (self.force_fast_rate_min +
                     (self.force_fast_rate_max - self.force_fast_rate_min) *
                     torch.sigmoid(pars[:, 1:2]))
        amp_slow = nn.functional.softplus(pars[:, 2:3]) + 0.02
        rate_slow = (self.force_slow_rate_min +
                     (self.force_slow_rate_max - self.force_slow_rate_min) *
                     torch.sigmoid(pars[:, 3:4]))
        hard = 0.12 * torch.tanh(pars[:, 4:5])

        dcrit = (self.force_dcrit_min +
                 (self.force_dcrit_max - self.force_dcrit_min) *
                 torch.sigmoid(pars[:, 5:6]))
        sharp = (self.force_sharp_min +
                 (self.force_sharp_max - self.force_sharp_min) *
                 torch.sigmoid(pars[:, 6:7]))

        base = (amp_fast * (1.0 - torch.exp(-rate_fast * dn)) +
                amp_slow * (1.0 - torch.exp(-rate_slow * dn)) +
                hard * dn)

        residual = (self.force_residual_limit *
                    torch.tanh(self.force_residual(feat)) *
                    dn * (1.0 - dn.clamp(max=1.0)))

        gate = torch.sigmoid((dcrit - dn) * sharp)
        force = torch.clamp(base + residual, min=0.0) * gate
        return force.squeeze(-1)

    def force_parameters(self):
        if not self.query_mode:
            return []
        modules = [self.force_mat_enc, self.force_param_head, self.force_residual]
        return [p for m in modules for p in m.parameters()]

    def primary_field_parameters(self):
        modules = [self.node_enc, self.edge_enc, self.cond_enc,
                   self.layers, self.stress_head, self.disp_head,
                   self.v41_guide_force_q]
        return [p for m in modules for p in m.parameters()]

    def eyy_parameters(self):
        if not self.enable_eyy:
            return []
        modules = [self.eyy_edge_enc, self.eyy_layers,
                   self.eyy_norm, self.eyy_head]
        return [p for m in modules for p in m.parameters()]

    def forward(self, x, ei, ea, cond, batch_idx, num_graphs, pooled_buffer=None,
                force_only=False, detach_force_trunk=False):
        if self.query_mode and force_only:
            force = self.force_from_cond(cond)
            return None, None, None, force, None, None, None, None

        cond_emb = self.encode_cond(cond, num_graphs)
        h_n = self.trunk(x, ei, ea, cond_emb, batch_idx)
        pooled = scatter_mean(h_n, batch_idx, num_graphs)
        graph_feat = torch.cat([pooled, cond_emb], -1)
        self._last_v41_guide_force = self.guide_force_from_features(
            pooled, cond_emb, cond)

        if self.query_mode:
            force = self.force_from_cond(cond)
        elif pooled_buffer:
            force = self.tf_force(torch.stack(pooled_buffer + [pooled], 0))
        else:
            force = self.force_first(pooled).squeeze(-1)

        stress = self.stress_head(h_n).squeeze(-1)
        disp = self.disp_head(h_n)
        eyy = self.eyy_branch(h_n, ei, ea, cond_emb, batch_idx)
        dead_logit = self.dead_head(h_n).squeeze(-1)
        damage_logit = self.damage_head(h_n).squeeze(-1)
        fracture_logit = self.fracture_head(graph_feat).squeeze(-1)
        return stress, eyy, disp, force, pooled, dead_logit, damage_logit, fracture_logit

def build_batch(job_list, t, sigma_override, shared, device, cfg):
    B = len(job_list)
    xs, conds, ys, yes, yds, yfs, yas, ygs, ws = [], [], [], [], [], [], [], [], []
    zero_prev = cfg['QUERY_MODE']
    for jid, job in enumerate(job_list):
        fr = job[t]
        if zero_prev:
            sp = shared.zero_prev
        else:
            sp = sigma_override.get(jid)
            sp = fr['sigma_prev'] if sp is None else sp
        xs.append(shared.build_x(sp))
        conds.append(fr['cond'])
        ys.append(fr['y_stress'])
        yes.append(fr['y_eyy'])
        yds.append(fr['y_disp'])
        yfs.append(fr['y_force'])
        yas.append(fr['alive'])
        ygs.append(fr['damage'])
        ws.append(fr['post_frac'])
    ei, ea, bi = shared.batched_topology(B, device)
    return (torch.cat(xs, 0).to(device, non_blocking=True), ei, ea,
            torch.stack(conds, 0).to(device), bi,
            torch.cat(ys, 0).to(device),
            torch.cat(yes, 0).to(device),
            torch.cat(yds, 0).to(device),
            torch.cat(yfs, 0).view(-1).to(device),
            torch.cat(yas, 0).to(device),
            torch.cat(ygs, 0).to(device), ws)

def _stratified_force_indices(job, k):

    """
    n = int(job.curve_disp.shape[0])
    if n <= k:
        return torch.arange(n, dtype=torch.long)

    f = np.asarray(job.curve_force, dtype=np.float64)
    peak_i = int(np.argmax(f))
    peak = max(float(f[peak_i]), 1e-8)
    post = np.where((np.arange(n) > peak_i) & (f <= 0.02 * peak))[0]
    frac_i = int(post[0]) if len(post) else n - 1
    pre_i = max(peak_i + 1, frac_i - max(3, n // 80))

    spans = [
        (0, max(1, peak_i)),
        (peak_i, max(peak_i + 1, pre_i)),
        (max(peak_i, pre_i - max(2, n // 100)), max(pre_i + 1, frac_i)),
        (max(peak_i, frac_i - max(3, n // 100)), min(n - 1, frac_i + max(2, n // 100))),
        (frac_i, n - 1),
    ]
    picks = []
    for lo, hi in spans:
        lo, hi = int(max(0, lo)), int(min(n - 1, hi))
        picks.append(lo if hi <= lo else random.randint(lo, hi))

    while len(picks) < k:
        lo = max(peak_i, pre_i - max(5, n // 50))
        hi = min(n - 1, frac_i + max(5, n // 50))
        picks.append(random.randint(lo, max(lo, hi)))

    picks = sorted(set(picks))
    while len(picks) < k:
        candidate = random.randrange(n)
        if candidate not in picks:
            picks.append(candidate)
            picks.sort()
    if len(picks) > k:
        pos = np.linspace(0, len(picks) - 1, k).round().astype(int)
        picks = [picks[i] for i in pos]
    return torch.tensor(picks, dtype=torch.long)

def build_dense_force_batch(job_list, k, shared, device, max_dense_graphs=None,
                            stratified=True, post_force_frac=0.02):

    """
    valid_jobs = [j for j in job_list if j.curve_cond is not None]
    if not valid_jobs:
        return None
    k = max(3, int(k))
    max_graphs = max(k, int(max_dense_graphs or k))
    n_jobs_use = max(1, min(len(valid_jobs), max_graphs // k))
    jobs_use = random.sample(valid_jobs, n_jobs_use) if len(valid_jobs) > n_jobs_use else valid_jobs

    conds, yfs, disps, groups, post_masks = [], [], [], [], []
    for gid, job in enumerate(jobs_use):
        if stratified:
            idx = _stratified_force_indices(job, k)
        else:
            idx = torch.randperm(job.curve_cond.shape[0])[:k].sort().values
        conds.append(job.curve_cond[idx])
        y = job.curve_force_n[idx]
        yfs.append(y)
        d = torch.tensor(np.asarray(job.curve_disp)[idx.numpy()], dtype=torch.float32)
        disps.append(d)
        groups.append(torch.full((len(idx),), gid, dtype=torch.long))
        peak = float(torch.max(job.curve_force_n).item())
        peak_idx = int(torch.argmax(job.curve_force_n).item())
        idx_np = idx.numpy()
        pm = (idx_np > peak_idx) & (y.numpy() <= post_force_frac * max(peak, 1e-8))
        post_masks.append(torch.tensor(pm, dtype=torch.bool))

    B = sum(len(x) for x in conds)
    ei, ea, bi = shared.batched_topology(B, device)
    x = shared.build_x(shared.zero_prev).repeat(B, 1).to(device, non_blocking=True)
    return (x, ei, ea, torch.cat(conds, 0).to(device), bi,
            torch.cat(yfs, 0).view(-1).to(device), B,
            torch.cat(disps, 0).to(device),
            torch.cat(groups, 0).to(device),
            torch.cat(post_masks, 0).to(device))

def dense_force_shape_losses(pred, target, disp, group_id, post_mask, cfg):
    beta = float(cfg.get('FORCE_LOSS_HUBER_BETA', 0.05))
    value_loss = nn.functional.smooth_l1_loss(pred, target, beta=beta)
    slope_losses, curve_losses = [], []
    for gid in torch.unique(group_id):
        m = group_id == gid
        p, y, d = pred[m], target[m], disp[m]
        order = torch.argsort(d)
        p, y, d = p[order], y[order], d[order]
        if len(p) >= 2:
            dd = (d[1:] - d[:-1]).clamp(min=1e-4)
            ps = (p[1:] - p[:-1]) / dd
            ys = (y[1:] - y[:-1]) / dd
            slope_losses.append(nn.functional.smooth_l1_loss(ps, ys, beta=beta))
        if len(p) >= 3:
            dd1 = (d[1:] - d[:-1]).clamp(min=1e-4)
            ps = (p[1:] - p[:-1]) / dd1
            ys = (y[1:] - y[:-1]) / dd1
            mid_d = 0.5 * (d[1:] + d[:-1])
            dmid = (mid_d[1:] - mid_d[:-1]).clamp(min=1e-4)
            pc = (ps[1:] - ps[:-1]) / dmid
            yc = (ys[1:] - ys[:-1]) / dmid
            curve_losses.append(nn.functional.smooth_l1_loss(pc, yc, beta=beta))
    zero = pred.new_zeros(())
    slope_loss = torch.stack(slope_losses).mean() if slope_losses else zero
    curve_loss = torch.stack(curve_losses).mean() if curve_losses else zero
    post_loss = (pred[post_mask] ** 2).mean() if post_mask.any() else zero
    return value_loss, slope_loss, curve_loss, post_loss

def expand_node_mask_by_edges(seed_mask, edge_index, rings=1):
    mask = seed_mask.bool().clone()
    if rings <= 0 or not mask.any():
        return mask
    r, c = edge_index
    for _ in range(int(rings)):
        grown = mask.clone()
        grown[c[mask[r]]] = True
        grown[r[mask[c]]] = True
        if torch.equal(grown, mask):
            break
        mask = grown
    return mask

def run_epoch(model, jobs, cfg, shared, optimizer=None, sched_p=0.0, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    device = cfg['DEVICE']
    if not jobs:
        if train and cfg.get('FAIL_ON_EMPTY_TRAIN', True):
        return {'loss': float('nan'), 'steps': 0, 'grad_norm': 0.0}
    if train:
        random.shuffle(jobs)
    bs = min(cfg['BATCH_JOBS'], len(jobs))
    accum = max(1, cfg['GRAD_ACCUM'] if train else 1)
    T = cfg['FRAMES_PER_JOB']
    total, nb, n_steps = 0.0, 0, 0
    grad_norm_last = 0.0

    pbar = tqdm(range(0, len(jobs), bs), total=(len(jobs) + bs - 1) // bs,
                desc='Train' if train else 'Eval', leave=False)
    for start_i in pbar:
        batch_jobs = jobs[start_i:start_i + bs]
        B = len(batch_jobs)
        sigma_state = {j: None for j in range(B)}
        pooled_buffer = []
        seq_loss = 0.0
        if train:
            optimizer.zero_grad(set_to_none=True)

        for t in range(T):
            override = {}
            for jid in range(B):
                use_model = (not cfg['QUERY_MODE']) and train and t > 0 and random.random() < sched_p
                override[jid] = sigma_state[jid] if (use_model and sigma_state[jid] is not None) else None

            x, ei, ea, cond, bi, y_s, y_e, y_d, y_f, y_a, y_g, pf = build_batch(
                batch_jobs, t, override, shared, device, cfg)

            with torch.set_grad_enabled(train):
                with amp_ctx(cfg['USE_AMP'] and device == 'cuda'):
                    s_pred, e_pred, d_pred, f_pred, pooled, dead_logit, damage_logit, frac_logit = model(
                        x, ei, ea, cond, bi, B, pooled_buffer=pooled_buffer)
                    if t == 0:
                        d_pred = torch.zeros_like(d_pred)

                    n = shared.n_nodes
                    r, c = ei
                    if cfg.get('INTACT_FIELD_V41_PROFILE', True):
                        crack_neigh = torch.zeros_like(y_a, dtype=torch.bool)
                        frame_w = 1.0
                    else:
                        crack_seed = ((y_a < 0.5) | (y_g >= 0.50))
                        crack_neigh = expand_node_mask_by_edges(
                            crack_seed, ei,
                            cfg.get('FIELD_CRACK_NEIGHBOR_RINGS', 2))
                        frame_w = 1.0
                        if t >= int(cfg.get('FIELD_LATE_FRAME_START', 21)):
                            frame_w = float(cfg.get('FIELD_LATE_FRAME_WEIGHT', 1.0))
                        if t == T - 1:
                            frame_w = float(cfg.get('FIELD_FINAL_FRAME_WEIGHT', frame_w))

                    w = torch.tensor(
                        [cfg['POST_FRACTURE_STRESS_W'] if p > 0.5 else 1.0 for p in pf],
                        device=device, dtype=s_pred.dtype
                    ).repeat_interleave(n)
                    w = w * frame_w
                    w = w * (1.0 + cfg['W_DAMAGE_REGION'] * y_g.clamp(0.0, 1.0))
                    w = w * (1.0 + cfg['W_HIGH_STRESS_REGION'] *
                             y_s.detach().abs().clamp(0.0, 3.0) / 3.0)
                    w = w * (1.0 +
                             cfg.get('FIELD_CRACK_STRESS_WEIGHT', 0.0) *
                             crack_neigh.float())
                    if t == 0:
                        w = w * cfg['T0_STRESS_WEIGHT']
                    loss_s = (w * (s_pred - y_s) ** 2).sum() / w.sum().clamp(min=1.0)

                    if cfg.get('ENABLE_EYY', False):
                        e_mask = (y_a > 0.5).float()
                        ew = e_mask * frame_w
                        if t == 0:
                            ew = ew * 5.0
                        loss_e = ((ew * (e_pred - y_e) ** 2).sum() /
                                  ew.sum().clamp(min=1.0))
                    else:
                        loss_e = torch.zeros((), device=device)
                    loss_u = ((d_pred - y_d) ** 2).mean()
                    loss_f = ((f_pred - y_f) ** 2).mean()
                    loss_guide_f = ((model._last_v41_guide_force - y_f) ** 2).mean()
                    loss_var0 = s_pred.var() if t == 0 else torch.zeros((), device=device)

                    if cfg.get('INTACT_FIELD_V41_PROFILE', True):
                        em = torch.ones_like(r, dtype=torch.bool)
                    else:
                        em = (y_a[r] > 0.5) & (y_a[c] > 0.5)
                    if em.any():
                        loss_smooth = (((d_pred[r[em]] - d_pred[c[em]]) -
                                       (y_d[r[em]] - y_d[c[em]])) ** 2).mean()
                        loss_ss = (((s_pred[r[em]] - s_pred[c[em]]) -
                                   (y_s[r[em]] - y_s[c[em]])) ** 2).mean()
                    else:
                        loss_smooth = torch.zeros((), device=device)
                        loss_ss = torch.zeros((), device=device)

                    gt_grad = (y_s[r] - y_s[c]).abs().detach()
                    valid_grad = em & torch.isfinite(gt_grad)
                    if valid_grad.any():
                        qg = torch.quantile(
                            gt_grad[valid_grad],
                            float(cfg.get('FIELD_GT_GRAD_QUANTILE', 0.88)))
                        detail_edge = valid_grad & (
                            crack_neigh[r] | crack_neigh[c] |
                            (gt_grad >= qg))
                    else:
                        detail_edge = valid_grad
                    if detail_edge.any():
                        loss_local_grad = nn.functional.smooth_l1_loss(
                            s_pred[r[detail_edge]] - s_pred[c[detail_edge]],
                            y_s[r[detail_edge]] - y_s[c[detail_edge]])
                    else:
                        loss_local_grad = torch.zeros((), device=device)

                    dead_target = 1.0 - y_a
                    pos = dead_target.sum().detach()
                    neg = dead_target.numel() - pos
                    pw = (neg / (pos + 1.0)).clamp(cfg['DEAD_POS_WEIGHT_MIN'],
                                                    cfg['DEAD_POS_WEIGHT_MAX'])
                    bce = nn.functional.binary_cross_entropy_with_logits(
                        dead_logit, dead_target, pos_weight=pw, reduction='none')
                    prob = torch.sigmoid(dead_logit)
                    pt = prob * dead_target + (1.0 - prob) * (1.0 - dead_target)
                    loss_alive = (((1.0 - pt) ** cfg['DEAD_FOCAL_GAMMA']) * bce).mean()

                    eps_tv = torch.tensor(1e-6, device=device, dtype=prob.dtype)
                    tp_soft = (prob * dead_target).sum()
                    fp_soft = (prob * (1.0 - dead_target)).sum()
                    fn_soft = ((1.0 - prob) * dead_target).sum()
                    tv = ((tp_soft + eps_tv) /
                          (tp_soft + cfg['DEAD_TVERSKY_ALPHA'] * fp_soft +
                           cfg['DEAD_TVERSKY_BETA'] * fn_soft + eps_tv))
                    loss_dead_tversky = 1.0 - tv

                    pred_edge = prob[r] - prob[c]
                    true_edge = dead_target[r] - dead_target[c]
                    loss_crack_edge = nn.functional.smooth_l1_loss(pred_edge, true_edge)

                    damage_pred = torch.sigmoid(damage_logit)
                    dw = 1.0 + 6.0 * y_g.clamp(0.0, 1.0)
                    loss_damage_reg = (dw * nn.functional.smooth_l1_loss(
                        damage_pred, y_g.clamp(0.0, 1.0), reduction='none')).mean()
                    dmg_target = (y_g >= cfg['DAMAGE_CRACK_THRESHOLD']).float()
                    dpos = dmg_target.sum().detach()
                    dneg = dmg_target.numel() - dpos
                    dpw = (dneg / (dpos + 1.0)).clamp(2.0, 60.0)
                    loss_damage_cls = nn.functional.binary_cross_entropy_with_logits(
                        damage_logit, dmg_target, pos_weight=dpw)
                    loss_damage = loss_damage_reg + 0.5 * loss_damage_cls

                    frac_target = dead_target.view(B, n).amax(dim=1)
                    loss_frac = nn.functional.binary_cross_entropy_with_logits(frac_logit, frac_target)
                    local_score = torch.maximum(prob, damage_pred).view(B, n)
                    ktop = max(1, n // 100)
                    local_score = torch.topk(local_score, ktop, dim=1).values.mean(dim=1)
                    loss_cons = ((local_score - torch.sigmoid(frac_logit)) ** 2).mean()

                    loss = (cfg['W_STRESS'] * loss_s + cfg['W_EYY'] * loss_e +
                            cfg['W_DISP'] * loss_u + cfg['W_FORCE'] * loss_f +
                            cfg.get('W_V41_GUIDE_FORCE', 0.50) * loss_guide_f +
                            cfg['W_SMOOTH'] * loss_smooth +
                            cfg['W_STRESS_SMOOTH'] * loss_ss +
                            cfg.get('FIELD_LOCAL_GRAD_WEIGHT', 0.0) * loss_local_grad +
                            cfg['W_VAR0'] * loss_var0 +
                            cfg['W_ALIVE'] * loss_alive +
                            cfg['W_DEAD_TVERSKY'] * loss_dead_tversky +
                            cfg['W_CRACK_EDGE'] * loss_crack_edge +
                            cfg['W_DAMAGE'] * loss_damage +
                            cfg['W_FRACTURE_STATE'] * loss_frac +
                            cfg['W_CRACK_CONSISTENCY'] * loss_cons)

                    if (cfg['QUERY_MODE'] and cfg['N_DENSE_DISP'] > 0 and
                            (not cfg.get('DENSE_FORCE_ONLY_LAST_FRAME', True) or t == T - 1)):
                        db = build_dense_force_batch(
                            batch_jobs, cfg['N_DENSE_DISP'], shared, device,
                            cfg.get('MAX_DENSE_GRAPHS', 5),
                            stratified=cfg.get('DENSE_STRATIFIED_SAMPLING', True),
                            post_force_frac=cfg.get('POST_FORCE_FRAC', 0.02))
                        if db is not None:
                            dx, dei, dea, dcond, dbi, dyf, dB, ddisp, dgroup, dpost = db
                            _, _, _, df_pred, _, _, _, _ = model(
                                dx, dei, dea, dcond, dbi, dB, force_only=True,
                                detach_force_trunk=cfg.get('DENSE_FORCE_DETACH_TRUNK', True))
                            lv, lslope, lcurve, lpost = dense_force_shape_losses(
                                df_pred, dyf, ddisp, dgroup, dpost, cfg)
                            guide_df = model.guide_force_forward(
                                dx, dei, dea, dcond, dbi, dB)
                            loss_guide_dense = ((guide_df - dyf) ** 2).mean()
                            loss = (loss + cfg['W_DENSE_FORCE'] * lv
                                    + cfg.get('W_V41_GUIDE_DENSE_FORCE', 2.0) * loss_guide_dense
                                    + cfg.get('W_FORCE_SLOPE', 0.0) * lslope
                                    + cfg.get('W_FORCE_CURVATURE', 0.0) * lcurve
                                    + cfg.get('W_POST_FRACTURE_ZERO_FORCE', 0.0) * lpost)

                if not torch.isfinite(loss):
                if train:
                    ls = loss / accum
                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(ls).backward()
                    else:
                        ls.backward()
                    do_step = ((t + 1) % accum == 0) or (t == T - 1)
                    if do_step:
                        if scaler is not None and scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        gp = torch.nn.utils.clip_grad_norm_(
                            model.primary_field_parameters(), 1.0)
                        eyy_params = model.eyy_parameters()
                        if eyy_params:
                            torch.nn.utils.clip_grad_norm_(eyy_params, 1.0)
                        if model.query_mode:
                            torch.nn.utils.clip_grad_norm_(model.force_parameters(), 1.0)
                        grad_norm_last = float(gp.detach().cpu())
                        if scaler is not None and scaler.is_enabled():
                            scaler.step(optimizer); scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        n_steps += 1

            seq_loss += float(loss.detach().cpu())
            if not cfg['QUERY_MODE']:
                sp_det = s_pred.detach().float().cpu()
                for jid in range(B):
                    sigma_state[jid] = sp_det[jid * shared.n_nodes:(jid + 1) * shared.n_nodes]
                pooled_buffer.append(pooled.detach())
                if len(pooled_buffer) > T:
                    pooled_buffer.pop(0)

        total += seq_loss / T
        nb += 1
        pbar.set_postfix(loss='{:.3e}'.format(seq_loss / T))

    if train and n_steps == 0:
    return {'loss': total / max(nb, 1), 'steps': n_steps, 'grad_norm': grad_norm_last}

def alive_from_outputs(dead_logit, damage_logit, fracture_logit, load_disp, cfg):
    dead_prob = torch.sigmoid(dead_logit.float())
    damage_prob = torch.sigmoid(damage_logit.float())
    frac_prob = float(torch.sigmoid(fracture_logit.float()).reshape(-1)[0])
    if float(load_disp) < cfg['CRACK_ENABLE_DISP']:
        return torch.ones_like(dead_prob), dead_prob, damage_prob, frac_prob
    den = max(cfg['CRACK_FULL_DISP'] - cfg['CRACK_ENABLE_DISP'], 1e-6)
    progress = min(max((float(load_disp) - cfg['CRACK_ENABLE_DISP']) / den, 0.0), 1.0)
    dead_thr = cfg['DEAD_PROB_THRESHOLD'] - 0.08 * progress
    dmg_thr = cfg['DAMAGE_CRACK_THRESHOLD'] - 0.08 * progress
    local_dead = (dead_prob >= dead_thr) | (damage_prob >= dmg_thr)
    if frac_prob < cfg['FRACTURE_PROB_THRESHOLD']:
        local_dead = (dead_prob >= 0.90) | (damage_prob >= 0.95)
    return (~local_dead).float(), dead_prob, damage_prob, frac_prob

@torch.no_grad()
def rollout_job(model, job, cfg, shared, device):
    T = cfg['FRAMES_PER_JOB']
    sigma_state, pooled_buffer = None, []
    SP, ST, EP, ET, FP, FT, DP, DT, AP, GP, QP = [], [], [], [], [], [], [], [], [], [], []
    for t in range(T):
        x, ei, ea, cond, bi, y_s, y_e, y_d, y_f, y_a, y_g, pf = build_batch(
            [job], t, {0: sigma_state}, shared, device, cfg)
        s_pred, e_pred, d_pred, f_pred, pooled, dead_logit, damage_logit, frac_logit = model(
            x, ei, ea, cond, bi, 1, pooled_buffer=pooled_buffer)
        if t == 0:
            d_pred = torch.zeros_like(d_pred)
        if not cfg['QUERY_MODE']:
            sigma_state = s_pred.detach().float().cpu(); pooled_buffer.append(pooled.detach())
        alive, _dp, dmg, frac = alive_from_outputs(
            dead_logit.float().cpu(), damage_logit.float().cpu(), frac_logit.float().cpu(),
            job[t]['raw_load_disp'], cfg)
        SP.append(s_pred.float().cpu()); ST.append(y_s.float().cpu())
        EP.append(e_pred.float().cpu()); ET.append(y_e.float().cpu())
        FP.append(f_pred.float().cpu().view(-1)); FT.append(y_f.float().cpu().view(-1))
        DP.append(d_pred.float().cpu()); DT.append(y_d.float().cpu())
        AP.append(alive); GP.append(dmg); QP.append(frac)
    return SP, ST, EP, ET, FP, FT, DP, DT, AP, GP, QP

@torch.no_grad()
def query(model, shared, stats, cfg, disps, mat_params_raw):
    if not cfg['QUERY_MODE']:
        raise RuntimeError('query() requires QUERY_MODE=True')
    model.eval(); device = cfg['DEVICE']
    d = np.atleast_1d(np.asarray(disps, dtype=np.float64))
    mp = np.asarray(mat_params_raw, dtype=np.float64)
    mat_n = ((mp - stats['mat_mean']) / stats['mat_std']).astype(np.float32)
    cond_all = torch.tensor(make_cond(d, mat_n, stats, cfg))
    n = shared.n_nodes; chunk = max(1, int(cfg['QUERY_CHUNK']))
    F, S, E, A, U, G, Q = [], [], [], [], [], [], []
    for i in range(0, len(d), chunk):
        c = cond_all[i:i + chunk]; B = c.shape[0]
        ei, ea, bi = shared.batched_topology(B, device)
        x = shared.build_x(shared.zero_prev).repeat(B, 1).to(device)
        with amp_ctx(cfg['USE_AMP'] and device == 'cuda'):
            s, e, u, f, _p, dl, gl, ql = model(x, ei, ea, c.to(device), bi, B)
        F.append(f.float().cpu().view(-1)); S.append(s.float().cpu().view(B, n))
        E.append(e.float().cpu().view(B, n)); U.append(u.float().cpu().view(B, n, 3))
        dlc = dl.float().cpu().view(B, n); glc = gl.float().cpu().view(B, n)
        qlc = ql.float().cpu().view(B)
        ac, gc, qc = [], [], []
        for j in range(B):
            al, _dp, dmg, frac = alive_from_outputs(dlc[j], glc[j], qlc[j:j+1], d[i+j], cfg)
            ac.append(al); gc.append(dmg); qc.append(frac)
        A.append(torch.stack(ac)); G.append(torch.stack(gc)); Q.extend(qc)
    force = torch.cat(F).numpy() * stats['force_std'] + stats['force_mean']
    stress = torch.cat(S).numpy() * stats['stress_std'] + stats['stress_mean']
    eyy = torch.cat(E).numpy() * stats['eyy_std'] + stats['eyy_mean']
    u = torch.cat(U).numpy() * stats['disp_std'] + stats['disp_mean']
    alive = torch.cat(A).numpy(); damage = torch.cat(G).numpy()
    force[d <= 1e-9] = 0.0
    eyy_percent = eyy * 100.0 if cfg.get('DIC_OUTPUT_UNIT') == 'strain' else eyy
    return {'disp': d, 'force': force, 'stress': stress,
            'eyy_display': eyy, 'eyy_percent': eyy_percent,
            'alive': alive, 'damage': damage,
            'fracture_prob': np.asarray(Q), 'u': u}

@torch.no_grad()
def evaluate(model, jobs, cfg, stats, shared):
    """Validation/test metrics.

    """
    model.eval()
    device = cfg['DEVICE']
    sp, st, ep, et, fp, ft, dp, dt = [], [], [], [], [], [], [], []
    hot_abs = []
    hot_sq_sum = hot_n = 0.0
    hot_true_sq_sum = 0.0
    late_sq_sum = late_n = 0.0
    crack_sq_sum = crack_n = 0.0
    frame_r2 = []

    dead_inter = dead_union = dead_tp = dead_pred_n = dead_true_n = 0.0
    frac_ok = frac_n = 0

    T = cfg['FRAMES_PER_JOB']
    late_start = max(1, T - max(4, T // 5))

    for job in jobs:
        a, b, ce, de, e, f, g, h, ap, gp, qp = rollout_job(
            model, job, cfg, shared, device)

        for t, (ps, ts) in enumerate(zip(a[1:], b[1:]), start=1):
            p_phys = ps * stats['stress_std'] + stats['stress_mean']
            t_phys = ts * stats['stress_std'] + stats['stress_mean']
            err_phys = p_phys - t_phys

            q = torch.quantile(t_phys, 0.90)
            hot_mask = t_phys >= q
            if hot_mask.any():
                he = err_phys[hot_mask]
                hot_abs.append(he.abs())
                hot_sq_sum += float((he ** 2).sum())
                hot_true_sq_sum += float((t_phys[hot_mask] ** 2).sum())
                hot_n += float(hot_mask.sum())

            den = ((ts - ts.mean()) ** 2).sum()
            if den > 1e-8:
                frame_r2.append(1.0 - ((ts - ps) ** 2).sum() / den)

            if t >= late_start:
                late_sq_sum += float((err_phys ** 2).sum())
                late_n += float(err_phys.numel())

            crack_mask = ((job[t]['damage'] >= 0.50) |
                          (job[t]['alive'] < 0.50))
            if crack_mask.any():
                ce_phys = err_phys[crack_mask]
                crack_sq_sum += float((ce_phys ** 2).sum())
                crack_n += float(ce_phys.numel())

        sp += a[1:]
        st += b[1:]
        ep += ce[1:]
        et += de[1:]
        fp += e[1:]
        ft += f[1:]
        dp += g[1:]
        dt += h[1:]

        for t in range(1, T):
            pd = ap[t] < 0.5
            td = job[t]['alive'] < 0.5
            dead_inter += float((pd & td).sum())
            dead_union += float((pd | td).sum())
            dead_tp += float((pd & td).sum())
            dead_pred_n += float(pd.sum())
            dead_true_n += float(td.sum())
            gt_frac = bool(td.any())
            pr_frac = qp[t] >= cfg['FRACTURE_PROB_THRESHOLD']
            frac_ok += int(gt_frac == pr_frac)
            frac_n += 1

    sp, st = torch.cat(sp), torch.cat(st)
    ep, et = torch.cat(ep), torch.cat(et)
    fp, ft = torch.cat(fp), torch.cat(ft)
    dp, dt = torch.cat(dp), torch.cat(dt)

    rel_l2 = (torch.norm(sp - st) / (torch.norm(st) + 1e-8)).item()
    r2 = (1 - ((st - sp) ** 2).sum() /
          (((st - st.mean()) ** 2).sum() + 1e-8)).item()

    if cfg.get('ENABLE_EYY', False):
        eyy_rel = (torch.norm(ep - et) / (torch.norm(et) + 1e-8)).item()
        eyy_r2 = (1 - ((et - ep) ** 2).sum() /
                  (((et - et.mean()) ** 2).sum() + 1e-8)).item()
    else:
        eyy_rel = 0.0
        eyy_r2 = float('nan')

    force_rel = (torch.norm(fp - ft) / (torch.norm(ft) + 1e-8)).item()
    force_r2 = (1 - ((ft - fp) ** 2).sum() /
                (((ft - ft.mean()) ** 2).sum() + 1e-8)).item()

    stress_err = (sp - st) * stats['stress_std']
    eyy_err = ((ep - et) * stats['eyy_std']
               if cfg.get('ENABLE_EYY', False)
               else torch.zeros_like(ep))
    force_err = (fp - ft) * stats['force_std']
    disp_err = (dp - dt) * torch.tensor(stats['disp_std'])

    stress_mae = stress_err.abs().mean().item()
    stress_rmse = torch.sqrt((stress_err ** 2).mean()).item()
    if cfg.get('ENABLE_EYY', False):
        eyy_mae = eyy_err.abs().mean().item()
        eyy_rmse = torch.sqrt((eyy_err ** 2).mean()).item()
    else:
        eyy_mae = float('nan')
        eyy_rmse = float('nan')
    force_mae = force_err.abs().mean().item()
    force_rmse = torch.sqrt((force_err ** 2).mean()).item()
    disp_rmse = torch.sqrt((disp_err ** 2).mean()).item()

    hot_mae = torch.cat(hot_abs).mean().item() if hot_abs else float('nan')
    hot_rmse = math.sqrt(hot_sq_sum / max(hot_n, 1.0))
    hot_rel = math.sqrt(hot_sq_sum / max(hot_true_sq_sum, 1e-8))
    late_rmse = math.sqrt(late_sq_sum / max(late_n, 1.0))
    crack_rmse = math.sqrt(crack_sq_sum / max(crack_n, 1.0)) if crack_n > 0 else float('nan')

    dead_iou = dead_inter / max(dead_union, 1.0)
    dead_precision = dead_tp / max(dead_pred_n, 1.0)
    dead_recall = dead_tp / max(dead_true_n, 1.0)
    dead_f1 = (2.0 * dead_precision * dead_recall /
               max(dead_precision + dead_recall, 1e-12))

    stress_scale = max(float(stats['stress_std']), 1e-8)
    crack_norm = (crack_rmse / stress_scale
                  if np.isfinite(crack_rmse) else late_rmse / stress_scale)
    late_norm = late_rmse / stress_scale

    crack_select = 0.0
    if cfg.get('FIELD_SELECTION_USE_CRACK_METRICS', False):
        crack_select = (
            cfg.get('FIELD_SELECTION_CRACK_WEIGHT', 0.20) * crack_norm
            + 0.15 * (1.0 - dead_iou)
        )
    selection_score = (
        rel_l2
        + (cfg.get('FIELD_SELECTION_EYY_WEIGHT', 0.0) * eyy_rel
           if cfg.get('ENABLE_EYY', False) else 0.0)
        + cfg.get('FIELD_SELECTION_HOT_WEIGHT', 0.35) * hot_rel
        + cfg.get('FIELD_SELECTION_LATE_WEIGHT', 0.15) * late_norm
        + crack_select
    )

    eyy_scale_to_percent = 100.0 if cfg.get('DIC_OUTPUT_UNIT') == 'strain' else 1.0

    return {
        'rel_l2': rel_l2,
        'r2': r2,
        'frame_r2': float(torch.stack(frame_r2).mean()) if frame_r2 else float('nan'),
        'hotspot_rel_l2': hot_rel,
        'hotspot_mae': hot_mae,
        'hotspot_rmse': hot_rmse,
        'late_stress_rmse': late_rmse,
        'crack_zone_rmse': crack_rmse,

        'eyy_rel_l2': eyy_rel,
        'eyy_r2': eyy_r2,
        'eyy_mae': eyy_mae,
        'eyy_rmse': eyy_rmse,
        'eyy_mae_percent': eyy_mae * eyy_scale_to_percent,
        'eyy_rmse_percent': eyy_rmse * eyy_scale_to_percent,

        'force_rel_l2': force_rel,
        'force_r2': force_r2,
        'force_mae': force_mae,
        'force_rmse': force_rmse,

        'selection_score': selection_score,
        'mises_mae': stress_mae,
        'stress_rmse': stress_rmse,
        'disp_rel_l2': (torch.norm(dp - dt) /
                        (torch.norm(dt) + 1e-8)).item(),
        'disp_rmse': disp_rmse,

        'dead_iou': dead_iou,
        'dead_precision': dead_precision,
        'dead_recall': dead_recall,
        'dead_f1': dead_f1,
        'fracture_acc': frac_ok / max(frac_n, 1),
        'dead_true_count': dead_true_n,
        'dead_pred_count': dead_pred_n,
    }

def _crack_threshold_path(cfg):
    return os.path.join(cfg['CKPT_DIR'], cfg.get('CRACK_THRESHOLD_FILE',
                                                 'crack_thresholds.json'))

def load_calibrated_crack_thresholds(cfg):
    p = _crack_threshold_path(cfg)
    if not os.path.exists(p):
        return False
    try:
        d = json.load(open(p, 'r'))
        for k in ('DEAD_PROB_THRESHOLD', 'DAMAGE_CRACK_THRESHOLD',
                  'FRACTURE_PROB_THRESHOLD'):
            if k in d:
                cfg[k] = float(d[k])
        print('  [CrackThreshold] loaded dead={:.3f}, damage={:.3f}, fracture={:.3f}'
              .format(cfg['DEAD_PROB_THRESHOLD'], cfg['DAMAGE_CRACK_THRESHOLD'],
                      cfg['FRACTURE_PROB_THRESHOLD']))
        return True
    except Exception as e:
        print('  [CrackThreshold][WARNING] load failed: {}'.format(e))
        return False

@torch.no_grad()
def calibrate_crack_thresholds(model, jobs, cfg, shared, save=True):

    """
    if not jobs:
        return None
    model.eval()
    device = cfg['DEVICE']
    records = []
    true_total = 0
    for job in tqdm(jobs, desc='[CrackCalibrate]', leave=False):
        for t in range(cfg['FRAMES_PER_JOB']):
            x, ei, ea, cond, bi, _ys, _ye, _yd, _yf, y_a, _yg, _pf = build_batch(
                [job], t, {0: None}, shared, device, cfg)
            _s, _e, _u, _f, _p, dl, gl, ql = model(x, ei, ea, cond, bi, 1)
            dead_p = torch.sigmoid(dl.float()).cpu().numpy().reshape(-1)
            dmg_p = torch.sigmoid(gl.float()).cpu().numpy().reshape(-1)
            frac_p = float(torch.sigmoid(ql.float()).cpu().reshape(-1)[0])
            target = (y_a.float().cpu().numpy().reshape(-1) < 0.5)
            disp = float(job[t]['raw_load_disp'])
            records.append((dead_p, dmg_p, frac_p, target, disp))
            true_total += int(target.sum())

    if true_total == 0:
        return None

    dead_grid = np.linspace(0.48, 0.90, 15)
    dmg_grid = np.linspace(0.70, 0.97, 10)
    min_recall = float(cfg.get('CRACK_CALIBRATE_MIN_RECALL', 0.90))
    candidates = []
    for dthr in dead_grid:
        for gthr in dmg_grid:
            tp = fp = fn = 0
            for dead_p, dmg_p, frac_p, target, disp in records:
                if disp < cfg['CRACK_ENABLE_DISP']:
                    pred = np.zeros_like(target, dtype=bool)
                else:
                    den = max(cfg['CRACK_FULL_DISP'] - cfg['CRACK_ENABLE_DISP'], 1e-6)
                    progress = min(max((disp - cfg['CRACK_ENABLE_DISP']) / den, 0.0), 1.0)
                    pred = ((dead_p >= dthr - 0.08 * progress) |
                            (dmg_p >= gthr - 0.08 * progress))
                    if frac_p < cfg['FRACTURE_PROB_THRESHOLD']:
                        pred = (dead_p >= 0.90) | (dmg_p >= 0.95)
                tp += int(np.logical_and(pred, target).sum())
                fp += int(np.logical_and(pred, ~target).sum())
                fn += int(np.logical_and(~pred, target).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            iou = tp / max(tp + fp + fn, 1)
            feasible = recall >= min_recall
            score = iou + 0.05 * precision - (0.5 * max(min_recall - recall, 0.0))
            candidates.append((feasible, score, iou, precision, recall, dthr, gthr))

    feasible = [x for x in candidates if x[0]]
    best = max(feasible if feasible else candidates,
               key=lambda x: (x[1], x[2], x[3]))
    _, _score, iou, precision, recall, dthr, gthr = best
    cfg['DEAD_PROB_THRESHOLD'] = float(dthr)
    cfg['DAMAGE_CRACK_THRESHOLD'] = float(gthr)
    result = {
        'DEAD_PROB_THRESHOLD': float(dthr),
        'DAMAGE_CRACK_THRESHOLD': float(gthr),
        'FRACTURE_PROB_THRESHOLD': float(cfg['FRACTURE_PROB_THRESHOLD']),
        'val_iou': float(iou), 'val_precision': float(precision),
        'val_recall': float(recall), 'true_dead_nodes': int(true_total),
    }
    if save:
        with open(_crack_threshold_path(cfg), 'w') as f:
            json.dump(result, f, indent=2)
    print('  [CrackCalibrate] dead={:.3f} damage={:.3f} | IoU={:.3f} '
          'precision={:.3f} recall={:.3f}'.format(
              dthr, gthr, iou, precision, recall))
    return result

def get_front_surface(shared, cfg):
    coords = shared.raw_coords
    tris = shared.surface_tris
    ax = cfg['SURFACE_AXIS']
    vals = coords[:, ax]
    v0 = vals.min() if cfg['SURFACE_SIDE'] == 'min' else vals.max()
    h = get_axis_step_np(coords, ax)
    tol = max(0.75 * h, 1e-8)
    tv = vals[tris]
    keep = np.all(tv <= v0 + tol, 1) if cfg['SURFACE_SIDE'] == 'min' \
        else np.all(tv >= v0 - tol, 1)

    if cfg['VIZ_CROP'] is not None:
        lo, hi = cfg['VIZ_CROP']
        yv = coords[:, 1][tris]
        keep = keep & np.all(yv >= lo, 1) & np.all(yv <= hi, 1)

    front_idx = np.where(keep)[0]
    ftris = tris[keep]
    ids = np.unique(ftris.reshape(-1))
    remap = {int(g): i for i, g in enumerate(ids)}
    local = np.array([[remap[int(a)], remap[int(b)], remap[int(c)]]
                      for a, b, c in ftris], dtype=np.int64)
    print('  [FrontSurface] nodes {} | triangles {}'.format(len(ids), len(local)))
    return ids, local, front_idx

@torch.no_grad()
def _collect_viz_cache(model, job, cfg, stats, shared, show):
    device = cfg['DEVICE']; cache = {}; fp_list=[]; ft_list=[]; ld_list=[]
    sigma_state=None; pooled_buffer=[]
    for t in range(cfg['FRAMES_PER_JOB']):
        x,ei,ea,cond,bi,y_s,y_e,y_d,y_f,y_a,y_g,pf = build_batch(
            [job], t, {0:sigma_state}, shared, device, cfg)
        s,e,u,f,pooled,dl,gl,ql = model(x,ei,ea,cond,bi,1,pooled_buffer=pooled_buffer)
        if t == 0: u = torch.zeros_like(u)
        if not cfg['QUERY_MODE']:
            sigma_state=s.detach().float().cpu(); pooled_buffer.append(pooled.detach())
        fp_list.append(float(f.item())*stats['force_std']+stats['force_mean'])
        ft_list.append(job[t]['raw_force']); ld_list.append(job[t]['raw_load_disp'])
        if t in show:
            al,_dp,dmg,frac = alive_from_outputs(dl.float().cpu(), gl.float().cpu(),
                                                  ql.float().cpu(), job[t]['raw_load_disp'], cfg)
            cache[t] = {
                'stress_p': s.detach().float().cpu().numpy()*stats['stress_std']+stats['stress_mean'],
                'stress_g': y_s.cpu().numpy()*stats['stress_std']+stats['stress_mean'],
                'eyy_p': e.detach().float().cpu().numpy()*stats['eyy_std']+stats['eyy_mean'],
                'eyy_g': y_e.cpu().numpy()*stats['eyy_std']+stats['eyy_mean'],
                'u_p': u.detach().float().cpu().numpy()*stats['disp_std']+stats['disp_mean'],
                'u_g': y_d.cpu().numpy()*stats['disp_std']+stats['disp_mean'],
                'alive_g': y_a.cpu().numpy(), 'alive_p': al.numpy(),
                'damage_p': dmg.numpy(), 'frac_p': frac,
                'tri_alive': job[t].get('tri_alive')}
    return cache, fp_list, ft_list, ld_list

def _triangle_components(mask, triangles):
    valid = np.where(~np.asarray(mask, dtype=bool))[0]
    if len(valid) == 0:
        return []
    node_to_tris = {}
    for ti in valid:
        for node in triangles[ti]:
            node_to_tris.setdefault(int(node), []).append(int(ti))
    valid_set = set(int(x) for x in valid)
    seen = set()
    comps = []
    for seed in valid:
        seed = int(seed)
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for node in triangles[cur]:
                for nb in node_to_tris.get(int(node), []):
                    if nb in valid_set and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps

def _dominant_crack_band_filter(mask, triangles, vertical_coord, cfg):

    """
    mask = np.asarray(mask, dtype=bool).copy()
    if not cfg.get('CRACK_BAND_FILTER', True):
        return mask
    idx = np.where(mask)[0]
    if len(idx) < int(cfg.get('CRACK_BAND_MIN_TRIANGLES', 6)):
        return mask
    tri_y = np.asarray(vertical_coord, dtype=np.float64)[triangles].mean(axis=1)
    ys = tri_y[idx]
    if not np.all(np.isfinite(ys)) or np.ptp(ys) < 1e-8:
        return mask

    half = float(cfg.get('CRACK_BAND_HALF_WIDTH', 1.25))
    best_center = float(np.median(ys))
    best_count = -1
    for c in ys:
        count = int(np.sum(np.abs(ys - c) <= half))
        if count > best_count:
            best_count = count
            best_center = float(c)
    keep_band = np.abs(tri_y - best_center) <= half
    return mask & keep_band

def _build_triangle_adjacency(triangles):
    edge_to_tris = {}
    for ti, tri in enumerate(np.asarray(triangles, dtype=np.int64)):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for e in ((a,b), (b,c), (c,a)):
            e = tuple(sorted(e))
            edge_to_tris.setdefault(e, []).append(int(ti))
    adj = [[] for _ in range(len(triangles))]
    for tri_ids in edge_to_tris.values():
        if len(tri_ids) < 2:
            continue
        for i in tri_ids:
            for j in tri_ids:
                if i != j:
                    adj[i].append(j)
    adj = [np.asarray(sorted(set(x)), dtype=np.int64) for x in adj]
    return adj

def _adaptive_damage_tighten(mask, triangles, damage, ids, cfg):
    candidate = np.asarray(mask, dtype=bool).copy()
    if damage is None or not candidate.any():
        return candidate
    tri_damage = np.asarray(damage[ids][triangles], dtype=np.float64).mean(axis=1)
    vals = tri_damage[candidate]
    if len(vals) == 0:
        return candidate
    q = float(cfg.get('DISPLAY_DAMAGE_PERCENTILE', 0.0))
    base_thr = float(cfg.get('DISPLAY_DAMAGE_THRESHOLD', 0.86))
    if q > 0.0 and len(vals) >= 4:
        qthr = float(np.quantile(vals, q))
        thr = max(base_thr, min(qthr, 0.985))
    else:
        thr = base_thr
    return candidate & (tri_damage >= thr)

def _shave_mask_edges(mask, triangles, passes=1, min_neighbors=2):
    out = np.asarray(mask, dtype=bool).copy()
    if not out.any() or passes <= 0:
        return out
    adj = _build_triangle_adjacency(triangles)
    for _ in range(int(passes)):
        idx = np.where(out)[0]
        if len(idx) == 0:
            break
        remove = []
        for ti in idx:
            nb = adj[int(ti)]
            cnt = int(np.sum(out[nb])) if len(nb) else 0
            if cnt < int(min_neighbors):
                remove.append(int(ti))
        if not remove:
            break
        if len(remove) > 0.4 * len(idx):
            break
        out[np.asarray(remove, dtype=np.int64)] = False
    return out

def _masked_triangle_components(mask, triangles):
    valid = np.where(np.asarray(mask, dtype=bool))[0]
    if len(valid) == 0:
        return []
    node_to_tris = {}
    for ti in valid:
        for node in triangles[ti]:
            node_to_tris.setdefault(int(node), []).append(int(ti))
    valid_set = set(int(x) for x in valid)
    seen = set(); comps = []
    for seed in valid:
        seed = int(seed)
        if seed in seen:
            continue
        stack = [seed]; seen.add(seed); comp = []
        while stack:
            cur = stack.pop(); comp.append(cur)
            for node in triangles[cur]:
                for nb in node_to_tris.get(int(node), []):
                    if nb in valid_set and nb not in seen:
                        seen.add(nb); stack.append(nb)
        comps.append(np.asarray(comp, dtype=np.int64))
    return comps

def _cleanup_predicted_triangle_mask(mask, triangles, vertical_coord,
                                     frac_prob, load_disp, cfg,
                                     damage=None, ids=None):

    Approach:

    """
    candidate = np.asarray(mask, dtype=bool).copy()
    if not cfg.get('PRED_MASK_CLEANUP', True):
        return candidate
    if float(load_disp or 0.0) < float(cfg['CRACK_ENABLE_DISP']):
        return np.zeros_like(candidate)
    if frac_prob is None or float(frac_prob) < float(cfg['FRACTURE_PROB_THRESHOLD']):
        return np.zeros_like(candidate)

    candidate = _dominant_crack_band_filter(candidate, triangles, vertical_coord, cfg)
    if not candidate.any():
        return candidate

    if damage is not None and ids is not None:
        tightened = _adaptive_damage_tighten(candidate, triangles, damage, ids, cfg)
        if tightened.any():
            candidate = tightened

    comps = _masked_triangle_components(candidate, triangles)
    min_comp = int(cfg.get('SMOOTH_CRACK_MIN_COMPONENT_TRIANGLES', 6))
    if len(comps) > 1:
        comps = [c for c in comps if len(c) >= min_comp] or comps
        sizes = [len(c) for c in comps]
        keep = int(np.argmax(sizes))
        out = np.zeros_like(candidate)
        out[comps[keep]] = True
        candidate = out

    candidate = _shave_mask_edges(
        candidate, triangles,
        passes=int(cfg.get('CRACK_EDGE_SHAVE_PASSES', 1)),
        min_neighbors=int(cfg.get('CRACK_EDGE_SHAVE_MIN_NEIGHBORS', 2))
    )

    min_keep = max(1, int(cfg.get('CRACK_REMOVE_SMALL_MASK_ISLANDS', 2)))
    comps = _masked_triangle_components(candidate, triangles)
    if len(comps) > 1:
        out = np.zeros_like(candidate)
        for comp in comps:
            if len(comp) >= min_keep:
                out[comp] = True
        candidate = out

    return candidate

def _median_positive_step(values):
    vals = np.unique(np.round(np.asarray(values, dtype=np.float64), 7))
    dv = np.diff(np.sort(vals))
    dv = dv[dv > 1e-8]
    if len(dv) == 0:
        return 0.10
    return float(np.median(dv))

def _terminal_crack_display_gate(load_disp, pred_force, peak_force,
                                 frac_prob, cfg):
    if not cfg.get('CRACK_DISPLAY_TERMINAL_ONLY', True):
        return True
    if load_disp is None or pred_force is None or peak_force is None:
        return False

    max_disp = float(cfg.get('_CURRENT_MAX_DISP_FOR_VIZ',
                             cfg.get('CRACK_FULL_DISP', 5.9)))
    late_limit = max(
        float(cfg.get('CRACK_ENABLE_DISP', 4.0)),
        float(cfg.get('CRACK_DISPLAY_MIN_DISP_RATIO', 0.875)) * max_disp)
    if float(load_disp) < late_limit:
        return False

    force_limit = max(
        float(cfg.get('CRACK_DISPLAY_FORCE_ABS_N', 240.0)),
        float(cfg.get('CRACK_DISPLAY_FORCE_RATIO', 0.10)) *
        max(abs(float(peak_force)), 1.0))
    if abs(float(pred_force)) > force_limit:
        return False

    if cfg.get('CRACK_DISPLAY_REQUIRE_FRAC_PROB', False):
        if frac_prob is None or float(frac_prob) < float(
                cfg.get('FRACTURE_PROB_THRESHOLD', 0.45)):
            return False
    return True

def _terminal_damage_locator_mask(triangles, ref_vertical, damage, ids, cfg):
    triangles = np.asarray(triangles, dtype=np.int64)
    ref_vertical = np.asarray(ref_vertical, dtype=np.float64)
    tri_y = ref_vertical[triangles].mean(axis=1)
    qlo, qhi = cfg.get('CRACK_TERMINAL_SEARCH_QUANTILES', (0.25, 0.75))
    lo, hi = np.quantile(ref_vertical, [float(qlo), float(qhi)])
    central = (tri_y >= lo) & (tri_y <= hi)
    if not central.any():
        central = np.ones(len(triangles), dtype=bool)

    if damage is not None and ids is not None:
        dmg = np.asarray(damage, dtype=np.float64)
        tri_score = dmg[ids][triangles].mean(axis=1)
    else:
        tri_score = np.zeros(len(triangles), dtype=np.float64)

    valid = np.where(central)[0]
    if len(valid) == 0:
        return np.zeros(len(triangles), dtype=bool)
    vals = tri_score[valid]
    best_tri = int(valid[int(np.nanargmax(vals))]) if np.isfinite(vals).any() else int(valid[len(valid)//2])
    center = float(tri_y[best_tri])
    step = _median_positive_step(ref_vertical)
    half = float(cfg.get('CRACK_TERMINAL_FALLBACK_HALF_ROWS', 1.10)) * step
    return central & (np.abs(tri_y - center) <= max(half, 0.55 * step))

def _terminal_stress_locator_mask(triangles, ref_vertical, stress_field, ids, cfg):
    triangles = np.asarray(triangles, dtype=np.int64)
    ref_vertical = np.asarray(ref_vertical, dtype=np.float64)
    tri_y = ref_vertical[triangles].mean(axis=1)
    qlo, qhi = cfg.get('CRACK_LOCATOR_SEARCH_QUANTILES', (0.25, 0.75))
    lo, hi = np.quantile(ref_vertical, [float(qlo), float(qhi)])
    central = (tri_y >= lo) & (tri_y <= hi)
    if not central.any():
        central = np.ones(len(triangles), dtype=bool)

    center = float(np.median(ref_vertical))
    if stress_field is not None:
        sf = np.asarray(stress_field, dtype=np.float64).reshape(-1)
        tri_s = sf[ids][triangles]
        pct = float(cfg.get('CRACK_LOCATOR_TOP_PERCENTILE', 75.0))
        tri_score = np.nanpercentile(tri_s, pct, axis=1)
        step = _median_positive_step(ref_vertical)
        row_key = np.round(tri_y / max(step, 1e-8)) * max(step, 1e-8)
        rows = np.unique(row_key[central])
        best_score = -np.inf
        for yy in rows:
            m = central & (np.abs(row_key - yy) <= 0.51 * max(step, 1e-8))
            if not m.any():
                continue
            score = float(np.nanmean(tri_score[m]))
            if np.isfinite(score) and score > best_score:
                best_score = score
                center = float(yy)

    step = _median_positive_step(ref_vertical)
    half = float(cfg.get('CRACK_LOCATOR_BAND_ROWS', 1.10)) * step
    return central & (np.abs(tri_y - center) <= max(half, 0.55 * step))

def _build_smooth_crack_overlay(gx, gy, ref_vertical, triangles, mask, cfg, pred_force=None, peak_force=None):

    v10.7 Key points:
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return None

    triangles = np.asarray(triangles, dtype=np.int64)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    ref_vertical = np.asarray(ref_vertical, dtype=np.float64)

    tri_ref_y = ref_vertical[triangles].mean(axis=1)
    crack_y = tri_ref_y[mask]
    if len(crack_y) == 0:
        return None

    center_ref = float(np.median(crack_y))
    step = _median_positive_step(ref_vertical)
    band_rows = float(cfg.get('SMOOTH_CRACK_BAND_ROWS', 3.0))
    node_band = max(band_rows * step, 1e-5)

    above = ((ref_vertical > center_ref) &
             (ref_vertical <= center_ref + node_band))
    below = ((ref_vertical < center_ref) &
             (ref_vertical >= center_ref - node_band))

    if above.sum() >= 3 and below.sum() >= 3:
        upper_edge = float(np.nanpercentile(gy[above], 5.0))
        lower_edge = float(np.nanpercentile(gy[below], 95.0))
    else:
        near = np.abs(ref_vertical - center_ref) <= max(1.8 * step, 1e-6)
        shift = float(np.median(gy[near] - ref_vertical[near])) if near.any() else 0.0
        center_plot = center_ref + shift
        nominal_half = float(np.clip(
            cfg.get('SMOOTH_CRACK_GAP_FACTOR', 0.62) * step,
            cfg.get('SMOOTH_CRACK_GAP_MIN', 0.060),
            cfg.get('SMOOTH_CRACK_GAP_MAX', 0.220)
        ))
        upper_edge = center_plot + nominal_half
        lower_edge = center_plot - nominal_half

    actual_gap = upper_edge - lower_edge
    nominal_gap = 2.0 * float(np.clip(
        cfg.get('SMOOTH_CRACK_GAP_FACTOR', 0.62) * step,
        cfg.get('SMOOTH_CRACK_GAP_MIN', 0.060),
        cfg.get('SMOOTH_CRACK_GAP_MAX', 0.220)
    ))
    if (not np.isfinite(actual_gap)) or actual_gap < 0.55 * nominal_gap:
        center_plot = 0.5 * (upper_edge + lower_edge)
        upper_edge = center_plot + 0.5 * nominal_gap
        lower_edge = center_plot - 0.5 * nominal_gap

    if cfg.get('SMOOTH_CRACK_FORCE_FULL_OPEN', True) and pred_force is not None and peak_force is not None:
        force_lim = max(float(cfg.get('SMOOTH_CRACK_FORCE_ZERO_ABS', 240.0)),
                        float(cfg.get('SMOOTH_CRACK_FORCE_ZERO_RATIO', 0.10)) * max(float(peak_force), 1.0))
        if abs(float(pred_force)) <= force_lim:
            center_plot = 0.5 * (upper_edge + lower_edge)
            open_gap = max(upper_edge - lower_edge, 1.15 * nominal_gap)
            upper_edge = center_plot + 0.5 * open_gap
            lower_edge = center_plot - 0.5 * open_gap

    buffer = max(float(cfg.get('SMOOTH_CRACK_EXTRA_BUFFER', 0.040)), 0.18 * step)
    upper_edge += buffer
    lower_edge -= buffer

    xnear = np.abs(ref_vertical - center_ref) <= max(3.0 * step, 1e-6)
    if xnear.sum() < 4:
        xnear = np.ones_like(ref_vertical, dtype=bool)
    xvals = gx[xnear]
    qlo, qhi = cfg.get('SMOOTH_CRACK_COVERAGE_Q', (0.5, 99.5))
    xmin = float(np.nanpercentile(xvals, qlo))
    xmax = float(np.nanpercentile(xvals, qhi))
    pad = 0.035 * max(xmax - xmin, 1e-6)
    xmin -= pad
    xmax += pad

    xx = np.linspace(xmin, xmax, 220)
    style = str(cfg.get('SMOOTH_CRACK_STYLE', 'flat')).lower()
    waviness = float(cfg.get('SMOOTH_CRACK_WAVINESS', 0.0))
    half_gap = max(0.5 * (upper_edge - lower_edge), 1e-6)
    amp = waviness * half_gap
    if style == 'mild_wave' and amp > 0.0:
        q = (xx - xmin) / max(xmax - xmin, 1e-12)
        top = upper_edge + amp * (0.65 * np.sin(2.0 * np.pi * q + 0.2) + 0.15 * np.sin(4.0 * np.pi * q + 0.7))
        bottom = lower_edge + amp * (0.55 * np.sin(2.0 * np.pi * q + 1.0) + 0.12 * np.sin(4.0 * np.pi * q + 0.1))
    else:
        top = np.full_like(xx, upper_edge)
        bottom = np.full_like(xx, lower_edge)

    px = np.concatenate([xx, xx[::-1]])
    py = np.concatenate([top, bottom[::-1]])
    return px, py

def _draw_smooth_crack_overlay(ax, triang):
    poly = getattr(triang, '_smooth_crack_overlay', None)
    if poly is None:
        return
    px, py = poly
    lw = float(getattr(triang, '_smooth_crack_edge_lw', 2.5))
    ax.fill(px, py, facecolor='white', edgecolor='white',
            linewidth=lw, joinstyle='round', zorder=80, antialiased=True)

def _surface_laplacian_smooth(values, triangles, passes=1, alpha=0.14):
    arr = np.asarray(values, dtype=np.float64).copy()
    if int(passes) <= 0 or float(alpha) <= 0.0 or len(arr) == 0:
        return arr
    tri = np.asarray(triangles, dtype=np.int64)
    if tri.size == 0:
        return arr

    edges = np.concatenate([
        tri[:, [0, 1]], tri[:, [1, 0]],
        tri[:, [1, 2]], tri[:, [2, 1]],
        tri[:, [2, 0]], tri[:, [0, 2]],
    ], axis=0)
    r, c = edges[:, 0], edges[:, 1]
    n = arr.shape[0]
    a = float(np.clip(alpha, 0.0, 0.45))

    for _ in range(int(passes)):
        sums = np.zeros_like(arr, dtype=np.float64)
        counts = np.zeros(n, dtype=np.float64)
        np.add.at(sums, r, arr[c])
        np.add.at(counts, r, 1.0)
        valid = counts > 0
        avg = arr.copy()
        if arr.ndim == 1:
            avg[valid] = sums[valid] / counts[valid]
        else:
            avg[valid] = sums[valid] / counts[valid, None]
        before_mean = np.nanmean(arr, axis=0)
        arr = (1.0 - a) * arr + a * avg
        arr += before_mean - np.nanmean(arr, axis=0)
    return arr

def _surface_neighbor_average(values, triangles):
    arr = np.asarray(values, dtype=np.float64)
    tri = np.asarray(triangles, dtype=np.int64)
    if arr.size == 0 or tri.size == 0:
        return arr.copy()
    edges = np.concatenate([
        tri[:, [0, 1]], tri[:, [1, 0]],
        tri[:, [1, 2]], tri[:, [2, 1]],
        tri[:, [2, 0]], tri[:, [0, 2]],
    ], axis=0)
    r, c = edges[:, 0], edges[:, 1]
    avg = arr.copy()
    sums = np.zeros_like(arr, dtype=np.float64)
    cnts = np.zeros(arr.shape[0], dtype=np.float64)
    np.add.at(sums, r, arr[c])
    np.add.at(cnts, r, 1.0)
    valid = cnts > 0
    avg[valid] = sums[valid] / cnts[valid]
    return avg

def _stress_field_for_display(field_front, ftris, cfg, pred=False):
    arr = _surface_laplacian_smooth(
        field_front, ftris,
        passes=int(cfg.get('STRESS_DISPLAY_SMOOTH_PASSES', 1)),
        alpha=float(cfg.get('STRESS_DISPLAY_SMOOTH_ALPHA', 0.14)))

    if pred and cfg.get('PRED_STRESS_DISPLAY_DEJAG', True):
        avg = _surface_neighbor_average(arr, ftris)
        dev = arr - avg
        base_cap = float(cfg.get('PRED_STRESS_DISPLAY_DEJAG_CAP_MPA', 4.5))
        rel_cap = float(cfg.get('PRED_STRESS_DISPLAY_DEJAG_REL', 1.35))
        p90 = float(np.nanpercentile(np.abs(dev), 90.0)) if len(dev) else 0.0
        cap = max(base_cap, rel_cap * p90)
        clipped = avg + np.clip(dev, -cap, cap)
        lam = float(np.clip(cfg.get('PRED_STRESS_DISPLAY_DEJAG_BLEND', 0.65), 0.0, 1.0))
        arr = (1.0 - lam) * arr + lam * clipped
        arr = _surface_laplacian_smooth(
            arr, ftris,
            passes=int(cfg.get('PRED_STRESS_DISPLAY_EXTRA_SMOOTH_PASSES', 1)),
            alpha=float(cfg.get('PRED_STRESS_DISPLAY_EXTRA_SMOOTH_ALPHA', 0.10)))
    return arr

def _add_sparse_stress_contours(ax, triang, field, levels, cfg):
    every = max(1, int(cfg.get('STRESS_CONTOUR_LINE_EVERY', 4)))
    line_levels = np.asarray(levels)[::every]
    if len(line_levels) < 2:
        return
    try:
        ax.tricontour(
            triang, field, levels=line_levels,
            colors=cfg.get('STRESS_CONTOUR_LINE_COLOR', 'white'),
            linewidths=float(cfg.get('STRESS_CONTOUR_LINE_WIDTH', 0.22)),
            alpha=float(cfg.get('STRESS_CONTOUR_LINE_ALPHA', 0.22)),
            antialiased=True, zorder=15)
    except (ValueError, RuntimeError):
        pass

def _make_tri(shared, cfg, surf, u, alive, damage=None, tri_alive=None, gt=False,
              frac_prob=None, load_disp=None, pred_force=None, peak_force=None,
              locator_stress=None):
    ids, ftris, front_idx = surf
    pax = [i for i in range(3) if i != cfg['SURFACE_AXIS']]
    u_front = np.asarray(u[ids], dtype=np.float64)
    if not gt:
        u_front = _surface_laplacian_smooth(
            u_front, ftris,
            passes=int(cfg.get('PRED_GEOMETRY_DISPLAY_SMOOTH_PASSES', 2)),
            alpha=float(cfg.get('PRED_GEOMETRY_DISPLAY_SMOOTH_ALPHA', 0.18)))
    gx = shared.raw_coords[ids, pax[0]] + u_front[:, pax[0]]
    gy = shared.raw_coords[ids, pax[1]] + u_front[:, pax[1]]
    tr = mtri.Triangulation(gx, gy, triangles=ftris)
    if gt and tri_alive is not None and len(tri_alive) == len(shared.surface_tris):
        mask = tri_alive[front_idx] < 0.5
    else:
        dead_count = (alive[ids][ftris] < 0.5).sum(axis=1)
        mask = dead_count >= int(cfg['DEAD_TRI_MIN_NODES'])
        if damage is not None:
            tri_damage = np.asarray(damage[ids][ftris], dtype=np.float64).mean(axis=1)
            strong = (tri_damage >= float(cfg.get('DISPLAY_DAMAGE_THRESHOLD', 0.86)))
            strong = strong | (dead_count >= max(2, int(cfg['DEAD_TRI_MIN_NODES']) + 1))
            mask = mask & strong
        ref_vertical = shared.raw_coords[ids, cfg['LOAD_AXIS']]
        terminal_gate = _terminal_crack_display_gate(
            load_disp, pred_force, peak_force, frac_prob, cfg)

        if cfg.get('CRACK_DISPLAY_MODE') == 'terminal_overlay':
            if terminal_gate:
                if cfg.get('CRACK_DISPLAY_FROM_STRESS_ONLY', True):
                    mask = _terminal_stress_locator_mask(
                        ftris, ref_vertical, locator_stress, ids, cfg)
                else:
                    mask = _cleanup_predicted_triangle_mask(
                        mask, ftris, ref_vertical, frac_prob, load_disp, cfg,
                        damage=damage, ids=ids)
                    if (not mask.any() and
                            cfg.get('CRACK_TERMINAL_FALLBACK_FROM_DAMAGE', True)):
                        mask = _terminal_damage_locator_mask(
                            ftris, ref_vertical, damage, ids, cfg)
                if mask.any():
                    tr._smooth_crack_overlay = _build_smooth_crack_overlay(
                        gx, gy, ref_vertical, ftris, mask, cfg,
                        pred_force=pred_force, peak_force=peak_force)
                    tr._smooth_crack_edge_lw = float(
                        cfg.get('SMOOTH_CRACK_EDGE_LW', 2.5))
            display_mask = np.zeros_like(mask, dtype=bool)

        elif cfg.get('CRACK_DISPLAY_MODE') == 'smooth_overlay':
            mask = _cleanup_predicted_triangle_mask(
                mask, ftris, ref_vertical, frac_prob, load_disp, cfg,
                damage=damage, ids=ids)
            if mask.any():
                tr._smooth_crack_overlay = _build_smooth_crack_overlay(
                    gx, gy, ref_vertical, ftris, mask, cfg,
                    pred_force=pred_force, peak_force=peak_force)
                tr._smooth_crack_edge_lw = float(
                    cfg.get('SMOOTH_CRACK_EDGE_LW', 2.5))
            display_mask = np.zeros_like(mask, dtype=bool)
        else:
            mask = _cleanup_predicted_triangle_mask(
                mask, ftris, ref_vertical, frac_prob, load_disp, cfg,
                damage=damage, ids=ids)
            display_mask = mask

    if gt:
        display_mask = mask
    if display_mask.all():
        display_mask[:] = False
    tr.set_mask(display_mask)
    return tr

def _select_prefracture_locator_stress(cache, show, fp_list, ld_list, cfg):
    show = list(show)
    if not show:
        return None
    peak = max([abs(float(x)) for x in fp_list] + [1.0])
    dmax = max([float(x) for x in ld_list] + [1.0])
    force_min = max(
        float(cfg.get('CRACK_LOCATOR_FORCE_ABS_N', 700.0)),
        float(cfg.get('CRACK_LOCATOR_FORCE_RATIO', 0.30)) * peak)
    disp_min = float(cfg.get('CRACK_LOCATOR_DISP_MIN_RATIO', 0.55)) * dmax
    candidates = [
        int(t) for t, f, d in zip(show, fp_list, ld_list)
        if abs(float(f)) >= force_min and float(d) >= disp_min
    ]
    if not candidates:
        candidates = [int(t) for t, f in zip(show, fp_list)
                      if abs(float(f)) >= force_min]
    if not candidates:
        candidates = [int(t) for t in (show[:-1] if len(show) > 1 else show)]
    order = {int(t): float(d) for t, d in zip(show, ld_list)}
    tloc = max(candidates, key=lambda t: order.get(int(t), -1.0))
    return cache[tloc]['stress_p']

@torch.no_grad()
def visualize_job(model, job, cfg, stats, shared, surf, save_path):
    """Stress Pred/GT/Error + full force-displacement curve."""
    model.eval()
    # surf = (front_node_ids, front_surface_triangles, front_triangle_indices)
    ids, ftris, _ = surf
    show = np.linspace(0, cfg['FRAMES_PER_JOB'] - 1,
                       cfg['VIZ_FRAMES'], dtype=int)
    cache, fp_list, ft_list, ld_list = _collect_viz_cache(
        model, job, cfg, stats, shared, show)
    cfg['_CURRENT_MAX_DISP_FOR_VIZ'] = float(max(ld_list)) if ld_list else float(
        stats.get('max_load_disp', cfg.get('CRACK_FULL_DISP', 5.9)))

    vals, errs = [], []
    for t in show:
        c = cache[t]
        m = c['alive_g'][ids] > 0.5
        vals.append(c['stress_g'][ids][m])
        errs.append(np.abs(c['stress_p'][ids][m] - c['stress_g'][ids][m]))

    vals = np.concatenate(vals)
    if cfg.get('STRESS_PLOT_RANGE') is not None:
        vmin, vmax = map(float, cfg['STRESS_PLOT_RANGE'])
    else:
        pp = cfg['STRESS_PLOT_PERCENTILES']
        vmin = max(0.0, float(np.nanpercentile(vals, pp[0])))
        vmax = float(np.nanpercentile(vals, pp[1]))
        if vmax <= vmin + 1e-6:
            vmax = vmin + 1.0

    err_pct = float(cfg.get('ERROR_PERCENTILE', 99.0))
    fixed_err = cfg.get('ERROR_PLOT_FIXED_MAX_MPA')
    errmax = (float(fixed_err) if fixed_err is not None else
              max(float(np.nanpercentile(np.concatenate(errs), err_pct)), 1.0))
    lv = np.linspace(vmin, vmax, cfg['STRESS_PLOT_LEVELS'])
    elv = np.linspace(0.0, errmax, cfg['STRESS_PLOT_LEVELS'])
    error_norm = make_error_norm(errmax, cfg)

    n = len(show)
    fig = plt.figure(figsize=(2.75 * n, 13))
    gs = gridspec.GridSpec(4, n, height_ratios=[4, 4, 4, 2])
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(3)]
    maps = [None] * 3
    peak_force = max(fp_list) if len(fp_list) else None
    locator_stress = _select_prefracture_locator_stress(
        cache, show, fp_list, ld_list, cfg)

    for col, t in enumerate(show):
        c = cache[t]
        tp = _make_tri(shared, cfg, surf, c['u_p'], c['alive_p'], c.get('damage_p'),
                       frac_prob=c['frac_p'], load_disp=ld_list[t],
                       pred_force=fp_list[t], peak_force=peak_force,
                       locator_stress=locator_stress)
        tg = _make_tri(shared, cfg, surf, c['u_g'], c['alive_g'],
                       tri_alive=c['tri_alive'], gt=True)
        err = np.abs(c['stress_p'][ids] - c['stress_g'][ids])
        pred_vis = _stress_field_for_display(c['stress_p'][ids], ftris, cfg, pred=True)
        gt_vis = _stress_field_for_display(c['stress_g'][ids], ftris, cfg, pred=False)
        panels = [
            (tp, pred_vis, cfg.get('STRESS_FIELD_CMAP', 'turbo'),
             ('Pred t={}\nD={:.2f}mm'.format(t, ld_list[t]) if not cfg.get('TRAIN_LEARNED_CRACK_HEAD', False) else 'Pred t={}\nD={:.2f}mm  Pfrac={:.2f}'.format(t, ld_list[t], c['frac_p']))),
            (tg, gt_vis, cfg.get('STRESS_FIELD_CMAP', 'turbo'),
             'GT t={}'.format(t)),
            (tg, err, ABS_ERROR_CMAP, '|Error| t={}'.format(t)),
        ]
        for r, (tr, field, cmap, title) in enumerate(panels):
            display_mode = str(cfg.get('STRESS_DISPLAY_MODE', 'balanced_contour')).lower()
            if r < 2 and display_mode == 'gouraud':
                cs = axes[r][col].tripcolor(
                    tr, field, shading='gouraud', cmap=cmap,
                    vmin=vmin, vmax=vmax)
            else:
                if r < 2:
                    cs = axes[r][col].tricontourf(
                        tr, field, levels=lv, cmap=cmap,
                        extend='both', antialiased=True)
                else:
                    cs = axes[r][col].tricontourf(
                        tr, field, levels=elv, cmap=cmap,
                        norm=error_norm, extend='max', antialiased=True)
                if r < 2 and display_mode == 'balanced_contour':
                    _add_sparse_stress_contours(
                        axes[r][col], tr, field, lv, cfg)
            maps[r] = cs
            if r == 0:
                _draw_smooth_crack_overlay(axes[r][col], tr)
            axes[r][col].set_title(title, fontsize=8)
            axes[r][col].set_aspect('equal')
            axes[r][col].axis('off')

    for r in range(3):
        cb = fig.colorbar(maps[r], ax=axes[r], fraction=0.015, pad=0.01)
        cb.set_label(cfg['STRESS_PLOT_LABEL'] if r < 2 else '|Error| (MPa)')
        if r < 2:
            ticks = cfg.get('STRESS_PLOT_TICKS')
            cb.set_ticks(ticks if ticks is not None else np.linspace(vmin, vmax, 7))
        else:
            eticks = cfg.get('ERROR_PLOT_TICKS_MPA')
            if eticks is not None:
                cb.set_ticks([x for x in eticks if 0.0 <= float(x) <= errmax + 1e-9])

    axc = fig.add_subplot(gs[3, :])
    if job.curve_disp is not None:
        axc.plot(job.curve_disp, job.curve_force, 'k-', lw=1.35,
                 label='GT dense')
    dq = np.linspace(0, max(ld_list), cfg['QUERY_CURVE_N'])
    mp = job.mat_n * stats['mat_std'] + stats['mat_mean']
    qr = query(model, shared, stats, cfg, dq, mp)
    qr_force_raw = np.asarray(qr['force'], dtype=float)
    qr_plot_x, qr_force_plot, qr_force_eval, qr_drop_x = _prepare_force_curve_for_plot(
        qr['disp'], qr_force_raw, cfg, fracture_prob=qr.get('fracture_prob'), smooth=True
    )
    if cfg.get('FORCE_PLOT_RAW_TOO', True):
        qr_raw_x, qr_raw_y, _, _ = _prepare_force_curve_for_plot(
            qr['disp'], qr_force_raw, cfg, fracture_prob=qr.get('fracture_prob'), smooth=False
        )
        axc.plot(
            qr_raw_x, qr_raw_y,
            color='salmon', ls='--',
            lw=0.8, alpha=0.55,
            label='Pred raw'
        )
    axc.plot(
        qr_plot_x, qr_force_plot,
        'r-', lw=1.5, label='Pred query'
    )
    axc.plot(ld_list, ft_list, 'ks', ms=3, label='GT @28')
    axc.plot(ld_list, fp_list, 'r^', ms=4, ls='none', label='Pred @28')
    axc.set(xlabel='Displacement (mm)', ylabel='Force (N)',
            title=('Force-Displacement comparison' if not cfg.get('SHOW_JOB_NAME_IN_TITLES', False) else 'Force-Displacement [{}]'.format(job.job_name)))
    axc.grid(alpha=.3)
    axc.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg['VIZ_DPI'])
    plt.close()
    print('  [Viz-Stress] saved {}'.format(save_path))

@torch.no_grad()
def visualize_dic_job(model, job, cfg, stats, shared, surf, save_path):
    model.eval()
    ids, _, _ = surf
    show = np.linspace(0, cfg['FRAMES_PER_JOB'] - 1,
                       cfg['VIZ_FRAMES'], dtype=int)
    cache, fp_list, _, ld_list = _collect_viz_cache(
        model, job, cfg, stats, shared, show)
    cfg['_CURRENT_MAX_DISP_FOR_VIZ'] = float(max(ld_list)) if ld_list else float(
        stats.get('max_load_disp', cfg.get('CRACK_FULL_DISP', 5.9)))

    vmin, vmax = cfg['DIC_EYY_RANGE']
    lv = np.linspace(vmin, vmax, cfg['DIC_PLOT_LEVELS'])
    errvals = []
    for t in show:
        c = cache[t]
        m = c['alive_g'][ids] > 0.5
        errvals.append(np.abs(c['eyy_p'][ids][m] - c['eyy_g'][ids][m]))

    err_floor = 5e-4 if cfg.get('DIC_OUTPUT_UNIT') == 'strain' else 0.05
    errmax = max(float(np.nanpercentile(
        np.concatenate(errvals), cfg.get('ERROR_PERCENTILE', 99.0))), err_floor)
    elv = np.linspace(0.0, errmax, cfg['DIC_PLOT_LEVELS'])

    n = len(show)
    fig = plt.figure(figsize=(2.75 * n, 11.5))
    gs = gridspec.GridSpec(3, n)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(n)] for r in range(3)]
    maps = [None] * 3
    locator_stress = _select_prefracture_locator_stress(
        cache, show, fp_list, ld_list, cfg)

    for col, t in enumerate(show):
        c = cache[t]
        tp = _make_tri(shared, cfg, surf, c['u_p'], c['alive_p'], c.get('damage_p'),
                       frac_prob=c['frac_p'], load_disp=ld_list[t],
                       pred_force=fp_list[t], peak_force=max(fp_list) if len(fp_list) else None,
                       locator_stress=locator_stress)
        tg = _make_tri(shared, cfg, surf, c['u_g'], c['alive_g'],
                       tri_alive=c['tri_alive'], gt=True)
        err = np.abs(c['eyy_p'][ids] - c['eyy_g'][ids])
        panels = [
            (tp, c['eyy_p'][ids], cfg.get('DIC_FIELD_CMAP', 'turbo'),
             'Pred eyy t={}\nD={:.2f}mm'.format(t, ld_list[t])),
            (tg, c['eyy_g'][ids], cfg.get('DIC_FIELD_CMAP', 'turbo'),
             'GT eyy t={}'.format(t)),
            (tg, err, ABS_ERROR_CMAP, '|Error eyy| t={}'.format(t)),
        ]
        for r, (tr, field, cmap, title) in enumerate(panels):
            cs = axes[r][col].tricontourf(
                tr, field, levels=lv if r < 2 else elv, cmap=cmap,
                extend='both' if r < 2 else 'max', antialiased=True)
            maps[r] = cs
            if r == 0:
                _draw_smooth_crack_overlay(axes[r][col], tr)
            axes[r][col].set_title(title, fontsize=8)
            axes[r][col].set_aspect('equal')
            axes[r][col].axis('off')

    for r in range(3):
        cb = fig.colorbar(maps[r], ax=axes[r], fraction=.015, pad=.01)
        if r < 2:
            cb.set_ticks(cfg['DIC_EYY_TICKS'])
            cb.set_label(dic_unit_label(cfg))
        else:
            cb.set_label('|Error eyy| [{}]'.format(
                '%' if cfg.get('DIC_OUTPUT_UNIT') == 'percent' else '1'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg['VIZ_DPI'])
    plt.close()
    print('  [Viz-DIC] saved {}'.format(save_path))

def _curve_peak_and_fracture(disp, force):
    d = np.asarray(disp, dtype=np.float64)
    f = np.asarray(force, dtype=np.float64)
    peak = float(np.max(f))
    ip = int(np.argmax(f))
    frac_disp = float(d[-1])
    for k in range(ip, len(f)):
        if f[k] <= 0.05 * max(peak, 1e-8):
            frac_disp = float(d[k])
            break
    return peak, frac_disp

def _curve_smooth_for_metric(y):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 9:
        return y.copy()
    kernel = np.array([-2., 3., 6., 7., 6., 3., -2.],
                      dtype=np.float64) / 21.0
    pad = len(kernel) // 2
    return np.convolve(np.pad(y, (pad, pad), mode='edge'),
                       kernel, mode='valid')

def _curve_metric_region(d, gt):
    gt = np.asarray(gt, dtype=np.float64)
    ip = int(np.argmax(gt))
    peak = float(np.max(gt))
    after = np.where(
        (np.arange(len(gt)) > ip) &
        (gt < 0.85 * max(peak, 1e-8))
    )[0]
    end = int(after[0]) if len(after) else len(gt)
    end = max(end, min(len(gt), 9))
    return np.arange(end, dtype=np.int64)

@torch.no_grad()
def query_force_only(model, stats, cfg, load_disps, mat_params_raw):
    d = np.atleast_1d(
        np.asarray(load_disps, dtype=np.float32))
    mp = np.asarray(
        mat_params_raw, dtype=np.float32).reshape(-1)
    mp_n = (
        (mp - np.asarray(stats['mat_mean'], dtype=np.float32)) /
        np.asarray(stats['mat_std'], dtype=np.float32)
    )
    cond_np = make_cond(d, mp_n, stats, cfg)
    cond = torch.tensor(
        cond_np, dtype=torch.float32,
        device=cfg['DEVICE'])
    vals = []
    chunk = max(64, int(cfg.get(
        'FORCE_QUERY_CHUNK', 2048)))
    model.eval()
    for i in range(0, len(cond), chunk):
        f = model.force_from_cond(cond[i:i + chunk])
        vals.append(f.detach().float().cpu())
    force_n = torch.cat(vals).numpy()
    force = (
        force_n * float(stats['force_std']) +
        float(stats['force_mean'])
    )
    force[d <= 1e-9] = 0.0
    return force

@torch.no_grad()
def evaluate_dense_force_selection(model, jobs, cfg, stats, shared,
                                   points_override=None):

    """
    model.eval()
    use_jobs = [j for j in jobs if j.curve_disp is not None]
    n_jobs_cfg = int(cfg.get('DENSE_SELECT_JOBS', 0))
    if n_jobs_cfg > 0:
        use_jobs = use_jobs[:n_jobs_cfg]
    if not use_jobs:
        return None

    npoints = max(
        41,
        int(points_override or cfg.get('DENSE_SELECT_POINTS', 201))
    )

    all_pred, all_gt = [], []
    per_rel, per_rmse_frac, per_r2 = [], [], []
    peak_errs, frac_errs = [], []
    slope_errs, curvature_errs, ripple_errs = [], [], []
    elastic_rmse_fracs, elastic_slope_errs = [], []

    for job in use_jobs:
        cd = np.asarray(job.curve_disp, dtype=np.float64)
        cf = np.asarray(job.curve_force, dtype=np.float64)
        idx = np.unique(
            np.rint(np.linspace(0, len(cd) - 1, npoints)).astype(int)
        )
        d = cd[idx]
        gt = cf[idx]

        mat_raw = job.mat_n * stats['mat_std'] + stats['mat_mean']
        pred = np.asarray(
            query_force_only(
                model, stats, cfg, d, mat_raw),
            dtype=np.float64)

        all_pred.append(pred)
        all_gt.append(gt)

        err = pred - gt
        gt_norm = max(float(np.linalg.norm(gt)), 1e-8)
        peak_gt = max(float(np.max(gt)), 1e-8)

        per_rel.append(float(np.linalg.norm(err) / gt_norm))
        per_rmse_frac.append(
            float(np.sqrt(np.mean(err ** 2)) / peak_gt)
        )
        den = float(np.sum((gt - np.mean(gt)) ** 2)) + 1e-8
        per_r2.append(float(1.0 - np.sum(err ** 2) / den))

        pp, pd = _curve_peak_and_fracture(d, pred)
        gp, gd = _curve_peak_and_fracture(d, gt)
        peak_errs.append(abs(pp - gp) / peak_gt)
        frac_errs.append(
            abs(pd - gd) / max(float(d[-1] - d[0]), 1e-8)
        )

        region = _curve_metric_region(d, gt)
        pseg = (pred / peak_gt)[region]
        gseg = (gt / peak_gt)[region]

        dn_all = (d - d[0]) / max(float(d[-1] - d[0]), 1e-8)
        em = dn_all <= float(cfg.get('DENSE_ELASTIC_RATIO', 0.16))
        if np.count_nonzero(em) >= 3:
            ep = pred[em] / peak_gt
            eg = gt[em] / peak_gt
            ed = d[em]
            elastic_rmse_fracs.append(float(np.sqrt(np.mean((ep - eg) ** 2))))
            dep = np.diff(ep) / np.maximum(np.diff(ed), 1e-8)
            deg = np.diff(eg) / np.maximum(np.diff(ed), 1e-8)
            slope_scale = max(float(np.max(np.abs(deg))), 1e-8)
            elastic_slope_errs.append(float(
                np.sqrt(np.mean(((dep - deg) / slope_scale) ** 2))))
        else:
            elastic_rmse_fracs.append(0.0)
            elastic_slope_errs.append(0.0)

        if len(pseg) >= 4:
            dp, dg = np.diff(pseg), np.diff(gseg)
            d2p, d2g = np.diff(pseg, n=2), np.diff(gseg, n=2)
            slope_errs.append(
                float(np.sqrt(np.mean((dp - dg) ** 2)))
            )
            curvature_errs.append(
                float(np.sqrt(np.mean((d2p - d2g) ** 2)))
            )

            hp = pseg - _curve_smooth_for_metric(pseg)
            hg = gseg - _curve_smooth_for_metric(gseg)
            ripple_errs.append(
                float(np.sqrt(np.mean((hp - hg) ** 2)))
            )
        else:
            slope_errs.append(0.0)
            curvature_errs.append(0.0)
            ripple_errs.append(0.0)

    pred_all = np.concatenate(all_pred)
    gt_all = np.concatenate(all_gt)
    err_all = pred_all - gt_all

    rmse = float(np.sqrt(np.mean(err_all ** 2)))
    mae = float(np.mean(np.abs(err_all)))
    rel_l2 = float(
        np.linalg.norm(err_all) /
        max(np.linalg.norm(gt_all), 1e-8)
    )
    den = float(np.sum((gt_all - np.mean(gt_all)) ** 2)) + 1e-8
    r2 = float(1.0 - np.sum(err_all ** 2) / den)

    mean_peak = float(np.mean([
        np.max(np.asarray(j.curve_force, dtype=np.float64))
        for j in use_jobs
    ]))
    rmse_global_frac = rmse / max(mean_peak, 1e-8)
    mae_global_frac = mae / max(mean_peak, 1e-8)

    q = float(cfg.get('DENSE_WORST_QUANTILE', 0.90))
    worst_rel = float(np.quantile(per_rel, q))
    worst_rmse_frac = float(np.quantile(per_rmse_frac, q))
    mean_slope = float(np.mean(slope_errs))
    mean_curvature = float(np.mean(curvature_errs))
    mean_ripple = float(np.mean(ripple_errs))
    mean_elastic_rmse = float(np.mean(elastic_rmse_fracs))
    mean_elastic_slope = float(np.mean(elastic_slope_errs))
    mean_peak_err = float(np.mean(peak_errs))
    mean_frac_err = float(np.mean(frac_errs))

    curve_score = (
        0.24 * rel_l2 +
        0.12 * worst_rel +
        0.10 * rmse_global_frac +
        0.08 * worst_rmse_frac +
        0.05 * mae_global_frac +
        0.06 * mean_peak_err +
        0.08 * mean_frac_err +
        0.10 * mean_slope +
        0.07 * mean_curvature +
        0.06 * mean_ripple +
        cfg.get('DENSE_ELASTIC_RMSE_WEIGHT', 0.10) * mean_elastic_rmse +
        cfg.get('DENSE_ELASTIC_SLOPE_WEIGHT', 0.10) * mean_elastic_slope
    )

    return {
        'curve_score': float(curve_score),
        'dense_force_rel_l2': rel_l2,
        'dense_force_r2': r2,
        'dense_force_rmse': rmse,
        'dense_force_mae': mae,
        'dense_force_worst_rel_l2': worst_rel,
        'dense_force_worst_rmse_frac': worst_rmse_frac,
        'dense_peak_force_error_frac': mean_peak_err,
        'dense_fracture_disp_error_frac': mean_frac_err,
        'dense_slope_rmse': mean_slope,
        'dense_curvature_rmse': mean_curvature,
        'dense_ripple_rmse': mean_ripple,
        'dense_elastic_rmse_frac': mean_elastic_rmse,
        'dense_elastic_slope_rmse': mean_elastic_slope,
        'dense_min_job_r2': float(np.min(per_r2)),
        'dense_mean_job_r2': float(np.mean(per_r2)),
        'dense_jobs': len(use_jobs),
        'dense_points_per_job': npoints,
    }

@torch.no_grad()
def final_report(model, jobs, cfg, stats, shared, save_dir, tag='test'):
    model.eval()
    device = cfg['DEVICE']
    allp, allt, curves, rows = [], [], [], []
    dense_err = []
    for job in jobs:
        sp, st, ep, et, fp, ft, _dp, _dt, _al, _dg, _fq = rollout_job(model, job, cfg, shared, device)
        allp += sp[1:]
        allt += st[1:]
        ld = [job[t]['raw_load_disp'] for t in range(cfg['FRAMES_PER_JOB'])]
        fpr = [float(v) * stats['force_std'] + stats['force_mean'] for v in fp]
        ftr = [job[t]['raw_force'] for t in range(cfg['FRAMES_PER_JOB'])]

        if cfg['QUERY_MODE'] and job.curve_disp is not None:
            mp_raw = job.mat_n * stats['mat_std'] + stats['mat_mean']
            qr = query(model, shared, stats, cfg, job.curve_disp, mp_raw)
            dense_err.append(np.abs(qr['force'] - job.curve_force))
            ld, fpr, ftr = job.curve_disp, qr['force'], job.curve_force
        curves.append((ld, fpr, ftr))

        def peak_and_frac(f, d):
            f = np.asarray(f)
            pk = float(np.max(f))
            i = int(np.argmax(f))
            fd = d[-1]
            for k in range(i, len(f)):
                if f[k] < 0.05 * pk:
                    fd = d[k]
                    break
            return pk, fd

        pp, pd_ = peak_and_frac(fpr, ld)
        tp, td_ = peak_and_frac(ftr, ld)
        rows.append((job.job_name, tp, pp, td_, pd_))

    sp = torch.cat(allp) * stats['stress_std'] + stats['stress_mean']
    st = torch.cat(allt) * stats['stress_std'] + stats['stress_mean']
    rmse = torch.sqrt(((sp - st) ** 2).mean()).item()
    r2 = (1 - ((st - sp) ** 2).sum() / (((st - st.mean()) ** 2).sum() + 1e-8)).item()

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    idx = np.random.choice(len(sp), size=min(8000, len(sp)), replace=False)
    ax[0].scatter(st.numpy()[idx], sp.numpy()[idx], s=3, alpha=0.25)
    lims = [float(st.min()), float(st.max())]
    ax[0].plot(lims, lims, 'r--', lw=1)
    ax[0].set_xlabel('GT stress (MPa)')
    ax[0].set_ylabel('Pred stress (MPa)')
    ax[0].set_title('R2={:.4f}  RMSE={:.2f} MPa'.format(r2, rmse))

    for ld, fpr, ftr in curves:
        ax[1].plot(ld, ftr, 'k-', alpha=0.5, lw=1)
        ax[1].plot(ld, fpr, 'r--', alpha=0.8, lw=1)
    ax[1].set_xlabel('Displacement (mm)')
    ax[1].set_ylabel('Force (N)')
    ax[1].set_title('Force-Displacement (black=GT, red=Pred)')
    ax[1].grid(alpha=0.3)

    tp = [r[1] for r in rows]; pp = [r[2] for r in rows]
    td = [r[3] for r in rows]; pd_ = [r[4] for r in rows]
    ax[2].scatter(tp, pp, label='Peak force (N)', s=40)
    ax2b = ax[2].twinx()
    ax2b.scatter(td, pd_, color='tab:orange', marker='^', s=40,
                 label='Fracture disp (mm)')
    lim = [min(tp) * 0.95, max(tp) * 1.05]
    ax[2].plot(lim, lim, 'k--', lw=1)
    ax[2].set_xlabel('GT'); ax[2].set_ylabel('Pred peak force (N)')
    ax2b.set_ylabel('Pred fracture disp (mm)')
    ax[2].set_title('Peak force / fracture displacement')

    plt.tight_layout()
    p = os.path.join(save_dir, 'final_report_{}.png'.format(tag))
    plt.savefig(p, dpi=150)
    plt.close()

    pf_err = float(np.mean([abs(a - b) for a, b in zip(tp, pp)]))
    fd_err = float(np.mean([abs(a - b) for a, b in zip(td, pd_)]))
    metrics = evaluate(model, jobs, cfg, stats, shared)
    print('  [{}] Stress R2={:.4f} RMSE={:.2f} MPa | '
          'Force R2={:.4f} RMSE={:.1f} N | Eyy=OFF'
          .format(tag, metrics['r2'], metrics['stress_rmse'],
                  metrics['force_r2'], metrics['force_rmse']))
    print('  [{}] late stress RMSE={:.2f} MPa | crack-zone RMSE={} | '
          'dead IoU={:.3f} recall={:.3f}'
          .format(tag, metrics['late_stress_rmse'],
                  '{:.2f} MPa'.format(metrics['crack_zone_rmse'])
                  if np.isfinite(metrics['crack_zone_rmse']) else 'N/A',
                  metrics['dead_iou'], metrics['dead_recall']))
    print('  [{}] Peak force MAE={:.1f} N | Fracture disp MAE={:.3f} mm'
          .format(tag, pf_err, fd_err))
    out = {
        'r2': metrics['r2'],
        'rmse': metrics['stress_rmse'],
        'stress_rmse_mpa': metrics['stress_rmse'],
        'eyy_rmse': metrics['eyy_rmse'],
        'eyy_rmse_percent': metrics['eyy_rmse_percent'],
        'force_rmse_n': metrics['force_rmse'],
        'late_stress_rmse_mpa': metrics['late_stress_rmse'],
        'crack_zone_rmse_mpa': metrics['crack_zone_rmse'],
        'dead_iou': metrics['dead_iou'],
        'dead_recall': metrics['dead_recall'],
        'peak_force_mae': pf_err,
        'fracture_disp_mae': fd_err,
    }
    if dense_err:
        de = np.concatenate(dense_err)
        pk = float(np.mean(tp))
        out['dense_force_mae'] = float(de.mean())
        print('  [{}] Full curve(501 pts/case) force MAE={:.1f} N  ({:.2f}% Peak force)'
              .format(tag, de.mean(), 100 * de.mean() / max(pk, 1e-6)))
    json_path = os.path.join(save_dir, 'final_metrics_{}.json'.format(tag))
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out

def _gt_frame_raw(job, t, stats):
    fr = job[int(t)]
    return {
        'stress': fr['y_stress'].numpy() * stats['stress_std'] + stats['stress_mean'],
        'eyy': np.asarray(fr['raw_eyy'], dtype=np.float32),
        'u': fr['y_disp'].numpy() * stats['disp_std'] + stats['disp_mean'],
        'alive': fr['alive'].numpy(),
        'damage': fr['damage'].numpy(),
        'tri_alive': fr.get('tri_alive'),
        'force': float(fr['raw_force']),
        'disp': float(fr['raw_load_disp']),
        'post_frac': float(fr.get('post_frac', 0.0)),
        'frame': int(t),
    }

def gt_state_at_displacement(job, disp_value, stats, cfg):

    """
    ds = np.asarray([fr['raw_load_disp'] for fr in job.frames], dtype=float)
    d = float(np.clip(disp_value, ds.min(), ds.max()))
    hi = int(np.searchsorted(ds, d, side='left'))
    if hi <= 0:
        st = _gt_frame_raw(job, 0, stats)
        st['source'] = 'GT frame 0'
        return st
    if hi >= len(ds):
        st = _gt_frame_raw(job, len(ds) - 1, stats)
        st['source'] = 'GT frame {}'.format(len(ds) - 1)
        return st
    lo = hi - 1
    a = _gt_frame_raw(job, lo, stats)
    b = _gt_frame_raw(job, hi, stats)
    if abs(ds[hi] - d) < 1e-10:
        b['source'] = 'GT frame {}'.format(hi)
        return b

    same_tri = ((a['tri_alive'] is None and b['tri_alive'] is None) or
                (a['tri_alive'] is not None and b['tri_alive'] is not None and
                 np.array_equal(a['tri_alive'], b['tri_alive'])))
    same_alive = np.array_equal(a['alive'] > 0.5, b['alive'] > 0.5)
    can_interp = (cfg.get('QUERY_GT_INTERPOLATE', True) and same_tri and same_alive)
    if not can_interp:
        st = a if abs(d - ds[lo]) <= abs(ds[hi] - d) else b
        st = dict(st)
        st['source'] = 'nearest GT frame {}'.format(st['frame'])
        return st

    w = float((d - ds[lo]) / max(ds[hi] - ds[lo], 1e-12))
    out = {
        'stress': (1 - w) * a['stress'] + w * b['stress'],
        'eyy': (1 - w) * a['eyy'] + w * b['eyy'],
        'u': (1 - w) * a['u'] + w * b['u'],
        'alive': a['alive'].copy(),
        'damage': (1 - w) * a['damage'] + w * b['damage'],
        'tri_alive': a['tri_alive'],
        'force': (1 - w) * a['force'] + w * b['force'],
        'disp': d,
        'post_frac': max(a['post_frac'], b['post_frac']),
        'frame': None,
        'source': 'linear GT frames {}-{}'.format(lo, hi),
    }
    return out

def _smooth_force_for_display(disps, forces, cfg):

    """
    x = np.asarray(disps, dtype=np.float64)
    y = np.asarray(forces, dtype=np.float64).copy()
    if (not cfg.get('FORCE_DISPLAY_SMOOTH', True)) or len(y) < 9:
        return y

    peak = float(np.nanmax(y))
    if not np.isfinite(peak) or peak <= 0:
        return y

    ip = int(np.nanargmax(y))
    candidates = np.where(
        (np.arange(len(y)) > ip) & (y < 0.65 * peak)
    )[0]
    drop = int(candidates[0]) if len(candidates) else len(y)

    k = int(cfg.get('FORCE_DISPLAY_KERNEL', 7))
    if k % 2 == 0:
        k += 1
    k = max(5, min(k, 11))
    half = k // 2

    if k == 5:
        kernel = np.array([-3, 12, 17, 12, -3], dtype=float) / 35.0
    elif k == 7:
        kernel = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=float) / 21.0
    else:
        kernel = np.ones(k, dtype=float) / float(k)

    out = y.copy()
    passes = max(1, int(cfg.get('FORCE_DISPLAY_SMOOTH_PASSES', 1)))
    end = max(drop - 2, 2 * half + 1)
    segment = out[:end].copy()
    for _ in range(passes):
        pad = np.pad(segment, (half, half), mode='edge')
        segment = np.convolve(pad, kernel, mode='valid')
    out[:end] = segment
    out = np.maximum(out, 0.0)
    return out

def _find_display_fracture_index(disps, forces, fracture_prob, cfg):
    x = np.asarray(disps, dtype=np.float64)
    y = np.asarray(forces, dtype=np.float64)
    if len(y) < 4:
        return None
    peak = float(np.nanmax(y))
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    ip = int(np.nanargmax(y))

    if (fracture_prob is not None) and cfg.get('FORCE_DISPLAY_DROP_USE_FRAC_PROB', True):
        q = np.asarray(fracture_prob, dtype=np.float64).reshape(-1)
        th = float(cfg.get('FORCE_DISPLAY_DROP_FRAC_PROB_TH', 0.55))
        cand = np.where((np.arange(len(q)) >= ip) & (q >= th))[0]
        if len(cand):
            return int(cand[0])

    force_th = max(
        float(cfg.get('FORCE_DISPLAY_DROP_FORCE_ABS_N', 80.0)),
        float(cfg.get('FORCE_DISPLAY_DROP_FORCE_RATIO', 0.05)) * peak)
    cand = np.where((np.arange(len(y)) > ip) & (y <= force_th))[0]
    if len(cand):
        return int(cand[0])
    return None

def _manual_drop_value(cfg):
    value = cfg.get('FORCE_DISPLAY_MANUAL_DROP_DISP', None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None

def _prepare_force_curve_for_plot(disps, forces, cfg,
                                  fracture_prob=None, smooth=True):

    Raw query output, CSV or checkpoint.
    """
    x = np.asarray(disps, dtype=np.float64)
    y_raw = np.asarray(forces, dtype=np.float64)
    y = _smooth_force_for_display(x, y_raw, cfg) if smooth else y_raw.copy()

    if not cfg.get('FORCE_DISPLAY_VERTICAL_DROP', True):
        return x, y, y, None

    mode = str(cfg.get('FORCE_DISPLAY_DROP_MODE', 'manual')).strip().lower()
    if mode in ('off', 'none', 'false', '0'):
        return x, y, y, None

    x0 = None
    if mode == 'manual':
        x0 = _manual_drop_value(cfg)
        if x0 is None:
            return x, y, y, None
        x0 = float(np.clip(x0, float(np.min(x)), float(np.max(x))))
    elif mode == 'auto':
        idx = _find_display_fracture_index(x, y, fracture_prob, cfg)
        if idx is None or idx <= 0 or idx >= len(x):
            return x, y, y, None
        x0 = float(x[idx])
    else:
        return x, y, y, None

    y0 = float(np.interp(x0, x, y))
    before = x < x0
    after = x > x0
    xp = np.concatenate([x[before], [x0, x0], x[after]])
    yp = np.concatenate([
        y[before], [y0, 0.0],
        np.zeros(int(np.sum(after)), dtype=np.float64)
    ])

    y_eval = y.copy()
    y_eval[x >= x0] = 0.0
    return xp, yp, y_eval, x0

def _draw_force_curve(ax, curve_cache, gt_curve, disp_value, f_pred, f_gt=None):
    if gt_curve is not None:
        gd = np.asarray(gt_curve[0], dtype=float)
        gf = np.asarray(gt_curve[1], dtype=float)
        ax.plot(gd, gf, 'k-', lw=1.8, label='GT / Abaqus')
        if f_gt is None:
            f_gt = float(np.interp(disp_value, gd, gf))
        ax.scatter([disp_value], [f_gt], marker='s', s=65,
                   facecolor='black', edgecolor='white', linewidth=0.8,
                   zorder=6, label='GT at query')
    pred_disp = np.asarray(curve_cache['disp'], dtype=float)
    pred_force_raw = np.asarray(curve_cache['force'], dtype=float)
    pred_frac_prob = curve_cache.get('fracture_prob')
    pred_plot_x, pred_plot_y, pred_force_eval, pred_drop_x = _prepare_force_curve_for_plot(
        pred_disp, pred_force_raw, CFG, fracture_prob=pred_frac_prob, smooth=True
    )
    f_pred_plot = float(
        np.interp(disp_value, pred_disp, pred_force_eval)
    )
    if CFG.get('FORCE_PLOT_RAW_TOO', True):
        raw_plot_x, raw_plot_y, _, _ = _prepare_force_curve_for_plot(
            pred_disp, pred_force_raw, CFG, fracture_prob=pred_frac_prob, smooth=False
        )
        ax.plot(
            raw_plot_x, raw_plot_y,
            color='salmon', lw=0.9, ls='--',
            alpha=0.60, label='GNN raw'
        )
    ax.plot(
        pred_plot_x, pred_plot_y,
        color='red', lw=1.8,
        label='GNN selected/display'
    )
    ax.axvline(disp_value, color='#3b5bdb', ls='--', lw=1.4,
               alpha=0.9, label='Query displacement')
    ax.scatter([disp_value], [f_pred_plot], s=85, color='red', edgecolor='white',
               linewidth=1.0, zorder=7, label='Pred at query')
    lines = ['D = {:.3f} mm'.format(disp_value),
             'Fpred = {:.1f} N'.format(f_pred_plot)]
    if pred_drop_x is not None:
        lines.append('Dfract(pred) = {:.3f} mm'.format(pred_drop_x))
    if f_gt is not None:
        lines.append('Fgt = {:.1f} N'.format(f_gt))
    ax.text(0.985, 0.96, '\n'.join(lines), transform=ax.transAxes,
            ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.35', alpha=0.92))
    ax.set_xlabel('Displacement (mm)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Force-displacement curve with queried position', pad=8)
    ax.grid(alpha=0.28)
    ax.legend(ncol=3, fontsize=9, loc='lower center')

def export_curve(model, shared, stats, cfg, mat_raw, save_dir, tag='query',
                 gt=None):
    dmax = float(stats['max_load_disp'])
    dq = np.linspace(0.0, dmax, cfg['QUERY_CURVE_N'])
    r = query(model, shared, stats, cfg, dq, mat_raw)

    csv_p = os.path.join(save_dir, 'curve_{}.csv'.format(tag))
    with open(csv_p, 'w') as f:
        f.write('displacement_mm,force_N\n')
        for a, b in zip(r['disp'], r['force']):
            f.write('{:.6f},{:.4f}\n'.format(a, b))

    plt.figure(figsize=(7, 4.5))
    if gt is not None:
        plt.plot(gt[0], gt[1], 'k-', lw=1.5, label='Abaqus')
    qx, qy, _qeval, qdrop = _prepare_force_curve_for_plot(
        r['disp'], r['force'], cfg, fracture_prob=r.get('fracture_prob'), smooth=True
    )
    plt.plot(qx, qy, 'r-', lw=1.5,
             label='GNN query ({} pts)'.format(len(dq)))
    plt.xlabel('Displacement (mm)'); plt.ylabel('Force (N)')
    plt.title('Predicted force-displacement' if not cfg.get('SHOW_JOB_NAME_IN_TITLES', False) else 'Predicted force-displacement  [{}]'.format(tag))
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    png_p = os.path.join(save_dir, 'curve_{}.png'.format(tag))
    plt.savefig(png_p, dpi=140); plt.close()
    print('  [Query] Curve saved {} / {}'.format(csv_p, png_p))
    return r

def _field_comparison_report(shared, cfg, surf, disp_value, point, gt_state,
                             curve_cache, gt_curve, save_path, kind):
    ids, ftris, _ = surf
    pred_u = point['u'][0]
    pred_alive = point['alive'][0]
    frac = float(point['fracture_prob'][0])
    pred_tri = _make_tri(shared, cfg, surf, pred_u, pred_alive, point.get('damage', [None])[0],
                         frac_prob=frac, load_disp=disp_value,
                         pred_force=float(point['force'][0]),
                         peak_force=float(np.nanmax(np.asarray(curve_cache['force'], dtype=float))))
    gt_tri = _make_tri(shared, cfg, surf, gt_state['u'], gt_state['alive'],
                       tri_alive=gt_state.get('tri_alive'), gt=True)

    if kind == 'stress':
        pred = point['stress'][0]
        gt = gt_state['stress']
        common = (pred_alive > 0.5) & (gt_state['alive'] > 0.5)
        vals = np.concatenate([pred[common], gt[common]]) if common.any() else np.r_[pred, gt]
        if cfg.get('STRESS_PLOT_RANGE') is not None:
            v0, v1 = map(float, cfg['STRESS_PLOT_RANGE'])
        else:
            pp = cfg['STRESS_PLOT_PERCENTILES']
            v0 = max(0.0, float(np.nanpercentile(vals, pp[0])))
            v1 = float(np.nanpercentile(vals, pp[1]))
            if v1 <= v0 + 1e-6:
                v1 = v0 + 1.0
        levels = np.linspace(v0, v1, cfg['STRESS_PLOT_LEVELS'])
        cmap = cfg.get('STRESS_FIELD_CMAP', 'turbo')
        label = 'Stress (MPa)'
        err_label = '|Error| (MPa)'
        title = 'Stress comparison @ {:.3f} mm'.format(disp_value)
        suffix = 'stress'
    elif kind == 'eyy':
        pred = point['eyy_display'][0]
        gt = gt_state['eyy']
        common = (pred_alive > 0.5) & (gt_state['alive'] > 0.5)
        v0, v1 = cfg['DIC_EYY_RANGE']
        levels = np.linspace(v0, v1, cfg['DIC_PLOT_LEVELS'])
        cmap = cfg.get('DIC_FIELD_CMAP', 'turbo')
        label = dic_unit_label(cfg)
        err_label = '|Error eyy| [{}]'.format(
            '%' if cfg.get('DIC_OUTPUT_UNIT') == 'percent' else '1')
        title = 'DIC eyy comparison @ {:.3f} mm'.format(disp_value)
        suffix = 'eyy'
    else:
        raise ValueError('unknown field kind: {}'.format(kind))

    err = np.abs(pred - gt)
    valid_err = err[common] if common.any() else err
    floor = 1.0 if kind == 'stress' else (
        5e-4 if cfg.get('DIC_OUTPUT_UNIT') == 'strain' else 0.05)
    fixed_err = (cfg.get('ERROR_PLOT_FIXED_MAX_MPA')
                 if kind == 'stress' else None)
    emax = (float(fixed_err) if fixed_err is not None else
            max(float(np.nanpercentile(
                valid_err, cfg.get('QUERY_ERROR_PERCENTILE', 99.0))), floor))
    err_levels = np.linspace(0.0, emax,
                             cfg['STRESS_PLOT_LEVELS'] if kind == 'stress'
                             else cfg['DIC_PLOT_LEVELS'])
    query_error_norm = make_error_norm(emax, cfg) if kind == 'stress' else None
    rmse = float(np.sqrt(np.mean(valid_err ** 2))) if len(valid_err) else float('nan')
    mae = float(np.mean(valid_err)) if len(valid_err) else float('nan')

    fig = plt.figure(figsize=(15.5, 10.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.2, 2.0],
                           hspace=0.24, wspace=0.18)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    display_mode = str(cfg.get('STRESS_DISPLAY_MODE', 'balanced_contour')).lower()
    if kind == 'stress':
        pred_display = _stress_field_for_display(pred[ids], ftris, cfg, pred=True)
        gt_display = _stress_field_for_display(gt[ids], ftris, cfg, pred=False)
    else:
        pred_display = pred[ids]
        gt_display = gt[ids]

    if kind == 'stress' and display_mode == 'gouraud':
        p0 = axs[0].tripcolor(pred_tri, pred_display, shading='gouraud', cmap=cmap,
                              vmin=v0, vmax=v1)
        p1 = axs[1].tripcolor(gt_tri, gt_display, shading='gouraud', cmap=cmap,
                              vmin=v0, vmax=v1)
    else:
        p0 = axs[0].tricontourf(pred_tri, pred_display, levels=levels, cmap=cmap,
                                extend='both', antialiased=True)
        p1 = axs[1].tricontourf(gt_tri, gt_display, levels=levels, cmap=cmap,
                                extend='both', antialiased=True)
        if kind == 'stress' and display_mode == 'balanced_contour':
            _add_sparse_stress_contours(axs[0], pred_tri, pred_display, levels, cfg)
            _add_sparse_stress_contours(axs[1], gt_tri, gt_display, levels, cfg)
    _draw_smooth_crack_overlay(axs[0], pred_tri)
    p2 = axs[2].tricontourf(
        gt_tri, err[ids], levels=err_levels,
        cmap=ABS_ERROR_CMAP,
        norm=query_error_norm if kind == 'stress' else None,
        extend='max', antialiased=True)
    cb0 = fig.colorbar(p0, ax=axs[0], shrink=.75, label=label)
    cb1 = fig.colorbar(p1, ax=axs[1], shrink=.75, label=label)
    if kind == 'stress' and cfg.get('STRESS_PLOT_TICKS') is not None:
        cb0.set_ticks(cfg['STRESS_PLOT_TICKS'])
        cb1.set_ticks(cfg['STRESS_PLOT_TICKS'])
    if kind == 'eyy':
        cb1.set_ticks(cfg['DIC_EYY_TICKS'])
    cb2 = fig.colorbar(p2, ax=axs[2], shrink=.75, label=err_label)
    if kind == 'stress' and cfg.get('ERROR_PLOT_TICKS_MPA') is not None:
        cb2.set_ticks([
            x for x in cfg['ERROR_PLOT_TICKS_MPA']
            if 0.0 <= float(x) <= emax + 1e-9
        ])
    axs[0].set_title('Pred @ {:.3f} mm'.format(disp_value))
    axs[1].set_title('GT @ {:.3f} mm\n{}'.format(disp_value, gt_state['source']))
    axs[2].set_title('|Pred - GT|\nRMSE={:.4g}, MAE={:.4g}'.format(rmse, mae))
    for ax in axs:
        ax.set_aspect('equal')
        ax.axis('off')

    axc = fig.add_subplot(gs[1, :])
    _draw_force_curve(axc, curve_cache, gt_curve, disp_value,
                      float(point['force'][0]), float(gt_state['force']))
    fig.suptitle('{} | fracture P={:.3f} | pred dead nodes={}'
                 .format(title, frac, int((pred_alive < 0.5).sum())), fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    plt.savefig(save_path, dpi=cfg.get('QUERY_REPORT_DPI', cfg['VIZ_DPI']))
    plt.close()
    return {'path': save_path, 'rmse': rmse, 'mae': mae, 'kind': suffix}

def _crack_comparison_report(shared, cfg, surf, disp_value, point, gt_state,
                             curve_cache, gt_curve, save_path):
    ids, ftris, _ = surf
    pax = [i for i in range(3) if i != cfg['SURFACE_AXIS']]
    u = gt_state['u']
    gx = shared.raw_coords[ids, pax[0]] + u[ids, pax[0]]
    gy = shared.raw_coords[ids, pax[1]] + u[ids, pax[1]]
    tri = mtri.Triangulation(gx, gy, triangles=ftris)
    pred_dead = (point['alive'][0] < 0.5).astype(float)
    gt_dead = (gt_state['alive'] < 0.5).astype(float)
    dmg = point['damage'][0]
    fig = plt.figure(figsize=(15.5, 10.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[3.2, 2.0], hspace=.24, wspace=.18)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    maps = [
        axs[0].tricontourf(tri, dmg[ids], levels=np.linspace(0, 1, 51),
                           cmap=cfg.get('DAMAGE_CMAP', 'YlOrRd'), extend='max'),
        axs[1].tricontourf(tri, pred_dead[ids], levels=[-0.01, .5, 1.01], cmap='Greys'),
        axs[2].tricontourf(tri, gt_dead[ids], levels=[-0.01, .5, 1.01], cmap='Greys'),
    ]
    fig.colorbar(maps[0], ax=axs[0], shrink=.75, label='Pred SDEG')
    axs[0].set_title('Pred damage')
    axs[1].set_title('Pred crack mask')
    axs[2].set_title('GT crack mask')
    for ax in axs:
        ax.set_aspect('equal'); ax.axis('off')
    axc = fig.add_subplot(gs[1, :])
    _draw_force_curve(axc, curve_cache, gt_curve, disp_value,
                      float(point['force'][0]), float(gt_state['force']))
    plt.tight_layout()
    plt.savefig(save_path, dpi=cfg.get('QUERY_REPORT_DPI', cfg['VIZ_DPI']))
    plt.close()
    return save_path

def export_query_report(model, shared, stats, cfg, surf, mat_raw,
                        disp_value, save_path, gt_curve=None,
                        curve_cache=None, gt_job=None):
    point = query(model, shared, stats, cfg, [disp_value], mat_raw)
    if curve_cache is None:
        dq = np.linspace(0.0, float(stats['max_load_disp']), cfg['QUERY_CURVE_N'])
        curve_cache = query(model, shared, stats, cfg, dq, mat_raw)
    if gt_job is None:
    gt_state = gt_state_at_displacement(gt_job, disp_value, stats, cfg)
    root, ext = os.path.splitext(save_path)
    if not ext:
        root = save_path
    stress_path = root + '_stress_compare.png'
    stress_res = _field_comparison_report(
        shared, cfg, surf, disp_value, point, gt_state, curve_cache, gt_curve,
        stress_path, 'stress')
    crack_path = None
    if cfg.get('QUERY_SAVE_CRACK_REPORT', True):
        crack_path = root + '_crack_compare.png'
        _crack_comparison_report(shared, cfg, surf, disp_value, point, gt_state,
                                 curve_cache, gt_curve, crack_path)
    print('  [Query] D={:.3f}mm -> Fpred={:.1f}N Fgt={:.1f}N | '
          'stress RMSE={:.3f}MPa | Eyy=OFF'
          .format(disp_value, float(point['force'][0]), float(gt_state['force']),
                  stress_res['rmse']))
    print('          {}'.format(stress_path))
    return {'point': point, 'gt': gt_state, 'stress': stress_res,
            'eyy': None, 'crack_path': crack_path}

def export_contour(model, shared, stats, cfg, surf, mat_raw,
                   disp_value, save_path, gt_curve=None,
                   curve_cache=None, gt_job=None):
    return export_query_report(
        model, shared, stats, cfg, surf, mat_raw, disp_value,
        save_path, gt_curve=gt_curve, curve_cache=curve_cache,
        gt_job=gt_job)

def load_for_inference(cfg):
    load_calibrated_crack_thresholds(cfg)
    stats = load_stats(os.path.join(cfg['CKPT_DIR'], 'stats.json'))
    jobs_dict = group_files_by_job(cfg['DATA_DIR'])
    any_file = jobs_dict[sorted(jobs_dict.keys())[0]][0]
    shared = SharedTopology(any_file, stats)
    model = GNNModel(in_n=shared.in_node_dim, in_e=4, in_cond=cond_dim(cfg), h=cfg['HIDDEN'],
                     layers=cfg['LAYERS'], cd=cfg['COND_DIM'],
                     tf_layers=cfg['TF_LAYERS'], tf_heads=cfg['TF_HEADS'],
                     tf_dropout=cfg['TF_DROPOUT'],
                     max_len=max(64, cfg['FRAMES_PER_JOB'] + 4),
                     query_mode=cfg['QUERY_MODE'],
                     force_fourier_k=cfg['DISP_FOURIER_K'],
                     force_mat_dim=cfg['N_MATERIAL_PARAMS'] if cfg['USE_MATERIAL_COND'] else 0,
                     force_residual_limit=cfg['FORCE_RESIDUAL_LIMIT'],
                     force_dcrit_min=cfg['FORCE_DCRIT_MIN'],
                     force_dcrit_max=cfg['FORCE_DCRIT_MAX'],
                     force_sharp_min=cfg['FORCE_SHARPNESS_MIN'],
                     force_sharp_max=cfg['FORCE_SHARPNESS_MAX']).to(cfg['DEVICE'])
    model.load_state_dict(torch.load(os.path.join(cfg['CKPT_DIR'], 'best_model.pt'),
                                     map_location=cfg['DEVICE']))
    model.eval()
    return model, shared, stats

def save_training_history(rows, save_dir):
    if not rows:
        return
    csv_path = os.path.join(save_dir, 'training_history.csv')
    fields = [
        'epoch', 'train_loss',
        'train_stress_rmse', 'val_stress_rmse',
        'train_eyy_rmse_percent', 'val_eyy_rmse_percent',
        'train_force_rmse', 'val_force_rmse',
        'val_r2_stress', 'val_r2_eyy', 'val_r2_force',
        'val_dead_iou', 'val_dead_precision', 'val_dead_recall',
        'val_late_stress_rmse', 'val_crack_zone_rmse',
        'lr'
    ]
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fields})

    def arr(key):
        out = []
        for r in rows:
            v = r.get(key, np.nan)
            out.append(np.nan if v is None else float(v))
        return np.asarray(out, dtype=float)

    ep = arr('epoch')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(ep, arr('val_stress_rmse'), label='Val')
    axes[0, 0].plot(ep, arr('train_stress_rmse'), '--', label='Train subset')
    axes[0, 0].set_title('Stress RMSE (MPa)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].grid(alpha=.25)
    axes[0, 0].legend()

    axes[0, 1].plot(ep, arr('val_late_stress_rmse'), label='Late-frame RMSE')
    axes[0, 1].plot(ep, arr('val_crack_zone_rmse'), label='Damage-zone RMSE')
    axes[0, 1].set_title('Stress field diagnostics (MPa)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(alpha=.25)
    axes[0, 1].legend()

    axes[1, 0].plot(ep, arr('val_force_rmse'), label='Val')
    axes[1, 0].plot(ep, arr('train_force_rmse'), '--', label='Train subset')
    axes[1, 0].set_title('Force RMSE (N)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(alpha=.25)
    axes[1, 0].legend()

    if CFG.get('TRAIN_LEARNED_CRACK_HEAD', False):
        axes[1, 1].plot(ep, arr('val_dead_iou'), label='Dead IoU')
        axes[1, 1].plot(ep, arr('val_dead_precision'), label='Dead precision')
        axes[1, 1].plot(ep, arr('val_dead_recall'), label='Dead recall')
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[1, 1].set_title('Fracture metrics')
        axes[1, 1].legend()
    else:
        axes[1, 1].plot(ep, arr('val_late_stress_rmse'), label='Late-frame RMSE')
        axes[1, 1].plot(ep, arr('val_crack_zone_rmse'), label='GT damage-zone RMSE')
        axes[1, 1].set_title('Late-field diagnostics\n(crack head disabled)')
        axes[1, 1].legend()
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(alpha=.25)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=170)
    plt.close()

def resolve_split_counts(n_jobs, cfg):
    fixed_need = cfg['N_VAL_JOBS'] + cfg['N_TEST_JOBS'] + cfg['MIN_TRAIN_JOBS']
    if n_jobs >= fixed_need:
        return cfg['N_VAL_JOBS'], cfg['N_TEST_JOBS']

    if n_jobs < cfg['MIN_TRAIN_JOBS'] + 2:
        raise RuntimeError(
            .format(cfg['MIN_TRAIN_JOBS'] + 2, cfg['MIN_TRAIN_JOBS'], n_jobs))

    nv = max(1, int(round(n_jobs * cfg['SMALL_DATA_VAL_RATIO'])))
    nt = max(1, int(round(n_jobs * cfg['SMALL_DATA_TEST_RATIO'])))
    while n_jobs - nv - nt < cfg['MIN_TRAIN_JOBS']:
        if nv >= nt and nv > 1:
            nv -= 1
        elif nt > 1:
            nt -= 1
        else:
            break
    return nv, nt

def select_visualization_job(jobs, cfg):
    wanted = str(cfg.get('VIZ_JOB_NAME', '')).strip()
    if wanted:
        for job in jobs:
            if job.job_name == wanted:
                return job
    return jobs[0]

def _save_curve_candidate(model, epoch, dense_vm, vm, cfg, records):
    if epoch < int(cfg.get('BEST_CURVE_MIN_EPOCH', 10)):
        return records
    if vm['r2'] < float(cfg.get('BEST_MIN_STRESS_R2', 0.965)):
        return records
    if (cfg.get('TRAIN_LEARNED_CRACK_HEAD', False) and
            vm['dead_iou'] < float(cfg.get('BEST_MIN_DEAD_IOU', 0.35))):
        return records

    cand_dir = os.path.join(cfg['CKPT_DIR'], 'curve_candidates')
    os.makedirs(cand_dir, exist_ok=True)
    path = os.path.join(cand_dir, 'epoch_{:03d}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

    rec = {
        'epoch': int(epoch),
        'path': path,
        'train_curve_score': float(dense_vm['curve_score']),
        'field_score': float(vm['selection_score']),
        'stress_r2': float(vm['r2']),
        'dead_iou': float(vm['dead_iou']),
    }
    records = [r for r in records if r['epoch'] != epoch]
    records.append(rec)
    records.sort(
        key=lambda r: (r['train_curve_score'], r['field_score'])
    )

    keep = max(2, int(cfg.get('DENSE_TOPK', 6)))
    removed = records[keep:]
    records = records[:keep]
    for r in removed:
        try:
            if os.path.exists(r['path']):
                os.remove(r['path'])
        except OSError:
            pass

    with open(
        os.path.join(cand_dir, 'topk_candidates.json'),
        'w', encoding='utf-8'
    ) as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records

def _average_checkpoints(paths, out_path):
    if not paths:
        return None
    states = [torch.load(p, map_location='cpu') for p in paths]
    avg = {}
    for k in states[0]:
        vals = [s[k] for s in states]
        if torch.is_floating_point(vals[0]):
            avg[k] = torch.stack(
                [v.float() for v in vals], dim=0
            ).mean(dim=0).to(vals[0].dtype)
        else:
            avg[k] = vals[0]
    torch.save(avg, out_path)
    return out_path

def finalize_curve_checkpoint(model, records, val_jobs, cfg, stats, shared,
                              best_model_path, best_field_path):
    candidates = [
        dict(r) for r in records if os.path.exists(r['path'])
    ]
    if not candidates:
        if os.path.exists(best_field_path):
            shutil.copy2(best_field_path, best_model_path)
            return 0, 'best_field_fallback', []
        return 0, 'existing_best', []

    final_points = int(cfg.get('DENSE_FINAL_POINTS', 301))
    ranking = []

    for rec in candidates:
        model.load_state_dict(torch.load(rec['path']))
        field = evaluate(model, val_jobs, cfg, stats, shared)
        curve = evaluate_dense_force_selection(
            model, val_jobs, cfg, stats, shared,
            points_override=final_points
        )
        qualified = (
            field['r2'] >= float(cfg.get('BEST_MIN_STRESS_R2', 0.965)) and
            (not cfg.get('TRAIN_LEARNED_CRACK_HEAD', False) or
             field['dead_iou'] >= float(cfg.get('BEST_MIN_DEAD_IOU', 0.35)))
        )
        ranking.append({
            **rec,
            'source': 'single',
            'final_curve_score': float(curve['curve_score']),
            'final_dense_r2': float(curve['dense_force_r2']),
            'final_dense_rmse': float(curve['dense_force_rmse']),
            'final_worst_rel_l2':
                float(curve['dense_force_worst_rel_l2']),
            'final_slope_rmse': float(curve['dense_slope_rmse']),
            'final_curvature_rmse':
                float(curve['dense_curvature_rmse']),
            'final_ripple_rmse': float(curve['dense_ripple_rmse']),
            'final_elastic_rmse_frac': float(curve['dense_elastic_rmse_frac']),
            'final_elastic_slope_rmse': float(curve['dense_elastic_slope_rmse']),
            'final_stress_r2': float(field['r2']),
            'final_dead_iou': float(field['dead_iou']),
            'qualified': bool(qualified),
        })

    n_avg = min(
        int(cfg.get('FINAL_TOPK_AVERAGE', 3)),
        len(candidates)
    )
    if n_avg >= 2:
        singles = sorted(
            ranking, key=lambda x: x['final_curve_score']
        )
        avg_path = os.path.join(
            cfg['CKPT_DIR'], 'best_curve_average.pt'
        )
        _average_checkpoints(
            [r['path'] for r in singles[:n_avg]], avg_path
        )
        model.load_state_dict(torch.load(avg_path))
        field = evaluate(model, val_jobs, cfg, stats, shared)
        curve = evaluate_dense_force_selection(
            model, val_jobs, cfg, stats, shared,
            points_override=final_points
        )
        qualified = (
            field['r2'] >= float(cfg.get('BEST_MIN_STRESS_R2', 0.965)) and
            (not cfg.get('TRAIN_LEARNED_CRACK_HEAD', False) or
             field['dead_iou'] >= float(cfg.get('BEST_MIN_DEAD_IOU', 0.35)))
        )
        ranking.append({
            'epoch': int(singles[0]['epoch']),
            'path': avg_path,
            'source': 'average_top{}'.format(n_avg),
            'final_curve_score': float(curve['curve_score']),
            'final_dense_r2': float(curve['dense_force_r2']),
            'final_dense_rmse': float(curve['dense_force_rmse']),
            'final_worst_rel_l2':
                float(curve['dense_force_worst_rel_l2']),
            'final_slope_rmse': float(curve['dense_slope_rmse']),
            'final_curvature_rmse':
                float(curve['dense_curvature_rmse']),
            'final_ripple_rmse': float(curve['dense_ripple_rmse']),
            'final_elastic_rmse_frac': float(curve['dense_elastic_rmse_frac']),
            'final_elastic_slope_rmse': float(curve['dense_elastic_slope_rmse']),
            'final_stress_r2': float(field['r2']),
            'final_dead_iou': float(field['dead_iou']),
            'qualified': bool(qualified),
        })

    qualified_rows = [r for r in ranking if r['qualified']]
    pool = qualified_rows if qualified_rows else ranking
    pool.sort(key=lambda r: (
        r['final_curve_score'],
        r['final_worst_rel_l2'],
        r['final_elastic_rmse_frac'],
        r['final_ripple_rmse']
    ))
    winner = pool[0]
    shutil.copy2(winner['path'], best_model_path)

    ranking.sort(key=lambda r: (
        not r['qualified'],
        r['final_curve_score']
    ))
    with open(
        os.path.join(
            cfg['CKPT_DIR'],
            'final_curve_candidate_ranking.json'
        ),
        'w', encoding='utf-8'
    ) as f:
        json.dump(ranking, f, indent=2, ensure_ascii=False)

    print(
        '  [FinalCurveSelect] source={} epoch={} score={:.6f} '
        'R2={:.5f} RMSE={:.1f}N worstRel={:.4f} ripple={:.5f} elastic={:.5f}'
        .format(
            winner['source'], winner['epoch'],
            winner['final_curve_score'],
            winner['final_dense_r2'],
            winner['final_dense_rmse'],
            winner['final_worst_rel_l2'],
            winner['final_ripple_rmse'],
            winner['final_elastic_rmse_frac']
        )
    )
    return int(winner['epoch']), str(winner['source']), ranking

def set_force_only_trainable(model, enabled=True):
    for p in model.parameters():
        p.requires_grad_(not enabled)
    if enabled:
        for p in model.force_parameters():
            p.requires_grad_(True)

def _full_force_batch(jobs, device):
    conds, targets, groups, disps, posts = [], [], [], [], []
    for gid, job in enumerate(jobs):
        if job.curve_cond is None:
            continue
        conds.append(job.curve_cond)
        targets.append(job.curve_force_n)
        d = torch.tensor(
            np.asarray(job.curve_disp), dtype=torch.float32)
        disps.append(d)
        groups.append(torch.full(
            (len(d),), gid, dtype=torch.long))
        f = np.asarray(job.curve_force, dtype=np.float64)
        ip = int(np.argmax(f))
        peak = max(float(f[ip]), 1e-8)
        post = ((np.arange(len(f)) > ip) &
                (f <= 0.02 * peak))
        posts.append(torch.tensor(post, dtype=torch.bool))
    if not conds:
        return None
    return (
        torch.cat(conds, 0).to(device),
        torch.cat(targets, 0).to(device),
        torch.cat(disps, 0).to(device),
        torch.cat(groups, 0).to(device),
        torch.cat(posts, 0).to(device),
    )

def force_stage_losses(pred, target, disp, group_id, post_mask, cfg):
    beta = float(cfg.get('FORCE_LOSS_HUBER_BETA', 0.05))
    point_loss = nn.functional.smooth_l1_loss(
        pred, target, beta=beta, reduction='none')
    weights = torch.ones_like(point_loss)
    d1_losses, d2_losses, ripple_losses = [], [], []
    drop_d1_losses, peak_losses = [], []
    elastic_slope_losses, elastic_curv_losses, elastic_anchor_losses = [], [], []

    for gid in torch.unique(group_id):
        idx = torch.where(group_id == gid)[0]
        p, y, d = pred[idx], target[idx], disp[idx]
        order = torch.argsort(d)
        idx = idx[order]
        p, y, d = p[order], y[order], d[order]
        if len(p) < 2:
            continue

        peak_y, peak_idx = torch.max(y, dim=0)
        peak_losses.append(nn.functional.smooth_l1_loss(
            torch.max(p), peak_y, beta=beta))

        dn = (d - d[0]) / (d[-1] - d[0]).clamp(min=1e-8)
        early = dn <= float(cfg.get('FORCE_FINETUNE_EARLY_RATIO', 0.18))
        if cfg.get('FORCE_FINETUNE_REGION_BALANCE', True):
            weights[idx[early]] *= float(
                cfg.get('FORCE_FINETUNE_EARLY_WEIGHT', 2.80))

        early_idx = torch.where(early)[0]
        if len(early_idx) >= 3:
            pe, ye, de = p[early_idx], y[early_idx], d[early_idx]
            dde = (de[1:] - de[:-1]).clamp(min=1e-8)
            sp = (pe[1:] - pe[:-1]) / dde
            sy = (ye[1:] - ye[:-1]) / dde
            slope_scale = (peak_y / (d[-1] - d[0]).clamp(min=1e-8)).abs().clamp(min=1e-6)
            elastic_slope_losses.append(nn.functional.smooth_l1_loss(
                sp / slope_scale, sy / slope_scale, beta=beta))
            elastic_anchor_losses.append(nn.functional.smooth_l1_loss(
                pe, ye, beta=beta))
            if len(pe) >= 4:
                elastic_curv_losses.append(nn.functional.smooth_l1_loss(
                    (sp[1:] - sp[:-1]) / slope_scale,
                    (sy[1:] - sy[:-1]) / slope_scale,
                    beta=beta))

        dd = (d[1:] - d[:-1]).clamp(min=1e-8)
        slope_y = (y[1:] - y[:-1]) / dd
        after_peak = torch.arange(len(slope_y), device=y.device) >= peak_idx
        neg_mag = torch.where(
            after_peak, (-slope_y).clamp(min=0.0), torch.zeros_like(slope_y))
        if torch.any(neg_mag > 0):
            q = torch.quantile(neg_mag[neg_mag > 0], 0.65)
            drop_edge = neg_mag >= q
            drop_node = torch.zeros_like(y, dtype=torch.bool)
            drop_node[:-1] |= drop_edge
            drop_node[1:] |= drop_edge
            weights[idx[drop_node]] *= float(
                cfg.get('FORCE_FINETUNE_DROP_WEIGHT', 2.50))
            dp, dy = p[1:] - p[:-1], y[1:] - y[:-1]
            if drop_edge.any():
                drop_d1_losses.append(nn.functional.smooth_l1_loss(
                    dp[drop_edge], dy[drop_edge], beta=beta))

        post_local = post_mask[idx]
        if post_local.any():
            weights[idx[post_local]] *= float(
                cfg.get('FORCE_FINETUNE_POST_VALUE_WEIGHT', 2.0))

        if len(p) >= 5:
            dp, dy = p[1:] - p[:-1], y[1:] - y[:-1]
            d2p = p[2:] - 2.0 * p[1:-1] + p[:-2]
            d2y = y[2:] - 2.0 * y[1:-1] + y[:-2]
            d1_losses.append(nn.functional.smooth_l1_loss(dp, dy, beta=beta))
            d2_losses.append(nn.functional.smooth_l1_loss(d2p, d2y, beta=beta))
            kernel = p.new_tensor([-2., 3., 6., 7., 6., 3., -2.]) / 21.0
            pp = nn.functional.pad(p.view(1, 1, -1), (3, 3), mode='replicate')
            yy = nn.functional.pad(y.view(1, 1, -1), (3, 3), mode='replicate')
            ps = nn.functional.conv1d(pp, kernel.view(1, 1, -1)).view(-1)
            ys = nn.functional.conv1d(yy, kernel.view(1, 1, -1)).view(-1)
            ripple_losses.append(nn.functional.smooth_l1_loss(
                p - ps, y - ys, beta=beta))

    value = (weights * point_loss).sum() / weights.sum().clamp(min=1.0)
    zero = pred.new_zeros(())
    d1 = torch.stack(d1_losses).mean() if d1_losses else zero
    d2 = torch.stack(d2_losses).mean() if d2_losses else zero
    ripple = torch.stack(ripple_losses).mean() if ripple_losses else zero
    drop_d1 = torch.stack(drop_d1_losses).mean() if drop_d1_losses else zero
    peak = torch.stack(peak_losses).mean() if peak_losses else zero
    elastic_slope = (torch.stack(elastic_slope_losses).mean()
                     if elastic_slope_losses else zero)
    elastic_curv = (torch.stack(elastic_curv_losses).mean()
                    if elastic_curv_losses else zero)
    elastic_anchor = (torch.stack(elastic_anchor_losses).mean()
                      if elastic_anchor_losses else zero)
    post = (pred[post_mask] ** 2).mean() if post_mask.any() else zero

    total = (
        value
        + cfg.get('FORCE_FINETUNE_D1_WEIGHT', 0.16) * d1
        + cfg.get('FORCE_FINETUNE_D2_WEIGHT', 0.06) * d2
        + cfg.get('FORCE_FINETUNE_RIPPLE_WEIGHT', 0.10) * ripple
        + cfg.get('FORCE_FINETUNE_POST_WEIGHT', 0.50) * post
        + cfg.get('FORCE_FINETUNE_PEAK_WEIGHT', 0.10) * peak
        + cfg.get('FORCE_FINETUNE_DROP_D1_WEIGHT', 0.25) * drop_d1
        + cfg.get('FORCE_FINETUNE_ELASTIC_SLOPE_WEIGHT', 0.45) * elastic_slope
        + cfg.get('FORCE_FINETUNE_ELASTIC_CURV_WEIGHT', 0.10) * elastic_curv
        + cfg.get('FORCE_FINETUNE_ELASTIC_ANCHOR_WEIGHT', 0.25) * elastic_anchor)
    return total, {
        'value': float(value.detach().cpu()),
        'd1': float(d1.detach().cpu()),
        'd2': float(d2.detach().cpu()),
        'ripple': float(ripple.detach().cpu()),
        'post': float(post.detach().cpu()),
        'peak': float(peak.detach().cpu()),
        'drop_d1': float(drop_d1.detach().cpu()),
        'elastic_slope': float(elastic_slope.detach().cpu()),
        'elastic_curv': float(elastic_curv.detach().cpu()),
        'elastic_anchor': float(elastic_anchor.detach().cpu()),
    }

def finetune_force_branch(model, train_jobs, val_jobs, cfg, stats, shared,
                          best_field_ckpt, final_ckpt):
    model.load_state_dict(torch.load(
        best_field_ckpt, map_location=cfg['DEVICE']))

    force_ids = {id(p) for p in model.force_parameters()}
    frozen_snapshot = {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if id(p) not in force_ids
    }
    set_force_only_trainable(model, True)

    optimizer = optim.AdamW(
        model.force_parameters(),
        lr=float(cfg.get('FORCE_FINETUNE_LR', 2.5e-4)),
        weight_decay=float(cfg.get('FORCE_FINETUNE_WEIGHT_DECAY', 1e-6)))
    force_sched = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=int(cfg.get('FORCE_FINETUNE_LR_PATIENCE', 8)),
        min_lr=float(cfg.get('FORCE_FINETUNE_MIN_LR', 2e-5)))
    best_score = float('inf')
    best_epoch = 0
    bad = 0
    force_ckpt = os.path.join(
        cfg['CKPT_DIR'], 'best_force_finetuned_model.pt')
    rows = []

    batch_jobs = max(1, int(cfg.get('FORCE_FINETUNE_BATCH_JOBS', 8)))
    total_epochs = int(cfg.get('FORCE_FINETUNE_EPOCHS', 80))

    for epoch in range(1, total_epochs + 1):
        model.train()
        order = list(train_jobs)
        random.shuffle(order)
        losses = []

        for i in range(0, len(order), batch_jobs):
            jb = order[i:i + batch_jobs]
            packed = _full_force_batch(jb, cfg['DEVICE'])
            if packed is None:
                continue
            cond, target, disp, gid, post = packed
            optimizer.zero_grad(set_to_none=True)
            pred = model.force_from_cond(cond)
            loss, parts = force_stage_losses(
                pred, target, disp, gid, post, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.force_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        vm = evaluate_dense_force_selection(
            model, val_jobs, cfg, stats, shared,
            points_override=int(
                cfg.get('FORCE_FINETUNE_EVAL_POINTS', 301)))
        score = vm['curve_score']
        force_sched.step(score)
        rows.append({
            'epoch': epoch,
            'train_force_loss': float(np.mean(losses)) if losses else np.nan,
            **vm
        })
        pd.DataFrame(rows).to_csv(
            os.path.join(cfg['CKPT_DIR'],
                         'force_finetune_history.csv'),
            index=False)

        print(
            '  [ForceStage] {}/{} loss={:.5f} score={:.6f} '
            'R2={:.5f} RMSE={:.1f}N worstRel={:.4f} ripple={:.5f} elastic={:.5f}'
            .format(
                epoch, total_epochs,
                float(np.mean(losses)) if losses else float('nan'),
                score, vm['dense_force_r2'],
                vm['dense_force_rmse'],
                vm['dense_force_worst_rel_l2'],
                vm['dense_ripple_rmse'],
                vm['dense_elastic_rmse_frac']) +
            ' LR={:.1e}'.format(optimizer.param_groups[0]['lr']))

        if score < best_score:
            best_score = score
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), force_ckpt)
            with open(os.path.join(
                    cfg['CKPT_DIR'],
                    'best_force_finetune_metrics.json'),
                    'w', encoding='utf-8') as f:
                json.dump(
                    {'epoch': epoch, **vm},
                    f, indent=2, ensure_ascii=False)
        else:
            bad += 1
            if bad >= int(cfg.get(
                    'FORCE_FINETUNE_PATIENCE', 16)):
                print('  [ForceStage EarlyStop] best epoch={}'
                      .format(best_epoch))
                break

    if os.path.exists(force_ckpt):
        shutil.copy2(force_ckpt, final_ckpt)
        model.load_state_dict(torch.load(
            final_ckpt, map_location=cfg['DEVICE']))
    else:
        shutil.copy2(best_field_ckpt, final_ckpt)
        model.load_state_dict(torch.load(
            final_ckpt, map_location=cfg['DEVICE']))

    max_field_change = 0.0
    for name, p in model.named_parameters():
        if name in frozen_snapshot:
            diff = float(
                (p.detach().cpu() - frozen_snapshot[name])
                .abs().max().item())
            max_field_change = max(max_field_change, diff)
    print('  [FieldFreezeCheck] max field parameter change={:.3e}'
          .format(max_field_change))
    if max_field_change > 1e-12:
        raise RuntimeError(

    set_force_only_trainable(model, False)
    return best_epoch, best_score

def load_v41_primary_warmstart(model, path, device):
    if not path or not os.path.exists(path):
        return 0
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    own = model.state_dict()
    prefixes = ('node_enc.', 'edge_enc.', 'cond_enc.', 'layers.',
                'stress_head.', 'disp_head.')
    matched = {
        k: v for k, v in state.items()
        if k.startswith(prefixes) and k in own and own[k].shape == v.shape
    }
    own.update(matched)
    model.load_state_dict(own, strict=True)
        len(matched), path))
    return len(matched)

def configure_manual_fracture_drop(cfg, max_disp):
    mode = str(cfg.get('FORCE_DISPLAY_DROP_MODE', 'manual')).strip().lower()
    if mode != 'manual':
        return _manual_drop_value(cfg)

    current = _manual_drop_value(cfg)
    if not cfg.get('ASK_MANUAL_FRACTURE_DISP_AFTER_TRAIN', True):
        if current is None:
        else:
        return current

    if current is not None:
    try:
    except EOFError:
        return current

    if raw.lower() == 'auto':
        cfg['FORCE_DISPLAY_DROP_MODE'] = 'auto'
        cfg['FORCE_DISPLAY_MANUAL_DROP_DISP'] = None
        value = None
    elif raw == '' or raw.lower() in ('none', 'off', 'no'):
        cfg['FORCE_DISPLAY_DROP_MODE'] = 'manual'
        cfg['FORCE_DISPLAY_MANUAL_DROP_DISP'] = None
        value = None
    else:
        try:
            value = float(raw)
        except ValueError:
            return current
        if not np.isfinite(value) or value <= 0.0 or value > float(max_disp):
            return current
        cfg['FORCE_DISPLAY_DROP_MODE'] = 'manual'
        cfg['FORCE_DISPLAY_MANUAL_DROP_DISP'] = float(value)

    try:
        payload = {
            'mode': str(cfg.get('FORCE_DISPLAY_DROP_MODE', 'manual')),
            'manual_drop_disp_mm': cfg.get('FORCE_DISPLAY_MANUAL_DROP_DISP', None),
            'note': 'display-only; does not alter training, metrics, raw CSV or checkpoint'
        }
        with open(os.path.join(cfg['CKPT_DIR'],
                               cfg.get('MANUAL_FRACTURE_DISP_FILE',
                                       'manual_fracture_drop.json')),
                  'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
    return value

def main():
    set_global_seed(CFG.get('TRAIN_SEED', 42), CFG.get('DETERMINISTIC_TRAINING', False))
    print('=' * 64)
    print('=' * 64)

    stats_path = os.path.join(CFG['CKPT_DIR'], 'stats.json')
    current_sig = dataset_signature(CFG['DATA_DIR'])
    use_cache = False
    if os.path.exists(stats_path):
        old_stats = load_stats(stats_path)
        use_cache = ((not CFG['RECOMPUTE_STATS_IF_DATA_CHANGED'] or
                      old_stats.get('dataset_signature') == current_sig) and
                     int(old_stats.get('stats_version', -1)) == int(CFG['STATS_VERSION']))
        if use_cache:
            stats = old_stats
        else:
    if not use_cache:
        stats = compute_stats(CFG['DATA_DIR'], CFG)
        save_stats(stats, stats_path)
    print('  stress mean/std={:.3f}/{:.3f} | force scale={:.1f} | max_disp={:.4f} | Eyy=OFF'
          .format(stats['stress_mean'], stats['stress_std'],
                  stats['force_std'], stats['max_load_disp']))

    jobs_dict = group_files_by_job(CFG['DATA_DIR'])
    job_ids = sorted(jobs_dict.keys())
    if not job_ids:
    random.Random(CFG['SPLIT_SEED']).shuffle(job_ids)
    nv, nt = resolve_split_counts(len(job_ids), CFG)
    val_ids, test_ids, train_ids = job_ids[:nv], job_ids[nv:nv + nt], job_ids[nv + nt:]
    print('  Split: train {} / val {} / test {}'.format(
        len(train_ids), len(val_ids), len(test_ids)))
    if len(job_ids) < CFG['N_VAL_JOBS'] + CFG['N_TEST_JOBS'] + CFG['MIN_TRAIN_JOBS']:
    if not train_ids:

    shared = SharedTopology(jobs_dict[job_ids[0]][0], stats)

    print('--- Loading data ---')
    train_jobs = [JobGroup(jobs_dict[j], stats, CFG, shared) for j in tqdm(train_ids, desc='[Train]')]
    val_jobs = [JobGroup(jobs_dict[j], stats, CFG, shared) for j in tqdm(val_ids, desc='[Val]')]
    test_jobs = [JobGroup(jobs_dict[j], stats, CFG, shared) for j in tqdm(test_ids, desc='[Test]')]

    train_dead_labels = int(sum(float((fr['alive'] < 0.5).sum())
                                for job in train_jobs for fr in job.frames))
    val_dead_labels = int(sum(float((fr['alive'] < 0.5).sum())
                              for job in val_jobs for fr in job.frames))
          .format(train_dead_labels, val_dead_labels))
    if train_dead_labels == 0:
              'Please ensure NPZ contains surface_tri_alive.')

    in_cond = cond_dim(CFG)
    model = GNNModel(in_n=shared.in_node_dim, in_e=4, in_cond=in_cond, h=CFG['HIDDEN'],
                     layers=CFG['LAYERS'], cd=CFG['COND_DIM'],
                     tf_layers=CFG['TF_LAYERS'], tf_heads=CFG['TF_HEADS'],
                     tf_dropout=CFG['TF_DROPOUT'],
                     max_len=max(64, CFG['FRAMES_PER_JOB'] + 4),
                     query_mode=CFG['QUERY_MODE'],
                     force_fourier_k=CFG['DISP_FOURIER_K'],
                     force_mat_dim=CFG['N_MATERIAL_PARAMS'] if CFG['USE_MATERIAL_COND'] else 0,
                     force_residual_limit=CFG['FORCE_RESIDUAL_LIMIT'],
                     force_dcrit_min=CFG['FORCE_DCRIT_MIN'],
                     force_dcrit_max=CFG['FORCE_DCRIT_MAX'],
                     force_sharp_min=CFG['FORCE_SHARPNESS_MIN'],
                     force_sharp_max=CFG['FORCE_SHARPNESS_MAX'],
                     eyy_aux_layers=CFG['EYY_AUX_LAYERS'],
                     enable_eyy=CFG.get('ENABLE_EYY', False),
                     force_fast_rate_min=CFG['FORCE_FAST_RATE_MIN'],
                     force_fast_rate_max=CFG['FORCE_FAST_RATE_MAX'],
                     force_slow_rate_min=CFG['FORCE_SLOW_RATE_MIN'],
                     force_slow_rate_max=CFG['FORCE_SLOW_RATE_MAX']).to(CFG['DEVICE'])
    load_v41_primary_warmstart(
        model, str(CFG.get('V41_PRIMARY_WARMSTART_PATH', '')).strip(),
        CFG['DEVICE'])
    if CFG['USE_COMPILE'] and hasattr(torch, 'compile'):
        model = torch.compile(model)
        print('  torch.compile enabled')
    print('  Mode: {} | Cond dim {} | Params {:,}'.format(
        in_cond, sum(p.numel() for p in model.parameters())))
    print('  [Force-v10.14.1-exactV41-field] dual-rate elastic basis K={} | dense={}pts/batch | max graphs={} | '
          'slope/curve/post={:.3f}/{:.3f}/{:.2f}'.format(
              CFG['DISP_FOURIER_K'], CFG['N_DENSE_DISP'], CFG['MAX_DENSE_GRAPHS'],
              CFG.get('W_FORCE_SLOPE', 0.0), CFG.get('W_FORCE_CURVATURE', 0.0),
              CFG.get('W_POST_FRACTURE_ZERO_FORCE', 0.0)))

    if CFG.get('TRAIN_LEARNED_CRACK_HEAD', False):
        stage1_params = list(model.parameters())
    else:
        blocked = ('dead_head.', 'damage_head.', 'fracture_head.')
        stage1_params = [
            p for name, p in model.named_parameters()
            if not name.startswith(blocked)
        ]
        for name, p in model.named_parameters():
            if name.startswith(blocked):
                p.requires_grad_(False)
        print('  [ExactV4.1Field] crack/Eyy disabled; Stage1 trains stress/disp + V4.1 guide-force.')
    optimizer = optim.AdamW(stage1_params, lr=CFG['LR'], weight_decay=1e-5)
    scaler = make_scaler(CFG['USE_AMP'] and CFG['DEVICE'] == 'cuda')
    ckpt = os.path.join(CFG['CKPT_DIR'], 'best_model.pt')
    surf = get_front_surface(shared, CFG)

    best, best_epoch, bad, stopped = float('inf'), 0, 0, False
    best_field, best_field_epoch = float('inf'), 0
    best_force, best_force_epoch = float('inf'), 0
    best_field_ckpt = os.path.join(CFG['CKPT_DIR'], 'best_field_model.pt')
    best_force_ckpt = os.path.join(CFG['CKPT_DIR'], 'best_force_model.pt')
    last_dense_metrics = None
    curve_candidates = []
    viz_job = select_visualization_job(val_jobs, CFG)
        viz_job.job_name))
    p1, s0, s1 = CFG['PHASE1_EPOCHS'], CFG['PHASE2_SCHED_START'], CFG['PHASE2_SCHED_END']
    dropped = False
    history_rows = []
    diag_pairs = []
    train_diag_jobs = train_jobs[:max(1, min(len(train_jobs), int(CFG['TRAIN_DIAG_JOBS'])))]

    clean_field_path = str(CFG.get('CLEAN_FIELD_CKPT_PATH', '')).strip()
    use_external_clean_field = bool(
        clean_field_path and os.path.exists(clean_field_path))
    resume_force_only = bool(
        CFG.get('RESUME_FORCE_STAGE_ONLY', False) and
        os.path.exists(best_field_ckpt))

    if use_external_clean_field:
        state = torch.load(clean_field_path, map_location=CFG['DEVICE'])
        model.load_state_dict(state)
        torch.save(model.state_dict(), best_field_ckpt)
        best_field_epoch = int(CFG.get(
            'RESUME_FIELD_EPOCH_HINT', CFG['EPOCHS']))
        best_epoch = best_field_epoch
              .format(clean_field_path))
        field_epoch_iter = []
    elif resume_force_only:
        model.load_state_dict(torch.load(
            best_field_ckpt, map_location=CFG['DEVICE']))
        best_field_epoch = int(CFG.get(
            'RESUME_FIELD_EPOCH_HINT', CFG['EPOCHS']))
        best_epoch = best_field_epoch
        print('  [Resume] loaded best_field_model.pt; skip Stage1, continue Stage2.')
        field_epoch_iter = []
    else:
        field_epoch_iter = range(1, CFG['EPOCHS'] + 1)

    for epoch in field_epoch_iter:
        if CFG['QUERY_MODE']:
            sched_p = 0.0

        elif epoch <= p1:
            sched_p = 0.0
        elif epoch <= s1:
            sched_p = (epoch - s0) / float(s1 - s0) * 0.5
        else:
            sched_p = 0.5

        tr = run_epoch(model, train_jobs, CFG, shared, optimizer, sched_p, scaler)
        tl = tr['loss']
        vm = evaluate(model, val_jobs, CFG, stats, shared)

        dense_vm = None
        dense_every = max(1, int(CFG.get('DENSE_SELECT_EVERY', 5)))
        if (epoch == 1 or epoch % dense_every == 0 or
                epoch == CFG['EPOCHS']):
            dense_vm = evaluate_dense_force_selection(
                model, val_jobs, CFG, stats, shared)
            if dense_vm is not None:
                last_dense_metrics = dense_vm
                print(
                    '  [DenseCurve] score={:.6f} R2={:.5f} '
                    'RMSE={:.1f}N worstRel={:.4f} '
                    'slope={:.5f} curve={:.5f} ripple={:.5f} elastic={:.5f}'
                    .format(
                        dense_vm['curve_score'],
                        dense_vm['dense_force_r2'],
                        dense_vm['dense_force_rmse'],
                        dense_vm['dense_force_worst_rel_l2'],
                        dense_vm['dense_slope_rmse'],
                        dense_vm['dense_curvature_rmse'],
                        dense_vm['dense_ripple_rmse'],
                        dense_vm['dense_elastic_rmse_frac']
                    )
                )
                curve_candidates = _save_curve_candidate(
                    model, epoch, dense_vm, vm,
                    CFG, curve_candidates
                )

        if epoch == CFG['PHASE2_LR_DROP'] and not dropped:
            for g in optimizer.param_groups:
                g['lr'] = CFG['PHASE2_LR']
            dropped = True
            print('  [LR] dropped to {:.2e}'.format(CFG['PHASE2_LR']))

        tm = None
        if epoch == 1 or epoch % int(CFG['TRAIN_DIAG_EVERY']) == 0:
            tm = evaluate(model, train_diag_jobs, CFG, stats, shared)
            diag_pairs.append((epoch, tm['stress_rmse'], vm['stress_rmse']))
            if len(diag_pairs) >= 3:
                prev = diag_pairs[:-1]
                best_prev_train = min(v[1] for v in prev)
                best_prev_val = min(v[2] for v in prev)
                if (tm['stress_rmse'] < best_prev_train and
                        vm['stress_rmse'] > CFG['OVERFIT_WARN_RATIO'] * best_prev_val):
                          '{:.0f}%；%; check training_history.png training_history.png。'
                          .format((vm['stress_rmse'] / best_prev_val - 1.0) * 100.0))

        crack_text = (
            'deadIoU:{:.3f} precision:{:.3f} recall:{:.3f}'.format(
                vm['dead_iou'], vm['dead_precision'], vm['dead_recall'])
            if CFG.get('TRAIN_LEARNED_CRACK_HEAD', False)
            else 'crackHead:OFF (terminal display only)')
        print('E{:3d}/{} {} | L:{:.3e} | R2s:{:.4f} R2f:{:.4f} | '
              'sRMSE:{:.2f}MPa FRMSE:{:.1f}N | '
              'late:{:.2f} damageZone:{:.2f}MPa | {} | '
              'steps:{} grad:{:.2e} LR:{:.1e}'.format(
                  epoch, CFG['EPOCHS'], 'P1' if epoch <= p1 else 'P2', tl,
                  vm['r2'], vm['force_r2'], vm['stress_rmse'], vm['force_rmse'],
                  vm['late_stress_rmse'],
                  vm['crack_zone_rmse'] if np.isfinite(vm['crack_zone_rmse']) else -1.0,
                  crack_text,
                  tr['steps'], tr['grad_norm'], optimizer.param_groups[0]['lr']))

        history_rows.append({
            'epoch': epoch,
            'train_loss': tl,
            'train_stress_rmse': np.nan if tm is None else tm['stress_rmse'],
            'val_stress_rmse': vm['stress_rmse'],
            'train_eyy_rmse_percent': np.nan,
            'val_eyy_rmse_percent': np.nan,
            'train_force_rmse': np.nan if tm is None else tm['force_rmse'],
            'val_force_rmse': vm['force_rmse'],
            'val_r2_stress': vm['r2'],
            'val_r2_eyy': np.nan,
            'val_r2_force': vm['force_r2'],
            'val_dead_iou': vm['dead_iou'],
            'val_dead_precision': vm['dead_precision'],
            'val_dead_recall': vm['dead_recall'],
            'val_late_stress_rmse': vm['late_stress_rmse'],
            'val_crack_zone_rmse': vm['crack_zone_rmse'],
            'dense_curve_score': np.nan if dense_vm is None else dense_vm['curve_score'],
            'dense_force_rmse': np.nan if dense_vm is None else dense_vm['dense_force_rmse'],
            'dense_force_r2': np.nan if dense_vm is None else dense_vm['dense_force_r2'],
            'dense_worst_rel_l2': np.nan if dense_vm is None else dense_vm['dense_force_worst_rel_l2'],
            'dense_min_job_r2': np.nan if dense_vm is None else dense_vm['dense_min_job_r2'],
            'dense_slope_rmse': np.nan if dense_vm is None else dense_vm['dense_slope_rmse'],
            'dense_curvature_rmse': np.nan if dense_vm is None else dense_vm['dense_curvature_rmse'],
            'dense_ripple_rmse': np.nan if dense_vm is None else dense_vm['dense_ripple_rmse'],
            'lr': optimizer.param_groups[0]['lr'],
        })
        save_training_history(history_rows, CFG['CKPT_DIR'])

        if vm['selection_score'] < best_field:
            best_field = vm['selection_score']
            best = best_field
            best_field_epoch = epoch
            best_epoch = epoch
            bad = 0
            torch.save(model.state_dict(), best_field_ckpt)
            print('  [Save-Field] epoch={} field={:.6f}'
                  .format(epoch, best_field))
        elif epoch >= CFG['MIN_EPOCHS']:
            bad += 1

        if os.path.exists(best_field_ckpt):
            shutil.copy2(best_field_ckpt, ckpt)

        if epoch >= CFG['MIN_EPOCHS'] and bad >= CFG['PATIENCE']:
            print('  [Early Stop] epoch {}, best composite={:.4f}'.format(epoch, best))
            stopped = True
            break

        if epoch % CFG['SAVE_EVERY_EPOCHS'] == 0:
            visualize_job(model, viz_job, CFG, stats, shared, surf,
                          os.path.join(CFG['CKPT_DIR'], 'viz_e{:03d}.png'.format(epoch)))
            if CFG.get('ENABLE_EYY', False) and CFG.get('SAVE_DIC_VIZ', False):
                visualize_dic_job(model, viz_job, CFG, stats, shared, surf,
                                  os.path.join(CFG['CKPT_DIR'], 'viz_e{:03d}_dic.png'.format(epoch)))
            model.train()

    if not os.path.exists(best_field_ckpt):

    if CFG.get('FORCE_FINETUNE', True):
        force_best_epoch, force_best_score = finetune_force_branch(
            model, train_jobs, val_jobs, CFG, stats, shared,
            best_field_ckpt, ckpt)
        best_source = 'best_field + force_finetune'
    else:
        shutil.copy2(best_field_ckpt, ckpt)
        model.load_state_dict(torch.load(
            ckpt, map_location=CFG['DEVICE']))
        force_best_epoch, force_best_score = 0, float('nan')
        best_source = 'best_field_only'

    best_epoch = best_field_epoch
    if resume_force_only:
        print('Stage1 Using existing best_field_model.pt。')
    elif not stopped:
        print('Completed {} epochs'.format(CFG['EPOCHS']))

    if (CFG.get('TRAIN_LEARNED_CRACK_HEAD', False) and
            CFG.get('CRACK_THRESHOLD_CALIBRATE', True)):
        calibrate_crack_thresholds(model, val_jobs, CFG, shared, save=True)

    configure_manual_fracture_drop(CFG, stats['max_load_disp'])

    best_viz = os.path.join(
        CFG['CKPT_DIR'], 'viz_best_e{:03d}.png'.format(best_epoch))
    visualize_job(model, viz_job, CFG, stats, shared, surf, best_viz)
    visualize_job(model, viz_job, CFG, stats, shared, surf,
                  os.path.join(CFG['CKPT_DIR'], 'viz_best.png'))
    if CFG.get('ENABLE_EYY', False) and CFG.get('SAVE_DIC_VIZ', False):
        visualize_dic_job(model, viz_job, CFG, stats, shared, surf,
                          os.path.join(CFG['CKPT_DIR'], 'viz_best_dic.png'))
        visualize_dic_job(model, viz_job, CFG, stats, shared, surf,
                          os.path.join(CFG['CKPT_DIR'], 'viz_best_e{:03d}_dic.png'.format(best_epoch)))
    print(
        'source={}；source={}; result figure={}={}'
        .format(best_field_epoch, force_best_epoch,
                best_source, best_viz)
    )

    model.load_state_dict(torch.load(
        best_field_ckpt, map_location=CFG['DEVICE']))
    visualize_job(
        model, viz_job, CFG, stats, shared, surf,
        os.path.join(CFG['CKPT_DIR'],
                     'viz_best_field_e{:03d}.png'.format(best_field_epoch)))
    model.load_state_dict(torch.load(
        ckpt, map_location=CFG['DEVICE']))

    model.load_state_dict(torch.load(ckpt))
    print('\n--- Validation set ---')
    final_report(model, val_jobs, CFG, stats, shared, CFG['CKPT_DIR'], 'val')
    final_report(model, test_jobs, CFG, stats, shared, CFG['CKPT_DIR'], 'test')
    for j, job in enumerate(test_jobs[:3]):
        visualize_job(model, job, CFG, stats, shared, surf,
                      os.path.join(CFG['CKPT_DIR'], 'viz_test{}.png'.format(j)))

    if CFG['QUERY_DEMO'] and CFG['QUERY_MODE']:
        print('\n--- Query demo (arbitrary displacement -> force + contour) ---')
        job = test_jobs[0]
        mat_raw = job.mat_n * stats['mat_std'] + stats['mat_mean']
        gt = (job.curve_disp, job.curve_force) if job.curve_disp is not None else None
        curve_cache = export_curve(
            model, shared, stats, CFG, mat_raw, CFG['CKPT_DIR'],
            tag=job.job_name, gt=gt)
        for dv in CFG['QUERY_DEMO_DISPS']:
            export_query_report(
                model, shared, stats, CFG, surf, mat_raw, dv,
                os.path.join(CFG['CKPT_DIR'],
                             'query_report_{}_d{:.2f}.png'.format(job.job_name, dv)),
                gt_curve=gt, curve_cache=curve_cache, gt_job=job)

    if CFG['INTERACTIVE_QUERY_AFTER_TRAIN'] and CFG['QUERY_MODE']:
        job = test_jobs[0] if test_jobs else val_jobs[0]
        mat_raw = job.mat_n * stats['mat_std'] + stats['mat_mean']
        gt_curve = (job.curve_disp, job.curve_force) if job.curve_disp is not None else None
        dense_disp = np.linspace(0.0, float(stats['max_load_disp']), CFG['QUERY_CURVE_N'])
        curve_cache = query(model, shared, stats, CFG, dense_disp, mat_raw)
        while True:
            try:
            except EOFError:
                break
            if text.lower() in ('q', 'quit', 'exit', ''):
                break
            if text.lower().startswith('drop'):
                parts = text.replace(',', ' ').split()
                if len(parts) != 2:
                    print('  Usage: drop 5.53 or drop none')
                    continue
                if parts[1].lower() in ('none', 'off', 'no'):
                    CFG['FORCE_DISPLAY_DROP_MODE'] = 'manual'
                    CFG['FORCE_DISPLAY_MANUAL_DROP_DISP'] = None
                elif parts[1].lower() == 'auto':
                    CFG['FORCE_DISPLAY_DROP_MODE'] = 'auto'
                    CFG['FORCE_DISPLAY_MANUAL_DROP_DISP'] = None
                else:
                    try:
                        drop_value = float(parts[1])
                    except ValueError:
                        continue
                    if drop_value <= 0.0 or drop_value > float(stats['max_load_disp']):
                            float(stats['max_load_disp'])))
                        continue
                    CFG['FORCE_DISPLAY_DROP_MODE'] = 'manual'
                    CFG['FORCE_DISPLAY_MANUAL_DROP_DISP'] = drop_value
                continue
            try:
                vals = [float(x) for x in text.replace(',', ' ').split()]
            except ValueError:
                continue
            rr = query(model, shared, stats, CFG, vals, mat_raw)
            for i, dv in enumerate(rr['disp']):
                outp = os.path.join(
                    CFG['CKPT_DIR'], 'manual_query_report_d{:.3f}.png'.format(dv))
                rep = export_query_report(
                    model, shared, stats, CFG, surf, mat_raw, dv, outp,
                    gt_curve=gt_curve, curve_cache=curve_cache, gt_job=job)
                print('  D={:.3f} mm -> F={:.2f} N, max stress={:.2f} MPa, '
                      'Pfract={:.3f}, dead={} | stressRMSE={:.3f}MPa | Eyy=OFF'
                      .format(dv, rr['force'][i], float(rr['stress'][i].max()),
                              float(rr['fracture_prob'][i]),
                              int((rr['alive'][i] < 0.5).sum()),
                              rep['stress']['rmse']))

    print('\nAll done. Results in: {}'.format(CFG['CKPT_DIR']))

def load_validation_jobs_for_calibration(cfg, shared, stats):
    jobs_dict = group_files_by_job(cfg['DATA_DIR'])
    ids = sorted(jobs_dict.keys())
    random.Random(cfg['SPLIT_SEED']).shuffle(ids)
    nv, nt = resolve_split_counts(len(ids), cfg)
    val_ids = ids[:nv]
    return [JobGroup(jobs_dict[j], stats, cfg, shared)
            for j in tqdm(val_ids, desc='[CalibVal]')]

def load_closest_reference_job(cfg, shared, stats, mat_raw):
    jobs_dict = group_files_by_job(cfg['DATA_DIR'])
    best_id = None
    best_dist = float('inf')
    target = np.asarray(mat_raw, dtype=np.float64).reshape(-1)
    for jid, files in jobs_dict.items():
        fp = sorted(files, key=lambda x: int(re.search(r'frame(\d+)', x).group(1)))[0]
        d = np.load(fp, allow_pickle=True)
        mp = np.asarray(d['material_params'], dtype=np.float64).reshape(-1) \
            if 'material_params' in d else np.array([1.0, .30, 1.0, 1.0, 1.0])
        scale = np.maximum(np.asarray(stats['mat_std'], dtype=float), 1e-6)
        dist = float(np.linalg.norm((mp - target) / scale))
        if dist < best_dist:
            best_dist = dist
            best_id = jid
    if best_id is None:
        return None
    return JobGroup(jobs_dict[best_id], stats, cfg, shared)

def cli_query(args):
    """
    Example:
      python 03_train_gnn_dualhead.py --query 5.2
      python 03_train_gnn_dualhead.py --query 1.0 3.0 5.2 --material 1.02 0.30 0.95 1.10 1.03
      python 03_train_gnn_dualhead.py --curve   --material 1.0 0.3 1.0 1.0 1.0
    """
    model, shared, stats = load_for_inference(CFG)
    surf = get_front_surface(shared, CFG)
    mat = np.array(args.material, dtype=np.float64) if args.material else \
        np.array([1.0, 0.30, 1.0, 1.0, 1.0])
    print('  Material params: E_ratio={:.4f} nu={:.4f} sy_ratio={:.4f} n_ratio={:.4f} ef_ratio={:.4f}'
          .format(*mat))

    if getattr(args, 'calibrate_crack', False):
        val_jobs = load_validation_jobs_for_calibration(CFG, shared, stats)
        calibrate_crack_thresholds(model, val_jobs, CFG, shared, save=True)

    ref_job = load_closest_reference_job(CFG, shared, stats, mat)
    gt_curve = ((ref_job.curve_disp, ref_job.curve_force)
                if ref_job is not None and ref_job.curve_disp is not None else None)
    curve_cache = None
    if args.curve or args.query:
        curve_cache = export_curve(
            model, shared, stats, CFG, mat, CFG['CKPT_DIR'], tag='cli', gt=gt_curve)

    if args.query:
        r = query(model, shared, stats, CFG, args.query, mat)
        for i, dv in enumerate(r['disp']):
            print('   {:8.3f}  {:10.1f}  {:12.2f}   {:8d}  Pfrac={:.3f}'.format(
                dv, r['force'][i], float(r['stress'][i].max()),
                int((r['alive'][i] < 0.5).sum()),
                float(r['fracture_prob'][i])))
            export_query_report(
                model, shared, stats, CFG, surf, mat, dv,
                os.path.join(CFG['CKPT_DIR'],
                             'cli_query_report_d{:.3f}.png'.format(dv)),
                curve_cache=curve_cache, gt_curve=gt_curve, gt_job=ref_job)

if __name__ == '__main__':
    import argparse
    ap.add_argument('--query', type=float, nargs='+', default=None,
    ap.add_argument('--material', type=float, nargs=5, default=None,
                    help='E_ratio nu sy_ratio n_ratio ef_ratio')
    ap.add_argument('--calibrate-crack', action='store_true',
    a = ap.parse_args()
    if a.query or a.curve or a.calibrate_crack:
        cli_query(a)
    else:
        main()
