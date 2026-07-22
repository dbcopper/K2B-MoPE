# few_shot_cellpose_mope_ranking.py
"""
MoPE-Rank: Mixture of Phenotypic Experts for Pairwise Ranking-based
Fusarium Head Blight Severity Assessment in Wheat Kernels

Key Innovation:
1. Three domain-specific experts (Color, Texture, Morphology) with adaptive gating
2. Pairwise ranking loss trained with only binary labels (healthy/diseased)
3. Continuous severity score with data-driven threshold determination
4. Interpretable expert contribution analysis per kernel
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman'],
    'axes.unicode_minus': False,
    'font.size':          12,
    'axes.titlesize':     15,
    'axes.labelsize':     14,
    'xtick.labelsize':    13,
    'ytick.labelsize':    13,
    'legend.fontsize':    13,
})
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import warnings
from itertools import product
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import color as skcolor
warnings.filterwarnings('ignore')

# ==========================================
# Configuration
# ==========================================
CELLPOSE_MODEL = "cyto"
CELLPOSE_FLOW_THRESHOLD = 0.4
CELLPOSE_FIXED_DIAMETER = 116
ROI_DOWNSAMPLE_MAX_SIDE = 1600
EXAMPLES_DIR = "./examples"
HEALTHY_EXAMPLES_DIR = os.path.join(EXAMPLES_DIR, "Healthy")
DISEASED_EXAMPLES_DIR = os.path.join(EXAMPLES_DIR, "Diseased")

# MoPE Training Config
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 300
RANKING_MARGIN = 0.3
BATCH_SIZE = 128

# Feature group definitions (indices into the 25-dim feature vector)
COLOR_INDICES = list(range(0, 11))    # 11 color features
TEXTURE_INDICES = list(range(11, 22)) # 11 texture features
MORPH_INDICES = list(range(22, 25))   # 3 morphology features

# ==========================================
# Feature Extraction (same as v2)
# ==========================================

def extract_fhb_features(seed_rgb, seed_gray, seed_mask):
    """Extract phenotypic features for FHB severity assessment"""
    features = {}

    # ========== 1. Color/Whiteness Features ==========
    lab = skcolor.rgb2lab(seed_rgb / 255.0)
    L = lab[:, :, 0][seed_mask]
    a = lab[:, :, 1][seed_mask]
    b = lab[:, :, 2][seed_mask]

    features['L_mean'] = np.mean(L)
    features['L_std'] = np.std(L)
    features['L_median'] = np.median(L)
    features['L_max'] = np.max(L)
    features['L_high_ratio'] = np.sum(L > 60) / len(L)
    features['L_very_high_ratio'] = np.sum(L > 70) / len(L)

    chroma = np.sqrt(a**2 + b**2)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)

    hsv = cv2.cvtColor(seed_rgb, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0][seed_mask]
    S = hsv[:, :, 1][seed_mask]
    V = hsv[:, :, 2][seed_mask]

    features['S_mean'] = np.mean(S)
    features['S_std'] = np.std(S)
    features['S_low_ratio'] = np.sum(S < 80) / len(S)
    features['V_mean'] = np.mean(V)
    features['V_std'] = np.std(V)
    features['H_mean'] = np.mean(H)
    features['H_std'] = np.std(H)

    R = seed_rgb[:, :, 0][seed_mask].astype(np.float32)
    G = seed_rgb[:, :, 1][seed_mask].astype(np.float32)
    B = seed_rgb[:, :, 2][seed_mask].astype(np.float32)

    total = R + G + B + 1e-10
    features['r_ratio'] = np.mean(R / total)
    features['g_ratio'] = np.mean(G / total)
    features['b_ratio'] = np.mean(B / total)
    features['rb_diff'] = np.mean((R - B) / (total / 3))

    # ========== 2. Texture/Wrinkle Features ==========
    # Fill background with seed mean to avoid false edges at boundary
    seed_mean_gray = np.mean(seed_gray[seed_mask])
    gray_filled = seed_gray.copy()
    gray_filled[~seed_mask] = int(seed_mean_gray)

    # Erode mask to exclude boundary pixels from texture sampling
    erode_kernel = np.ones((3, 3), np.uint8)
    inner_mask = cv2.erode(seed_mask.astype(np.uint8), erode_kernel, iterations=1) > 0
    # Fallback: if erosion removes too many pixels, use original mask
    if np.sum(inner_mask) < 50:
        inner_mask = seed_mask

    # Laplacian (computed on mean-filled image, sampled from inner mask)
    laplacian = cv2.Laplacian(gray_filled, cv2.CV_64F)
    lap_masked = laplacian[inner_mask]
    features['laplacian_var'] = np.var(lap_masked)
    features['laplacian_mean'] = np.mean(np.abs(lap_masked))

    # Sobel edges
    sobel_x = cv2.Sobel(gray_filled, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_filled, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_masked = sobel_mag[inner_mask]
    features['edge_mean'] = np.mean(sobel_masked)
    features['edge_std'] = np.std(sobel_masked)

    # LBP (computed on mean-filled image, sampled from inner mask)
    for radius in [1, 2]:
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_filled, P=n_points, R=radius, method='uniform')
        lbp_masked = lbp[inner_mask]
        lbp_hist, _ = np.histogram(lbp_masked, bins=n_points + 2, density=True)
        entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        features[f'lbp_entropy_r{radius}'] = entropy
        features[f'lbp_uniform_ratio_r{radius}'] = np.sum(lbp_hist[:-1])

    # GLCM (background already filled with seed mean, so transitions are smooth)
    gray_filled_scaled = (gray_filled / 16).astype(np.uint8)
    try:
        glcm = graycomatrix(gray_filled_scaled, distances=[1, 2],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                           levels=16, symmetric=True, normed=True)
        features['glcm_contrast'] = graycoprops(glcm, 'contrast').mean()
        features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity').mean()
        features['glcm_energy'] = graycoprops(glcm, 'energy').mean()
        features['glcm_correlation'] = graycoprops(glcm, 'correlation').mean()
    except:
        features['glcm_contrast'] = 0
        features['glcm_homogeneity'] = 1
        features['glcm_energy'] = 1
        features['glcm_correlation'] = 0

    # Gray statistics (only seed pixels)
    gray_values = seed_gray[seed_mask]
    features['gray_mean'] = np.mean(gray_values)
    features['gray_std'] = np.std(gray_values)
    features['gray_range'] = np.max(gray_values) - np.min(gray_values)

    # ========== 3. Morphology Features ==========
    contours, _ = cv2.findContours(seed_mask.astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter > 0:
            features['circularity'] = 4 * np.pi * area / (perimeter ** 2)
        else:
            features['circularity'] = 0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        features['solidity'] = area / (hull_area + 1e-10)

        rect = cv2.minAreaRect(cnt)
        w, h = rect[1]
        if min(w, h) > 0:
            features['aspect_ratio'] = max(w, h) / min(w, h)
        else:
            features['aspect_ratio'] = 1

        features['compactness'] = (perimeter ** 2) / (4 * np.pi * area + 1e-10)
    else:
        features['circularity'] = 0
        features['solidity'] = 0
        features['aspect_ratio'] = 1
        features['compactness'] = 1

    return features


def get_feature_vector(features):
    """Convert feature dict to vector, grouped by expert domain"""
    key_features = [
        # Color Expert (index 0-10)
        'L_mean', 'L_std', 'L_high_ratio', 'L_very_high_ratio',
        'chroma_mean', 'S_mean', 'S_low_ratio',
        'V_mean', 'r_ratio', 'g_ratio', 'rb_diff',
        # Texture Expert (index 11-21)
        'laplacian_var', 'laplacian_mean', 'edge_mean', 'edge_std',
        'lbp_entropy_r1', 'lbp_entropy_r2',
        'glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
        'gray_std', 'gray_range',
        # Morphology Expert (index 22-24)
        'circularity', 'solidity', 'aspect_ratio',
    ]
    vec = [features.get(k, 0) for k in key_features]
    return np.array(vec, dtype=np.float64), key_features


def extract_features_from_dir(examples_dir):
    """Extract features from all seed images in example directory"""
    all_features = []

    if not os.path.exists(examples_dir):
        return np.zeros((0, 25)), []

    for filename in sorted(os.listdir(examples_dir)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(examples_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = binary > 0

        if np.sum(mask) < 100:
            continue

        try:
            # mask already segments seed from black background
            # extract_fhb_features handles background via mean-filling + inner_mask
            features = extract_fhb_features(rgb, gray, mask)
            vec, names = get_feature_vector(features)
            all_features.append(vec)
        except Exception as e:
            print(f"  Warning: {filename} feature extraction failed - {e}")
            continue

    if len(all_features) == 0:
        return np.zeros((0, 25)), []

    return np.array(all_features), names


# ==========================================
# MoPE Model (Mixture of Phenotypic Experts)
# ==========================================

class PhenotypicExpert(nn.Module):
    """Single expert: maps domain-specific features to a severity score"""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class DeviationGuidedGating(nn.Module):
    """
    Feature-Deviation-Guided Gating Network.

    Gating weights are driven by a biologically motivated, parameter-free
    deviation signal:
      signal_k = mean( |x_k| )

    After z-score normalisation, healthy kernels cluster near 0 and diseased
    kernels exhibit larger absolute feature values. The expert whose feature
    group deviates most from the healthy baseline therefore receives the
    highest gating weight -- no labelled data required to compute this prior.

    A single learnable temperature T controls the sharpness of expert
    selection (T → 0: hard assignment to one expert; T → ∞: equal weights).
    This is the only trained parameter in the gating module, minimising
    the risk of overfitting on small reference sets.

    gate = softmax( mean|x_k| / T )
    """
    def __init__(self, input_dim=25):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)

    def forward(self, color_feats, texture_feats, morph_feats, x_full):
        return torch.softmax(self.fc(x_full), dim=-1)


class MoPERanker(nn.Module):
    """
    Mixture of Phenotypic Experts for FHB severity ranking.

    Three domain experts (Color / Texture / Morphology) each map their
    respective feature group to a scalar severity score via a linear layer
    + sigmoid.  A DeviationGuidedGating network combines their scores using
    weights proportional to the per-group feature deviation from the healthy
    baseline, refined by a small learned adjustment.
    """
    def __init__(self, color_dim=11, texture_dim=11, morph_dim=3):
        super().__init__()
        self.expert_color   = PhenotypicExpert(color_dim)
        self.expert_texture = PhenotypicExpert(texture_dim)
        self.expert_morph   = PhenotypicExpert(morph_dim)

        total_dim = color_dim + texture_dim + morph_dim
        self.gating = DeviationGuidedGating(total_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, 25] z-score normalised feature vector
        Returns:
            severity:      [batch, 1] continuous severity score in (0, 1)
            gates:         [batch, 3] gating weights (color, texture, morph)
            expert_scores: [batch, 3] individual expert scores
        """
        color_feats   = x[:, COLOR_INDICES]
        texture_feats = x[:, TEXTURE_INDICES]
        morph_feats   = x[:, MORPH_INDICES]

        s_color   = self.expert_color(color_feats)
        s_texture = self.expert_texture(texture_feats)
        s_morph   = self.expert_morph(morph_feats)

        expert_scores = torch.cat([s_color, s_texture, s_morph], dim=-1)  # [batch, 3]

        gates = self.gating(color_feats, texture_feats, morph_feats, x)   # [batch, 3]

        severity = (gates * expert_scores).sum(dim=-1, keepdim=True)       # [batch, 1]

        return severity, gates, expert_scores


# ==========================================
# Pairwise Ranking Training
# ==========================================

class PairwiseRankingTrainer:
    """Train MoPE model with pairwise ranking loss"""

    def __init__(self, model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                 margin=RANKING_MARGIN):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.margin = margin

    def generate_pairs(self, healthy_feats, diseased_feats, silent=False):
        """Generate all cross-class pairwise training samples"""
        n_h = len(healthy_feats)
        n_d = len(diseased_feats)

        # All (healthy, diseased) pairs: diseased should rank higher
        pairs_h = []
        pairs_d = []
        for i, j in product(range(n_h), range(n_d)):
            pairs_h.append(healthy_feats[i])
            pairs_d.append(diseased_feats[j])

        pairs_h = np.array(pairs_h)
        pairs_d = np.array(pairs_d)

        if not silent:
            print(f"  Generated {len(pairs_h)} pairwise training samples "
                  f"({n_h} healthy x {n_d} diseased)")

        return pairs_h, pairs_d

    def train(self, healthy_feats, diseased_feats, num_epochs=NUM_EPOCHS,
              batch_size=BATCH_SIZE, verbose=True, silent=False):
        """
        Train with pairwise margin ranking loss.
        For each pair: loss = max(0, margin - (score_diseased - score_healthy))
        """
        pairs_h, pairs_d = self.generate_pairs(healthy_feats, diseased_feats,
                                                silent=silent)

        pairs_h_t = torch.FloatTensor(pairs_h)
        pairs_d_t = torch.FloatTensor(pairs_d)

        n_pairs = len(pairs_h)
        history = {'loss': [], 'pair_acc': []}

        for epoch in range(num_epochs):
            # Shuffle pairs each epoch
            perm = torch.randperm(n_pairs)
            pairs_h_t = pairs_h_t[perm]
            pairs_d_t = pairs_d_t[perm]

            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0

            for start in range(0, n_pairs, batch_size):
                end = min(start + batch_size, n_pairs)
                batch_h = pairs_h_t[start:end]
                batch_d = pairs_d_t[start:end]

                score_h, _, _ = self.model(batch_h)
                score_d, _, _ = self.model(batch_d)

                # Pairwise margin ranking loss
                diff = score_d - score_h  # should be positive
                loss = torch.clamp(self.margin - diff, min=0).mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                epoch_correct += (diff > 0).sum().item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            pair_acc = epoch_correct / n_pairs
            history['loss'].append(avg_loss)
            history['pair_acc'].append(pair_acc)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:>4d}/{num_epochs}  "
                      f"Loss: {avg_loss:.4f}  Pairwise Acc: {pair_acc:.4f}")

        return history

    def evaluate_loo(self, healthy_feats, diseased_feats, num_epochs=200):
        """Leave-One-Out cross-validation for ranking performance"""
        all_feats = np.vstack([healthy_feats, diseased_feats])
        all_labels = np.array([0]*len(healthy_feats) + [1]*len(diseased_feats))
        n = len(all_feats)

        correct_ranking = 0
        total_pairs_tested = 0
        correct_binary = 0

        print(f"  Running LOO cross-validation ({n} folds)...")

        for i in range(n):
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    LOO progress: {i+1}/{n}...")

            # Fix seed per fold for reproducibility
            torch.manual_seed(42 + i)
            np.random.seed(42 + i)

            # Leave one out
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False

            train_feats = all_feats[train_mask]
            train_labels = all_labels[train_mask]
            test_feat = all_feats[i:i+1]
            test_label = all_labels[i]

            h_feats = train_feats[train_labels == 0]
            d_feats = train_feats[train_labels == 1]

            # Train a fresh model
            loo_model = MoPERanker()
            loo_trainer = PairwiseRankingTrainer(loo_model, margin=self.margin)
            loo_trainer.train(h_feats, d_feats, num_epochs=num_epochs,
                             verbose=False, silent=True)

            # Predict
            loo_model.eval()
            with torch.no_grad():
                test_t = torch.FloatTensor(test_feat)
                score, _, _ = loo_model(test_t)
                score = score.item()

                # Get training score statistics for threshold
                h_scores, _, _ = loo_model(torch.FloatTensor(h_feats))
                d_scores, _, _ = loo_model(torch.FloatTensor(d_feats))
                threshold = (h_scores.mean() + d_scores.mean()).item() / 2

            # Binary accuracy (using threshold)
            pred_label = 1 if score > threshold else 0
            if pred_label == test_label:
                correct_binary += 1

            # Pairwise ranking accuracy
            if test_label == 1:
                # Diseased sample should score higher than all healthy
                with torch.no_grad():
                    h_all_scores, _, _ = loo_model(torch.FloatTensor(h_feats))
                n_correct = (score > h_all_scores.numpy()).sum()
                correct_ranking += n_correct
                total_pairs_tested += len(h_feats)
            else:
                # Healthy sample should score lower than all diseased
                with torch.no_grad():
                    d_all_scores, _, _ = loo_model(torch.FloatTensor(d_feats))
                n_correct = (score < d_all_scores.numpy()).sum()
                correct_ranking += n_correct
                total_pairs_tested += len(d_feats)

        binary_acc = correct_binary / n
        ranking_acc = correct_ranking / total_pairs_tested if total_pairs_tested > 0 else 0

        print(f"  LOO Binary Accuracy: {binary_acc:.3f} ({correct_binary}/{n})")
        print(f"  LOO Pairwise Ranking Accuracy: {ranking_acc:.3f}")

        return binary_acc, ranking_acc


# ==========================================
# Severity Threshold Determination
# ==========================================

class SeverityCalibrator:
    """Determine severity level thresholds from training data distribution"""

    def __init__(self, model):
        self.model = model
        self.thresholds = {}
        self.level_names = []
        self.h_stats = {}
        self.d_stats = {}

    def calibrate(self, healthy_feats, diseased_feats):
        """
        Determine severity thresholds based on training score distribution.

        Levels are determined by the gap between healthy and diseased score ranges:
          - Healthy zone:     score < mu_h + 2*sigma_h
          - Mild zone:        between healthy and midpoint
          - Moderate zone:    between midpoint and diseased lower bound
          - Severe zone:      score > mu_d - 2*sigma_d
        """
        self.model.eval()
        with torch.no_grad():
            h_scores, _, _ = self.model(torch.FloatTensor(healthy_feats))
            d_scores, _, _ = self.model(torch.FloatTensor(diseased_feats))

        h_scores = h_scores.numpy().flatten()
        d_scores = d_scores.numpy().flatten()

        # Statistics
        mu_h, sigma_h = np.mean(h_scores), np.std(h_scores)
        mu_d, sigma_d = np.mean(d_scores), np.std(d_scores)

        self.h_stats = {'mean': mu_h, 'std': sigma_h,
                        'min': np.min(h_scores), 'max': np.max(h_scores)}
        self.d_stats = {'mean': mu_d, 'std': sigma_d,
                        'min': np.min(d_scores), 'max': np.max(d_scores)}

        # Threshold determination
        t_low = mu_h + 2 * sigma_h    # upper bound of healthy
        t_high = mu_d - 2 * sigma_d   # lower bound of clearly diseased
        t_mid = (t_low + t_high) / 2   # midpoint of transition zone

        # Ensure monotonic ordering
        if t_low >= t_mid:
            t_mid = t_low + (t_high - t_low) * 0.5
        if t_mid >= t_high:
            t_high = t_mid + 0.01

        self.thresholds = {
            'healthy_upper': float(t_low),
            'mild_upper': float(t_mid),
            'moderate_upper': float(t_high),
        }
        self.level_names = ['Healthy', 'Mild', 'Moderate', 'Severe']

        print(f"\n  Severity Calibration (data-driven thresholds):")
        print(f"    Healthy scores:  {mu_h:.3f} +/- {sigma_h:.3f} "
              f"(range: {np.min(h_scores):.3f} ~ {np.max(h_scores):.3f})")
        print(f"    Diseased scores: {mu_d:.3f} +/- {sigma_d:.3f} "
              f"(range: {np.min(d_scores):.3f} ~ {np.max(d_scores):.3f})")
        print(f"    Thresholds: Healthy<{t_low:.3f} | Mild<{t_mid:.3f} | "
              f"Moderate<{t_high:.3f} | Severe>={t_high:.3f}")

        return self.thresholds

    def get_level(self, score):
        """Map continuous score to severity level"""
        if score < self.thresholds['healthy_upper']:
            return 0, 'Healthy'
        elif score < self.thresholds['mild_upper']:
            return 1, 'Mild'
        elif score < self.thresholds['moderate_upper']:
            return 2, 'Moderate'
        else:
            return 3, 'Severe'

    def get_level_color(self, level):
        """Color map for severity levels (BGR for OpenCV)"""
        colors = {
            0: (0, 200, 0),      # Green - Healthy
            1: (0, 200, 200),    # Yellow - Mild
            2: (0, 130, 255),    # Orange - Moderate
            3: (0, 0, 220),      # Red - Severe
        }
        return colors.get(level, (128, 128, 128))

    def get_continuous_color(self, score):
        """Continuous color gradient based on severity score (BGR)"""
        score = np.clip(score, 0, 1)
        if score < 0.5:
            # Green → Yellow
            t = score / 0.5
            r = int(t * 255)
            g = 200
            b = 0
        else:
            # Yellow → Red
            t = (score - 0.5) / 0.5
            r = 255
            g = int((1 - t) * 200)
            b = 0
        return (b, g, r)


# ==========================================
# Visualization
# ==========================================

def add_continuous_colorbar_bgr(image_bgr, calibrator, bar_width=32, pad=18):
    """Append a vertical severity color bar to the right side of a BGR image."""
    h, w = image_bgr.shape[:2]
    canvas = np.full((h, w + pad + bar_width + 70, 3), 255, dtype=np.uint8)
    canvas[:, :w] = image_bgr

    x0 = w + pad
    x1 = x0 + bar_width
    for y in range(h):
        score = 1.0 - (y / max(h - 1, 1))
        canvas[y:y+1, x0:x1] = np.array(calibrator.get_continuous_color(score), dtype=np.uint8)

    cv2.rectangle(canvas, (x0, 0), (x1, h - 1), (40, 40, 40), 1)

    tick_scores = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for tick in tick_scores:
        y = int(round((1.0 - tick) * (h - 1)))
        cv2.line(canvas, (x1 + 2, y), (x1 + 10, y), (40, 40, 40), 1)
        cv2.putText(
            canvas, f'{tick:.1f}', (x1 + 14, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA
        )

    cv2.putText(
        canvas, 'Severity', (x0 - 2, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas, 'score', (x0 + 2, 42),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA
    )
    return canvas

def plot_training_history(history, output_path):
    """Plot training loss and pairwise accuracy curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['loss'], color='#e74c3c', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Pairwise Ranking Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['pair_acc'], color='#2ecc71', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pairwise Accuracy')
    axes[1].set_title('Training Pairwise Ranking Accuracy')
    axes[1].set_ylim([0, 1.05])
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_severity_analysis(results, calibrator, output_path):
    """Compact dashboard: expert contribution, expert scores, and VSK summary."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    scores = [r['severity_score'] for r in results]
    gates = np.array([r['gates'] for r in results])
    expert_scores = np.array([r['expert_scores'] for r in results])
    score_arr = np.array(scores, dtype=np.float64)
    q25, q50, q75, q90 = np.quantile(score_arr, [0.25, 0.50, 0.75, 0.90])
    mean_score = float(np.mean(score_arr))
    std_score = float(np.std(score_arr))
    n_kernels = len(score_arr)
    train_h_n = calibrator.h_stats.get('n', 107)
    train_d_n = calibrator.d_stats.get('n', 34)
    ref_mean = (
        calibrator.h_stats['mean'] * train_h_n + calibrator.d_stats['mean'] * train_d_n
    ) / (train_h_n + train_d_n)
    vsk_threshold = calibrator.d_stats['mean'] + 0.25 * (mean_score - ref_mean)
    predicted_vsk_pct = float(np.mean(score_arr >= vsk_threshold) * 100.0)
    above_n = int(np.sum(score_arr >= vsk_threshold))
    below_n = int(n_kernels - above_n)

    # 1. Overall score distribution with VSK threshold
    ax = axes[0]
    ax.hist(scores, bins=30, color='#5dade2', alpha=0.78, edgecolor='white')
    ax.axvline(vsk_threshold, color='#e74c3c', linestyle='--', linewidth=2.4, label='VSK threshold')
    ax.axvline(mean_score, color='#34495e', linestyle='-.', linewidth=2.0, label='Image mean')
    ymax = ax.get_ylim()[1]
    ax.text(
        vsk_threshold + 0.01,
        ymax * 0.90,
        f'{vsk_threshold:.3f}',
        color='#e74c3c',
        fontsize=12,
        va='top',
        ha='left',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8),
    )
    ax.text(
        mean_score + 0.01,
        ymax * 0.90,
        f'{mean_score:.3f}',
        color='#34495e',
        fontsize=12,
        va='top',
        ha='left',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8),
    )
    ax.set_xlabel('Severity Score', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(f'Score Distribution and Threshold\nVSK score = {predicted_vsk_pct:.1f}% ({above_n}/{n_kernels})', fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.25, axis='y')

    # 2. Expert gating weights distribution (box plot)
    ax = axes[1]
    bp = ax.boxplot([gates[:, 0], gates[:, 1], gates[:, 2]],
                    labels=['Color\nExpert', 'Texture\nExpert', 'Morphology\nExpert'],
                    patch_artist=True, widths=0.6)
    expert_colors = ['#e74c3c', '#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], expert_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Gating Weight', fontsize=14)
    ax.set_title('Expert Contribution Distribution', fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Expert scores vs severity score (scatter)
    ax = axes[2]
    ax.scatter(scores, expert_scores[:, 0], alpha=0.5, s=30,
              c='#e74c3c', label='Color Expert')
    ax.scatter(scores, expert_scores[:, 1], alpha=0.5, s=30,
              c='#3498db', label='Texture Expert')
    ax.scatter(scores, expert_scores[:, 2], alpha=0.5, s=30,
              c='#2ecc71', label='Morphology Expert')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Final Severity Score', fontsize=14)
    ax.set_ylabel('Individual Expert Score', fontsize=14)
    ax.set_title('Expert Scores vs Final Severity', fontsize=16)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_severity_strip(results, output_path, max_seeds=48):
    """
    Severity grid: seeds sorted by severity score in a multi-row grid.
    Each seed displayed in its own cell with colored border and score label.
    """
    sorted_results = sorted(results, key=lambda r: r['severity_score'])

    # Sample evenly if too many
    if len(sorted_results) > max_seeds:
        indices = np.linspace(0, len(sorted_results)-1, max_seeds, dtype=int)
        sorted_results = [sorted_results[i] for i in indices]

    n = len(sorted_results)
    ncols = min(n, 16)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.6))

    # Handle single row case
    if nrows == 1:
        axes = axes.reshape(1, -1) if n > 1 else np.array([[axes]])

    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        if i < n:
            result = sorted_results[i]
            crop = result.get('crop_rgb', None)
            score = result['severity_score']

            if crop is not None:
                # Pad to square for uniform display
                h, w = crop.shape[:2]
                size = max(h, w)
                canvas = np.ones((size, size, 3), dtype=np.uint8) * 255
                y_off = (size - h) // 2
                x_off = (size - w) // 2
                canvas[y_off:y_off+h, x_off:x_off+w] = crop
                ax.imshow(canvas)

            # Colored border based on severity
            border_color = plt.cm.RdYlGn_r(score)
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f'{score:.2f}', fontsize=13, color='black',
                         fontweight='bold', labelpad=6)
        else:
            ax.axis('off')

    # Add vertical colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.02, pad=0.02, aspect=30)
    cbar.set_label('Severity Score', fontsize=13)

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_expert_weight_analysis(results, output_path):
    """Analyze learned feature weights from experts"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    feature_names_by_expert = {
        'Color Expert': [
            'L_mean', 'L_std', 'L_high_ratio', 'L_very_high',
            'chroma', 'S_mean', 'S_low_ratio',
            'V_mean', 'r_ratio', 'g_ratio', 'rb_diff'
        ],
        'Texture Expert': [
            'lap_var', 'lap_mean', 'edge_mean', 'edge_std',
            'lbp_ent_r1', 'lbp_ent_r2',
            'glcm_con', 'glcm_hom', 'glcm_eng',
            'gray_std', 'gray_range'
        ],
        'Morphology Expert': [
            'circularity', 'solidity', 'aspect_ratio'
        ]
    }

    expert_modules = ['expert_color', 'expert_texture', 'expert_morph']
    expert_colors = ['#e74c3c', '#3498db', '#2ecc71']

    for ax, (expert_name, feat_names), module_name, color in zip(
        axes, feature_names_by_expert.items(), expert_modules, expert_colors
    ):
        # Get weights from the expert
        weights = results['model_weights'][module_name]
        abs_weights = np.abs(weights)
        sorted_idx = np.argsort(abs_weights)

        sorted_names = [feat_names[i] for i in sorted_idx]
        sorted_weights = weights[sorted_idx]

        colors = [color if w > 0 else '#95a5a6' for w in sorted_weights]

        ax.barh(sorted_names, sorted_weights, color=colors, alpha=0.8)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title(f'{expert_name}\nLearned Weights', fontsize=15)
        ax.set_xlabel('Weight (+ = more severe)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# Utility Functions
# ==========================================

def estimate_diameter_from_rois(candidate_rois):
    """Estimate seed diameter from candidate ROIs"""
    if not candidate_rois:
        return None
    diameters = [np.sqrt(4 * roi.get('area', 0) / np.pi)
                 for roi in candidate_rois if roi.get('area', 0) > 50]
    return int(np.median(diameters)) if diameters else None


def build_candidate_rois_better(gray, margin=30, min_area=100,
                                downsample_max_side=ROI_DOWNSAMPLE_MAX_SIDE):
    """Better ROI proposal: downsample + Otsu + connected components."""
    h, w = gray.shape[:2]
    scale = 1.0
    longest_side = max(h, w)
    if longest_side > downsample_max_side:
        scale = downsample_max_side / float(longest_side)
        work = cv2.resize(
            gray,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA
        )
    else:
        work = gray

    blurred = cv2.GaussianBlur(work, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidate_rois = []
    for cid in range(1, n_labels):
        area = float(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        bw = int(stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(stats[cid, cv2.CC_STAT_HEIGHT])
        x1 = max(0, int(np.floor((x - margin) / scale)))
        y1 = max(0, int(np.floor((y - margin) / scale)))
        x2 = min(gray.shape[1], int(np.ceil((x + bw + margin) / scale)))
        y2 = min(gray.shape[0], int(np.ceil((y + bh + margin) / scale)))
        candidate_rois.append({'bbox': (x1, y1, x2, y2), 'area': area / (scale * scale)})

    return merge_overlapping_rois(candidate_rois)


def merge_overlapping_rois(rois):
    """Merge ROIs that have any intersection"""
    if len(rois) == 0:
        return []

    def intersects(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

    merged_groups = []
    used = [False] * len(rois)

    for i in range(len(rois)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        j = 0
        while j < len(group):
            for k in range(len(rois)):
                if not used[k] and intersects(rois[group[j]]['bbox'], rois[k]['bbox']):
                    group.append(k)
                    used[k] = True
            j += 1
        merged_groups.append(group)

    result = []
    for group in merged_groups:
        bboxes = [rois[idx]['bbox'] for idx in group]
        merged_bbox = (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes)
        )
        result.append({'bbox': merged_bbox, 'count': len(group)})

    return result


def stitch_roi_masks(cellpose_model, gray, rois, diameter, flow_threshold=CELLPOSE_FLOW_THRESHOLD):
    """Run Cellpose on each ROI independently and stitch labels back to full image."""
    full_masks = np.zeros(gray.shape, dtype=np.uint32)
    next_label = 1

    for roi in rois:
        x1, y1, x2, y2 = roi['bbox']
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        roi_masks, _, _ = cellpose_model.eval(
            crop,
            diameter=diameter,
            flow_threshold=flow_threshold
        )
        if roi_masks is None or np.max(roi_masks) == 0:
            continue
        roi_masks = roi_masks.astype(np.uint32)
        roi_masks[roi_masks > 0] += next_label - 1
        target = full_masks[y1:y2, x1:x2]
        target[roi_masks > 0] = roi_masks[roi_masks > 0]
        full_masks[y1:y2, x1:x2] = target
        next_label = int(full_masks.max()) + 1

    return full_masks


def normalize_features(healthy_feats, diseased_feats):
    """Z-score normalization using training set statistics"""
    all_feats = np.vstack([healthy_feats, diseased_feats])
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-10
    return (healthy_feats - mean) / std, (diseased_feats - mean) / std, mean, std


# ==========================================
# Train / Load Logic
# ==========================================

def train_and_save(base_output, stages_dir):
    """Train MoPE model from examples and save everything"""
    model_path = os.path.join(base_output, 'mope_model.pth')
    feats_path = os.path.join(base_output, 'training_features.npz')

    # Step 1: Extract training features
    print("\n[Train Step 1] Extracting features from training examples...")
    healthy_feats, feature_names = extract_features_from_dir(HEALTHY_EXAMPLES_DIR)
    diseased_feats, _ = extract_features_from_dir(DISEASED_EXAMPLES_DIR)

    n_healthy = len(healthy_feats)
    n_diseased = len(diseased_feats)

    print(f"  Healthy samples: {n_healthy}")
    print(f"  Diseased samples: {n_diseased}")
    print(f"  Feature dimensions: {len(feature_names)}")
    print(f"  Potential training pairs: {n_healthy * n_diseased}")

    if n_healthy < 1 or n_diseased < 1:
        exit("Error: Insufficient training samples")

    # Normalize
    healthy_norm, diseased_norm, feat_mean, feat_std = normalize_features(
        healthy_feats, diseased_feats
    )

    # Save extracted features
    np.savez(feats_path,
             healthy_feats=healthy_feats, diseased_feats=diseased_feats,
             healthy_norm=healthy_norm, diseased_norm=diseased_norm,
             feat_mean=feat_mean, feat_std=feat_std,
             feature_names=feature_names)
    print(f"  Training features saved: {feats_path}")

    # Step 2: Train MoPE
    print("\n[Train Step 2] Training MoPE pairwise ranking model...")
    print(f"  Model: 3 experts (Color={len(COLOR_INDICES)}d, "
          f"Texture={len(TEXTURE_INDICES)}d, Morphology={len(MORPH_INDICES)}d)")
    print(f"  Training: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}, margin={RANKING_MARGIN}")

    model = MoPERanker()

    trainer = PairwiseRankingTrainer(model, lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY,
                                     margin=RANKING_MARGIN)

    history = trainer.train(healthy_norm, diseased_norm,
                           num_epochs=NUM_EPOCHS, verbose=True)

    plot_training_history(history, os.path.join(stages_dir, 'training_history.png'))
    print(f"\n  Final Pairwise Ranking Accuracy: {history['pair_acc'][-1]:.4f}")

    # Step 2b: LOO Cross-validation
    print("\n[Train Step 2b] Leave-One-Out Cross-validation...")
    loo_binary_acc, loo_ranking_acc = trainer.evaluate_loo(
        healthy_norm, diseased_norm, num_epochs=200
    )

    # Step 3: Calibrate thresholds
    print("\n[Train Step 3] Calibrating severity thresholds...")
    calibrator = SeverityCalibrator(model)
    model.eval()
    thresholds = calibrator.calibrate(healthy_norm, diseased_norm)

    # Save model + everything
    torch.save({
        'model_state_dict': model.state_dict(),
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'thresholds': thresholds,
        'calibrator_h_stats': calibrator.h_stats,
        'calibrator_d_stats': calibrator.d_stats,
        'feature_names': feature_names,
        'history': history,
        'loo_binary_acc': loo_binary_acc,
        'loo_ranking_acc': loo_ranking_acc,
        'config': {
            'color_dim': len(COLOR_INDICES),
            'texture_dim': len(TEXTURE_INDICES),
            'morph_dim': len(MORPH_INDICES),
            'ranking_margin': RANKING_MARGIN,
        }
    }, model_path)
    print(f"\n  Model saved: {model_path}")

    return model, calibrator, feat_mean, feat_std, feature_names, history, \
           loo_binary_acc, loo_ranking_acc


def load_model(base_output):
    """Load pre-trained model and all saved state"""
    model_path = os.path.join(base_output, 'mope_model.pth')

    if not os.path.exists(model_path):
        return None

    print(f"\n  Found saved model: {model_path}")
    checkpoint = torch.load(model_path, weights_only=False)

    config = checkpoint.get('config', {})
    model = MoPERanker(
        color_dim=config.get('color_dim', len(COLOR_INDICES)),
        texture_dim=config.get('texture_dim', len(TEXTURE_INDICES)),
        morph_dim=config.get('morph_dim', len(MORPH_INDICES)),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    calibrator = SeverityCalibrator(model)
    calibrator.thresholds = checkpoint['thresholds']
    calibrator.h_stats = checkpoint['calibrator_h_stats']
    calibrator.d_stats = checkpoint['calibrator_d_stats']
    calibrator.level_names = ['Healthy', 'Mild', 'Moderate', 'Severe']

    feat_mean = checkpoint['feat_mean']
    feat_std = checkpoint['feat_std']
    feature_names = checkpoint['feature_names']
    history = checkpoint.get('history', None)
    loo_binary_acc = checkpoint.get('loo_binary_acc', None)
    loo_ranking_acc = checkpoint.get('loo_ranking_acc', None)

    print(f"  Model loaded successfully!")
    print(f"  Thresholds: Healthy<{calibrator.thresholds['healthy_upper']:.3f} | "
          f"Mild<{calibrator.thresholds['mild_upper']:.3f} | "
          f"Moderate<{calibrator.thresholds['moderate_upper']:.3f} | Severe")
    if loo_binary_acc is not None:
        print(f"  LOO Binary Acc: {loo_binary_acc:.3f} | "
              f"LOO Ranking Acc: {loo_ranking_acc:.3f}")

    return model, calibrator, feat_mean, feat_std, feature_names, history, \
           loo_binary_acc, loo_ranking_acc


def process_image(image_path, model, calibrator, feat_mean, feat_std,
                  feature_names, base_output):
    """Process a single image: segmentation + severity scoring"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Cannot read {image_path}")
        return None

    img_name = os.path.splitext(os.path.basename(image_path))[0]
    img_output = os.path.join(base_output, 'results', img_name)
    seeds_dir = os.path.join(img_output, 'seeds')
    for level_name in ['Healthy', 'Mild', 'Moderate', 'Severe']:
        os.makedirs(os.path.join(seeds_dir, level_name), exist_ok=True)
    os.makedirs(img_output, exist_ok=True)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Cellpose Segmentation ---
    print(f"\n  [{img_name}] Cellpose segmentation...")
    candidate_rois = build_candidate_rois_better(gray)
    print(f"  Better ROI proposals: {len(candidate_rois)} | fixed diameter={CELLPOSE_FIXED_DIAMETER}")

    cache_dir = os.path.join(base_output, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir,
                              f"cellpose_betterroi_d{CELLPOSE_FIXED_DIAMETER}_{img_name}_{gray.shape[0]}x{gray.shape[1]}.npz")

    if os.path.exists(cache_path):
        data = np.load(cache_path)
        masks = data['masks'].astype(np.uint32)
        num_seeds = int(masks.max())
        print(f"  Loaded from cache: {num_seeds} seeds")
    else:
        use_gpu = torch.cuda.is_available()
        from cellpose import models
        try:
            cellpose_model = models.CellposeModel(gpu=use_gpu)
        except:
            cellpose_model = models.Cellpose(gpu=use_gpu, model_type=CELLPOSE_MODEL)

        diameter = CELLPOSE_FIXED_DIAMETER
        print(f"  Using fixed diameter: {diameter}")

        if len(candidate_rois) == 0:
            masks = np.zeros(gray.shape, dtype=np.uint32)
        else:
            masks = stitch_roi_masks(
                cellpose_model,
                gray,
                candidate_rois,
                diameter=diameter,
                flow_threshold=CELLPOSE_FLOW_THRESHOLD
            )

        num_seeds = int(masks.max())
        np.savez_compressed(cache_path, masks=masks)
        print(f"  Detected {num_seeds} seeds")

    # --- Feature Extraction & Severity Scoring ---
    print(f"  [{img_name}] Scoring seeds...")

    results = []
    model.eval()

    for seed_id in range(1, num_seeds + 1):
        mask = (masks == seed_id)
        area = np.sum(mask)

        if not (100 < area < 10000):
            continue

        y_coords, x_coords = np.where(mask)
        ymin, ymax = y_coords.min(), y_coords.max()
        xmin, xmax = x_coords.min(), x_coords.max()

        seed_rgb = rgb_image[ymin:ymax+1, xmin:xmax+1].copy()
        seed_gray = gray[ymin:ymax+1, xmin:xmax+1].copy()
        seed_mask = mask[ymin:ymax+1, xmin:xmax+1].copy()

        try:
            features = extract_fhb_features(seed_rgb, seed_gray, seed_mask)
            feat_vec, _ = get_feature_vector(features)
            feat_norm = (feat_vec - feat_mean) / feat_std

            with torch.no_grad():
                feat_tensor = torch.FloatTensor(feat_norm).unsqueeze(0)
                severity, gates, expert_scores = model(feat_tensor)

            severity_score = severity.item()
            gates_np = gates.numpy().flatten()
            expert_scores_np = expert_scores.numpy().flatten()
            level_id, level_name = calibrator.get_level(severity_score)

            # Clean crops for visualization
            crop_rgb_clean = seed_rgb.copy()
            crop_rgb_clean[~seed_mask] = [255, 255, 255]  # white bg for strip
            crop_bgr = cv2.cvtColor(seed_rgb, cv2.COLOR_RGB2BGR)
            crop_bgr[~seed_mask] = [0, 0, 0]  # black bg for saved files

            results.append({
                'seed_id': seed_id,
                'mask': mask,
                'centroid': (int(np.mean(x_coords)), int(np.mean(y_coords))),
                'features': features,
                'feat_vec': feat_vec,
                'severity_score': severity_score,
                'gates': gates_np,
                'expert_scores': expert_scores_np,
                'level_id': level_id,
                'level_name': level_name,
                'crop_rgb': crop_rgb_clean,
                'crop_bgr': crop_bgr,
                'bbox': (xmin, ymin, xmax, ymax),
            })
        except:
            continue

        if seed_id % 50 == 0:
            print(f"    Processed {seed_id}/{num_seeds}...")

    print(f"  [{img_name}] Valid seeds: {len(results)}")

    if len(results) == 0:
        print(f"  [{img_name}] No valid seeds found, skipping.")
        return None

    score_arr = np.array([r['severity_score'] for r in results], dtype=np.float64)
    train_h_n = calibrator.h_stats.get('n', 107)
    train_d_n = calibrator.d_stats.get('n', 34)
    ref_mean = (
        calibrator.h_stats['mean'] * train_h_n + calibrator.d_stats['mean'] * train_d_n
    ) / (train_h_n + train_d_n)
    vsk_threshold = calibrator.d_stats['mean'] + 0.25 * (float(score_arr.mean()) - ref_mean)
    vsk_positive = int(np.sum(score_arr >= vsk_threshold))
    vsk_negative = int(len(score_arr) - vsk_positive)

    # Level statistics
    level_counts = {}
    for r in results:
        name = r['level_name']
        level_counts[name] = level_counts.get(name, 0) + 1

    print(f"  [{img_name}] Severity Distribution:")
    for lname in ['Healthy', 'Mild', 'Moderate', 'Severe']:
        cnt = level_counts.get(lname, 0)
        pct = cnt / len(results) * 100
        print(f"    {lname}: {cnt} ({pct:.1f}%)")
    print(f"  [{img_name}] Binary VSK split: below={vsk_negative} ({vsk_negative/len(results)*100:.1f}%) | "
          f"at/above={vsk_positive} ({vsk_positive/len(results)*100:.1f}%), threshold={vsk_threshold:.3f}")

    # --- Visualization ---
    # Gradient annotated image
    viz_gradient = image.copy()
    viz_levels = image.copy()
    level_overlay = image.copy()

    for result in results:
        mask_r = result['mask']
        score = result['severity_score']
        level = result['level_id']
        is_vsk_positive = score >= vsk_threshold

        mask_uint8 = (mask_r * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        grad_color = calibrator.get_continuous_color(score)
        cv2.drawContours(viz_gradient, contours, -1, grad_color, 2)
        level_color = (0, 200, 0) if not is_vsk_positive else (0, 0, 220)
        level_overlay[mask_r] = level_color

    viz_levels = cv2.addWeighted(level_overlay, 0.32, viz_levels, 0.68, 0)

    cv2.imwrite(os.path.join(img_output, 'severity_gradient.jpg'), viz_gradient)
    cv2.imwrite(os.path.join(img_output, 'severity_levels.jpg'), viz_levels)

    # Severity strip
    plot_severity_strip(results, os.path.join(img_output, 'severity_strip.png'))

    # Comprehensive analysis
    plot_severity_analysis(results, calibrator,
                          os.path.join(img_output, 'severity_analysis.png'))

    # Save cropped seeds by level
    for result in results:
        lname = result['level_name']
        sid = result['seed_id']
        cv2.imwrite(os.path.join(seeds_dir, lname, f'seed_{sid:03d}.jpg'),
                    result['crop_bgr'])

    # Save CSV
    csv_path = os.path.join(img_output, 'severity_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['seed_id', 'severity_score', 'level', 'level_id',
                  'gate_color', 'gate_texture', 'gate_morph',
                  'expert_color', 'expert_texture', 'expert_morph'] + feature_names
        writer.writerow(header)
        for r in results:
            row = [
                r['seed_id'],
                f"{r['severity_score']:.4f}",
                r['level_name'],
                r['level_id'],
                f"{r['gates'][0]:.4f}",
                f"{r['gates'][1]:.4f}",
                f"{r['gates'][2]:.4f}",
                f"{r['expert_scores'][0]:.4f}",
                f"{r['expert_scores'][1]:.4f}",
                f"{r['expert_scores'][2]:.4f}",
            ]
            row.extend([f"{r['feat_vec'][j]:.4f}" for j in range(len(feature_names))])
            writer.writerow(row)

    return {
        'image_name': img_name,
        'results': results,
        'level_counts': level_counts,
        'output_dir': img_output,
    }


# ==========================================
# Main Pipeline
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MoPE-Rank Severity Assessment')
    parser.add_argument('--images', nargs='+', default=['./origin.jpg'],
                       help='Image paths to process')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain even if model exists')
    args = parser.parse_args()

    base_output = os.path.join('output_results', 'MoPE_Ranking')
    stages_dir = os.path.join(base_output, 'stages')
    os.makedirs(stages_dir, exist_ok=True)

    print("=" * 70)
    print("MoPE-Rank: Mixture of Phenotypic Experts Severity Ranking")
    print("=" * 70)

    start_total = time.time()

    # ==========================================
    # Phase 1: Train or Load Model
    # ==========================================
    model_path = os.path.join(base_output, 'mope_model.pth')
    loaded = None

    if not args.retrain and os.path.exists(model_path):
        print("\n[Phase 1] Loading pre-trained model (use --retrain to force retrain)")
        loaded = load_model(base_output)

    if loaded is None:
        print("\n[Phase 1] Training new model from examples...")
        loaded = train_and_save(base_output, stages_dir)

    model, calibrator, feat_mean, feat_std, feature_names, \
        history, loo_binary_acc, loo_ranking_acc = loaded

    # Expert weight analysis (always regenerate)
    model_weights = {}
    for name, param in model.named_parameters():
        if 'fc.weight' in name:
            module_name = name.split('.')[0]
            model_weights[module_name] = param.detach().numpy().flatten()
    plot_expert_weight_analysis(
        {'model_weights': model_weights},
        os.path.join(stages_dir, 'expert_weights.png')
    )

    train_time = time.time() - start_total
    print(f"\n  Phase 1 completed in {train_time:.1f}s")

    # ==========================================
    # Phase 2: Process Images (Inference Only)
    # ==========================================
    print("\n" + "=" * 70)
    print(f"[Phase 2] Processing {len(args.images)} image(s)...")
    print("=" * 70)

    all_image_results = []

    for image_path in args.images:
        print(f"\n{'─' * 50}")
        print(f"Processing: {image_path}")
        print(f"{'─' * 50}")

        img_result = process_image(image_path, model, calibrator,
                                    feat_mean, feat_std, feature_names,
                                    base_output)
        if img_result is not None:
            all_image_results.append(img_result)

    # ==========================================
    # Final Report
    # ==========================================
    total_time = time.time() - start_total

    print("\n" + "=" * 70)
    print("MoPE-Rank Complete!")
    print("=" * 70)

    if loo_binary_acc is not None:
        print(f"\nModel Performance:")
        if history:
            print(f"  Training Pairwise Acc: {history['pair_acc'][-1]:.4f}")
        print(f"  LOO Binary Acc:       {loo_binary_acc:.4f}")
        print(f"  LOO Pairwise Rank:    {loo_ranking_acc:.4f}")

    for img_res in all_image_results:
        img_name = img_res['image_name']
        results = img_res['results']
        level_counts = img_res['level_counts']

        print(f"\n[{img_name}] {len(results)} seeds:")
        for lname in ['Healthy', 'Mild', 'Moderate', 'Severe']:
            cnt = level_counts.get(lname, 0)
            pct = cnt / len(results) * 100 if results else 0
            print(f"  {lname:>10s}: {cnt:>4d} ({pct:>5.1f}%)")

        scores_all = [r['severity_score'] for r in results]
        print(f"  Score: {min(scores_all):.3f} ~ {max(scores_all):.3f} "
              f"(mean={np.mean(scores_all):.3f})")

        all_gates = np.array([r['gates'] for r in results])
        print(f"  Expert weights: Color={all_gates[:, 0].mean():.3f}  "
              f"Texture={all_gates[:, 1].mean():.3f}  "
              f"Morph={all_gates[:, 2].mean():.3f}")

    print(f"\nOutput: {base_output}/")
    print(f"Time: {total_time:.1f}s")
    print("=" * 70)
