"""
ART PRICE PREDICTION - VERSION OPTIMISÉE AVEC XGBOOST TUNING
Intègre art_price_prediction_4files.py + xgboost_optimization.py

USAGE:
    python art_price_prediction_optimized.py

    Cela va:
    1. Charger tous les fichiers CSV (5 fichiers)
    2. Préparer les données
    3. AVANT: Trainer le modèle baseline
    4. OPTIMISER: Hyperparamètres XGBoost (2 phases)
    5. APRÈS: Afficher les améliorations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOADING & PREPROCESSING (from art_price_prediction_4files.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_all_csv_files():
    """Load and consolidate ALL 5 CSV files"""

    files = {
        'up_to_2012': 'df_for_ml_improved_up_to_2012.csv',
        'mloutfull': 'Df_mloutfull.csv',
    }

    datasets = {}
    total_records = 0

    for name, filepath in files.items():
        try:
            print(f"[{name}] Loading {filepath}...")
            df = pd.read_csv(filepath)

            # Handle Df_mloutfull.csv specific cleaning
            if name == 'full_mlout':
                df = df.drop(columns=['_merge', '_merge_01'], errors='ignore')
                if 'Ind' in df.columns:
                    df = df[df['Ind'] != 'left_only'].copy()

            # Basic cleaning
            df = df[df['price_usd'] > 0].copy()
            df = df.dropna(subset=['price_usd'])
            df['log_price'] = np.log10(df['price_usd'])
            df['source'] = name

            df = identify_market_type(df)
            datasets[name] = df
            total_records += len(df)

            est_count = sum(df['market_type'] == "Established")
            emer_count = sum(df['market_type'] == "Emerging")
            n_cols = df.shape[1]
            print(f" {len(df):,} records ({n_cols} cols) | Est: {est_count:,} | Emer: {emer_count:,}")

        except FileNotFoundError:
            print(f" Error: {filepath} NOT FOUND")
        except Exception as e:
            print(f" Error: {str(e)[:60]}")

    if datasets:
        df_combined = pd.concat(datasets.values(), ignore_index=True, sort=False)
        print(f"\n{'='*70}")
        print(f"COMBINED: {total_records:,} records from {len(datasets)} sources")
        print(f"FINAL SHAPE: {df_combined.shape[0]:,} rows × {df_combined.shape[1]} columns")
        print(f"{'='*70}")
        return df_combined, datasets
    else:
        raise ValueError("Error: No CSV files loaded successfully")

def identify_market_type(df):
    """Classify transactions into established and emerging markets"""
    established_countries = ['USA', 'UK', 'France', 'Germany']

    market_type = []
    for idx, row in df.iterrows():
        is_established = False
        for country in established_countries:
            col_candidates = [
                f'Country - {country}',
                f'Country_{country}',
                f'Country___{country}',
                f'dealcntry{country.lower()}',
                f'collcntry{country.lower()}'
            ]
            for col_name in col_candidates:
                if col_name in df.columns and row.get(col_name, 0) == 1:
                    is_established = True
                    break
            if is_established:
                break
        market_type.append('Established' if is_established else 'Emerging')

    df['market_type'] = market_type
    return df

def select_social_features():
    """All social/metadata features"""
    return [
        'age', 'ranking', 'fest_biennal', 'private_inst', 'public_inst',
        'solo_show', 'group_show', 'artwork_order', 'log_ranking',
        'price_usd_prev_5_mean', 'price_usd_prev_5_median', 'price_usd_prev_5_max',
        'price_usd_prev_10_mean', 'price_usd_prev_10_median', 'price_usd_prev_10_max',
        'price_same_size_prev_5_mean', 'price_same_size_prev_5_median',
        'price_same_size_prev_10_mean', 'price_same_size_prev_10_median',
        'gender_male', 'gender_female', 'gender_NA',
        'edu_abroad', 'edu_domestic', 'edu_both',
        'elite_school', 'elite_award_received',
        'Tier - 1', 'Tier - 2', 'Tier - 3', 'Tier - 4',
        'Tier___1', 'Tier___2', 'Tier___3', 'Tier___4',
        'Continent - Europe', 'Continent - North America', 'Continent - East Asia',
        'Continent___Europe', 'Continent___North_America', 'Continent___East_Asia',
        'Genre___Painting', 'Genre___Print', 'Genre___Photography', 'Genre___Sculpture',
        'matched_genre', 'matched_country',
        'size_inchsqr', 'log10_estimate_geo_mean_usd',
        'friend_ranking', 'log_frirank', 'nat_mean1', 'nonat_mean1',
        'natcon_mean1', 'nonatcon_mean1',
        'hofstead1_home', 'hofstead2_home', 'hofstead3_home', 'hofstead4_home',
        'hofstead1_market', 'hofstead2_market', 'hofstead3_market', 'hofstead4_market',
        'hofdistance1', 'hofdistance2', 'hofdistance3', 'hofdistance4',
    ]

def prepare_training_data(df, features):
    """Prepare data for training - ROBUST handling"""
    available_features = [f for f in features if f in df.columns]
    print(f"\n[Prep] Using {len(available_features)}/{len(features)} features")
    print(f"[Prep] Missing features: {len(features) - len(available_features)}")

    X = df[available_features].copy()
    y = df['log_price'].copy()

    # Convert to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    print(f"[Prep] NaN before fillna: {X.isna().sum().sum():,}")

    # Fill NaN per column
    for col in X.columns:
        col_median = X[col].median()
        if pd.isna(col_median):
            col_median = 0
        X[col] = X[col].fillna(col_median)

    # Replace inf
    for col in X.columns:
        col_median = X[col].median()
        X[col] = X[col].replace([np.inf, -np.inf], col_median)

    # Final mask
    mask = np.isfinite(X.values).all(axis=1) & np.isfinite(y)
    X_final = X[mask].reset_index(drop=True)
    y_final = y[mask].reset_index(drop=True)

    retention = len(X_final) / len(df) * 100
    print(f"[Prep] FINAL: {len(X_final):,} rows × {X_final.shape[1]} features ({retention:.1f}% retention)")
    return X_final, y_final

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: BASELINE MODEL (AVANT optimisation)
# ═════════════════════════════════════════════════════════════════════════════

def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train baseline XGBoost model (before optimization)"""
    print(f"\n{'='*70}")
    print(f"BASELINE MODEL TRAINING (Before Optimization)")
    print(f"{'='*70}")
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Default XGBoost parameters
    baseline_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='rmse',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = baseline_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    y_test_orig = 10 ** y_test
    y_pred_orig = 10 ** y_pred
    rmse_usd = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print(f"\n[BASELINE RESULTS]")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE (log): {rmse_log:.4f}")
    print(f"  RMSE (USD): ${rmse_usd:,.0f}")
    print(f"  MAPE: {mape:.1f}%")

    baseline_metrics = {
        'r2': r2,
        'rmse_log': rmse_log,
        'mae': mae,
        'rmse_usd': rmse_usd,
        'mape': mape
    }

    return baseline_model, baseline_metrics

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: OPTIMIZATION (Phase 1 + Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

def phase1_exploration(X_train, y_train, X_val, y_val):
    """Phase 1: Quick exploration with large parameter ranges"""
    print(f"\n{'='*70}")
    print(f"PHASE 1: QUICK EXPLORATION")
    print(f"{'='*70}")

    param_dist = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3]
    }

    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
        param_dist,
        n_iter=30,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("\n[Fitting Phase 1...]")
    search.fit(X_train, y_train)

    print(f"\n[PHASE 1 RESULTS]")
    print(f"  Best CV Score: {search.best_score_:.4f}")
    for param in sorted(search.best_params_.keys()):
        print(f"  {param}: {search.best_params_[param]}")

    return search.best_params_

def phase2_finetuning(X_train, y_train, X_val, y_val, best_params_1):
    """Phase 2: Fine tuning with narrow parameter ranges"""
    print(f"\n{'='*70}")
    print(f"PHASE 2: FINE TUNING")
    print(f"{'='*70}")

    # Narrow ranges around Phase 1 best params
    depth = best_params_1['max_depth']
    lr = best_params_1['learning_rate']
    n_est = best_params_1['n_estimators']
    subsamp = best_params_1['subsample']
    colsamp = best_params_1['colsample_bytree']
    min_ch = best_params_1['min_child_weight']

    param_dist = {
        'n_estimators': [max(100, n_est-100), n_est, n_est+100, n_est+200],
        'max_depth': [max(2, depth-1), depth, depth+1, depth+2],
        'learning_rate': [lr*0.5, lr*0.75, lr, lr*1.25, lr*1.5],
        'subsample': [max(0.5, subsamp-0.1), subsamp, min(1.0, subsamp+0.1)],
        'colsample_bytree': [max(0.5, colsamp-0.1), colsamp, min(1.0, colsamp+0.1)],
        'min_child_weight': [max(1, min_ch-1), min_ch, min_ch+1, min_ch+2],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.1, 1.0, 10.0]
    }

    search = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
        param_dist,
        n_iter=25,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("\n[Fitting Phase 2...]")
    search.fit(X_train, y_train)

    print(f"\n[PHASE 2 RESULTS]")
    print(f"  Best CV Score: {search.best_score_:.4f}")
    for param in sorted(search.best_params_.keys()):
        print(f"  {param}: {search.best_params_[param]}")

    return search.best_params_

def train_optimized_model(X_train, y_train, X_test, y_test, best_params):
    """Train optimized XGBoost model with best hyperparameters"""
    print(f"\n{'='*70}")
    print(f"TRAINING OPTIMIZED MODEL")
    print(f"{'='*70}")

    optimized_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        verbosity=0
    )

    optimized_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    y_pred = optimized_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    y_test_orig = 10 ** y_test
    y_pred_orig = 10 ** y_pred
    rmse_usd = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mape = np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100

    print(f"\n[OPTIMIZED MODEL RESULTS]")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE (log): {rmse_log:.4f}")
    print(f"  RMSE (USD): ${rmse_usd:,.0f}")
    print(f"  MAPE: {mape:.1f}%")

    optimized_metrics = {
        'r2': r2,
        'rmse_log': rmse_log,
        'mae': mae,
        'rmse_usd': rmse_usd,
        'mape': mape
    }

    return optimized_model, optimized_metrics

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPARISON & RESULTS
# ═════════════════════════════════════════════════════════════════════════════

def compare_results(baseline_metrics, optimized_metrics):
    """Display comparison BEFORE vs AFTER optimization"""
    print(f"\n{'='*70}")
    print(f"OPTIMIZATION RESULTS: BEFORE vs AFTER")
    print(f"{'='*70}")

    print(f"\n{'Metric':<20} {'BASELINE':<15} {'OPTIMIZED':<15} {'IMPROVEMENT':<15}")
    print("-"*70)

    for metric in ['r2', 'rmse_log', 'rmse_usd', 'mape']:
        before = baseline_metrics[metric]
        after = optimized_metrics[metric]

        if metric == 'r2':
            improvement = ((after - before) / abs(before) * 100) if before != 0 else 0
            print(f"{metric:<20} {before:.4f}        {after:.4f}        +{improvement:+.2f}%")
        else:
            improvement = ((before - after) / abs(before) * 100) if before != 0 else 0
            print(f"{metric:<20} {before:,.0f}       {after:,.0f}        +{improvement:+.2f}%")

    r2_gain = ((optimized_metrics['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2']) * 100)
    print(f"\n🎯 OVERALL R² IMPROVEMENT: +{r2_gain:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" ART PRICE PREDICTION - XGBOOST OPTIMIZATION PIPELINE")
    print("="*70)

    # Step 1: Load data
    df_combined, datasets = load_all_csv_files()

    # Step 2: Prepare features
    social_features = select_social_features()
    X, y = prepare_training_data(df_combined, social_features)

    # Step 3: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Baseline model
    baseline_model, baseline_metrics = train_baseline_model(
        X_train, y_train, X_test, y_test
    )

    # Step 5: Optimization (Phase 1 + 2)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    best_params_1 = phase1_exploration(X_train_split, y_train_split, X_val, y_val)
    best_params_2 = phase2_finetuning(X_train_split, y_train_split, X_val, y_val, best_params_1)

    # Step 6: Optimized model
    optimized_model, optimized_metrics = train_optimized_model(
        X_train, y_train, X_test, y_test, best_params_2
    )

    # Step 7: Comparison
    compare_results(baseline_metrics, optimized_metrics)

    print("\n✅ OPTIMIZATION COMPLETE!")
