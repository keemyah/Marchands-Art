"""
XGBOOST HYPERPARAMETER OPTIMIZATION MODULE
Méthode RandomizedSearchCV (Stratégie 2 - Recommandée)
2 phases: Exploration rapide + Fine tuning

Compatible avec n'importe quel projet XGBoost/ML
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 1: EXPLORATION RAPIDE (Large parameter ranges)
# ============================================================================

def phase1_quick_exploration(X_train, y_train, X_val, y_val):
    """
    Phase 1: Teste larges ranges de paramètres
    Temps: ~5-10 minutes (45M rows)
    Output: Meilleure région de l'espace des paramètres
    """
    print("\n" + "="*80)
    print("PHASE 1: QUICK EXPLORATION (Large parameter ranges)")
    print("="*80)

    # Large ranges - explorer la région générale optimale
    param_dist_1 = {
        'n_estimators': [100, 200, 300, 500, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 2, 3]
    }

    # 30 itérations = bon compromis temps/performance
    search_1 = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
        param_dist_1,
        n_iter=30,  # Test 30 combinaisons aléatoires
        cv=3,       # 3-fold CV (rapide)
        scoring='r2',
        n_jobs=-1,  # Paralléliser sur tous les cores
        random_state=42,
        verbose=1
    )

    print("\n[Fitting Phase 1 model...]")
    search_1.fit(X_train, y_train)

    # Résultats Phase 1
    best_params_1 = search_1.best_params_
    best_score_1 = search_1.best_score_

    print(f"\n[PHASE 1 RESULTS]")
    print(f"  Best CV Score: {best_score_1:.4f}")
    print(f"  Best Parameters:")
    for param, value in sorted(best_params_1.items()):
        print(f"    - {param}: {value}")

    # Validation sur set de validation
    model_1 = XGBRegressor(**best_params_1, random_state=42, n_jobs=-1, eval_metric='rmse')
    model_1.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_val = model_1.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    print(f"  Validation R²: {r2_val:.4f}")

    return best_params_1, search_1.cv_results_

# ============================================================================
# PHASE 2: FINE TUNING (Narrow ranges autour du meilleur)
# ============================================================================

def phase2_fine_tuning(X_train, y_train, X_val, y_val, best_params_1):
    """
    Phase 2: Fine tune les paramètres autour des best_params_ de Phase 1
    Temps: ~3-5 minutes (45M rows)
    Output: Hyperparamètres quasi-optimaux
    """
    print("\n" + "="*80)
    print("PHASE 2: FINE TUNING (Narrow parameter ranges)")
    print("="*80)

    # Extraire les meilleurs paramètres de Phase 1
    best_depth = best_params_1['max_depth']
    best_lr = best_params_1['learning_rate']
    best_n_est = best_params_1['n_estimators']
    best_subsample = best_params_1['subsample']
    best_colsample = best_params_1['colsample_bytree']
    best_min_child = best_params_1['min_child_weight']

    # Narrower ranges (±1 ou ×0.5-2)
    param_dist_2 = {
        'n_estimators': [
            max(100, best_n_est - 100),
            best_n_est,
            best_n_est + 100,
            best_n_est + 200
        ],
        'max_depth': [
            max(2, best_depth - 1),
            best_depth,
            best_depth + 1,
            best_depth + 2
        ],
        'learning_rate': [
            best_lr * 0.5,
            best_lr * 0.75,
            best_lr,
            best_lr * 1.25,
            best_lr * 1.5
        ],
        'subsample': [
            max(0.5, best_subsample - 0.1),
            best_subsample,
            min(1.0, best_subsample + 0.1)
        ],
        'colsample_bytree': [
            max(0.5, best_colsample - 0.1),
            best_colsample,
            min(1.0, best_colsample + 0.1)
        ],
        'min_child_weight': [
            max(1, best_min_child - 1),
            best_min_child,
            best_min_child + 1,
            best_min_child + 2
        ],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0.1, 1.0, 10.0]
    }

    # 25 itérations sur narrow ranges
    search_2 = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'),
        param_dist_2,
        n_iter=25,  # Test 25 combinaisons aléatoires (fine tuning)
        cv=5,       # 5-fold CV (plus robuste)
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("\n[Fitting Phase 2 model...]")
    search_2.fit(X_train, y_train)

    # Résultats Phase 2
    best_params_2 = search_2.best_params_
    best_score_2 = search_2.best_score_

    print(f"\n[PHASE 2 RESULTS]")
    print(f"  Best CV Score: {best_score_2:.4f}")
    print(f"  CV Score Improvement: +{(best_score_2 - best_params_1.get('cv_score', best_score_2))*100:.2f}%")
    print(f"  Best Parameters:")
    for param, value in sorted(best_params_2.items()):
        print(f"    - {param}: {value}")

    # Validation sur set de validation
    model_2 = XGBRegressor(**best_params_2, random_state=42, n_jobs=-1, eval_metric='rmse')
    model_2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred_val = model_2.predict(X_val)
    r2_val = r2_score(y_val, y_pred_val)
    print(f"  Validation R²: {r2_val:.4f}")

    return best_params_2

# ============================================================================
# PHASE 3: TRAIN FINAL MODEL + EVALUATION
# ============================================================================

def phase3_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Phase 3: Train modèle final avec best_params optimisés
    Évalue sur test set
    """
    print("\n" + "="*80)
    print("PHASE 3: FINAL MODEL TRAINING & EVALUATION")
    print("="*80)

    print(f"\n[Creating final model with optimized hyperparameters...]")

    final_model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric='rmse',
        verbosity=0
    )

    print(f"[Training on full training set ({len(X_train):,} samples)...]")
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Prédictions
    y_pred = final_model.predict(X_test)

    # Métriques
    r2 = r2_score(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_log = mean_absolute_error(y_test, y_pred)

    # Conversion en prix USD original (si y est en log10)
    y_test_original = 10 ** y_test
    y_pred_original = 10 ** y_pred
    rmse_usd = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    # Affichage résultats
    print(f"\n[FINAL MODEL PERFORMANCE ON TEST SET]")
    print(f"  ════════════════════════════════════════════")
    print(f"  R² Score:           {r2:.4f}")
    print(f"  RMSE (log scale):   {rmse_log:.4f}")
    print(f"  MAE (log scale):    {mae_log:.4f}")
    print(f"  ════════════════════════════════════════════")
    print(f"  RMSE (USD):         ${rmse_usd:,.0f}")
    print(f"  MAPE:               {mape:.1f}%")
    print(f"  ════════════════════════════════════════════")
    print(f"\n  Test set size: {len(X_test):,} samples")
    print(f"  Mean prediction error: {np.mean(np.abs(y_test - y_pred)):.4f} (log)")

    return final_model, {
        'r2': r2,
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'rmse_usd': rmse_usd,
        'mape': mape,
        'y_pred': y_pred,
        'y_test': y_test,
        'best_params': best_params
    }

# ============================================================================
# COMPARAISON BEFORE/AFTER
# ============================================================================

def compare_models(old_metrics, new_metrics):
    """Affiche la comparaison avant/après optimisation"""
    print("\n" + "="*80)
    print("BEFORE vs AFTER OPTIMIZATION")
    print("="*80)

    metrics_to_compare = ['r2', 'rmse_log', 'rmse_usd', 'mape']

    print(f"{'Metric':<20} {'BEFORE':<15} {'AFTER':<15} {'CHANGE':<15}")
    print("-"*80)

    for metric in metrics_to_compare:
        if metric in old_metrics and metric in new_metrics:
            before = old_metrics[metric]
            after = new_metrics[metric]

            if metric == 'r2':
                change = after - before
                pct = (change / before * 100) if before != 0 else 0
                print(f"{metric:<20} {before:.4f}        {after:.4f}        +{pct:+.1f}%")
            else:
                change = after - before
                pct = (change / before * 100) if before != 0 else 0
                print(f"{metric:<20} {before:,.0f}        {after:,.0f}        {pct:+.1f}%")

# ============================================================================
# FULL OPTIMIZATION PIPELINE
# ============================================================================

def optimize_xgboost_full(X_train, y_train, X_test, y_test, old_metrics=None):
    """
    Pipeline complet d'optimisation XGBoost

    Arguments:
        X_train, y_train: Training data
        X_test, y_test: Test data
        old_metrics: Dictionnaire avec anciennes métriques (optionnel)

    Returns:
        final_model: Modèle XGBoost optimisé
        new_metrics: Dictionnaire avec nouvelles métriques

    Example:
        from xgboost_optimization import optimize_xgboost_full

        final_model, metrics = optimize_xgboost_full(
            X_train, y_train, X_test, y_test,
            old_metrics=baseline_metrics
        )
    """

    print("\n" + "="*80)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION - 3 PHASE PIPELINE")
    print("="*80)
    print(f"\nTraining samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:         {X_train.shape[1]}")

    # Split training en train + validation (pour Phase 1 & 2)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # PHASE 1: Quick Exploration
    best_params_1, cv_results_1 = phase1_quick_exploration(
        X_train_split, y_train_split, X_val, y_val
    )

    # PHASE 2: Fine Tuning
    best_params_2 = phase2_fine_tuning(
        X_train_split, y_train_split, X_val, y_val, best_params_1
    )

    # PHASE 3: Final Model
    final_model, new_metrics = phase3_final_model(
        X_train, y_train, X_test, y_test, best_params_2
    )

    # Comparaison before/after si old_metrics fourni
    if old_metrics:
        compare_models(old_metrics, new_metrics)

    return final_model, new_metrics

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\nIMPORTANT: Use this module in your ML projects")
    print("\nExample usage:")
    print("""
    from xgboost_optimization import optimize_xgboost_full
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    # Prepare your data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # OPTION 1: Just optimize
    final_model, metrics = optimize_xgboost_full(
        X_train, y_train, X_test, y_test
    )

    # OPTION 2: Compare with baseline
    # Train baseline first
    baseline = XGBRegressor(n_estimators=500, max_depth=5)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    baseline_metrics = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Then optimize
    final_model, new_metrics = optimize_xgboost_full(
        X_train, y_train, X_test, y_test,
        old_metrics=baseline_metrics
    )
    """)
