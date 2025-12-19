import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sqlalchemy import text  # <-- FIX 1 (Corregido)

def save_backtest_results_to_db(
    engine,
    projection_name: str,
    scope: str,              # ej: "Total_Filtrado", "SUV", etc.
    horizon_months: int,
    train_start,
    train_end,
    test_start,
    test_end,
    leaderboard_dict: dict,
):
    """
    Guarda en BD la metadata de la corrida (forecast_runs)
    y el leaderboard por modelo (forecast_model_metrics).

    - forecast_model_metrics: NO usa columna 'scope' (asumimos que no existe).
    - forecast_runs: SÍ usa columna 'scope' (asumimos que es NOT NULL).
    """

    # 1) Pasar el dict de métricas a DataFrame
    rows = []
    for model_name, m in leaderboard_dict.items():
        if not isinstance(m, dict) or "MAPE" not in m:
            continue

        # sklearn MAPE viene como proporción (0–1). Lo llevamos a %
        mape_pct = float(m["MAPE"]) * 100.0

        rows.append({
            "Modelo": str(model_name),
            "MAPE": float(mape_pct),
            "MAE": float(m["MAE"]),
            "RMSE": float(m["RMSE"]),
        })

    df_lb = pd.DataFrame(rows)

    if df_lb.empty:
        print(f"⚠️ Leaderboard vacío, no se guardan métricas en BD para '{projection_name}'.")
        return

    # 2) Mejor modelo por menor MAPE
    best_row = df_lb.sort_values("MAPE").iloc[0]
    best_model = str(best_row["Modelo"])
    best_mape  = float(best_row["MAPE"])
    best_mae   = float(best_row["MAE"])
    best_rmse  = float(best_row["RMSE"])

    with engine.begin() as conn:
        # 3) Borrar métricas anteriores de forecast_model_metrics SOLO por projection_name
        conn.execute(
            text("""
                DELETE FROM forecast_model_metrics
                WHERE projection_name = :projection_name
            """),
            {"projection_name": projection_name},
        )

        # 4) Borrar corrida anterior en forecast_runs por projection_name + scope
        conn.execute(
            text("""
                DELETE FROM forecast_runs
                WHERE projection_name = :projection_name
                  AND scope = :scope
            """),
            {"projection_name": projection_name, "scope": scope},
        )

        # 5) Insertar nueva corrida en forecast_runs
        conn.execute(
            text("""
                INSERT INTO forecast_runs
                (projection_name, scope, horizon_months,
                 train_start, train_end, test_start, test_end,
                 best_model, best_mape, best_mae, best_rmse)
                VALUES
                (:projection_name, :scope, :horizon_months,
                 :train_start, :train_end, :test_start, :test_end,
                 :best_model, :best_mape, :best_mae, :best_rmse)
            """),
            {
                "projection_name": projection_name,
                "scope": scope,
                "horizon_months": int(horizon_months),
                "train_start": train_start.date(),
                "train_end": train_end.date(),
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "best_model": best_model,
                "best_mape": best_mape,
                "best_mae": best_mae,
                "best_rmse": best_rmse,
            },
        )

        # 6) Insertar leaderboard en forecast_model_metrics (sin scope)
        for _, row in df_lb.iterrows():
            conn.execute(
                text("""
                    INSERT INTO forecast_model_metrics
                    (projection_name, model_name, mape, mae, rmse)
                    VALUES (:projection_name, :model_name, :mape, :mae, :rmse)
                """),
                {
                    "projection_name": projection_name,
                    "model_name": row["Modelo"],
                    "mape": float(row["MAPE"]),
                    "mae": float(row["MAE"]),
                    "rmse": float(row["RMSE"]),
                },
            )

    print(
        f"✅ Métricas de backtest guardadas en BD para "
        f"projection_name='{projection_name}', scope='{scope}' (best={best_model})"
    )

try:
    from pmdarima import auto_arima
    PMDARIMA_INSTALLED = True
    print("pmdarima encontrado. Se usará auto_arima.")
except ImportError:
    PMDARIMA_INSTALLED = False
    print("pmdarima no encontrado. Se usará SARIMAX simple.")

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split 
import warnings
import traceback # Para imprimir errores

# Suprimir advertencias
warnings.filterwarnings("ignore")

# --- 🔧 INICIO v6.8: Definir fechas de SHOCK para excluirlas del cálculo de error ---
SHOCK_DATES_TO_EXCLUDE_FROM_MAPE = [
    '2019-10-01', # Paro 2019
    '2020-03-01', '2020-04-01', '2020-05-01', # COVID Lockdown
    '2022-06-01', # Paro 2022
    '2024-03-01', # Shock Cambio IVA
    '2024-04-01'  # Resaca Cambio IVA
]
SHOCK_DATES_TO_EXCLUDE_FROM_MAPE = pd.to_datetime(SHOCK_DATES_TO_EXCLUDE_FROM_MAPE)
# --- 🔧 FIN v6.8 ---


def load_data_from_db(engine):
    """
    Carga los datos históricos y los une.
    """
    print("Cargando 'sales_granular' (por segmento) desde la BD...")
    df_sales_sql = """
    SELECT
        DATE_FORMAT(fecha_proceso, '%Y-%m-01') as Fecha,
        segmento,
        SUM(unidades) as Ventas_Industria
    FROM
        sales_granular
    GROUP BY 1, 2
    """
    df_sales = pd.read_sql(text(df_sales_sql), engine, parse_dates=['Fecha'])

    print("Cargando 'historical_drivers' desde la BD...")
    df_drivers = pd.read_sql("SELECT * FROM historical_drivers", engine, parse_dates=['Fecha'])

    df_drivers.columns = [
        col.replace(' ', '_').replace('/', '_').replace('.', '_')
        for col in df_drivers.columns
    ]
    DUMMY_VARS_LIST = ['CUP_IMPORT', 'CAMBIO_IVA', 'ELEC_PRESIDENCIALES', 'PARO', 'UTILIDADES', 'DUMMY_COVID_LOCKDOWN']
    df_drivers.columns = [col.upper() if col.upper() in DUMMY_VARS_LIST else col for col in df_drivers.columns]

    df_master = pd.merge(df_sales, df_drivers, on='Fecha', how='inner')
    df_master = df_master.set_index('Fecha')
    print("DataFrame maestro (con segmentos) creado.")
    return df_master

def project_future_drivers(df_master_hist, horizon_months=18):
    """
    Proyecta drivers usando el motor robusto SARIMAX v3.3.
    """
    print(f"Iniciando proyección de drivers con SARIMAX v3.3 para {horizon_months} meses...")

    df_drivers_hist = df_master_hist.drop(columns=['segmento', 'Ventas_Industria'], errors='ignore')
    df_drivers_hist = df_drivers_hist[~df_drivers_hist.index.duplicated(keep='first')]

    DUMMY_VARS_LIST = ['UTILIDADES', 'PARO', 'CUP_IMPORT', 'ELEC_PRESIDENCIALES', 'CAMBIO_IVA', 'DUMMY_COVID_LOCKDOWN']
    
    vars_to_project = [
        col for col in df_drivers_hist.columns
        if col.upper() not in DUMMY_VARS_LIST
    ]
    
    dummy_vars_in_data = [col for col in df_drivers_hist.columns if col.upper() in DUMMY_VARS_LIST]

    future_index = pd.date_range(
        start=df_drivers_hist.index.max() + pd.offsets.MonthBegin(1),
        periods=horizon_months,
        freq='MS'
    )

    df_scenarios = pd.DataFrame(index=future_index)

    print(f"Proyectando {len(vars_to_project)} variables en 3 escenarios...")
    for var in vars_to_project:
        ts_data = df_drivers_hist[var].dropna().asfreq('MS')
        pred_mean, conf_int_lower, conf_int_upper = None, None, None
        model_used = "Fallback (Último Valor)"

        if var.upper() == 'IVA':
            print(f"  > {var} tratado como constante.")
            val = ts_data.iloc[-1] if not ts_data.empty else 0.15
            df_scenarios[f'{var}_normal'] = val
            df_scenarios[f'{var}_pesimista'] = val
            df_scenarios[f'{var}_optimista'] = val
            continue

        if ts_data.nunique(dropna=False) <= 1:
            print(f"  > {var} es constante o vacía. Rellenando.")
            val = ts_data.iloc[-1] if not ts_data.empty else 0
            df_scenarios[f'{var}_normal'] = val
            df_scenarios[f'{var}_pesimista'] = val
            df_scenarios[f'{var}_optimista'] = val
            continue

        try:
            if PMDARIMA_INSTALLED:
                try:
                    print(f"  > Intentando auto_arima para {var}...")
                    auto_model = auto_arima(
                        ts_data, seasonal=True, m=12, stepwise=True,
                        suppress_warnings=True, error_action='raise', maxiter=10
                    )
                    pred_mean_np, conf_int_np = auto_model.predict(
                        n_periods=horizon_months, return_conf_int=True, alpha=0.05
                    )
                    pred_mean = pd.Series(pred_mean_np, index=future_index)
                    conf_int_lower = conf_int_np[:, 0]
                    conf_int_upper = conf_int_np[:, 1]
                    model_used = "auto_arima"
                    print(f"    > auto_arima exitoso para {var}.")
                except Exception as e_auto:
                    print(f"    > auto_arima falló para {var}: {e_auto}. Intentando SARIMAX simple...")
                    pred_mean = None

            if pred_mean is None:
                try:
                    print(f"  > Intentando SARIMAX(1,1,1)(1,1,0,12) para {var}...")
                    ts_data_sarimax = ts_data.asfreq('MS').fillna(method='ffill').fillna(method='bfill')

                    if len(ts_data_sarimax) < 25:
                        print(f"    > Datos insuficientes. Usando ARIMA no estacional.")
                        model = SARIMAX(ts_data_sarimax, order=(1, 1, 1))
                    else:
                         model = SARIMAX(ts_data_sarimax, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))

                    fit = model.fit(disp=False)
                    pred_res = fit.get_forecast(steps=horizon_months)
                    pred_mean = pred_res.predicted_mean
                    conf_int_df = pred_res.conf_int(alpha=0.05)
                    conf_int_df.columns = ['lower', 'upper']
                    conf_int_lower = conf_int_df['lower'].values
                    conf_int_upper = conf_int_df['upper'].values
                    model_used = "SARIMAX Simple"
                    print(f"    > SARIMAX simple exitoso para {var}.")
                except Exception as e_sarimax:
                    raise e_sarimax

            df_scenarios[f'{var}_normal'] = pred_mean
            df_scenarios[f'{var}_pesimista'] = conf_int_lower
            df_scenarios[f'{var}_optimista'] = conf_int_upper

        except Exception as e:
            print(f"  > !!! ERROR FINAL al proyectar {var} con {model_used}: {e}. Usando el último valor.")
            last_val = ts_data.iloc[-1] if not ts_data.empty else 0
            df_scenarios[f'{var}_normal'] = last_val
            df_scenarios[f'{var}_pesimista'] = last_val
            df_scenarios[f'{var}_optimista'] = last_val
    
    for var in dummy_vars_in_data:
        df_scenarios[f'{var}_normal'] = 0
        df_scenarios[f'{var}_pesimista'] = 0
        df_scenarios[f'{var}_optimista'] = 0

    print("Proyección de drivers por escenario completada.")
    return df_scenarios


# --- Funciones de ML ---

def _create_features(df):
    """
    v6.6: REVERTIDO. Lags 2 y 3 demostraron ser ruido y empeoraron el MAPE.
    Volvemos al modelo v5.9 que tenía un MAPE de ~5%.
    """
    df_feat = df.copy()
    
    if not isinstance(df_feat.index, pd.DatetimeIndex):
         df_feat = df_feat.set_index(pd.to_datetime(df_feat.index))

    cols_to_preserve = ['Ventas_Industria']
    if 'segmento' in df_feat.columns:
        cols_to_preserve.append('segmento')
        
    df_preserved = df_feat[cols_to_preserve]
    df_X = df_feat.drop(columns=cols_to_preserve, errors='ignore')
    
    df_X['month'] = df_X.index.month
    df_X['quarter'] = df_X.index.quarter
    
    lags_to_create = [1, 2, 3, 12]
    df_X_with_lags = df_X.copy()
    for col in df_X.columns:
        for lag in lags_to_create:
            df_X_with_lags[f'{col}_lag_{lag}'] = df_X[col].shift(lag)
    
    # --- 🔧 INICIO DE LA REVERSIÓN (v6.6) ---
    df_X_with_lags['Ventas_Industria_lag_1'] = df_preserved['Ventas_Industria'].shift(1)
    # df_X_with_lags['Ventas_Industria_lag_2'] = df_preserved['Ventas_Industria'].shift(2) # <-- ELIMINADO
    # df_X_with_lags['Ventas_Industria_lag_3'] = df_preserved['Ventas_Industria'].shift(3) # <-- ELIMINADO
    df_X_with_lags['Ventas_Industria_lag_12'] = df_preserved['Ventas_Industria'].shift(12)
    # --- 🔧 FIN DE LA REVERSIÓN (v6.6) ---
    
    df_X_with_lags['Ventas_Industria_rolling_mean_3m'] = df_preserved['Ventas_Industria'].shift(1).rolling(window=3).mean()
    df_X_with_lags['Ventas_Industria_rolling_mean_6m'] = df_preserved['Ventas_Industria'].shift(1).rolling(window=6).mean()
    df_X_with_lags['Ventas_Industria_rolling_std_6m'] = df_preserved['Ventas_Industria'].shift(1).rolling(window=6).std()
    
    df_X_clean = df_X_with_lags.ffill().fillna(0)

    df_processed = pd.concat([df_preserved, df_X_clean], axis=1)

    historical_rows_clean = df_processed.loc[df_preserved['Ventas_Industria'].notna()].dropna()
    future_rows = df_processed.loc[df_preserved['Ventas_Industria'].isna()]
    
    return pd.concat([historical_rows_clean, future_rows])


def run_model_competition(
    df_master_hist,
    segment_name,
    engine=None,
    projection_name: str = None,
    horizon_months: int = None,
    scope: str = None,
):
    """
    v6.8: Excluir SHOCK_DATES del cálculo de MAPE para obtener un error "real".
    Si se pasa engine + projection_name, guarda los resultados de backtest en BD.
    """
    print(f"Iniciando competencia de modelos v6.8 (Walk-Forward CV) para: {segment_name}")
    
    df_segment = df_master_hist[df_master_hist['segmento'] == segment_name].copy()
    df_clean_no_lags = df_segment.drop(columns=['segmento'], errors='ignore')
    
    df_clean_with_features = _create_features(df_clean_no_lags) # <-- AHORA USA v6.6

    if len(df_clean_with_features) < 36: 
        print(f"Advertencia: Datos insuficientes (< 36 meses) para CV en {segment_name}")
        return {"Error": {"MAPE": 999}}, {}, 999, {}

    Y_ml_full = df_clean_with_features['Ventas_Industria'].dropna()
    X_ml_full = df_clean_with_features.loc[Y_ml_full.index].drop(columns=['Ventas_Industria'])

    df_clean_no_lags_aligned = df_clean_no_lags.loc[Y_ml_full.index]
    Y_stats_full = df_clean_no_lags_aligned['Ventas_Industria']
    X_stats_full = df_clean_no_lags_aligned.drop(columns=['Ventas_Industria'])

    N_SPLITS = 5 
    TEST_MONTHS_PER_FOLD = 4 
    
    total_obs = len(Y_ml_full)
    initial_train_size = total_obs - (N_SPLITS * TEST_MONTHS_PER_FOLD)
    
    if initial_train_size < 24: 
        print(f"Datos insuficientes para {N_SPLITS} splits de {TEST_MONTHS_PER_FOLD} meses.")
        return {"Error": {"MAPE": 999}}, {}, 999, {}

    # Metadatos del CV para poder guardarlos luego junto con la proyección final
    train_start = Y_ml_full.index[0]
    train_end = Y_ml_full.index[initial_train_size - 1]
    test_start = Y_ml_full.index[initial_train_size]
    test_end = Y_ml_full.index[-1]

    cv_metadata = {
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
    }
        
    print(f"Configuración de CV: {N_SPLITS} splits, {TEST_MONTHS_PER_FOLD} meses/fold. Entreno inicial: {initial_train_size} meses.")

    models = {
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SARIMAX": "SARIMAX_MODEL", 
        "Prophet": "PROPHET_MODEL" 
    }
    
    model_predictions = {name: [] for name in models.keys()}
    all_y_test_folds = [] 

    for i in range(N_SPLITS):
        train_end_idx = initial_train_size + (i * TEST_MONTHS_PER_FOLD)
        test_end_idx = train_end_idx + TEST_MONTHS_PER_FOLD
        
        print(f"  > Fold {i+1}/{N_SPLITS}: Train [0:{train_end_idx}], Test [{train_end_idx}:{test_end_idx}]")

        X_train_ml, Y_train_ml = X_ml_full.iloc[:train_end_idx], Y_ml_full.iloc[:train_end_idx]
        X_test_ml, Y_test_ml = X_ml_full.iloc[train_end_idx:test_end_idx], Y_ml_full.iloc[train_end_idx:test_end_idx]
        
        X_train_stats, Y_train_stats = X_stats_full.iloc[:train_end_idx], Y_stats_full.iloc[:train_end_idx]
        X_test_stats, Y_test_stats = X_stats_full.iloc[train_end_idx:test_end_idx], Y_stats_full.iloc[train_end_idx:test_end_idx]

        all_y_test_folds.append(Y_test_ml) 
        
        X_train_ml_clean = X_train_ml.replace([np.inf, -np.inf], np.nan).fillna(0)
        Y_train_ml_clean = Y_train_ml.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test_ml_clean = X_test_ml.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X_train_stats_clean = X_train_stats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        Y_train_stats_clean = Y_train_stats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        X_test_stats_clean = X_test_stats.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        for name, model_instance in models.items():
            try:
                Y_pred_test = None
                
                if name == "Prophet":
                    model = Prophet() 
                    df_prophet_train = X_train_stats_clean.copy()
                    df_prophet_train['ds'] = Y_train_stats_clean.index
                    df_prophet_train['y'] = Y_train_stats_clean.values
                    for col in X_train_stats_clean.columns: model.add_regressor(col)
                    
                    model.fit(df_prophet_train) 
                    
                    df_prophet_test = X_test_stats_clean.copy()
                    df_prophet_test['ds'] = Y_test_stats.index
                    Y_pred_test_df = model.predict(df_prophet_test)
                    Y_pred_test = Y_pred_test_df['yhat'].values

                elif name == "SARIMAX":
                    model = SARIMAX(Y_train_stats_clean, exog=X_train_stats_clean, order=(1,1,1), seasonal_order=(1,1,0,12))
                    fit = model.fit(disp=False)
                    Y_pred_test = fit.forecast(steps=len(Y_test_stats), exog=X_test_stats_clean)
                
                else: # XGBoost y RandomForest
                    model_instance.fit(X_train_ml_clean, Y_train_ml_clean)
                    Y_pred_test = model_instance.predict(X_test_ml_clean)
                
                pred_series = pd.Series(Y_pred_test, index=Y_test_ml.index)
                model_predictions[name].append(pred_series)
                
            except Exception as e:
                print(f"Error entrenando {name} en Fold {i+1}: {e}")
                nan_series = pd.Series(np.nan, index=Y_test_ml.index)
                model_predictions[name].append(nan_series)

    # --- 6. Calcular Métricas Agregadas ---
    metrics_dict = {}
    backtest_forecasts_dict = {}
    
    all_y_test_concat = pd.concat(all_y_test_folds).dropna()
    
    for name, preds_list in model_predictions.items():
        all_preds_concat = pd.concat(preds_list)
        
        # --- 🔧 INICIO v6.8: Filtrar Shocks ANTES de calcular el error ---
        print(f"Calculando métricas para {name}...")
        
        # 1. Alinear predicciones y reales
        aligned_preds, aligned_y = all_preds_concat.align(all_y_test_concat, join='inner')
        
        # 2. Crear un DataFrame temporal para filtrar
        df_metrics_temp = pd.DataFrame({'y_true': aligned_y, 'y_pred': aligned_preds}).dropna()
        
        # 3. Filtrar (excluir) las fechas de shock
        df_metrics_clean = df_metrics_temp[~df_metrics_temp.index.isin(SHOCK_DATES_TO_EXCLUDE_FROM_MAPE)]
        print(f"  > Original CV points: {len(df_metrics_temp)}, Clean CV points (no shocks): {len(df_metrics_clean)}")
        # --- 🔧 FIN v6.8 ---

        if df_metrics_clean.empty:
            mape, mae, rmse = 999, 999999, 999999
        else:
            # 4. Calcular métricas SÓLO en los datos limpios
            mape = mean_absolute_percentage_error(df_metrics_clean['y_true'], df_metrics_clean['y_pred'])
            mae = mean_absolute_error(df_metrics_clean['y_true'], df_metrics_clean['y_pred'])
            rmse = np.sqrt(mean_squared_error(df_metrics_clean['y_true'], df_metrics_clean['y_pred']))
            
        metrics_dict[name] = {"MAPE": mape, "MAE": mae, "RMSE": rmse}
        backtest_forecasts_dict[name] = all_preds_concat # Guardar TODAS las predicciones (incluyendo shocks) para el gráfico
        
    sorted_results = dict(sorted(metrics_dict.items(), key=lambda item: item[1]['MAPE']))
    
    winner_name = list(sorted_results.keys())[0]
    winner_rmse = sorted_results[winner_name]['RMSE'] # <-- Usamos el RMSE "limpio"
    winner_backtest_data = {}
    
    print(f"El ganador de CV (sin shocks) es: {winner_name}. Re-entrenando en datos completos para residuos...")
    
    try:
        if winner_name == "Prophet":
            X_train_stats_clean = X_stats_full.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            Y_train_stats_clean = Y_stats_full.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            
            model = Prophet()
            df_prophet_train = X_train_stats_clean.copy()
            df_prophet_train['ds'] = Y_train_stats_clean.index
            df_prophet_train['y'] = Y_train_stats_clean.values
            for col in X_train_stats_clean.columns: model.add_regressor(col)
            
            model.fit(df_prophet_train)
            
            Y_pred_train = model.predict(df_prophet_train)['yhat'].values
            winner_backtest_data = {"name": winner_name, "Y_train": Y_train_stats_clean, "Y_pred_train": Y_pred_train}

        elif winner_name == "SARIMAX":
            X_train_stats_clean = X_stats_full.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            Y_train_stats_clean = Y_stats_full.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            
            model = SARIMAX(Y_train_stats_clean, exog=X_train_stats_clean, order=(1,1,1), seasonal_order=(1,1,0,12))
            fit = model.fit(disp=False)
            Y_pred_train = fit.predict(start=Y_train_stats_clean.index[0], end=Y_train_stats_clean.index[-1], exog=X_train_stats_clean)
            winner_backtest_data = {"name": winner_name, "Y_train": Y_train_stats_clean, "Y_pred_train": Y_pred_train}

        else: # XGBoost o RandomForest
            X_train_ml_clean = X_ml_full.replace([np.inf, -np.inf], np.nan).fillna(0)
            Y_train_ml_clean = Y_ml_full.replace([np.inf, -np.inf], np.nan).fillna(0)

            if winner_name == "XGBoost":
                model = XGBRegressor(n_estimators=500, learning_rate=0.01, random_state=42)
            else: # RandomForest
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X_train_ml_clean, Y_train_ml_clean) 
            Y_pred_train = model.predict(X_train_ml_clean)
            winner_backtest_data = {"name": winner_name, "Y_train": Y_train_ml_clean, "Y_pred_train": Y_pred_train}
        
        print("Datos de residuos del ganador generados.")
        
    except Exception as e:
        print(f"Error al re-entrenar al ganador para residuos: {e}")
        winner_backtest_data = {"name": winner_name, "Y_train": pd.Series(), "Y_pred_train": pd.Series()}

    return sorted_results, backtest_forecasts_dict, winner_rmse, winner_backtest_data, cv_metadata


def run_final_forecast(df_master_hist, segment_name, df_future_drivers, model_name, winner_rmse, scenario_suffix='_normal'):
    """
    v6.6: Revertido a v6.0 (sin lags 2 y 3) para restaurar el MAPE bajo.
    """
    print(f"Ejecutando forecast final v6.6 con {model_name} para {segment_name} (Escenario: {scenario_suffix})")
    
    df_hist_segment = df_master_hist[df_master_hist['segmento'] == segment_name].copy()
    df_hist_clean = df_hist_segment.drop(columns=['segmento'], errors='ignore')
    
    if df_future_drivers is None or df_future_drivers.empty:
         print("ERROR: df_future_drivers (el input) está vacío. No se puede pronosticar.")
         raise ValueError("df_future_drivers está vacío o no fue generado.")

    print(f"Filtrando drivers para el escenario: {scenario_suffix}")
    df_future_drivers_clean = df_future_drivers.filter(like=scenario_suffix)

    if df_future_drivers_clean.empty:
        print(f"ADVERTENCIA: df_future_drivers no contenía columnas '{scenario_suffix}'.")
        if scenario_suffix != '_normal':
            print("Fallback a '_normal'.")
            df_future_drivers_clean = df_future_drivers.filter(like="_normal")
            scenario_suffix = '_normal'
        else:
            raise ValueError(f"df_future_drivers no contenía columnas '{scenario_suffix}'.")

    df_future_drivers_clean.columns = [col.replace(scenario_suffix, "") for col in df_future_drivers_clean.columns]

    hist_driver_cols = df_hist_clean.columns.drop('Ventas_Industria')
    
    missing_in_future = [col for col in hist_driver_cols if col not in df_future_drivers_clean.columns]
    if missing_in_future:
        print(f"Advertencia: Faltan {len(missing_in_future)} drivers en el escenario. Rellenando con 0.")
        for col in missing_in_future:
            df_future_drivers_clean[col] = 0
    
    df_future_drivers_clean = df_future_drivers_clean[hist_driver_cols]
    df_future_drivers_clean['Ventas_Industria'] = np.nan

    # --- Inicio de Lógica Específica del Modelo ---
    Y_train = None
    X_train = None
    X_forecast = None
    
    final_model = None
    X_train_clean_final = None
    Y_train_clean_final = None
    
    if model_name in ["XGBoost", "RandomForest"]:
        print("Aplicando pre-procesamiento de Features para modelo ML...")
        df_full_timeline = pd.concat([df_hist_clean, df_future_drivers_clean]) 
        df_processed = _create_features(df_full_timeline) # <-- AHORA USA v6.6

        data_to_train = df_processed.loc[df_processed['Ventas_Industria'].notna()]
        data_to_forecast = df_processed.loc[df_processed['Ventas_Industria'].isna()]
        
        if data_to_forecast.empty: 
             print("ERROR: data_to_forecast está vacío DESPUÉS de _create_features.")
             return pd.DataFrame(columns=['Ventas_Proyectadas', 'Ventas_Minimo', 'Ventas_Maximo']), None, None, None
        
        Y_train = data_to_train['Ventas_Industria']
        X_train = data_to_train.drop(columns=['Ventas_Industria']) # <-- Corregido
        X_forecast = data_to_forecast.drop(columns=['Ventas_Industria'])
    
    else: # SARIMAX o Prophet
        print("Usando pre-procesamiento simple (sin features) para modelo estadístico...")
        Y_train = df_hist_clean['Ventas_Industria']
        X_train = df_hist_clean.drop(columns=['Ventas_Industria'])
        X_forecast = df_future_drivers_clean.drop(columns=['Ventas_Industria'])

    if Y_train.empty or X_train.empty or X_forecast.empty:
         print(f"ERROR: Datos de entrenamiento (Y: {len(Y_train)}, X: {len(X_train)}) o pronóstico (X: {len(X_forecast)}) vacíos.")
         return pd.DataFrame(columns=['Ventas_Proyectadas', 'Ventas_Minimo', 'Ventas_Maximo']), None, None, None
         
    X_forecast = X_forecast[X_train.columns] # Asegura que las columnas coincidan

    if model_name in ["XGBoost", "RandomForest"]:
        X_train_clean_final = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        Y_train_clean_final = Y_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_forecast_clean_final = X_forecast.replace([np.inf, -np.inf], np.nan).fillna(0)
    else: 
        X_train_clean_final = X_train.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        Y_train_clean_final = Y_train.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
        X_forecast_clean_final = X_forecast.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    df_forecast_results = pd.DataFrame(index=df_future_drivers.index)

    try:
        if model_name == "Prophet":
            print("Preparando y entrenando Prophet (final)...")
            final_model = Prophet()
            df_prophet_train_final = X_train_clean_final.copy()
            df_prophet_train_final['ds'] = Y_train_clean_final.index
            df_prophet_train_final['y'] = Y_train_clean_final.values
            
            for col in X_train_clean_final.columns:
                final_model.add_regressor(col)
            
            final_model.fit(df_prophet_train_final)
            
            df_prophet_test_final = X_forecast_clean_final.copy()
            df_prophet_test_final['ds'] = X_forecast_clean_final.index
            
            pred_df = final_model.predict(df_prophet_test_final)
            
            df_forecast_results['Ventas_Proyectadas'] = pred_df['yhat'].values
            df_forecast_results['Ventas_Minimo'] = pred_df['yhat_lower'].values
            df_forecast_results['Ventas_Maximo'] = pred_df['yhat_upper'].values

        elif model_name == "SARIMAX":
            print("Preparando y entrenando SARIMAX (final)...")
            final_model = SARIMAX(Y_train_clean_final, exog=X_train_clean_final, order=(1,1,1), seasonal_order=(1,1,0,12))
            fit = final_model.fit(disp=False)
            
            pred_res = fit.get_forecast(steps=len(X_forecast_clean_final), exog=X_forecast_clean_final)
            conf_int_df = pred_res.conf_int(alpha=0.05)
            
            df_forecast_results['Ventas_Proyectadas'] = pred_res.predicted_mean
            df_forecast_results['Ventas_Minimo'] = conf_int_df.iloc[:, 0]
            df_forecast_results['Ventas_Maximo'] = conf_int_df.iloc[:, 1]
            final_model = fit 

        else: # XGBoost o RandomForest
            print(f"Preparando y entrenando {model_name} (final)...")
            if model_name == "XGBoost":
                final_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, random_state=42)
            else: 
                final_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            final_model.fit(X_train_clean_final, Y_train_clean_final) 
            
            # --- PRONÓSTICO DINÁMICO (v6.6 - Sincronizado) ---
            print("Iniciando pronóstico dinámico (v6.6)...")
            X_forecast_dynamic = X_forecast_clean_final.copy()
            predictions = []
            
            df_hist_for_roll = df_master_hist[
                (df_master_hist['segmento'] == segment_name)
            ]['Ventas_Industria'].to_frame()

            for i in range(len(X_forecast_dynamic)):
                current_row_df = X_forecast_dynamic.iloc[[i]].copy()
                
                if i > 0:
                    last_pred = predictions[-1]
                    last_pred_date = X_forecast_dynamic.index[i-1]
                    df_hist_for_roll.loc[last_pred_date] = last_pred
                    
                    # --- 🔧 INICIO DE LA REVERSIÓN (v6.6) ---
                    # El bucle ahora coincide con _create_features (sin lags 2 y 3)
                    current_row_df['Ventas_Industria_lag_1'] = last_pred
                    if i > 11: current_row_df['Ventas_Industria_lag_12'] = predictions[-12]
                    # --- 🔧 FIN DE LA REVERSIÓN (v6.6) ---

                    temp_rolling_3m = df_hist_for_roll['Ventas_Industria'].shift(1).rolling(window=3).mean()
                    temp_rolling_6m = df_hist_for_roll['Ventas_Industria'].shift(1).rolling(window=6).mean()
                    temp_rolling_std_6m = df_hist_for_roll['Ventas_Industria'].shift(1).rolling(window=6).std()
                    
                    if not temp_rolling_3m.loc[last_pred_date:].empty:
                        current_row_df['Ventas_Industria_rolling_mean_3m'] = temp_rolling_3m.loc[last_pred_date]
                    if not temp_rolling_6m.loc[last_pred_date:].empty:
                        current_row_df['Ventas_Industria_rolling_mean_6m'] = temp_rolling_6m.loc[last_pred_date]
                    if not temp_rolling_std_6m.loc[last_pred_date:].empty:
                        current_row_df['Ventas_Industria_rolling_std_6m'] = temp_rolling_std_6m.loc[last_pred_date]

                current_row_df_clean = current_row_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                current_pred = final_model.predict(current_row_df_clean)[0]
                predictions.append(current_pred)
            
            margin_of_error = 1.96 * winner_rmse
            df_forecast_results['Ventas_Proyectadas'] = predictions
            df_forecast_results['Ventas_Minimo'] = df_forecast_results['Ventas_Proyectadas'] - margin_of_error
            df_forecast_results['Ventas_Maximo'] = df_forecast_results['Ventas_Proyectadas'] + margin_of_error
            
    except Exception as e:
        print(f"ERROR FATAL en el pronóstico final con {model_name} para {scenario_suffix}: {e}")
        traceback.print_exc()
        
        df_forecast_results['Ventas_Proyectadas'] = 0
        df_forecast_results['Ventas_Minimo'] = 0
        df_forecast_results['Ventas_Maximo'] = 0
        
        return df_forecast_results, None, None, None

    print(f"Pronóstico de escenario {scenario_suffix} completado.")
    df_forecast_results = df_forecast_results.clip(lower=0).round(0).astype(int)

    return df_forecast_results, final_model, X_train_clean_final, Y_train_clean_final