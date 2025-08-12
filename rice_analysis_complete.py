# ===============================================================================
# ANÁLISE COMPLETA: PREDIÇÃO DE PRODUÇÃO DE ARROZ
# Autor: Seu Nome
# Data: Agosto 2025
# ===============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# Interpretabilidade
import shap

# XGBoost
import xgboost as xgb

# Configurações de visualização
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ===============================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO INICIAL DOS DADOS
# ===============================================================================

def load_and_explore_data(file_path):
    """Carrega e faz exploração inicial dos dados"""
    
    print("="*60)
    print("🌾 ANÁLISE DE PRODUÇÃO DE ARROZ")
    print("="*60)
    
    # Carregamento
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dados carregados com sucesso!")
        print(f"📊 Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None
    
    # Informações básicas
    print("\n📋 INFORMAÇÕES BÁSICAS:")
    print("-" * 40)
    print(df.info())
    
    print("\n📊 ESTATÍSTICAS DESCRITIVAS:")
    print("-" * 40)
    print(df.describe())
    
    print("\n🔍 PRIMEIRAS 5 LINHAS:")
    print("-" * 40)
    print(df.head())
    
    print("\n❌ VALORES NULOS:")
    print("-" * 40)
    print(df.isnull().sum())
    
    return df

# ===============================================================================
# 2. ENGENHARIA DE FEATURES
# ===============================================================================

def feature_engineering(df):
    """Cria novas features relevantes para agricultura"""
    
    print("\n🔧 ENGENHARIA DE FEATURES")
    print("="*40)
    
    df_fe = df.copy()
    
    # 1. Features de Nutrientes
    print("📊 Criando features de nutrientes...")
    df_fe['NPK_Total'] = df_fe['Nitrogen'] + df_fe['POTASH'] + df_fe['PHOSPHATE']
    df_fe['N_P_Ratio'] = df_fe['Nitrogen'] / (df_fe['PHOSPHATE'] + 1)  # +1 para evitar divisão por 0
    df_fe['N_K_Ratio'] = df_fe['Nitrogen'] / (df_fe['POTASH'] + 1)
    df_fe['P_K_Ratio'] = df_fe['PHOSPHATE'] / (df_fe['POTASH'] + 1)
    
    # 2. Features Hídricas
    print("💧 Criando features hídricas...")
    df_fe['Water_Efficiency'] = df_fe['RICE_PRODUCTION'] / (df_fe['ANNUAL'] + 1)
    df_fe['Rain_Intensity'] = df_fe['avg_rain'] / (df_fe['ANNUAL'] / 365)  # chuva diária média
    
    # 3. Índice de Fertilidade
    print("🌱 Criando índice de fertilidade...")
    # Normalizar nutrientes para criar índice
    scaler_temp = StandardScaler()
    nutrients_normalized = scaler_temp.fit_transform(df_fe[['Nitrogen', 'POTASH', 'PHOSPHATE']])
    df_fe['Fertility_Index'] = np.mean(nutrients_normalized, axis=1)
    
    # 4. Tipo de Solo Dominante
    print("🏞️ Identificando solo dominante...")
    soil_cols = [col for col in df_fe.columns if col not in ['ANNUAL', 'avg_rain', 'Nitrogen', 
                                                            'POTASH', 'PHOSPHATE', 'RICE_PRODUCTION']]
    
    # Encontrar solo com maior proporção
    df_fe['Dominant_Soil'] = df_fe[soil_cols].idxmax(axis=1)
    df_fe['Soil_Diversity'] = (df_fe[soil_cols] > 0).sum(axis=1)  # Número de tipos de solo presentes
    
    # 5. Interações importantes
    print("🔗 Criando interações...")
    df_fe['Nitrogen_x_Rain'] = df_fe['Nitrogen'] * df_fe['avg_rain']
    df_fe['NPK_x_Water'] = df_fe['NPK_Total'] * df_fe['avg_rain']
    
    print(f"✅ Features criadas! Shape anterior: {df.shape}, nova: {df_fe.shape}")
    
    return df_fe

# ===============================================================================
# 3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ===============================================================================

def perform_eda(df):
    """Análise exploratória completa"""
    
    print("\n📊 ANÁLISE EXPLORATÓRIA DE DADOS")
    print("="*50)
    
    # Configurar subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Distribuição da Produção', 'Correlações com Produção',
                       'Nutrientes vs Produção', 'Precipitação vs Produção',
                       'Distribuição de Nutrientes', 'Eficiência Hídrica'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Distribuição da variável target
    fig.add_trace(
        go.Histogram(x=df['RICE_PRODUCTION'], name='Produção de Arroz', 
                    marker_color='lightblue', opacity=0.7),
        row=1, col=1
    )
    
    # 2. Correlações principais
    correlations = df[['ANNUAL', 'avg_rain', 'Nitrogen', 'POTASH', 'PHOSPHATE', 'RICE_PRODUCTION']].corr()['RICE_PRODUCTION'].drop('RICE_PRODUCTION')
    
    fig.add_trace(
        go.Bar(x=correlations.index, y=correlations.values, 
               name='Correlação com Produção',
               marker_color=['red' if x < 0 else 'green' for x in correlations.values]),
        row=1, col=2
    )
    
    # 3. Scatter: Nutrientes vs Produção
    fig.add_trace(
        go.Scatter(x=df['NPK_Total'], y=df['RICE_PRODUCTION'],
                  mode='markers', name='NPK Total vs Produção',
                  marker=dict(size=8, color=df['avg_rain'], 
                            colorscale='Viridis', showscale=True,
                            colorbar=dict(title="Chuva Média"))),
        row=2, col=1
    )
    
    # 4. Scatter: Precipitação vs Produção
    fig.add_trace(
        go.Scatter(x=df['ANNUAL'], y=df['RICE_PRODUCTION'],
                  mode='markers', name='Precipitação vs Produção',
                  marker=dict(size=8, color='orange')),
        row=2, col=2
    )
    
    # 5. Box plot de nutrientes
    nutrients = ['Nitrogen', 'POTASH', 'PHOSPHATE']
    for i, nutrient in enumerate(nutrients):
        fig.add_trace(
            go.Box(y=df[nutrient], name=nutrient, 
                  marker_color=px.colors.qualitative.Set1[i]),
            row=3, col=1
        )
    
    # 6. Eficiência Hídrica
    fig.add_trace(
        go.Scatter(x=df['Water_Efficiency'], y=df['RICE_PRODUCTION'],
                  mode='markers', name='Eficiência Hídrica',
                  marker=dict(size=10, color='purple')),
        row=3, col=2
    )
    
    # Atualizar layout
    fig.update_layout(height=1200, showlegend=True, 
                     title_text="Análise Exploratória - Produção de Arroz")
    fig.show()
    
    # Matriz de Correlação
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                center=0, square=True, fmt='.2f')
    plt.title('Matriz de Correlação', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return correlations

# ===============================================================================
# 4. ANÁLISE DE OUTLIERS
# ===============================================================================

def detect_outliers(df):
    """Detecta e analisa outliers"""
    
    print("\n🎯 DETECÇÃO DE OUTLIERS")
    print("="*30)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
        
        if len(outliers) > 0:
            print(f"⚠️  {col}: {len(outliers)} outliers detectados")
    
    # Visualizar outliers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    main_cols = ['RICE_PRODUCTION', 'Nitrogen', 'POTASH', 'PHOSPHATE', 'ANNUAL', 'avg_rain']
    
    for i, col in enumerate(main_cols):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'Outliers: {col}')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.suptitle('Análise de Outliers por Variável', fontsize=16, y=1.02)
    plt.show()
    
    return outliers_summary

# ===============================================================================
# 5. PREPROCESSAMENTO
# ===============================================================================

def preprocess_data(df):
    """Preprocessa dados para machine learning"""
    
    print("\n🔄 PREPROCESSAMENTO DOS DADOS")
    print("="*40)
    
    # Separar features numéricas
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('RICE_PRODUCTION')
    target = 'RICE_PRODUCTION'
    
    X = df[numeric_features]
    y = df[target]
    
    print(f"Features selecionadas: {len(numeric_features)}")
    print(f"Samples: {len(X)}")
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=numeric_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=numeric_features, index=X_test.index)
    
    print("✅ Preprocessamento concluído!")
    print(f"Treino: {X_train_scaled.shape}, Teste: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ===============================================================================
# 6. MODELAGEM MACHINE LEARNING
# ===============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """Treina múltiplos modelos e compara performance"""
    
    print("\n🤖 TREINAMENTO DE MODELOS")
    print("="*40)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=100, gamma=0.1)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"🔄 Treinando {name}...")
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[name] = y_pred_test
        
        # Métricas
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'Train R²': train_r2,
            'Test R²': test_r2,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std(),
            'Model': model
        }
        
        print(f"   Test R²: {test_r2:.4f} | RMSE: {test_rmse:.2f} | CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Criar DataFrame com resultados
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('Test R²', ascending=False)
    
    print("\n🏆 RANKING DOS MODELOS:")
    print("-" * 50)
    print(results_df[['Test R²', 'Test RMSE', 'CV R² Mean']].round(4))
    
    return results, predictions, results_df

# ===============================================================================
# 7. INTERPRETABILIDADE COM SHAP
# ===============================================================================

def interpret_model(best_model, X_train, X_test, feature_names):
    """Análise de interpretabilidade usando SHAP"""
    
    print("\n🔍 ANÁLISE DE INTERPRETABILIDADE")
    print("="*40)
    
    try:
        # Criar explainer SHAP
        if hasattr(best_model, 'predict'):
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer(X_test)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title('SHAP Summary Plot - Importância das Features')
        plt.tight_layout()
        plt.show()
        
        # Feature importance
        feature_importance = np.abs(shap_values.values).mean(0)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Plotar importância
        fig = px.bar(importance_df.head(10), x='Importance', y='Feature', 
                    orientation='h', title='Top 10 Features Mais Importantes')
        fig.show()
        
        print("✅ Análise de interpretabilidade concluída!")
        return importance_df
        
    except Exception as e:
        print(f"⚠️  Erro na interpretabilidade: {e}")
        return None

# ===============================================================================
# 8. VISUALIZAÇÕES DE PERFORMANCE
# ===============================================================================

def plot_model_performance(results, predictions, y_test):
    """Visualiza performance dos modelos"""
    
    print("\n📈 VISUALIZAÇÕES DE PERFORMANCE")
    print("="*40)
    
    # 1. Comparação de métricas
    models_names = list(results.keys())
    r2_scores = [results[model]['Test R²'] for model in models_names]
    rmse_scores = [results[model]['Test RMSE'] for model in models_names]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score Comparison', 'RMSE Comparison', 
                       'Predictions vs Actual', 'Residuals Analysis')
    )
    
    # R² comparison
    fig.add_trace(
        go.Bar(x=models_names, y=r2_scores, name='R² Score',
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # RMSE comparison
    fig.add_trace(
        go.Bar(x=models_names, y=rmse_scores, name='RMSE',
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    # Best model predictions vs actual
    best_model_name = max(results.keys(), key=lambda k: results[k]['Test R²'])
    best_predictions = predictions[best_model_name]
    
    fig.add_trace(
        go.Scatter(x=y_test, y=best_predictions, mode='markers',
                  name=f'{best_model_name} Predictions',
                  marker=dict(size=8, color='green')),
        row=2, col=1
    )
    
    # Perfect prediction line
    min_val = min(min(y_test), min(best_predictions))
    max_val = max(max(y_test), max(best_predictions))
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    # Residuals
    residuals = y_test - best_predictions
    fig.add_trace(
        go.Scatter(x=best_predictions, y=residuals, mode='markers',
                  name='Residuals', marker=dict(size=8, color='purple')),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=[min(best_predictions), max(best_predictions)], y=[0, 0],
                  mode='lines', name='Zero Line',
                  line=dict(color='black', dash='dash')),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True,
                     title_text=f"Análise de Performance - Melhor Modelo: {best_model_name}")
    fig.show()
    
    return best_model_name

# ===============================================================================
# 9. INSIGHTS E RECOMENDAÇÕES AGRONÔMICAS
# ===============================================================================

def generate_agricultural_insights(df, importance_df, correlations):
    """Gera insights agronômicos baseados na análise"""
    
    print("\n🌾 INSIGHTS AGRONÔMICOS")
    print("="*40)
    
    insights = []
    
    # 1. Análise de Nutrientes
    nitrogen_corr = correlations.get('Nitrogen', 0)
    potash_corr = correlations.get('POTASH', 0)
    phosphate_corr = correlations.get('PHOSPHATE', 0)
    
    if nitrogen_corr > 0.5:
        insights.append("💡 NITROGÊNIO: Forte correlação positiva com produção. Aumentar aplicação de N pode incrementar rendimento.")
    elif nitrogen_corr < -0.3:
        insights.append("⚠️  NITROGÊNIO: Correlação negativa detectada. Possível excesso causando perdas.")
    
    # 2. Análise Hídrica
    rain_corr = correlations.get('avg_rain', 0)
    annual_corr = correlations.get('ANNUAL', 0)
    
    if rain_corr > 0.3:
        insights.append("💧 IRRIGAÇÃO: Chuva média mostra correlação positiva. Manejo hídrico é crucial.")
    
    # 3. Análise de Solo
    soil_diversity_mean = df['Soil_Diversity'].mean()
    insights.append(f"🏞️  SOLO: Diversidade média de tipos de solo: {soil_diversity_mean:.1f}")
    
    # 4. Eficiência
    water_eff_mean = df['Water_Efficiency'].mean()
    insights.append(f"⚡ EFICIÊNCIA HÍDRICA: Média de {water_eff_mean:.2f} kg/mm de chuva")
    
    # 5. Recomendações baseadas em ranges ótimos
    high_producers = df[df['RICE_PRODUCTION'] > df['RICE_PRODUCTION'].quantile(0.75)]
    
    optimal_nitrogen = high_producers['Nitrogen'].mean()
    optimal_potash = high_producers['POTASH'].mean()
    optimal_phosphate = high_producers['PHOSPHATE'].mean()
    optimal_rain = high_producers['avg_rain'].mean()
    
    insights.extend([
        f"🎯 FÓRMULA ÓTIMA (baseada nos 25% mais produtivos):",
        f"   • Nitrogênio: {optimal_nitrogen:,.0f} kg/ha",
        f"   • Potássio: {optimal_potash:,.0f} kg/ha", 
        f"   • Fósforo: {optimal_phosphate:,.0f} kg/ha",
        f"   • Chuva ideal: {optimal_rain:.1f} mm/mês"
    ])
    
    # 6. Alertas de Sustentabilidade
    max_nitrogen = df['Nitrogen'].max()
    if max_nitrogen > 100000:  # Threshold alto
        insights.append("⚠️  SUSTENTABILIDADE: Níveis muito altos de N detectados. Revisar para evitar lixiviação.")
    
    print("\n".join(insights))
    
    return insights, {
        'optimal_nitrogen': optimal_nitrogen,
        'optimal_potash': optimal_potash,
        'optimal_phosphate': optimal_phosphate,
        'optimal_rain': optimal_rain
    }

# ===============================================================================
# 10. FUNÇÃO PRINCIPAL
# ===============================================================================

def main():
    """Função principal que executa toda a análise"""
    
    print("🚀 INICIANDO ANÁLISE COMPLETA DE PRODUÇÃO DE ARROZ")
    print("="*60)
    
    # 1. Carregar dados
    df = load_and_explore_data('X1.csv')
    if df is None:
        return
    
    # 2. Feature Engineering
    df_enhanced = feature_engineering(df)
    
    # 3. EDA
    correlations = perform_eda(df_enhanced)
    
    # 4. Detecção de outliers
    outliers = detect_outliers(df_enhanced)
    
    # 5. Preprocessamento
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df_enhanced)
    
    # 6. Modelagem
    results, predictions, results_df = train_models(X_train, X_test, y_train, y_test)
    
    # 7. Melhor modelo
    best_model_name = max(results.keys(), key=lambda k: results[k]['Test R²'])
    best_model = results[best_model_name]['Model']
    
    print(f"\n🏆 MELHOR MODELO: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['Test R²']:.4f}")
    print(f"   RMSE: {results[best_model_name]['Test RMSE']:.2f}")
    
    # 8. Interpretabilidade
    importance_df = interpret_model(best_model, X_train, X_test, X_train.columns)
    
    # 9. Visualizações de performance
    plot_model_performance(results, predictions, y_test)
    
    # 10. Insights agronômicos
    insights, optimal_params = generate_agricultural_insights(df_enhanced, importance_df, correlations)
    
    # 11. Resumo executivo
    print("\n" + "="*60)
    print("📊 RESUMO EXECUTIVO")
    print("="*60)
    print(f"🎯 Melhor Modelo: {best_model_name}")
    print(f"📈 Acurácia (R²): {results[best_model_name]['Test R²']:.1%}")
    print(f"📉 Erro Médio: ±{results[best_model_name]['Test RMSE']:.0f} kg/ha")
    print(f"🌾 Dataset: {len(df)} observações, {len(X_train.columns)} features")
    
    if importance_df is not None:
        top_feature = importance_df.iloc[0]['Feature']
        print(f"🔑 Feature mais importante: {top_feature}")
    
    print(f"💧 Eficiência hídrica média: {df_enhanced['Water_Efficiency'].mean():.2f} kg/mm")
    print("✅ Análise completa finalizada!")
    
    return {
        'df': df_enhanced,
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'importance': importance_df,
        'insights': insights,
        'optimal_params': optimal_params
    }

# ===============================================================================
# EXECUTAR ANÁLISE
# ===============================================================================

if __name__ == "__main__":
    # Executar análise completa
    analysis_results = main()
    
    # Salvar resultados
    print("\n💾 Salvando resultados...")
    
    # Salvar modelo (exemplo com pickle)
    # import pickle
    # with open('best_rice_model.pkl', 'wb') as f:
    #     pickle.dump(analysis_results['best_model'], f)
    
    # Salvar insights
    # with open('agricultural_insights.txt', 'w') as f:
    #     f.write('\n'.join(analysis_results['insights']))
    
    print("🎉 Projeto concluído com sucesso!")
    print("📁 Arquivos prontos para portfólio!")