# ===============================================================================
# APLICAÇÃO WEB STREAMLIT - ANÁLISE DE PRODUÇÃO DE ARROZ
# Demonstração Interativa para Portfólio
# ===============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Configurações da página
st.set_page_config(
    page_title="🌾 Rice Production Analyzer",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f0f8ff;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# ===============================================================================

@st.cache_data
def load_data():
    """Carrega e processa os dados"""
    try:
        # Tentar carregar dados reais
        df = pd.read_csv('X1.csv')
    except:
        # Dados de exemplo se não encontrar o arquivo
        st.warning("⚠️ Arquivo X1.csv não encontrado. Usando dados de exemplo.")
        np.random.seed(42)
        n_samples = 30
        df = pd.DataFrame({
            'ANNUAL': np.random.normal(1300, 200, n_samples),
            'avg_rain': np.random.normal(75, 15, n_samples),
            'Nitrogen': np.random.normal(80000, 20000, n_samples),
            'POTASH': np.random.normal(15000, 5000, n_samples),
            'PHOSPHATE': np.random.normal(40000, 10000, n_samples),
            'LOAMY_ALFISOL': np.random.uniform(0, 1, n_samples),
            'USTALF_USTOLLS': np.random.uniform(0, 1, n_samples),
            'VERTISOLS': np.random.uniform(0, 0.3, n_samples),
            'RICE_PRODUCTION': np.random.normal(1200, 300, n_samples)
        })
        
        # Normalizar proporções de solo
        soil_cols = ['LOAMY_ALFISOL', 'USTALF_USTOLLS', 'VERTISOLS']
        df[soil_cols] = df[soil_cols].div(df[soil_cols].sum(axis=1), axis=0)
    
    return df

@st.cache_data
def feature_engineering(df):
    """Engenharia de features"""
    df_fe = df.copy()
    
    # Features de nutrientes
    df_fe['NPK_Total'] = df_fe['Nitrogen'] + df_fe['POTASH'] + df_fe['PHOSPHATE']
    df_fe['N_P_Ratio'] = df_fe['Nitrogen'] / (df_fe['PHOSPHATE'] + 1)
    df_fe['N_K_Ratio'] = df_fe['Nitrogen'] / (df_fe['POTASH'] + 1)
    df_fe['P_K_Ratio'] = df_fe['PHOSPHATE'] / (df_fe['POTASH'] + 1)
    
    # Features hídricas
    df_fe['Water_Efficiency'] = df_fe['RICE_PRODUCTION'] / (df_fe['ANNUAL'] + 1)
    df_fe['Rain_Intensity'] = df_fe['avg_rain'] / (df_fe['ANNUAL'] / 365)
    
    # Índice de fertilidade
    from sklearn.preprocessing import StandardScaler
    scaler_temp = StandardScaler()
    nutrients_normalized = scaler_temp.fit_transform(df_fe[['Nitrogen', 'POTASH', 'PHOSPHATE']])
    df_fe['Fertility_Index'] = np.mean(nutrients_normalized, axis=1)
    
    # Interações
    df_fe['Nitrogen_x_Rain'] = df_fe['Nitrogen'] * df_fe['avg_rain']
    
    return df_fe

@st.cache_resource
def train_model(df):
    """Treina o modelo preditivo"""
    # Preparar dados
    numeric_features = df.select_dtypes(include=[np.number]).columns.drop('RICE_PRODUCTION')
    X = df[numeric_features]
    y = df['RICE_PRODUCTION']
    
    # Split e normalização
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar modelo
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Métricas
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, scaler, r2, rmse, X_train.columns

# ===============================================================================
# INTERFACE PRINCIPAL
# ===============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Rice Production Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### 📊 Análise Inteligente para Otimização da Produção de Arroz")
    
    # Carregar dados
    with st.spinner("🔄 Carregando dados..."):
        df = load_data()
        df_enhanced = feature_engineering(df)
        model, scaler, r2, rmse, feature_names = train_model(df_enhanced)
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Painel de Controle")
    
    # Navegação
    page = st.sidebar.selectbox(
        "📍 Navegação",
        ["🏠 Dashboard", "📊 Análise Exploratória", "🤖 Predição", "🔍 Insights", "📈 Comparações"]
    )
    
    # Dashboard Principal
    if page == "🏠 Dashboard":
        dashboard_page(df_enhanced, model, scaler, r2, rmse)
     
    # Predição
    elif page == "🤖 Predição":
        prediction_page(df_enhanced, model, scaler, feature_names)
    
    # Insights
    elif page == "🔍 Insights":
        insights_page(df_enhanced)
    
    # Comparações
    elif page == "📈 Comparações":
        comparison_page(df_enhanced)

# ===============================================================================
# PÁGINAS DA APLICAÇÃO
# ===============================================================================

def dashboard_page(df, model, scaler, r2, rmse):
    """Página principal do dashboard"""
    
    st.markdown("## 🎯 Visão Geral")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_production = df['RICE_PRODUCTION'].mean()
        st.metric("🌾 Produção Média", f"{avg_production:.0f} kg/ha")
    
    with col2:
        best_efficiency = df['Water_Efficiency'].max()
        st.metric("💧 Melhor Eficiência", f"{best_efficiency:.2f} kg/mm")
    
    with col3:
        st.metric("🤖 Acurácia do Modelo", f"{r2:.1%}")
    
    with col4:
        st.metric("📉 Erro Médio", f"±{rmse:.0f} kg/ha")
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Distribuição da Produção")
        fig_hist = px.histogram(df, x='RICE_PRODUCTION', nbins=20,
                               title="Distribuição da Produção de Arroz",
                               color_discrete_sequence=['#2E8B57'])
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🔗 Correlações Principais")
        corr_data = df[['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain', 'RICE_PRODUCTION']].corr()['RICE_PRODUCTION'].drop('RICE_PRODUCTION')
        
        fig_corr = go.Figure(data=[
            go.Bar(x=corr_data.index, y=corr_data.values,
                  marker_color=['red' if x < 0 else 'green' for x in corr_data.values])
        ])
        fig_corr.update_layout(title="Correlação com Produção", showlegend=False)
        st.plotly_chart(fig_corr, use_container_width=True)
    

def eda_page(df):
    """Página de análise exploratória"""
    
    st.markdown("## 📊 Análise Exploratória Detalhada")
    
    # Seletor de variável
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_var = st.selectbox("Selecione uma variável para análise:", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Estatísticas descritivas
        st.subheader(f"📋 Estatísticas: {selected_var}")
        stats = df[selected_var].describe()
        st.dataframe(stats)
        
        # Histograma
        fig_hist = px.histogram(df, x=selected_var, nbins=15,
                               title=f"Distribuição de {selected_var}")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(df, y=selected_var, title=f"Box Plot: {selected_var}")
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Scatter com produção
        if selected_var != 'RICE_PRODUCTION':
            fig_scatter = px.scatter(df, x=selected_var, y='RICE_PRODUCTION',
                                   title=f"{selected_var} vs Produção de Arroz",
                                   trendline="ols")
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Análise multivariada
    st.subheader("🎨 Análise Multivariada")
    
    # Seletor de variáveis para scatter 3D
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("Eixo X:", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Eixo Y:", numeric_cols, index=1)
    with col3:
        color_var = st.selectbox("Cor:", numeric_cols, index=2)
    
    if x_var != y_var:
        fig_3d = px.scatter(df, x=x_var, y=y_var, color=color_var,
                           size='RICE_PRODUCTION', title="Análise Multivariada",
                           height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

def prediction_page(df, model, scaler, feature_names):
    """Página de predição interativa"""
    
    st.markdown("## 🤖 Preditor de Produção")
    st.markdown("### Ajuste os parâmetros e veja a predição em tempo real!")
    
    # Inputs do usuário
    st.subheader("🎛️ Parâmetros de Entrada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**☔ Fatores Climáticos**")
        annual_rain = st.slider("Precipitação Anual (mm)", 
                               float(df['ANNUAL'].min()), float(df['ANNUAL'].max()), 
                               float(df['ANNUAL'].mean()))
        
        avg_rain = st.slider("Chuva Média Mensal (mm)", 
                            float(df['avg_rain'].min()), float(df['avg_rain'].max()), 
                            float(df['avg_rain'].mean()))
        
        st.markdown("**🌱 Nutrientes**")
        nitrogen = st.slider("Nitrogênio (kg/ha)", 
                            float(df['Nitrogen'].min()), float(df['Nitrogen'].max()), 
                            float(df['Nitrogen'].mean()))
        
        potash = st.slider("Potássio (kg/ha)", 
                          float(df['POTASH'].min()), float(df['POTASH'].max()), 
                          float(df['POTASH'].mean()))
        
        phosphate = st.slider("Fósforo (kg/ha)", 
                             float(df['PHOSPHATE'].min()), float(df['PHOSPHATE'].max()), 
                             float(df['PHOSPHATE'].mean()))
    
    with col2:
        # Criar features engineered
        npk_total = nitrogen + potash + phosphate
        n_p_ratio = nitrogen / (phosphate + 1)
        water_eff_estimated = 1.0  # Placeholder
        fertility_index = (nitrogen + potash + phosphate) / 100000  # Simplificado
        nitrogen_x_rain = nitrogen * avg_rain
        
        st.markdown("**📊 Features Calculadas**")
        st.metric("NPK Total", f"{npk_total:,.0f} kg/ha")
        st.metric("Razão N:P", f"{n_p_ratio:.2f}")
        st.metric("Índice de Fertilidade", f"{fertility_index:.2f}")
        
        # Preparar dados para predição
        input_data = pd.DataFrame({
            'ANNUAL': [annual_rain],
            'avg_rain': [avg_rain],
            'Nitrogen': [nitrogen],
            'POTASH': [potash],
            'PHOSPHATE': [phosphate],
            'NPK_Total': [npk_total],
            'N_P_Ratio': [n_p_ratio],
            'N_K_Ratio': [nitrogen / (potash + 1)],
            'P_K_Ratio': [phosphate / (potash + 1)],
            'Water_Efficiency': [water_eff_estimated],
            'Rain_Intensity': [avg_rain / (annual_rain / 365)],
            'Fertility_Index': [fertility_index],
            'Nitrogen_x_Rain': [nitrogen_x_rain]
        })
        
        # Adicionar colunas faltantes com zeros
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reordenar colunas
        input_data = input_data[feature_names]
        
        # Fazer predição
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            st.markdown("### 🎯 Predição")
            
            # Classificar a predição
            if prediction > df['RICE_PRODUCTION'].quantile(0.75):
                status = "🟢 Excelente"
                color = "green"
            elif prediction > df['RICE_PRODUCTION'].quantile(0.5):
                status = "🟡 Boa"
                color = "orange"
            else:
                status = "🔴 Baixa"
                color = "red"
            
            st.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center; color: white; margin: 20px 0;'>
                <h2>🌾 {prediction:.0f} kg/ha</h2>
                <h3>{status}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Comparação com médias
            mean_production = df['RICE_PRODUCTION'].mean()
            difference = prediction - mean_production
            percentage = (difference / mean_production) * 100
            
            if difference > 0:
                st.success(f"📈 {difference:.0f} kg/ha acima da média ({percentage:+.1f}%)")
            else:
                st.error(f"📉 {abs(difference):.0f} kg/ha abaixo da média ({percentage:+.1f}%)")
                
        except Exception as e:
            st.error(f"Erro na predição: {str(e)}")

def insights_page(df):
    """Página de insights agronômicos"""
    
    st.markdown("## 🔍 Insights Agronômicos")
    
    # Top performers
    st.subheader("🏆 Análise dos Melhores Produtores")
    
    top_25_percent = df[df['RICE_PRODUCTION'] > df['RICE_PRODUCTION'].quantile(0.75)]
    bottom_25_percent = df[df['RICE_PRODUCTION'] < df['RICE_PRODUCTION'].quantile(0.25)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 TOP 25% Produtores**")
        top_stats = {
            'Produção Média': f"{top_25_percent['RICE_PRODUCTION'].mean():.0f} kg/ha",
            'Nitrogênio Médio': f"{top_25_percent['Nitrogen'].mean():,.0f} kg/ha",
            'Chuva Média': f"{top_25_percent['avg_rain'].mean():.1f} mm",
            'Eficiência Hídrica': f"{top_25_percent['Water_Efficiency'].mean():.2f} kg/mm"
        }
        
        for metric, value in top_stats.items():
            st.metric(metric, value)
    
    with col2:
        st.markdown("**🔴 BOTTOM 25% Produtores**")
        bottom_stats = {
            'Produção Média': f"{bottom_25_percent['RICE_PRODUCTION'].mean():.0f} kg/ha",
            'Nitrogênio Médio': f"{bottom_25_percent['Nitrogen'].mean():,.0f} kg/ha",
            'Chuva Média': f"{bottom_25_percent['avg_rain'].mean():.1f} mm",
            'Eficiência Hídrica': f"{bottom_25_percent['Water_Efficiency'].mean():.2f} kg/mm"
        }
        
        for metric, value in bottom_stats.items():
            st.metric(metric, value)
    
    # Recomendações
    st.subheader("💡 Recomendações Baseadas em Dados")
    
    optimal_nitrogen = top_25_percent['Nitrogen'].mean()
    optimal_potash = top_25_percent['POTASH'].mean()
    optimal_phosphate = top_25_percent['PHOSPHATE'].mean()
    optimal_rain = top_25_percent['avg_rain'].mean()
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>🎯 Fórmula Ótima (baseada nos 25% mais produtivos)</h4>
        <ul>
            <li><b>Nitrogênio:</b> {optimal_nitrogen:,.0f} kg/ha</li>
            <li><b>Potássio:</b> {optimal_potash:,.0f} kg/ha</li>
            <li><b>Fósforo:</b> {optimal_phosphate:,.0f} kg/ha</li>
            <li><b>Precipitação ideal:</b> {optimal_rain:.1f} mm/mês</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Análise de correlações
    st.subheader("🔗 Fatores Mais Impactantes")
    
    correlations = df.select_dtypes(include=[np.number]).corr()['RICE_PRODUCTION'].abs().sort_values(ascending=False).drop('RICE_PRODUCTION')
    
    fig_importance = px.bar(
        x=correlations.values[:8],
        y=correlations.index[:8],
        orientation='h',
        title="Top 8 Fatores Correlacionados com Produção"
    )
    st.plotly_chart(fig_importance, use_container_width=True)

def comparison_page(df):
    """Página de comparações"""
    
    st.markdown("## 📈 Análise Comparativa")
    
    # Comparação por quartis
    st.subheader("📊 Análise por Quartis de Produção")
    
    df['Production_Quartile'] = pd.qcut(df['RICE_PRODUCTION'], 4, labels=['Q1 (Baixa)', 'Q2', 'Q3', 'Q4 (Alta)'])
    
    # Selecionar variável para comparação
    compare_var = st.selectbox("Variável para Comparação:", 
                              ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain', 'ANNUAL', 'NPK_Total'])
    
    fig_box = px.box(df, x='Production_Quartile', y=compare_var,
                     title=f"Distribuição de {compare_var} por Quartil de Produção")
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Análise de eficiência
    st.subheader("⚡ Análise de Eficiência")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Eficiência hídrica vs produção
        fig_eff = px.scatter(df, x='Water_Efficiency', y='RICE_PRODUCTION',
                           color='Production_Quartile', size='NPK_Total',
                           title="Eficiência Hídrica vs Produção")
        st.plotly_chart(fig_eff, use_container_width=True)
    
    with col2:
        # NPK vs produção
        fig_npk = px.scatter(df, x='NPK_Total', y='RICE_PRODUCTION',
                           color='avg_rain', title="NPK Total vs Produção")
        st.plotly_chart(fig_npk, use_container_width=True)

# ===============================================================================
# FOOTER
# ===============================================================================

def show_footer():
    """Footer da aplicação"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🌾 <b>Rice Production Analyzer</b> | Desenvolvido para análise agrícola inteligente</p>
        <p>📊 Powered by Streamlit + Machine Learning | Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)

# ===============================================================================
# EXECUTAR APLICAÇÃO
# ===============================================================================

if __name__ == "__main__":
    main()

    show_footer()
