# ===============================================================================
# APLICAÇÃO WEB STREAMLIT - ANÁLISE DE PRODUÇÃO DE ARROZ
# Versão Corrigida - Deploy
# ===============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost não disponível. Usando RandomForest como alternativa.")

# Configurações da página
st.set_page_config(
    page_title="🌾 AgriTech Analytics | Rice Production AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-color: #00C851;
        --secondary-color: #2E8B57;
        --accent-color: #FF6B35;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #FFFFFF;
        --text-secondary: #B0B0B0;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #00C851 0%, #2E8B57 100%);
        --shadow-light: rgba(0, 200, 81, 0.1);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: var(--gradient-primary);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 700;
        text-align: center;
        margin: 1.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px var(--shadow-light);
        transition: transform 0.3s ease;
        position: relative;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px var(--shadow-light);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-success);
        border-radius: 12px 12px 0 0;
    }
    
    .insight-box {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 200, 81, 0.2);
        border-left: 4px solid var(--primary-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 20px var(--shadow-light);
    }
    
    .insight-title {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.4rem;
    }
    
    .status-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .status-card.excellent {
        border-color: var(--primary-color);
        box-shadow: 0 0 20px rgba(0, 200, 81, 0.2);
    }
    
    .status-card.good {
        border-color: #FFA500;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);
    }
    
    .status-card.poor {
        border-color: var(--accent-color);
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.2);
    }
    
    .status-value {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.8rem 0;
    }
    
    .status-label {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0.4rem 0;
    }
    
    .status-description {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 0.8rem;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-card {
            padding: 0.8rem;
            margin: 0.4rem 0;
        }
        
        .status-value {
            font-size: 2rem;
        }
    }
    
    .stMetric {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================================================
# FUNÇÕES DE CARREGAMENTO E PROCESSAMENTO
# ===============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Carrega e processa os dados com tratamento de erro"""
    try:
        # Tentar carregar o arquivo CSV
        df = pd.read_csv('X1.csv')
        st.success("✅ Dados carregados com sucesso do arquivo X1.csv")
    except FileNotFoundError:
        st.info("📄 Arquivo X1.csv não encontrado. Gerando dados sintéticos para demonstração.")
        # Gerar dados sintéticos realistas
        np.random.seed(42)
        n_samples = 100
        
        # Variáveis base
        annual_rain = np.random.normal(1400, 250, n_samples)
        avg_rain = annual_rain / 12 + np.random.normal(0, 5, n_samples)
        
        # Nutrientes com correlações realistas
        nitrogen = np.random.normal(85000, 15000, n_samples)
        potash = nitrogen * 0.15 + np.random.normal(0, 3000, n_samples)
        phosphate = nitrogen * 0.5 + np.random.normal(0, 8000, n_samples)
        
        # Solo (proporções que somam 1)
        soil_vals = np.random.dirichlet([5, 3, 2], n_samples)
        
        # Produção baseada em fatores realistas
        production_base = (nitrogen * 0.008 + 
                          potash * 0.05 + 
                          phosphate * 0.02 + 
                          avg_rain * 8 + 
                          np.random.normal(0, 150, n_samples))
        
        df = pd.DataFrame({
            'ANNUAL': np.clip(annual_rain, 800, 2200),
            'avg_rain': np.clip(avg_rain, 40, 150),
            'Nitrogen': np.clip(nitrogen, 40000, 130000),
            'POTASH': np.clip(potash, 8000, 25000),
            'PHOSPHATE': np.clip(phosphate, 20000, 70000),
            'LOAMY_ALFISOL': soil_vals[:, 0],
            'USTALF_USTOLLS': soil_vals[:, 1],
            'VERTISOLS': soil_vals[:, 2],
            'RICE_PRODUCTION': np.clip(production_base, 600, 1800)
        })
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")
        st.stop()
    
    # Validação básica dos dados
    if df.empty:
        st.error("❌ Dataset está vazio")
        st.stop()
    
    # Verificar colunas obrigatórias
    required_cols = ['RICE_PRODUCTION', 'Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Colunas obrigatórias ausentes: {missing_cols}")
        st.stop()
    
    # Limpeza de dados
    df = df.dropna()
    
    # Remover outliers extremos (IQR method)
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def feature_engineering(df):
    """Engenharia de features com tratamento de erro"""
    try:
        df_fe = df.copy()
        
        # Features de nutrientes
        df_fe['NPK_Total'] = df_fe['Nitrogen'] + df_fe['POTASH'] + df_fe['PHOSPHATE']
        df_fe['N_P_Ratio'] = df_fe['Nitrogen'] / (df_fe['PHOSPHATE'] + 1)
        df_fe['N_K_Ratio'] = df_fe['Nitrogen'] / (df_fe['POTASH'] + 1)
        df_fe['P_K_Ratio'] = df_fe['PHOSPHATE'] / (df_fe['POTASH'] + 1)
        
        # Features hídricas
        df_fe['Water_Efficiency'] = df_fe['RICE_PRODUCTION'] / (df_fe['avg_rain'] + 1)
        df_fe['Rain_Intensity'] = df_fe['avg_rain'] / (df_fe['ANNUAL'] / 365 + 1)
        
        # Índice de fertilidade normalizado
        nutrients = df_fe[['Nitrogen', 'POTASH', 'PHOSPHATE']]
        nutrients_norm = (nutrients - nutrients.min()) / (nutrients.max() - nutrients.min())
        df_fe['Fertility_Index'] = nutrients_norm.mean(axis=1)
        
        # Interações
        df_fe['Nitrogen_x_Rain'] = df_fe['Nitrogen'] * df_fe['avg_rain']
        df_fe['Optimal_Score'] = (df_fe['NPK_Total'] * df_fe['avg_rain']) / 1000000
        
        # Verificar se há valores infinitos ou NaN
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
        df_fe = df_fe.fillna(df_fe.median())
        
        return df_fe
        
    except Exception as e:
        st.error(f"❌ Erro na engenharia de features: {str(e)}")
        return df

@st.cache_resource(ttl=3600, show_spinner=False)
def train_model(df):
    """Treina o modelo preditivo com fallback"""
    try:
        # Selecionar features numéricas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'RICE_PRODUCTION' in numeric_features:
            numeric_features.remove('RICE_PRODUCTION')
        
        X = df[numeric_features]
        y = df['RICE_PRODUCTION']
        
        # Verificar se há dados suficientes
        if len(df) < 10:
            st.error("❌ Dados insuficientes para treinar o modelo")
            st.stop()
        
        # Split dos dados
        test_size = min(0.3, max(0.1, len(df) * 0.2 / len(df)))  # Ajuste dinâmico
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Tentar usar XGBoost, caso contrário usar RandomForest
        if XGBOOST_AVAILABLE:
            try:
                model = xgb.XGBRegressor(
                    n_estimators=100, 
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                model_name = "XGBoost"
            except Exception:
                # Fallback para RandomForest
                model = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
                model_name = "RandomForest (fallback)"
        else:
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            model_name = "RandomForest"
        
        # Avaliação
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        st.success(f"✅ Modelo {model_name} treinado com sucesso!")
        
        return model, scaler, r2, rmse, X_train.columns, model_name
        
    except Exception as e:
        st.error(f"❌ Erro no treinamento do modelo: {str(e)}")
        # Modelo dummy como fallback extremo
        from sklearn.dummy import DummyRegressor
        dummy_model = DummyRegressor(strategy="mean")
        dummy_scaler = StandardScaler()
        
        X = df.select_dtypes(include=[np.number]).drop(columns=['RICE_PRODUCTION'], errors='ignore')
        y = df['RICE_PRODUCTION']
        
        X_scaled = dummy_scaler.fit_transform(X)
        dummy_model.fit(X_scaled, y)
        
        return dummy_model, dummy_scaler, 0.0, y.std(), X.columns, "Dummy (fallback)"

def create_modern_metric_card(title, value, delta=None, delta_color="normal", icon="📊", description=""):
    """Cria um card de métrica moderno"""
    delta_html = ""
    if delta:
        color = "#00C851" if delta_color == "normal" else "#FF6B35" if delta_color == "inverse" else "#666"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 0.4rem;">▲ {delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.8rem;">
            <span style="font-size: 1.2rem; margin-right: 0.4rem;">{icon}</span>
            <span style="font-weight: 600; color: #B0B0B0; font-size: 0.9rem;">{title}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: #FFFFFF; line-height: 1;">
            {value}
        </div>
        {delta_html}
        {f'<div style="color: #B0B0B0; font-size: 0.75rem; margin-top: 0.4rem;">{description}</div>' if description else ''}
    </div>
    """

# ===============================================================================
# PÁGINAS DA APLICAÇÃO
# ===============================================================================

def dashboard_page(df, model, scaler, r2, rmse, model_name):
    """Dashboard executivo otimizado"""
    
    st.markdown("## 📊 Dashboard Executivo")
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_production = df['RICE_PRODUCTION'].mean()
        st.markdown(create_modern_metric_card(
            "Produção Média", f"{avg_production:.0f} kg/ha", 
            None, "normal", "🌾", "Rendimento médio"
        ), unsafe_allow_html=True)
    
    with col2:
        max_efficiency = df['Water_Efficiency'].max()
        st.markdown(create_modern_metric_card(
            "Máx. Eficiência", f"{max_efficiency:.2f}", 
            "kg/mm", "normal", "💧", "Eficiência hídrica"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_modern_metric_card(
            f"{model_name}", f"{r2:.1%}", 
            "R²", "normal", "🤖", "Acurácia do modelo"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_modern_metric_card(
            "Erro (RMSE)", f"±{rmse:.0f}", 
            "kg/ha", "inverse", "📉", "Margem de erro"
        ), unsafe_allow_html=True)
    
    # Gráficos principais
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📈 Distribuição da Produção")
        
        fig = px.histogram(
            df, 
            x='RICE_PRODUCTION',
            nbins=20,
            title='Histograma da Produção de Arroz',
            template='plotly_dark',
            color_discrete_sequence=['#00C851']
        )
        
        fig.add_vline(
            x=avg_production, 
            line_dash="dash", 
            line_color="white",
            annotation_text=f"Média: {avg_production:.0f}"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Top Correlações")
        
        # Correlações com a produção
        correlations = df.select_dtypes(include=[np.number]).corr()['RICE_PRODUCTION'].abs()
        correlations = correlations.drop('RICE_PRODUCTION').sort_values(ascending=False).head(5)
        
        fig_bar = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title='Variáveis Mais Correlacionadas',
            template='plotly_dark',
            color_discrete_sequence=['#00C851']
        )
        
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Scatter plot principal
    st.markdown("### 🔍 Análise de Relacionamento")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        x_var = st.selectbox(
            "Eixo X:", 
            ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain', 'NPK_Total'],
            index=0
        )
        
        color_var = st.selectbox(
            "Colorir por:", 
            ['Water_Efficiency', 'Fertility_Index', 'NPK_Total'],
            index=0
        )
    
    with col1:
        fig_scatter = px.scatter(
            df,
            x=x_var,
            y='RICE_PRODUCTION',
            color=color_var,
            size='NPK_Total',
            hover_data=['Water_Efficiency'],
            title=f'Relação: {x_var} vs Produção',
            template='plotly_dark',
            color_continuous_scale='Viridis'
        )
        
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_production = df['RICE_PRODUCTION'].max()
        improvement_potential = ((best_production - avg_production) / avg_production) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🎯 Potencial de Melhoria</div>
            Com otimização, é possível aumentar a produção média em até 
            <strong>{improvement_potential:.1f}%</strong> (de {avg_production:.0f} para {best_production:.0f} kg/ha).
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_performers = df[df['RICE_PRODUCTION'] > df['RICE_PRODUCTION'].quantile(0.8)]
        avg_npk_top = high_performers['NPK_Total'].mean()
        avg_npk_all = df['NPK_Total'].mean()
        npk_diff = ((avg_npk_top - avg_npk_all) / avg_npk_all) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">🧪 Padrão dos Melhores</div>
            Top 20% dos produtores usam <strong>{npk_diff:+.1f}%</strong> mais NPK 
            que a média ({avg_npk_top:,.0f} vs {avg_npk_all:,.0f} kg/ha).
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        water_eff_top = high_performers['Water_Efficiency'].mean()
        water_eff_all = df['Water_Efficiency'].mean()
        eff_improvement = ((water_eff_top - water_eff_all) / water_eff_all) * 100
        
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">💧 Eficiência Superior</div>
            Produtores de elite são <strong>{eff_improvement:+.1f}%</strong> mais eficientes 
            no uso da água ({water_eff_top:.2f} vs {water_eff_all:.2f} kg/mm).
        </div>
        """, unsafe_allow_html=True)

def prediction_page(df, model, scaler, feature_names, model_name):
    """Página de predição simplificada"""
    
    st.markdown(f"## 🤖 Preditor IA ({model_name})")
    st.markdown("### Configure os parâmetros e obtenha predições instantâneas")
    
    # Layout em colunas
    col_input, col_result = st.columns([1.2, 0.8])
    
    with col_input:
        st.markdown("#### 🎛️ Parâmetros de Entrada")
        
        # Inputs organizados em tabs
        tab1, tab2 = st.tabs(["🌧️ Clima", "🧪 Nutrientes"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                annual_rain = st.slider(
                    "Precipitação Anual (mm)", 
                    float(df['ANNUAL'].min()), 
                    float(df['ANNUAL'].max()), 
                    float(df['ANNUAL'].mean())
                )
            
            with col2:
                avg_rain = st.slider(
                    "Chuva Mensal (mm)", 
                    float(df['avg_rain'].min()), 
                    float(df['avg_rain'].max()), 
                    float(df['avg_rain'].mean())
                )
        
        with tab2:
            nitrogen = st.slider(
                "Nitrogênio (kg/ha)", 
                float(df['Nitrogen'].min()), 
                float(df['Nitrogen'].max()), 
                float(df['Nitrogen'].mean())
            )
            
            col1, col2 = st.columns(2)
            with col1:
                potash = st.slider(
                    "Potássio (kg/ha)", 
                    float(df['POTASH'].min()), 
                    float(df['POTASH'].max()), 
                    float(df['POTASH'].mean())
                )
            
            with col2:
                phosphate = st.slider(
                    "Fósforo (kg/ha)", 
                    float(df['PHOSPHATE'].min()), 
                    float(df['PHOSPHATE'].max()), 
                    float(df['PHOSPHATE'].mean())
                )
    
    with col_result:
        st.markdown("#### 🎯 Resultado da Predição")
        
        try:
            # Preparar dados para predição
            input_data = pd.DataFrame({
                'ANNUAL': [annual_rain],
                'avg_rain': [avg_rain],
                'Nitrogen': [nitrogen],
                'POTASH': [potash],
                'PHOSPHATE': [phosphate]
            })
            
            # Adicionar features engineered
            input_data['NPK_Total'] = input_data['Nitrogen'] + input_data['POTASH'] + input_data['PHOSPHATE']
            input_data['N_P_Ratio'] = input_data['Nitrogen'] / (input_data['PHOSPHATE'] + 1)
            input_data['N_K_Ratio'] = input_data['Nitrogen'] / (input_data['POTASH'] + 1)
            input_data['P_K_Ratio'] = input_data['PHOSPHATE'] / (input_data['POTASH'] + 1)
            input_data['Water_Efficiency'] = 1.0  # Placeholder
            input_data['Rain_Intensity'] = input_data['avg_rain'] / (input_data['ANNUAL'] / 365 + 1)
            input_data['Fertility_Index'] = 0.5  # Placeholder
            input_data['Nitrogen_x_Rain'] = input_data['Nitrogen'] * input_data['avg_rain']
            input_data['Optimal_Score'] = (input_data['NPK_Total'] * input_data['avg_rain']) / 1000000
            
            # Adicionar colunas faltantes com médias
            for col in feature_names:
                if col not in input_data.columns:
                    if col in df.columns:
                        input_data[col] = df[col].mean()
                    else:
                        input_data[col] = 0
            
            # Reordenar e fazer predição
            input_data = input_data[feature_names]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            
            # Classificar resultado
            quartiles = df['RICE_PRODUCTION'].quantile([0.25, 0.5, 0.75])
            
            if prediction > quartiles[0.75]:
                status = "Excelente"
                status_color = "#00C851"
                status_icon = "🟢"
                status_class = "excellent"
            elif prediction > quartiles[0.5]:
                status = "Boa"
                status_color = "#FFA500"
                status_icon = "🟡"
                status_class = "good"
            else:
                status = "Baixa"
                status_color = "#FF6B35"
                status_icon = "🔴"
                status_class = "poor"
            
            # Card de resultado
            st.markdown(f"""
            <div class="status-card {status_class}">
                <div style="font-size: 2.5rem;">{status_icon}</div>
                <div class="status-value" style="color: {status_color};">
                    {prediction:.0f} <span style="font-size: 1.2rem;">kg/ha</span>
                </div>
                <div class="status-label" style="color: {status_color};">
                    {status}
                </div>
                <div class="status-description">
                    Produção predita pelo modelo {model_name}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas adicionais
            mean_production = df['RICE_PRODUCTION'].mean()
            difference = prediction - mean_production
            percentage = (difference / mean_production) * 100
            
            st.markdown("#### 📊 Comparação")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "vs Média", 
                    f"{difference:+.0f} kg/ha",
                    f"{percentage:+.1f}%"
                )
            
            with col2:
                npk_total = nitrogen + potash + phosphate
                cost_estimate = (nitrogen * 0.8 + potash * 1.2 + phosphate * 1.0) / 1000
                st.metric(
                    "Custo NPK", 
                    f"R$ {cost_estimate:,.0f}/ha"
                )
        
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")
            st.info("Verifique se todos os parâmetros estão dentro dos limites válidos.")

def insights_page(df):
    """Página de insights otimizada"""
    
    st.markdown("## 🔍 Insights Agronômicos")
    st.markdown("### Análise dos padrões de alta produtividade")
    
    # Criar quartis para análise
    quartile_75 = df['RICE_PRODUCTION'].quantile(0.75)
    quartile_25 = df['RICE_PRODUCTION'].quantile(0.25)
    
    top_performers = df[df['RICE_PRODUCTION'] > quartile_75]
    bottom_performers = df[df['RICE_PRODUCTION'] < quartile_25]
    
    # Comparação visual
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 TOP 25% PRODUTORES")
        
        avg_prod_top = top_performers['RICE_PRODUCTION'].mean()
        avg_npk_top = top_performers['NPK_Total'].mean()
        avg_rain_top = top_performers['avg_rain'].mean()
        avg_eff_top = top_performers['Water_Efficiency'].mean()
        
        st.metric("Produção Média", f"{avg_prod_top:.0f} kg/ha")
        st.metric("NPK Total", f"{avg_npk_top:,.0f} kg/ha")
        st.metric("Chuva Média", f"{avg_rain_top:.1f} mm/mês")
        st.metric("Eficiência Hídrica", f"{avg_eff_top:.2f} kg/mm")
    
    with col2:
        st.markdown("#### 🔴 BOTTOM 25% PRODUTORES")
        
        avg_prod_bot = bottom_performers['RICE_PRODUCTION'].mean()
        avg_npk_bot = bottom_performers['NPK_Total'].mean()
        avg_rain_bot = bottom_performers['avg_rain'].mean()
        avg_eff_bot = bottom_performers['Water_Efficiency'].mean()
        
        st.metric("Produção Média", f"{avg_prod_bot:.0f} kg/ha")
        st.metric("NPK Total", f"{avg_npk_bot:,.0f} kg/ha")
        st.metric("Chuva Média", f"{avg_rain_bot:.1f} mm/mês")
        st.metric("Eficiência Hídrica", f"{avg_eff_bot:.2f} kg/mm")
    
    # Análise de GAP
    st.markdown("### 📊 Análise de Gap - Oportunidades")
    
    variables = ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain']
    gaps = []
    gap_pcts = []
    
    for var in variables:
        gap = top_performers[var].mean() - bottom_performers[var].mean()
        gap_pct = (gap / bottom_performers[var].mean()) * 100
        gaps.append(gap)
        gap_pcts.append(gap_pct)
    
    fig_gap = px.bar(
        x=variables,
        y=gap_pcts,
        title="Diferença Percentual: Top 25% vs Bottom 25%",
        template='plotly_dark',
        color=gap_pcts,
        color_continuous_scale='RdYlGn'
    )
    
    fig_gap.update_layout(
        yaxis_title="Diferença (%)",
        xaxis_title="Variáveis",
        height=400
    )
    
    st.plotly_chart(fig_gap, use_container_width=True)
    
    # Receita de sucesso
    st.markdown("### 🎯 Receita de Sucesso (Baseada nos Top 25%)")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        optimal_nitrogen = top_performers['Nitrogen'].mean()
        optimal_potash = top_performers['POTASH'].mean()
        optimal_phosphate = top_performers['PHOSPHATE'].mean()
        optimal_rain = top_performers['avg_rain'].mean()
        
        st.markdown(f"""
        <div class="insight-box" style="padding: 1.5rem;">
            <div class="insight-title">🧪 FÓRMULA ÓTIMA</div>
            <div style="display: grid; gap: 0.8rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between;">
                    <span>🌱 Nitrogênio:</span>
                    <strong>{optimal_nitrogen:,.0f} kg/ha</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>🟡 Potássio:</span>
                    <strong>{optimal_potash:,.0f} kg/ha</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>🔵 Fósforo:</span>
                    <strong>{optimal_phosphate:,.0f} kg/ha</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span>💧 Chuva ideal:</span>
                    <strong>{optimal_rain:.1f} mm/mês</strong>
                </div>
                <hr style="border-color: var(--primary-color); margin: 0.5rem 0;">
                <div style="display: flex; justify-content: space-between; font-size: 1.1rem;">
                    <span>🎯 NPK Total:</span>
                    <strong style="color: var(--primary-color);">{optimal_nitrogen + optimal_potash + optimal_phosphate:,.0f} kg/ha</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Gráfico de composição NPK
        npk_values = [optimal_nitrogen, optimal_potash, optimal_phosphate]
        npk_labels = ['Nitrogênio', 'Potássio', 'Fósforo']
        
        fig_pie = px.pie(
            values=npk_values,
            names=npk_labels,
            title="Composição NPK Ótima",
            template='plotly_dark',
            color_discrete_sequence=['#00C851', '#FFA500', '#FF6B35']
        )
        
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Correlações mais importantes
    st.markdown("### 🔗 Fatores de Maior Impacto na Produção")
    
    correlations = df.select_dtypes(include=[np.number]).corr()['RICE_PRODUCTION'].abs()
    correlations = correlations.drop('RICE_PRODUCTION').sort_values(ascending=False).head(6)
    
    fig_corr = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="Correlações mais Fortes com a Produção",
        template='plotly_dark',
        color=correlations.values,
        color_continuous_scale='Viridis'
    )
    
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Insights acionáveis
    col1, col2, col3 = st.columns(3)
    
    nitrogen_corr = correlations.get('Nitrogen', 0)
    npk_corr = correlations.get('NPK_Total', 0)
    rain_corr = correlations.get('avg_rain', 0)
    
    with col1:
        if nitrogen_corr > 0.3:
            insight = "🟢 ALTA CORRELAÇÃO"
            desc = "Nitrogênio tem forte impacto na produção"
        else:
            insight = "🟡 CORRELAÇÃO MODERADA"
            desc = "Balancear com outros nutrientes"
            
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">{insight}</div>
            {desc}
            <br><br>
            <strong>Correlação:</strong> {nitrogen_corr:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        water_eff_avg = df['Water_Efficiency'].mean()
        if water_eff_avg > 10:
            insight = "💧 EFICIÊNCIA BOA"
            desc = "Boa utilização da água disponível"
        else:
            insight = "💧 MELHORAR EFICIÊNCIA"
            desc = "Otimizar uso de água pode aumentar produção"
            
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">{insight}</div>
            {desc}
            <br><br>
            <strong>Média:</strong> {water_eff_avg:.2f} kg/mm
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        production_std = df['RICE_PRODUCTION'].std()
        production_cv = production_std / df['RICE_PRODUCTION'].mean()
        
        if production_cv < 0.2:
            insight = "📊 BAIXA VARIABILIDADE"
            desc = "Produção consistente entre áreas"
        else:
            insight = "📊 ALTA VARIABILIDADE"
            desc = "Oportunidade de padronização"
            
        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">{insight}</div>
            {desc}
            <br><br>
            <strong>CV:</strong> {production_cv:.1%}
        </div>
        """, unsafe_allow_html=True)

def analysis_page(df):
    """Página de análise exploratória simplificada"""
    
    st.markdown("## 📊 Análise Exploratória")
    st.markdown("### Explore os dados de forma interativa")
    
    # Seletor de variável
    col1, col2 = st.columns([2, 1])
    
    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_var = st.selectbox("Selecione uma variável:", numeric_cols, index=0)
    
    with col2:
        chart_type = st.selectbox(
            "Tipo de gráfico:", 
            ["Histograma", "Box Plot", "Scatter vs Produção"]
        )
    
    # Visualização baseada na seleção
    if chart_type == "Histograma":
        fig = px.histogram(
            df, 
            x=selected_var, 
            nbins=20,
            title=f"Distribuição de {selected_var}",
            template='plotly_dark',
            color_discrete_sequence=['#00C851']
        )
        
        # Adicionar linha da média
        mean_val = df[selected_var].mean()
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color="white",
            annotation_text=f"Média: {mean_val:.2f}"
        )
        
    elif chart_type == "Box Plot":
        fig = px.box(
            df, 
            y=selected_var,
            title=f"Box Plot - {selected_var}",
            template='plotly_dark',
            color_discrete_sequence=['#00C851']
        )
        
    else:  # Scatter vs Produção
        if selected_var != 'RICE_PRODUCTION':
            fig = px.scatter(
                df, 
                x=selected_var, 
                y='RICE_PRODUCTION',
                title=f"{selected_var} vs Produção de Arroz",
                template='plotly_dark',
                trendline="ols",
                color_discrete_sequence=['#00C851']
            )
        else:
            fig = px.scatter(
                df, 
                x='NPK_Total', 
                y='RICE_PRODUCTION',
                title="NPK Total vs Produção de Arroz",
                template='plotly_dark',
                trendline="ols",
                color_discrete_sequence=['#00C851']
            )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Estatísticas descritivas
    st.markdown(f"### 📋 Estatísticas - {selected_var}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = df[selected_var].describe()
    
    with col1:
        st.metric("Média", f"{stats['mean']:.2f}")
        st.metric("Mínimo", f"{stats['min']:.2f}")
    
    with col2:
        st.metric("Mediana", f"{stats['50%']:.2f}")
        st.metric("Máximo", f"{stats['max']:.2f}")
    
    with col3:
        st.metric("Desvio Padrão", f"{stats['std']:.2f}")
        st.metric("1º Quartil", f"{stats['25%']:.2f}")
    
    with col4:
        st.metric("Coef. Variação", f"{(stats['std']/stats['mean']):.2%}")
        st.metric("3º Quartil", f"{stats['75%']:.2f}")
    
    # Matriz de correlação simplificada
    st.markdown("### 🔗 Matriz de Correlação (Top Variáveis)")
    
    # Selecionar apenas as variáveis mais importantes
    important_vars = ['RICE_PRODUCTION', 'Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain', 'NPK_Total', 'Water_Efficiency']
    available_vars = [var for var in important_vars if var in df.columns]
    
    corr_matrix = df[available_vars].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlações entre Variáveis Principais",
        template='plotly_dark',
        color_continuous_scale='RdBu'
    )
    
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ===============================================================================
# INTERFACE PRINCIPAL E NAVEGAÇÃO
# ===============================================================================

def main():
    """Função principal da aplicação"""
    
    # Header
    st.markdown('<h1 class="main-header">🌾 AgriTech Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análise Inteligente de Produção de Arroz com IA</p>', unsafe_allow_html=True)
    
    # Carregar dados com barra de progresso
    with st.spinner('🔄 Carregando e processando dados...'):
        df = load_data()
        df_enhanced = feature_engineering(df)
        model, scaler, r2, rmse, feature_names, model_name = train_model(df_enhanced)
    
    st.success(f"✅ Sistema pronto! Modelo {model_name} com R² = {r2:.3f}")
    
    # Sidebar para navegação
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; 
                    background: linear-gradient(135deg, #00C851 0%, #2E8B57 100%); 
                    border-radius: 12px; margin-bottom: 1.5rem; color: white;">
            <h3 style="margin: 0;">🎛️ Menu</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Navegue pelas análises</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Opções de página
        page_options = {
            "🏠 Dashboard": "dashboard",
            "🤖 Preditor IA": "prediction", 
            "🔍 Insights": "insights",
            "📊 Análise Exploratória": "analysis"
        }
        
        selected_page = st.radio("", list(page_options.keys()), key="navigation")
        page = page_options[selected_page]
        
        st.markdown("---")
        
        # Informações do modelo
        st.markdown("### 🤖 Status do Sistema")
        st.metric("Modelo", model_name)
        st.metric("Acurácia (R²)", f"{r2:.1%}")
        st.metric("Erro (RMSE)", f"±{rmse:.0f} kg/ha")
        st.metric("Amostras", f"{len(df_enhanced)}")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Resumo dos Dados")
        avg_prod = df_enhanced['RICE_PRODUCTION'].mean()
        max_prod = df_enhanced['RICE_PRODUCTION'].max()
        
        st.write(f"🌾 **Produção Média:** {avg_prod:.0f} kg/ha")
        st.write(f"🏆 **Produção Máxima:** {max_prod:.0f} kg/ha")
        
        improvement_potential = ((max_prod - avg_prod) / avg_prod) * 100
        st.write(f"📈 **Potencial:** +{improvement_potential:.0f}%")
    
    # Roteamento de páginas
    if page == "dashboard":
        dashboard_page(df_enhanced, model, scaler, r2, rmse, model_name)
    elif page == "prediction":
        prediction_page(df_enhanced, model, scaler, feature_names, model_name)
    elif page == "insights":
        insights_page(df_enhanced)
    elif page == "analysis":
        analysis_page(df_enhanced)


# ===============================================================================
# EXECUÇÃO DA APLICAÇÃO
# ===============================================================================

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        st.info("Por favor, recarregue a página ou entre em contato com o suporte.")
        
        # Debug info para desenvolvimento
        if st.checkbox("Mostrar detalhes do erro (debug)"):
            st.code(str(e))
            import traceback
            st.code(traceback.format_exc())


