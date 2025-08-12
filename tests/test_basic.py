"""
Testes básicos da aplicação
"""
import pytest
import pandas as pd
import sys
import os

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Testa se todas as dependências são importáveis"""
    import streamlit
    import pandas
    import numpy
    import plotly
    assert True

def test_data_loading():
    """Testa carregamento básico de dados"""
    # Implementar testes de dados
    assert True

if __name__ == "__main__":
    test_imports()
    test_data_loading()
    print("✅ All tests passed!")