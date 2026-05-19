import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="LIFT-UP Kestirimci Bakım", page_icon="✈️", layout="wide")

# --- CSS İLE TEMA VE DÜZEN ENJEKSİYONU ---
st.markdown("""
<style>
    /* ÜST TAVAN ŞERİDİ (PREMIUM BANNER) - Yan menüyle hizalı */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #004B87, #E31837) !important;
        height: 6px !important;
    }

    /* ÜST BOŞLUK HİZALAMASI */
    .block-container {
        padding-top: 1rem !important; 
        max-width: 98% !important;
    }

    /* METRİK KARTLARI */
    [data-testid="metric-container"] {
        border: 1px solid rgba(136, 136, 136, 0.2);
        padding: 15px;
        border-radius: 8px;
        background-color: rgba(248, 249, 250, 0.05); 
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #004B87 !important;
        font-weight: 800;
    }

    /* YAN PANEL (SIDEBAR) STİLİ */
    [data-testid="stSidebar"] {
        border-right: 3px solid #E31837;
    }

    [data-testid="stSidebar"]::before {
        content: "REMOVE BEFORE FLIGHT";
        display: block;
        background-color: #E31837;
        color: white;
        font-family: monospace;
        font-weight: bold;
        text-align: center;
        padding: 6px;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
        border-radius: 0 0 5px 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- İMZA YERİ ---
st.markdown("<div style='text-align: left; background-color: #E31837; color: white; display: inline-block; padding: 2px 10px; font-family: monospace; font-weight: bold; border-radius: 3px; font-size: 12px; margin-bottom: 10px;'>by Fuat Arıkan</div>", unsafe_allow_html=True)

# --- HEADER ---
col_baslik, col_logo = st.columns([5, 1])
with col_baslik:
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🛠️ LIFT-UP: Kestirimci Bakım Sistemi</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 14px; font-style: italic;'>Precision in Engineering, Excellence in Aviation.</p>", unsafe_allow_html=True)
with col_logo:
    if os.path.exists("agtoe.png"): st.image("agtoe.png", width=120)
    elif os.path.exists("logo.jpg"): st.image("logo.jpg", width=120)

st.markdown("<hr style='height: 2px; background: linear-gradient(90deg, #004B87, #E31837); border: none;'>", unsafe_allow_html=True)

# --- FİZİK MOTORU SINIFI ---
class AI_ToolLife:
    def __init__(self, tolerance, birim_ad):
        self.tolerance = tolerance  
        self.birim_ad = birim_ad
        self.scenarios = {}

    def add_scenario(self, name, mat_name, kc_ref, c_taylor, D, z, Lc, Vc, fz, ap, ae, blocks, wear_data, cam_cycle_time):
        # ... (Analiz mantığı aynı kalacak) ...
        daf = 1.4
        t_theo = (c_taylor / (Vc**3.5)) * (1 / (daf**1.5))
        X = np.array(blocks).reshape(-1, 1)
        y = np.array(wear_data)
        poly = PolynomialFeatures(degree=2)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        rmse_val = np.sqrt(mean_squared_error(y, model.predict(poly.fit_transform(X))))
        
        self.scenarios[name] = {
            'mat_name': mat_name, 't_theo': t_theo, 'rmse_val': rmse_val,
            'uretim_metni': "Analiz tamamlandı.", 'guven_araligi_metni': "Tahmin ediliyor",
            'sure_araligi_metni': "Tahmin ediliyor", 'veri_sayisi': len(blocks),
            'karsilastirma_durumu': "normal"
        }

    def plot_dashboard(self):
        # Grafik fonksiyonu...
        pass

# --- GÜNCEL SAĞ PANEL (Hatalı kodlardan arındırıldı) ---
def render_right_panel(tol, birim, mat_isim, mat_data):
    st.markdown("#### 🧠 Akıllı Fizik Motoru")
    st.markdown(f"**Aktif Tolerans:** <span style='color:#E31837; font-weight:bold;'>{tol} {birim}</span>", unsafe_allow_html=True)
    
    st.markdown("**Fiziksel Parametreler:**")
    if mat_isim:
        st.info(f"**Alaşım:** {mat_isim}\n\n**Kc:** {mat_data['kc']} MPa\n\n**Taylor:** {mat_data['c_taylor']:.1e}")
    else:
        st.warning("Malzeme bekleniyor...")
    
    st.caption("Sistem, anlık talaş kalınlığını hesaplar ve efektif kesme kuvvetini Taylor denklemi ile harmanlayarak Kırılma Ufkunu belirler.")

# --- ANA DÖNGÜ ---
# (Yukarıdaki fonksiyonları mevcut kodunla birleştirip, sağ sütun (colC) yerine render_right_panel çağırabilirsin)
