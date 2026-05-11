import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="TOMTAŞ LIFT-UP AI", layout="wide")

# --- HEADER: BAŞLIK VE LOGO YANYANA ---
col_baslik, col_logo = st.columns([5, 1])

with col_baslik:
    st.title("🛠️ TOMTAŞ LIFT-UP: AI Takım Ömrü Asistanı")

with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=150)
    elif os.path.exists("logo.jpg"):
        st.image("logo.jpg", width=150)

st.markdown("CMM ve CAM verileri ile Kestirimci Bakım ve Otonom Karar Mekanizması")

class AI_ToolLife:
    def __init__(self, tolerance):
        self.tolerance = tolerance  
        self.scenarios = {}

    def calculate_daf(self, ae, D):
        if ae >= D: return 1.8                
        elif ae <= (0.1 * D): return 1.1      
        else: return 1.4                           

    def add_scenario(self, name, mat_name, kc_ref, c_taylor, D, z, Lc, Vc, fz, ap, ae, blocks, wear_data, cam_cycle_time):
        daf = self.calculate_daf(ae, D)
        hm = fz * (np.pi * D / (2 * z)) * (ae / D)**2 if ae >= D else fz * np.sqrt(ae / D)
        kc_eff = kc_ref * (hm / 0.1)**-0.2
        t_theo = (c_taylor / (Vc**3.5)) * (1 / (daf**1.5))

        X = np.array(blocks).reshape(-1, 1)
        y = np.array(wear_data)
        max_blok = max(blocks)

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        
        mse = mean_squared_error(y, model.predict(X_poly))
        rmse_mm = np.sqrt(mse)

        a, b, c = model.coef_[2], model.coef_[1], model.intercept_ - self.tolerance
        coefs = [a, b, c] if abs(a) > 1e-15 else [b, c]
        roots = np.roots(coefs)
        valid_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]

        if valid_roots and min(valid_roots) <= (max_blok * 3):
            exact_cross = min(valid_roots) 
            grafik_son_blok = max(max_blok, int(np.ceil(exact_cross)) + 2)
            tam_blok = int(exact_cross)
            yuzde = int(round((exact_cross - tam_blok) * 100))
            
            if yuzde == 100:
                tam_blok += 1
                yuzde = 0
                
            exact_time_minutes = exact_cross * cam_cycle_time
            dk = int(exact_time_minutes)
            sn = int(round((exact_time_minutes - dk) * 60))
            
            if sn == 60:
                dk += 1
                sn = 0

            if yuzde == 0:
                guven_araligi_metni = f"{tam_blok}.00 Blok"
                uretim_metni = f"**{tam_blok} tam blok** risksiz üretilir. Takım tam **{tam_blok}. bloğun bitiminde** tolerans sınırını aşacaktır."
            else:
                guven_araligi_metni = f"{exact_cross:.2f} Blok"
                uretim_metni = f"**{tam_blok} tam blok** risksiz üretilir. Yeni bloğa geçildiğinde kesimin **%{yuzde}'sinde** (Eksen Değeri: **{exact_cross:.2f}**) tolerans aşılır."
                
            sure_araligi_metni = f"{dk} Dk {sn} Sn"
        else:
            grafik_son_blok = max_blok * 2
            guven_araligi_metni = f"{grafik_son_blok}+ Blok"
            sure_araligi_metni = f"{grafik_son_blok * cam_cycle_time:.1f}+ Dk"
            uretim_metni = "Analiz ufku boyunca takımda riskli aşınma gözlemlenmemiştir."

        future_blocks = np.arange(1, grafik_son_blok + 1).reshape(-1, 1)
        future_y = model.predict(poly.transform(future_blocks))

        self.scenarios[name] = {
            'mat_name': mat_name, 'b_raw': blocks, 'y_raw': wear_data,
            'b_fut': future_blocks.flatten(), 'y_fut': future_y,
            'rmse_mm': rmse_mm, 't_theo': t_theo,
            'guven_araligi_metni': guven_araligi_metni,
            'sure_araligi_metni': sure_araligi_metni,
            'uretim_metni': uretim_metni,
            'cam_cycle_time': cam_cycle_time
        }

    def plot_dashboard(self):
        if not self.scenarios:
            st.warning("Gösterilecek veri yok.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 18), facecolor='#f8f9fa')
        colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd']
        
        genel_max_blok = max([data['b_fut'][-1] for data in self.scenarios.values()])
        genel_max_time = max([data['b_fut'][-1] * data['cam_cycle_time'] for data in self.scenarios.values()])

        for i, (name, data) in enumerate(self.scenarios.items()):
            col = colors[i % len(colors)]
            etiket = f"{name} ({data['mat_name']})"

            ax1.scatter(data['b_raw'], data['y_raw'], color=col, s=120) 
            ax1.plot(data['b_fut'], data['y_fut'], color=col, linestyle='--', linewidth=3.5, label=f"{etiket}")
            guven_bandi = data['rmse_mm'] * 2 
            ax1.fill_between(data['b_fut'], data['y_fut'] - guven_bandi, data['y_fut'] + guven_bandi, color=col, alpha=0.15)

            time_raw = [b * data['cam_cycle_time'] for b in data['b_raw']]
            time_fut = [b * data['cam_cycle_time'] for b in data['b_fut']]
            ax2.scatter(time_raw, data['y_raw'], color=col, s=120)
            ax2.plot(time_fut, data['y_fut'], color=col, linestyle='-', linewidth=3.5, label=f"{etiket}")
            ax2.fill_between(time_fut, np.array(data['y_fut']) - guven_bandi, np.array(data['y_fut']) + guven_bandi, color=col, alpha=0.15)

            st.markdown(f"### 📌 {name.upper()} | Alaşım: {data['mat_name']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Teorik Katalog Ömrü", f"{data['t_theo']:.1f} Dakika")
            col2.metric("Tam Kırılma Noktası (Blok)", data['guven_araligi_metni'])
            col3.metric("Tam Kırılma Noktası (Zaman)", data['sure_araligi_metni'])
            
            st.info(f"**Operasyon Önerisi:** {data['uretim_metni']}")
            
            if data['rmse_mm'] > 0.01:
                st.warning(f"⚠️ **Veri Anomalyası:** CMM ölçümlerinde dalgalanma tespit edildi (Sapma: {data['rmse_mm']:.4f} mm). Ölçümleri teyit edin.")
            st.divider()

        y_limit = self.tolerance * 1.5
        for ax, title, xlabel in zip([ax1, ax2], 
                                     ["Blok Sayısına Göre Takım Aşınması", "CAM Süresine Göre Takım Aşınması"], 
                                     ["İşlenen Sütun / Blok Sayısı", "Aktif CAM İşleme Süresi (Dakika)"]):
            ax.axhline(y=self.tolerance, color='black', linewidth=4, label=f"Tolerans ({self.tolerance} mm)")
            ax.set_title(title, fontsize=18, fontweight='bold', pad=15)
            ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
            ax.set_ylabel("Boyutsal Sapma (mm)", fontsize=14, fontweight='bold')
            ax.set_ylim(0.0, y_limit) 
            ax.legend(loc='upper left', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.8)

        ax1.set_xlim(0, genel_max_blok)
        ax2.set_xlim(0, genel_max_time)
        fig.tight_layout(pad=3.0) 
        st.pyplot(fig)

MALZEMELER = {
    "Alüminyum 6061-T6": {"kc": 680, "c_taylor": 4.5e10},
    "Alüminyum 7075-T6": {"kc": 800, "c_taylor": 3.5e10},
    "Alüminyum 2024-T3": {"kc": 750, "c_taylor": 4.0e10},
    "Alüminyum 5052-H32": {"kc": 600, "c_taylor": 5.0e10},
    "Titanyum Ti-6Al-4V": {"kc": 1200, "c_taylor": 1.2e10},
    "Titanyum Ti-5Al-2.5Sn": {"kc": 1150, "c_taylor": 1.3e10},
    "Paslanmaz Çelik 304": {"kc": 1900, "c_taylor": 2.8e10},
    "17-4 PH Paslanmaz": {"kc": 2200, "c_taylor": 2.0e10},
    "AISI 4340 Alaşımlı Çelik": {"kc": 2400, "c_taylor": 1.8e10}
}

eksik_alanlar = []

with st.sidebar:
    st.header("⚙️ Genel Ayarlar")
    tol_siniri = st.number_input("Maksimum Tolerans (mm)", value=None, format="%.4f", placeholder="Örn: 0.0050")
    if tol_siniri is None:
        eksik_alanlar.append("Genel Ayarlar: Maksimum Tolerans")

    senaryo_sayisi = st.number_input("Karşılaştırılacak Senaryo Sayısı", min_value=1, max_value=5, value=1, step=1)
    
    st.markdown("---")
    st.subheader("🔗 Ortak Parametre Seçimleri")
    ortak_malzeme = st.checkbox("Tüm senaryolarda ortak MALZEME kullan", value=True)
    ortak_takim = st.checkbox("Tüm senaryolarda ortak TAKIM kullan", value=True)

    genel_malzeme = None
    if ortak_malzeme:
        genel_malzeme_secimi = st.selectbox("Ortak Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Malzeme Seçiniz...")
        if genel_malzeme_secimi: genel_malzeme = MALZEMELER[genel_malzeme_secimi]

    genel_t_cap, genel_t_dis, genel_t_boy = None, None, None
    if ortak_takim:
        genel_t_cap = st.number_input("Ortak Takım Çapı (D) [mm]", value=None, placeholder="Örn: 6.0")
        genel_t_dis = st.number_input("Ortak Takım Diş Sayısı (z)", value=None, placeholder="Örn: 4")
        genel_t_boy = st.number_input("Ortak Takım Kesme Boyu (Lc) [mm]", value=None, placeholder="Örn: 24.0")

st.markdown(f"### 📋 Test Verisi Girişi ({senaryo_sayisi} Senaryo)")
sekmeler = st.tabs([f"{i+1}. Senaryo" for i in range(senaryo_sayisi)])
senaryo_verileri = []

for i, sekme in enumerate(sekmeler):
    with sekme:
        isim = st.text_input(f"Senaryo Adı", value=f"Senaryo {i+1}", key=f"isim_{i}")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Malzeme ve Takım Ayarları**")
            if not ortak_malzeme:
                m_secim = st.selectbox("Hammadde Seçimi", list(MALZEMELER.keys()), index=None, placeholder="Malzeme Seçiniz...", key=f"mat_{i}")
                s_malzeme = MALZEMELER[m_secim] if m_secim else None
                s_malzeme_isim = m_secim
                if s_malzeme is None: eksik_alanlar.append(f"{isim}: Hammadde Seçimi")
            else:
                s_malzeme = genel_malzeme
                s_malzeme_isim = genel_malzeme_secimi
                if s_malzeme is None: eksik_alanlar.append(f"{isim}: Ortak Hammadde Seçimi")
                elif s_malzeme_isim: st.info(f"Kullanılan Ortak Malzeme: {s_malzeme_isim}")

            if not ortak_takim:
                t_cap = st.number_input("Takım Çapı (D) [mm]", value=None, placeholder="Örn: 6.0", key=f"tcap_{i}")
                t_dis = st.number_input("Takım Diş Sayısı (z)", value=None, placeholder="Örn: 4", key=f"tdis_{i}")
                t_boy = st.number_input("Takım Kesme Boyu (Lc) [mm]", value=None, placeholder="Örn: 24.0", key=f"tboy_{i}")
                if t_cap is None: eksik_alanlar.append(f"{isim}: Takım Çapı")
                if t_dis is None: eksik_alanlar.append(f"{isim}: Takım Diş Sayısı")
                if t_boy is None: eksik_alanlar.append(f"{isim}: Takım Kesme Boyu")
            else:
                t_cap, t_dis, t_boy = genel_t_cap, genel_t_dis, genel_t_boy
                if t_cap is None: eksik_alanlar.append(f"{isim}: Ortak Takım Çapı")
                if t_dis is None: eksik_alanlar.append(f"{isim}: Ortak Takım Diş Sayısı")
                if t_boy is None: eksik_alanlar.append(f"{isim}: Ortak Takım Kesme Boyu")

        with colB:
            st.markdown("**Kesme ve Ölçüm Parametreleri**")
            vc = st.number_input("Kesme Hızı (Vc) [m/min]", value=None, placeholder="Örn: 400", key=f"vc_{i}")
            if vc is None: eksik_alanlar.append(f"{isim}: Kesme Hızı (Vc)")
            fz = st.number_input("İlerleme (fz) [mm/diş]", value=None, placeholder="Örn: 0.08", format="%.4f", key=f"fz_{i}")
            if fz is None: eksik_alanlar.append(f"{isim}: İlerleme (fz)")
            ap = st.number_input("Eksenel Derinlik (ap) [mm]", value=None, placeholder="Örn: 5.0", key=f"ap_{i}")
            if ap is None: eksik_alanlar.append(f"{isim}: Eksenel Derinlik (ap)")
            ae = st.number_input("Radyal Derinlik (ae) [mm]", value=None, placeholder="Örn: 5.0", key=f"ae_{i}")
            if ae is None: eksik_alanlar.append(f"{isim}: Radyal Derinlik (ae)")
            
            st.markdown("**1 Sütun İşleme Süresi**")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                cam_dk = st.number_input("Dakika", value=None, placeholder="Örn: 2", min_value=0, step=1, key=f"cam_dk_{i}")
                if cam_dk is None: eksik_alanlar.append(f"{isim}: İşleme Süresi (Dakika)")
            with t_col2:
                cam_sn = st.number_input("Saniye (Opsiyonel)", value=None, placeholder="Örn: 15", min_value=0, max_value=59, step=1, key=f"cam_sn_{i}")
            
            cam_sure = cam_dk + (cam_sn if cam_sn else 0) / 60.0 if cam_dk is not None else None
            cmm_str = st.text_input("CMM Verileri (Boşluklu)", value="", placeholder="Örn: 0.0001 0.0002", key=f"cmm_{i}")
            if not cmm_str: eksik_alanlar.append(f"{isim}: CMM Verileri")

        senaryo_verileri.append({
            "isim": isim, "mat_isim": s_malzeme_isim, "mat_data": s_malzeme,
            "t_cap": t_cap, "t_dis": t_dis, "t_boy": t_boy,
            "vc": vc, "fz": fz, "ap": ap, "ae": ae, "cam_sure": cam_sure, "cmm_str": cmm_str
        })

st.markdown("---")

if st.button("🚀 Takım Ömrü Tahminini Başlat", use_container_width=True, type="primary"):
    if len(eksik_alanlar) > 0:
        hata_metni = "\n".join([f"- {alan}" for alan in list(set(eksik_alanlar))])
        st.error(f"⚠️ Lütfen analizi başlatmadan önce aşağıdaki eksik bilgileri doldurunuz:\n\n{hata_metni}")
    else:
        try:
            system = AI_ToolLife(tolerance=tol_siniri)
            for d in senaryo_verileri:
                cmm_vals = [float(x) for x in d["cmm_str"].replace(',', ' ').split()]
                system.add_scenario(d["isim"], d["mat_isim"], d["mat_data"]['kc'], d["mat_data"]['c_taylor'], d["t_cap"], d["t_dis"], d["t_boy"], d["vc"], d["fz"], d["ap"], d["ae"], list(range(1, len(cmm_vals) + 1)), cmm_vals, d["cam_sure"])
            system.plot_dashboard()
        except Exception as e:
            st.error(f"Hata: {e}")
