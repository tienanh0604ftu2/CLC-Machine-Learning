import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(
    page_title="Dự đoán CLC — EDO Prediction",
    page_icon="📦",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Be Vietnam Pro', sans-serif;
    }

    /* Hide default Streamlit header & footer */
    #MainMenu, header, footer { visibility: hidden; }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
    }
    .hero h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f1f5f9;
        margin: 0 0 0.4rem 0;
        line-height: 1.3;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 0;
        max-width: 700px;
        line-height: 1.6;
    }

    /* ── Section title ── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-title .icon {
        font-size: 1.1rem;
    }

    /* ── Input label ── */
    .input-label {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .input-label .label-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .input-label .label-hint {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* ── Stat cards ── */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-card {
        flex: 1;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
    }
    .stat-total .stat-value { color: #60a5fa; }
    .stat-blank .stat-value { color: #fbbf24; }
    .stat-edo .stat-value   { color: #34d399; }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: #334155;
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Text area ── */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
        background: #0f172a !important;
        color: #e2e8f0 !important;
        font-size: 0.85rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }
    .stTextArea textarea::placeholder {
        color: #475569 !important;
    }

    /* ── Primary button ── */
    .stButton > button[kind="primary"] {
        background: #3b82f6 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        color: white !important;
        transition: background 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2563eb !important;
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background: #334155 !important;
        border-color: #475569 !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }
    .sidebar-block {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-block h4 {
        margin: 0 0 0.4rem 0;
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .sidebar-block p, .sidebar-block li {
        color: #94a3b8;
        font-size: 0.8rem;
        line-height: 1.6;
        margin: 0;
    }
    .sidebar-block ol {
        padding-left: 1.2rem;
        margin: 0.3rem 0 0 0;
    }
    .sidebar-block code {
        background: #334155;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #93c5fd;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/model.pkl")
    enc   = joblib.load("artifacts/encoder_artifacts.pkl")
    return model, enc

model, enc = load_artifacts()


def build_features(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    X = pd.DataFrame()
    X['consignee_freq']     = df['CONSIGNEE_GROUP'].map(enc['consignee_freq_map']).fillna(0)
    X['consignee_edo_rate'] = df['CONSIGNEE_GROUP'].map(enc['consignee_rate_map']).fillna(enc['global_edo_rate'])
    X['is_prepaid']         = df['PAYMENT TYPE'].str.startswith('PPD').astype(int)
    X['is_collect']         = (df['PAYMENT TYPE'] == 'COL').astype(int)
    X['is_other']           = df['PAYMENT TYPE'].str.startswith('OTH').astype(int)
    X['is_ddp']             = df['PAYMENT TYPE'].str.contains('DDP').astype(int)
    for cat in enc['payment_categories']:
        X[f'pay_{cat}'] = (df['PAYMENT TYPE'] == cat).astype(int)
    return X[enc['feature_columns']]


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 CLC Predictor")
    st.markdown("---")

    st.markdown("""
    <div class="sidebar-block">
        <h4>Cách sử dụng</h4>
        <ol>
            <li>Nhập danh sách <b>Consignee</b> ở cột trái</li>
            <li>Nhập danh sách <b>Payment Type</b> ở cột phải</li>
            <li>Số dòng hai cột phải <b>bằng nhau</b></li>
            <li>Nhấn nút <b>Dự đoán</b></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-block">
        <h4>Payment Type phổ biến</h4>
        <p>
            <code>PPD</code> Prepaid &nbsp;·&nbsp;
            <code>COL</code> Collect<br>
            <code>OTH/DDU/S</code> Other DDU<br>
            <code>OTH/DDP/S</code> Other DDP
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-block">
        <h4>Kết quả dự đoán</h4>
        <p>
            <b style="color:#34d399">ONLY EDO</b> — Lô hàng cần phát hành EDO<br>
            <b style="color:#fbbf24">BLANK</b> — Lô hàng không cần EDO
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Hero Banner ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Machine Learning</div>
    <h1>Dự đoán lô hàng có cần EDO hay không?</h1>
    <p>Nhập thông tin Consignee và Payment Type để dự đoán cột CLC — xác định lô hàng thuộc nhóm <strong>ONLY EDO</strong> hay <strong>BLANK</strong>.</p>
</div>
""", unsafe_allow_html=True)


# ── Input Section ───────────────────────────────────────────────────
st.markdown('<div class="section-title"><span class="icon">📝</span> Nhập dữ liệu</div>', unsafe_allow_html=True)

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""
    <div class="input-label">
        <div class="label-name">CONSIGNEE (tên đầy đủ)</div>
        <div class="label-hint">Nhập tên đầy đủ — hệ thống tự động lấy <b>2 từ đầu</b> làm CONSIGNEE_GROUP</div>
    </div>
    """, unsafe_allow_html=True)

    consignee_input = st.text_area(
        label="consignee",
        height=200,
        placeholder="GREYSTONE DATA SYSTEMS VIETNAM CO LTD\nTOYOTA MOTOR VIETNAM CO LTD\nSAMSUNG ELECTRONICS VIETNAM",
        label_visibility="collapsed",
        key="consignee_input"
    )

with col_right:
    st.markdown("""
    <div class="input-label">
        <div class="label-name">PAYMENT TYPE</div>
        <div class="label-hint">Mỗi dòng một loại payment tương ứng với consignee bên trái</div>
    </div>
    """, unsafe_allow_html=True)

    payment_input = st.text_area(
        label="payment",
        height=200,
        placeholder="OTH/DDU/S\nPPD\nCOL",
        label_visibility="collapsed",
        key="payment_input"
    )


# ── Action ──────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

_, btn_col, _ = st.columns([1.2, 1, 1.2])
with btn_col:
    predict_clicked = st.button("Dự đoán", type="primary", use_container_width=True)


# ── Prediction Logic ────────────────────────────────────────────────
if predict_clicked:
    consignee_lines = [l.strip() for l in consignee_input.strip().split('\n') if l.strip()] if consignee_input.strip() else []
    payment_lines   = [l.strip() for l in payment_input.strip().split('\n') if l.strip()] if payment_input.strip() else []

    if not consignee_lines or not payment_lines:
        st.error("Vui lòng nhập dữ liệu vào **cả 2 cột** trước khi dự đoán.")
        st.stop()

    if len(consignee_lines) != len(payment_lines):
        st.error(
            f"Số dòng không khớp — **Consignee** có {len(consignee_lines)} dòng, "
            f"**Payment Type** có {len(payment_lines)} dòng."
        )
        st.stop()

    try:
        df_input = pd.DataFrame({
            'CONSIGNEE (gốc)': [s.upper() for s in consignee_lines],
            'PAYMENT TYPE':     [s.upper() for s in payment_lines]
        })

        # Lấy 2 từ đầu tiên làm CONSIGNEE_GROUP (giống lúc train model)
        df_input['CONSIGNEE_GROUP'] = (
            df_input['CONSIGNEE (gốc)']
            .str.split()
            .str[:2]
            .str.join(' ')
        )

        # Preview cho người dùng thấy kết quả chuyển đổi
        with st.expander("Xem trước: CONSIGNEE_GROUP (2 từ đầu)", expanded=False):
            preview_df = df_input[['CONSIGNEE (gốc)', 'CONSIGNEE_GROUP', 'PAYMENT TYPE']].copy()
            st.dataframe(preview_df, use_container_width=True, height=200)

        df_input['_is_oov'] = ~df_input['CONSIGNEE_GROUP'].isin(enc['consignee_rate_map'])

        X = build_features(df_input, enc)
        preds  = model.predict(X)
        probas = model.predict_proba(X)

        df_input['CLC_GROUP (dự đoán)']    = np.where(preds == 1, 'ONLY EDO', 'BLANK')
        df_input['Xác suất BLANK (%)']     = (probas[:, 0] * 100).round(1)
        df_input['Xác suất ONLY EDO (%)']  = (probas[:, 1] * 100).round(1)

        # ── Results ─────────────────────────────────────────────────
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span class="icon">📊</span> Kết quả</div>', unsafe_allow_html=True)

        total  = len(df_input)
        blanks = int((preds == 0).sum())
        edos   = int((preds == 1).sum())

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card stat-total">
                <div class="stat-label">Tổng dòng</div>
                <div class="stat-value">{total}</div>
            </div>
            <div class="stat-card stat-blank">
                <div class="stat-label">Blank</div>
                <div class="stat-value">{blanks}</div>
            </div>
            <div class="stat-card stat-edo">
                <div class="stat-label">Only EDO</div>
                <div class="stat-value">{edos}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Data table — hiển thị cả tên gốc và CONSIGNEE_GROUP
        display_df = df_input.drop(columns=['_is_oov'])

        def highlight_pred(row):
            n = len(row)
            if row['CLC_GROUP (dự đoán)'] == 'ONLY EDO':
                return [''] * (n - 3) + ['background-color:#064e3b; color:#6ee7b7'] * 3
            else:
                return [''] * (n - 3) + ['background-color:#422006; color:#fcd34d'] * 3

        st.dataframe(
            display_df.style.apply(highlight_pred, axis=1),
            use_container_width=True,
            height=400,
        )

        # OOV warning
        oov_list = df_input[df_input['_is_oov']]['CONSIGNEE_GROUP'].unique()
        if len(oov_list) > 0:
            with st.expander(f"⚠ {len(oov_list)} consignee chưa có trong tập huấn luyện"):
                st.write(list(oov_list))
                st.caption("Các dòng này dùng giá trị trung bình toàn cục để dự đoán — kết quả có thể kém chính xác hơn.")

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        csv_out = display_df.to_csv(index=False).encode('utf-8-sig')
        _, dl_col, _ = st.columns([1.2, 1, 1.2])
        with dl_col:
            st.download_button(
                label="Tải kết quả CSV",
                data=csv_out,
                file_name="clc_prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Lỗi khi xử lý: {e}")
        st.caption("Kiểm tra lại dữ liệu — đảm bảo format đúng và số dòng hai cột bằng nhau.")