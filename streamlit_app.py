import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

st.set_page_config(page_title="CLC Group Predictor", layout="wide")

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

# ── UI ──────────────────────────────────────────────────────────────
st.title("CLC Group Predictor")
st.caption("Paste dữ liệu trực tiếp từ Excel — 2 cột: CONSIGNEE_GROUP và PAYMENT TYPE")

st.markdown("**Hướng dẫn:** Trong Excel, chọn 2 cột `CONSIGNEE_GROUP` và `PAYMENT TYPE` (bao gồm header), copy và paste vào ô bên dưới.")

pasted = st.text_area(
    label="Dán dữ liệu từ Excel vào đây",
    height=200,
    placeholder="CONSIGNEE_GROUP\tPAYMENT TYPE\nGREYSTONE DATA SYSTEMS VIETNAM CO LTD\tOTH/DDU/S\nTOYOTA MOTOR VIETNAM CO LTD\tPPD\n..."
)

if st.button("Dự đoán", type="primary") and pasted.strip():
    try:
        # Parse tab-separated từ Excel paste
        df_input = pd.read_csv(
            io.StringIO(pasted.strip()),
            sep='\t',
            dtype=str
        ).fillna("")

        # Chuẩn hoá tên cột (trim spaces, upper)
        df_input.columns = df_input.columns.str.strip().str.upper()

        # Kiểm tra cột bắt buộc
        missing_cols = [c for c in ['CONSIGNEE_GROUP', 'PAYMENT TYPE'] if c not in df_input.columns]
        if missing_cols:
            st.error(f"Thiếu cột: {missing_cols}. Kiểm tra lại header trong Excel.")
            st.stop()

        df_input = df_input[['CONSIGNEE_GROUP', 'PAYMENT TYPE']].copy()
        df_input['CONSIGNEE_GROUP'] = df_input['CONSIGNEE_GROUP'].str.strip().str.upper()
        df_input['PAYMENT TYPE']    = df_input['PAYMENT TYPE'].str.strip().str.upper()

        # Đánh dấu OOV
        df_input['_is_oov'] = ~df_input['CONSIGNEE_GROUP'].isin(enc['consignee_rate_map'])

        # Build features & predict
        X = build_features(df_input, enc)
        preds  = model.predict(X)
        probas = model.predict_proba(X)

        # Gắn kết quả vào df
        df_input['CLC_GROUP (dự đoán)']  = np.where(preds == 1, 'ONLY EDO', 'BLANK')
        df_input['Xác suất BLANK (%)']  = (probas[:, 0] * 100).round(1)
        df_input['Xác suất ONLY EDO (%)'] = (probas[:, 1] * 100).round(1)

        # ── Hiển thị kết quả ──────────────────────────────────────
        st.success(f"Dự đoán xong {len(df_input)} dòng")

        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng dòng",    len(df_input))
        col2.metric("BLANK",        int((preds == 0).sum()))
        col3.metric("ONLY EDO",     int((preds == 1).sum()))

        # Tô màu kết quả
        def highlight_pred(row):
            if row['CLC_GROUP (dự đoán)'] == 'ONLY EDO':
                return [''] * (len(row) - 4) + ['background-color:#d4edda'] * 3 + ['']
            else:
                return [''] * (len(row) - 4) + ['background-color:#fff3cd'] * 3 + ['']

        display_df = df_input.drop(columns=['_is_oov'])

        st.dataframe(
            display_df.style.apply(highlight_pred, axis=1),
            use_container_width=True,
            height=400
        )

        # Cảnh báo OOV
        oov_list = df_input[df_input['_is_oov']]['CONSIGNEE_GROUP'].unique()
        if len(oov_list) > 0:
            with st.expander(f"Cảnh báo: {len(oov_list)} consignee chưa có trong tập train"):
                st.write(list(oov_list))
                st.caption("Các dòng này dùng global mean để dự đoán — kết quả kém tin cậy hơn.")

        # Download kết quả
        csv_out = display_df.to_csv(index=False).encode('utf-8-sig')  # utf-8-sig để Excel đọc được tiếng Việt
        st.download_button(
            label="Tải kết quả (.csv)",
            data=csv_out,
            file_name="clc_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Lỗi khi xử lý dữ liệu: {e}")
        st.caption("Kiểm tra lại: header có đúng tên cột không, dữ liệu có bị lỗi format không.")