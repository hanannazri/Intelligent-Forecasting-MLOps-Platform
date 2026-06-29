import streamlit as st


def load_css():
    st.markdown(
        """
<style>

.block-container {
    max-width: 95% !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
}

.section-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 20px;
}

.subsection-title {
    text-align: center;
    font-size: 28px;
    font-weight: 600;
    color: #f3f4f6;
    margin-top: 20px;
    margin-bottom: 15px;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    margin-bottom: 8px;
}

.subtitle {
    text-align: center;
    font-size: 24px;
    color: #b8b8b8;
    margin-bottom: 40px;
}

/* KPI Cards */

.kpi-card{
    background:linear-gradient(135deg,#1f2937 0%,#111827 100%);
    border:1px solid #374151;
    border-radius:16px;
    padding:18px 22px;
    height:180px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    box-shadow:0 6px 16px rgba(0,0,0,0.25);
}

.kpi-blue{
    background:linear-gradient(135deg,#1f2937,#111827);
}

.kpi-red{
    background:linear-gradient(135deg,#4c1d1d,#2b1010);
}

.kpi-green{
    background:linear-gradient(135deg,#153b2d,#0d241b);
}

.kpi-purple{
    background:linear-gradient(135deg,#2b244d,#181329);
}

.kpi-label{
    font-size:20px;
    text-align:center;
    margin-bottom:20px;
    font-weight:500;
    color:#cbd5e1;
    line-height:1.4;
    min-height:55px;
}

.kpi-value{
    font-size:30px;
    font-weight:700;
    text-align:center;
    color:#ffffff;
    font-family:Arial,sans-serif;
}

/* Recommendation Cards */

.recommend-card{
    background:linear-gradient(135deg,#1f2937,#111827);
    border:1px solid #374151;
    border-radius:14px;
    padding:22px;
    text-align:center;
    height:145px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.25);
}

.recommend-label{
    font-size:18px;
    color:#CBD5E1;
    font-weight:500;
    margin-bottom:18px;
}

.recommend-value{
    font-size:28px;
    font-weight:700;
    color:white;
    font-family:Arial,sans-serif;
}

/* Info Cards */

.info-card{
    background:linear-gradient(135deg,#1f2937,#111827);
    border:1px solid #374151;
    border-radius:14px;
    padding:18px;
    text-align:center;
    height:120px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    box-shadow:0 4px 10px rgba(0,0,0,0.25);
}

.info-label{
    font-size:16px;
    color:#CBD5E1;
    font-weight:500;
    margin-bottom:14px;
}

.info-value{
    font-size:22px;
    font-weight:700;
    color:white;
}

/* Footer */

.footer{
    margin-top:40px;
    margin-bottom:20px;
    padding:25px;
    background:#151923;
    border:1px solid #2f3542;
    border-radius:14px;
}

.footer-title{
    text-align:center;
    font-size:24px;
    font-weight:700;
    margin-bottom:25px;
    color:white;
}

.footer-grid{
    display:flex;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:20px;
}

.footer-item{
    flex:1;
    min-width:160px;
    text-align:center;
}

.footer-item b{
    color:#d1d5db;
    font-size:15px;
}

.footer-item span{
    display:block;
    margin-top:8px;
    font-size:18px;
    font-weight:600;
    color:white;
}

</style>
""",
        unsafe_allow_html=True,
    )