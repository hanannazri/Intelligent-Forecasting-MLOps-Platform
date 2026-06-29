import streamlit as st


def section_title(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def subsection_title(title):
    st.markdown(
        f'<div class="subsection-title">{title}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label, value, color_class):
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(label, value):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">{label}</div>
            <div class="info-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommend_card(label, value):
    st.markdown(
        f"""
        <div class="recommend-card">
            <div class="recommend-label">{label}</div>
            <div class="recommend-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )