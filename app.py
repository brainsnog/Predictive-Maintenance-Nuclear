import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

import tensorflow as tf

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="NACP MONITOR // UNIT-7",
    page_icon="☢",
    layout="wide"
)

from src.data_loader import load_processed_data

from src.sequence_builder import create_sequences
from src.model_utils import compute_reconstruction_error
from src.evaluation import (
    apply_consecutive_logic,
    compute_lead_times
)

# =========================
# PHASE 1: CSS FOUNDATION
# Pip-Boy / Retro Nuclear Dashboard Aesthetic
# =========================

st.markdown("""
<style>

/* ── FONTS ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');

/* ── COLOR PALETTE (CSS VARIABLES) ─────────────────── */
:root {
    --green-primary:   #39ff14;
    --green-bright:    #57ff2f;
    --green-dim:       #1f5c0a;
    --green-muted:     #0d2b05;
    --green-panel:     #071a02;
    --green-border:    #2a7a0e;
    --amber:           #ffb000;
    --amber-dim:       #7a5500;
    --red-alert:       #ff3c00;
    --red-dim:         #5a1400;
    --bg-dark:         #050a03;
    --bg-panel:        #070f04;
    --bg-sidebar:      #040802;
    --text-primary:    #39ff14;
    --text-dim:        #1e7a06;
    --text-muted:      #0f4004;
    --scanline-color:  rgba(0, 0, 0, 0.18);
    --glow-green:      0 0 8px #39ff14, 0 0 20px rgba(57,255,20,0.3);
    --glow-subtle:     0 0 4px rgba(57,255,20,0.4);
    --border-dim:      1px solid #1a4a08;
    --border-main:     1px solid #2a7a0e;
    --border-bright:   1px solid #39ff14;
}

/* ── GLOBAL RESET & BASE ────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* Remove default Streamlit white backgrounds */
[data-testid="stAppViewContainer"] > .main {
    background-color: var(--bg-dark) !important;
}

[data-testid="block-container"] {
    background-color: transparent !important;
    padding-top: 1.5rem !important;
}

/* ── CRT SCANLINE OVERLAY ───────────────────────────── */
/* Layered on top of the entire app as a pseudo-element via the body */
body::before {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    background: repeating-linear-gradient(
        0deg,
        var(--scanline-color) 0px,
        var(--scanline-color) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 9999;
    opacity: 0.45;
}

/* ── PHOSPHOR VIGNETTE ──────────────────────────────── */
body::after {
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    background: radial-gradient(
        ellipse at center,
        transparent 60%,
        rgba(0, 0, 0, 0.65) 100%
    );
    pointer-events: none;
    z-index: 9998;
}

/* ── TYPOGRAPHY GLOBAL ──────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'VT323', 'Courier New', monospace !important;
    color: var(--green-primary) !important;
    text-shadow: var(--glow-green) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

p, span, div, label {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: var(--text-primary) !important;
}

/* ── SCROLLBAR ──────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-dark);
    border-left: var(--border-dim);
}
::-webkit-scrollbar-thumb {
    background: var(--green-dim);
    border: 1px solid var(--green-border);
}
::-webkit-scrollbar-thumb:hover {
    background: var(--green-primary);
    box-shadow: var(--glow-subtle);
}

/* ── PAGE TITLE (st.title / .pip-page-title) ─────────── */
[data-testid="stMarkdownContainer"] h1,
.stApp h1 {
    font-family: 'VT323', monospace !important;
    font-size: 2.8rem !important;
    color: var(--green-primary) !important;
    text-shadow: var(--glow-green) !important;
    border-bottom: var(--border-bright) !important;
    padding-bottom: 0.4rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* Custom full-width page masthead */
.pip-masthead {
    border-bottom: 1px solid var(--green-border);
    padding-bottom: 0.9rem;
    margin-bottom: 0.4rem;
}

.pip-masthead-eyebrow {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.22em;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.pip-masthead-eyebrow::before {
    content: "";
    display: inline-block;
    width: 28px;
    height: 1px;
    background: var(--green-dim);
}

.pip-masthead-title {
    font-family: 'VT323', monospace;
    font-size: 3rem;
    color: var(--green-primary);
    text-shadow: var(--glow-green);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    line-height: 1.05;
    animation: flicker 10s infinite;
    display: block;
}

.pip-masthead-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 0.4rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
}

.pip-masthead-tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border: 1px solid var(--green-dim);
    padding: 1px 7px;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--text-dim);
}

/* ── SECTION HEADERS (st.subheader / .pip-section-header) */
.stApp h2, .stApp h3 {
    font-family: 'VT323', monospace !important;
    font-size: 1.55rem !important;
    color: var(--green-primary) !important;
    text-shadow: 0 0 6px rgba(57,255,20,0.4) !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-left: 3px solid var(--green-primary) !important;
    padding-left: 0.65rem !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.2rem !important;
}

/* Custom section header — replaces st.subheader */
.pip-section-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.6rem 0 0.5rem 0;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid var(--green-dim);
}

.pip-section-header-bar {
    display: inline-block;
    width: 3px;
    height: 1.3rem;
    background: var(--green-primary);
    box-shadow: 0 0 6px rgba(57,255,20,0.6);
    flex-shrink: 0;
}

.pip-section-header-text {
    font-family: 'VT323', monospace;
    font-size: 1.55rem;
    color: var(--green-primary);
    text-shadow: 0 0 6px rgba(57,255,20,0.4);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    line-height: 1;
}

.pip-section-header-tag {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border: 1px solid var(--green-dim);
    padding: 1px 6px;
    margin-left: auto;
    flex-shrink: 0;
}

/* ── SECTION DIVIDERS ───────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--green-dim) !important;
    box-shadow: 0 0 6px rgba(57,255,20,0.15) !important;
    margin: 1.4rem 0 !important;
    position: relative !important;
}

/* Custom scan-line divider */
.pip-scan-divider {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 1.4rem 0 0.8rem 0;
    opacity: 0.7;
}

.pip-scan-divider::before,
.pip-scan-divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--green-dim);
    box-shadow: 0 0 4px rgba(57,255,20,0.2);
}

.pip-scan-divider-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.55rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    white-space: nowrap;
}

/* ── SIDEBAR ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 2px solid var(--green-border) !important;
    box-shadow: 4px 0 18px rgba(57,255,20,0.08) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-primary) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'VT323', monospace !important;
    border-bottom: var(--border-main) !important;
    padding-bottom: 0.3rem !important;
    letter-spacing: 0.15em !important;
}

/* Sidebar selectbox label */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--text-dim) !important;
}

/* Sidebar selectbox input box */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #020501 !important;
    border: 1px solid var(--green-border) !important;
    border-radius: 0px !important;
    font-size: 0.82rem !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 0.3rem !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-size: 0.78rem !important;
    color: var(--text-primary) !important;
}

/* Sidebar slider track */
[data-testid="stSidebar"] [data-baseweb="slider"] {
    padding: 0.2rem 0 !important;
}

/* Sidebar internal section dividers */
.sidebar-section-rule {
    border: none;
    border-top: 1px solid var(--green-dim);
    margin: 0.8rem 0;
    opacity: 0.6;
}

/* Sidebar stat row */
.sb-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 0.18rem 0;
    border-bottom: 1px solid #0d2b05;
}

.sb-stat-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.sb-stat-value {
    font-family: 'VT323', monospace;
    font-size: 1.15rem;
    color: var(--green-primary);
    text-shadow: 0 0 6px rgba(57,255,20,0.5);
    letter-spacing: 0.05em;
}

.sb-stat-value.amber {
    color: var(--amber);
    text-shadow: 0 0 6px rgba(255,176,0,0.5);
}

.sb-stat-value.red {
    color: var(--red-alert);
    text-shadow: 0 0 6px rgba(255,60,0,0.5);
}

/* Sidebar facility ID plate */
.sb-facility-plate {
    background-color: #020501;
    border: 1px solid var(--green-border);
    border-top: 3px solid var(--green-primary);
    padding: 0.7rem 0.8rem 0.6rem 0.8rem;
    margin-bottom: 0.9rem;
    box-shadow: inset 0 0 10px rgba(57,255,20,0.04);
}

.sb-facility-name {
    font-family: 'VT323', monospace;
    font-size: 1.5rem;
    color: var(--green-primary);
    text-shadow: var(--glow-green);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    line-height: 1.1;
    animation: flicker 8s infinite;
}

.sb-facility-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 0.15rem;
}

/* Sidebar model info block */
.sb-model-block {
    background-color: #020501;
    border: var(--border-dim);
    border-left: 2px solid var(--green-dim);
    padding: 0.5rem 0.7rem;
    margin-top: 0.6rem;
    margin-bottom: 0.4rem;
}

.sb-model-line {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    line-height: 1.7;
}

.sb-model-line span {
    color: var(--green-primary);
    font-size: 0.68rem;
}

/* Sidebar system status badge */
.sb-status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--green-primary);
    border: 1px solid var(--green-border);
    padding: 3px 8px;
    margin-top: 0.5rem;
    background-color: #010300;
}

/* Sidebar section header label */
.sb-section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.35rem;
    margin-top: 0.8rem;
    border-bottom: 1px solid #0d2b05;
    padding-bottom: 0.2rem;
}

/* ── METRIC CARDS ───────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
    border-top: 2px solid var(--green-primary) !important;
    padding: 1rem 1.2rem 0.9rem 1.2rem !important;
    box-shadow: inset 0 0 18px rgba(57,255,20,0.05),
                0 0 8px rgba(57,255,20,0.06) !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Phosphor noise texture overlay on metric cards */
[data-testid="stMetric"]::after {
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.07) 0px,
        rgba(0,0,0,0.07) 1px,
        transparent 1px,
        transparent 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* Corner bracket — top-left */
[data-testid="stMetric"]::before {
    content: "◤";
    position: absolute;
    top: 3px; left: 4px;
    color: var(--green-dim);
    font-size: 0.45rem;
    line-height: 1;
    opacity: 0.7;
}

/* Ensure text renders above the scanline overlay */
[data-testid="stMetric"] > div {
    position: relative !important;
    z-index: 1 !important;
}

/* Card label — channel identifier */
[data-testid="stMetric"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    display: block !important;
    margin-bottom: 0.1rem !important;
}

/* Card value — large phosphor readout */
[data-testid="stMetricValue"] {
    font-family: 'VT323', monospace !important;
    font-size: 3.1rem !important;
    color: var(--green-primary) !important;
    text-shadow: var(--glow-green) !important;
    line-height: 1.05 !important;
    letter-spacing: 0.05em !important;
}

/* Delta — directional indicator */
[data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    opacity: 0.85 !important;
}

/* ── CUSTOM KPI READOUT PANELS (pip-kpi-* classes) ──── */

/* Outer wrapper — full instrument gauge panel */
.pip-kpi-panel {
    background-color: var(--bg-panel);
    border: var(--border-main);
    border-top-width: 2px;
    border-top-style: solid;
    padding: 0.9rem 1.1rem 0.85rem 1.1rem;
    position: relative;
    overflow: hidden;
    box-shadow: inset 0 0 18px rgba(57,255,20,0.04),
                0 0 8px rgba(57,255,20,0.05);
}

/* Scanline texture inside panel */
.pip-kpi-panel::after {
    content: "";
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,0,0,0.08) 0px,
        rgba(0,0,0,0.08) 1px,
        transparent 1px,
        transparent 4px
    );
    pointer-events: none;
    z-index: 0;
}

.pip-kpi-panel > * { position: relative; z-index: 1; }

/* Top-left corner bracket marker */
.pip-kpi-panel::before {
    content: "◤";
    position: absolute;
    top: 3px; left: 5px;
    font-size: 0.4rem;
    opacity: 0.5;
    z-index: 2;
}

/* Channel label — the metric title */
.pip-kpi-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 0.2rem;
    display: block;
}

/* Main readout value */
.pip-kpi-value {
    font-family: 'VT323', monospace;
    font-size: 3.1rem;
    line-height: 1.0;
    letter-spacing: 0.06em;
    display: block;
    text-shadow: var(--glow-green);
}

/* Unit suffix next to value */
.pip-kpi-unit {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    opacity: 0.6;
    margin-left: 4px;
    vertical-align: middle;
}

/* Status tag row below the value */
.pip-kpi-status {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border: 1px solid;
    padding: 1px 6px;
    margin-top: 0.45rem;
    opacity: 0.85;
}

/* Colour variants — applied via inline style on border-top & text */
.pip-kpi-green  { color: var(--green-primary); border-top-color: var(--green-primary); }
.pip-kpi-green  .pip-kpi-label  { color: var(--text-dim); }
.pip-kpi-green  .pip-kpi-value  { color: var(--green-primary); text-shadow: var(--glow-green); }
.pip-kpi-green  .pip-kpi-status { color: var(--green-primary); border-color: var(--green-border); }
.pip-kpi-green::before          { color: var(--green-dim); }

.pip-kpi-amber  { color: var(--amber); border-top-color: var(--amber) !important; border-color: var(--amber-dim); }
.pip-kpi-amber  .pip-kpi-label  { color: var(--amber-dim); }
.pip-kpi-amber  .pip-kpi-value  { color: var(--amber); text-shadow: 0 0 8px #ffb000, 0 0 20px rgba(255,176,0,0.3); }
.pip-kpi-amber  .pip-kpi-status { color: var(--amber); border-color: var(--amber-dim); }
.pip-kpi-amber::before          { color: var(--amber-dim); }

.pip-kpi-red    { color: var(--red-alert); border-top-color: var(--red-alert) !important; border-color: var(--red-dim); animation: pulse-alert 2.5s infinite; }
.pip-kpi-red    .pip-kpi-label  { color: #7a2a10; }
.pip-kpi-red    .pip-kpi-value  { color: var(--red-alert); text-shadow: 0 0 8px #ff3c00, 0 0 22px rgba(255,60,0,0.35); }
.pip-kpi-red    .pip-kpi-status { color: var(--red-alert); border-color: var(--red-dim); }
.pip-kpi-red::before            { color: var(--red-dim); }

/* ── BUTTONS ────────────────────────────────────────── */
.stButton > button {
    background-color: transparent !important;
    color: var(--green-primary) !important;
    border: var(--border-bright) !important;
    border-radius: 0px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.4rem !important;
    transition: background-color 0.12s ease,
                box-shadow      0.12s ease,
                color           0.12s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Bracket decorators flanking button text */
.stButton > button::before { content: "[ "; letter-spacing: 0; }
.stButton > button::after  { content: " ]"; letter-spacing: 0; }

.stButton > button:hover {
    background-color: var(--green-muted) !important;
    box-shadow: 0 0 10px rgba(57,255,20,0.4),
                inset 0 0 8px rgba(57,255,20,0.08) !important;
    color: var(--green-bright) !important;
    border-color: var(--green-bright) !important;
}

.stButton > button:active {
    background-color: var(--green-dim) !important;
    color: var(--bg-dark) !important;
    box-shadow: inset 0 0 12px rgba(57,255,20,0.5) !important;
}

/* ── SELECTBOX / DROPDOWNS ──────────────────────────── */

/* Outer label */
.stSelectbox label,
div[data-testid="stSelectbox"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
    margin-bottom: 0.15rem !important;
}

/* Input control box */
.stSelectbox > div > div,
div[data-testid="stSelectbox"] > div > div {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: border-color 0.12s ease, box-shadow 0.12s ease !important;
}

.stSelectbox > div > div:focus-within,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--green-primary) !important;
    box-shadow: 0 0 8px rgba(57,255,20,0.3) !important;
}

/* Selected value text */
.stSelectbox [data-baseweb="select"] span,
div[data-testid="stSelectbox"] [data-baseweb="select"] span {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
}

/* Dropdown chevron icon */
.stSelectbox svg, div[data-testid="stSelectbox"] svg {
    fill: var(--green-dim) !important;
}

/* Dropdown popup container */
[data-baseweb="popover"] {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.8),
                0 0 12px rgba(57,255,20,0.08) !important;
}

/* Dropdown menu list */
[data-baseweb="menu"] {
    background-color: var(--bg-panel) !important;
    border-radius: 0px !important;
    padding: 0 !important;
}

/* Individual menu options */
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    background-color: var(--bg-panel) !important;
    border-bottom: 1px solid #0a1f05 !important;
    padding: 0.45rem 0.9rem !important;
    letter-spacing: 0.06em !important;
    transition: background-color 0.08s ease !important;
}

[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background-color: var(--green-muted) !important;
    color: var(--green-bright) !important;
}

/* Currently selected item in the list */
[aria-selected="true"][data-baseweb="option"] {
    background-color: var(--green-dim) !important;
    color: var(--green-primary) !important;
    border-left: 2px solid var(--green-primary) !important;
}

/* ── TEXT & NUMBER INPUTS ───────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
    color: var(--green-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    caret-color: var(--green-primary) !important;
    transition: border-color 0.12s ease, box-shadow 0.12s ease !important;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--green-primary) !important;
    box-shadow: 0 0 8px rgba(57,255,20,0.3) !important;
    outline: none !important;
}

.stTextInput label, .stNumberInput label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

/* Number input stepper buttons (+/-) */
.stNumberInput button {
    background-color: var(--bg-panel) !important;
    border: var(--border-dim) !important;
    border-radius: 0px !important;
    color: var(--green-dim) !important;
    transition: all 0.1s ease !important;
}

.stNumberInput button:hover {
    background-color: var(--green-muted) !important;
    color: var(--green-primary) !important;
    border-color: var(--green-border) !important;
}

/* ── SLIDERS ────────────────────────────────────────── */
.stSlider label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

/* Track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background-color: var(--green-primary) !important;
    border-radius: 0px !important;
    width: 12px !important;
    height: 12px !important;
    box-shadow: var(--glow-subtle) !important;
    transition: box-shadow 0.12s ease !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:hover {
    box-shadow: var(--glow-green) !important;
}

/* Filled portion of track */
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(3) {
    background-color: var(--green-primary) !important;
    border-radius: 0px !important;
    height: 3px !important;
}

/* Unfilled portion of track */
[data-testid="stSlider"] [data-baseweb="slider"] div:nth-child(4) {
    background-color: var(--green-muted) !important;
    border-radius: 0px !important;
    height: 3px !important;
}

/* Tick value labels under slider */
[data-testid="stSlider"] span {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
}

/* ── RADIO BUTTONS ──────────────────────────────────── */
.stRadio label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

.stRadio > div > label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-primary) !important;
    text-transform: none !important;
    letter-spacing: 0.06em !important;
}

/* Radio circle — unchecked */
.stRadio input[type="radio"] + div {
    border-color: var(--green-border) !important;
    background-color: transparent !important;
}

/* Radio circle — checked */
.stRadio input[type="radio"]:checked + div {
    border-color: var(--green-primary) !important;
    background-color: var(--green-primary) !important;
    box-shadow: var(--glow-subtle) !important;
}

/* ── CHECKBOXES ─────────────────────────────────────── */
.stCheckbox label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--text-primary) !important;
    letter-spacing: 0.06em !important;
}

/* Checkbox box — unchecked */
.stCheckbox input[type="checkbox"] + div {
    border-color: var(--green-border) !important;
    border-radius: 0px !important;
    background-color: transparent !important;
}

/* Checkbox box — checked */
.stCheckbox input[type="checkbox"]:checked + div {
    background-color: var(--green-primary) !important;
    border-color: var(--green-primary) !important;
    border-radius: 0px !important;
    box-shadow: var(--glow-subtle) !important;
}

/* ── TEXTAREA ───────────────────────────────────────── */
.stTextArea textarea {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
    color: var(--green-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    caret-color: var(--green-primary) !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: var(--green-primary) !important;
    box-shadow: 0 0 8px rgba(57,255,20,0.3) !important;
    outline: none !important;
}

.stTextArea label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

/* ── MULTISELECT ────────────────────────────────────── */
.stMultiSelect label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.18em !important;
}

.stMultiSelect > div > div {
    background-color: var(--bg-panel) !important;
    border: var(--border-main) !important;
    border-radius: 0px !important;
}

/* Selected tags inside multiselect */
[data-baseweb="tag"] {
    background-color: var(--green-dim) !important;
    border: 1px solid var(--green-border) !important;
    border-radius: 0px !important;
    color: var(--green-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* Tag remove × button */
[data-baseweb="tag"] span[role="presentation"] {
    color: var(--green-primary) !important;
}

/* ── CONTROL PANEL WRAPPER (helper class) ───────────── */

/* Operator control group — wraps a label + widget */
.pip-control-group {
    border: var(--border-dim);
    border-left: 2px solid var(--green-dim);
    background-color: #030602;
    padding: 0.7rem 0.9rem 0.5rem 0.9rem;
    margin-bottom: 0.8rem;
    position: relative;
}

.pip-control-group::before {
    content: attr(data-channel);
    position: absolute;
    top: -0.55rem;
    left: 0.6rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.16em;
    background-color: #030602;
    padding: 0 4px;
}

/* Operator section header — sits above a group of controls */
.pip-controls-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #0d2b05;
    padding-bottom: 0.25rem;
    margin-bottom: 0.6rem;
    margin-top: 0.4rem;
}

/* ── DATAFRAMES / TABLES ────────────────────────────── */

/* Outer wrapper — instrument readout frame */
[data-testid="stDataFrame"] {
    border: var(--border-main) !important;
    border-top: 2px solid var(--green-border) !important;
    box-shadow: 0 0 18px rgba(57,255,20,0.07),
                inset 0 0 20px rgba(57,255,20,0.02) !important;
    background-color: var(--bg-panel) !important;
}

/* Inner scroll container */
[data-testid="stDataFrame"] .dvn-scroller,
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] iframe {
    background-color: var(--bg-panel) !important;
    color: var(--text-primary) !important;
}

/* Glide data editor — cell-level */
.glideDataEditor {
    background-color: var(--bg-panel) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* Individual cells */
.gdg-cell {
    background-color: var(--bg-panel) !important;
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    border-right: 1px solid #0d2b05 !important;
    border-bottom: 1px solid #0d2b05 !important;
}

/* Header row — column labels */
.gdg-cell.gdg-header,
[data-testid="stDataFrame"] .dvn-header {
    background-color: #040802 !important;
    color: var(--text-dim) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border-bottom: 1px solid var(--green-border) !important;
}

/* Row hover highlight */
.gdg-cell:hover {
    background-color: #0d2b05 !important;
    color: var(--green-bright) !important;
}

/* Scrollbar inside dataframe */
[data-testid="stDataFrame"] ::-webkit-scrollbar {
    width: 5px;
    height: 5px;
    background: var(--bg-panel);
}
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
    background: var(--green-dim);
    border: none;
}

/* st.table (static) */
[data-testid="stTable"] table {
    background-color: var(--bg-panel) !important;
    border-collapse: collapse !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    width: 100% !important;
    border: var(--border-main) !important;
}

[data-testid="stTable"] th {
    background-color: #040802 !important;
    color: var(--text-dim) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    border-bottom: 1px solid var(--green-border) !important;
    padding: 0.45rem 0.7rem !important;
    font-weight: normal !important;
}

[data-testid="stTable"] td {
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    border-bottom: 1px solid #0d2b05 !important;
    padding: 0.35rem 0.7rem !important;
}

[data-testid="stTable"] tr:hover td {
    background-color: #0d2b05 !important;
    color: var(--green-bright) !important;
}

/* DataFrame label / caption above table */
.pip-table-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.3rem;
    display: block;
    padding-left: 2px;
}

/* ── ALERTS / STATUS BOXES ──────────────────────────── */

/* Base shared rules for all alert types */
div[data-testid="stAlert"] {
    border-radius: 0px !important;
    border-top: 1px solid !important;
    border-right: 1px solid !important;
    border-bottom: 1px solid !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.7rem 1rem 0.7rem 1.2rem !important;
    position: relative !important;
}

/* Strip default Streamlit alert icons */
div[data-testid="stAlert"] svg {
    display: none !important;
}

/* ── st.success → STATUS: NOMINAL ── */
div[data-testid="stAlert"][data-baseweb="notification"],
div[data-testid="stAlert"] > div[class*="success"],
[data-testid="stAlert"] {
    background-color: var(--green-muted) !important;
    border-color: var(--green-border) !important;
    border-left: 3px solid var(--green-primary) !important;
    color: var(--green-primary) !important;
}

/* Selector overrides per type — success */
div[data-testid="stAlert"][kind="success"],
div[data-testid="stNotification"][kind="success"] {
    background-color: var(--green-muted) !important;
    border-color: var(--green-border) !important;
    border-left: 3px solid var(--green-primary) !important;
    color: var(--green-primary) !important;
}

/* ── st.warning → CAUTION ── */
div[data-testid="stAlert"][kind="warning"],
div[data-testid="stNotification"][kind="warning"],
div[data-testid="stAlert"] > div[class*="warning"] {
    background-color: #0d0800 !important;
    border-color: var(--amber-dim) !important;
    border-left: 3px solid var(--amber) !important;
    color: var(--amber) !important;
}

/* ── st.error → ALERT: CRITICAL ── */
div[data-testid="stAlert"][kind="error"],
div[data-testid="stNotification"][kind="error"],
div[data-testid="stAlert"] > div[class*="error"] {
    background-color: #0a0100 !important;
    border-color: var(--red-dim) !important;
    border-left: 3px solid var(--red-alert) !important;
    color: var(--red-alert) !important;
    animation: pulse-alert 2.2s infinite !important;
}

/* ── st.info → SYSTEM INFO ── */
div[data-testid="stAlert"][kind="info"],
div[data-testid="stNotification"][kind="info"],
div[data-testid="stAlert"] > div[class*="info"] {
    background-color: #030a03 !important;
    border-color: var(--green-dim) !important;
    border-left: 3px solid var(--green-dim) !important;
    color: var(--text-dim) !important;
}

/* Alert text content */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.05em !important;
    line-height: 1.5 !important;
}

/* ── CUSTOM PIP-BOY STATUS BROADCAST PANELS ─────────── */

/* Full-width system broadcast bar */
.pip-broadcast {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 0.75rem 1.1rem;
    margin-bottom: 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.06em;
    position: relative;
    border-radius: 0;
}

/* Left glyph / status code block */
.pip-broadcast-code {
    font-family: 'VT323', monospace;
    font-size: 1.5rem;
    line-height: 1;
    flex-shrink: 0;
    padding-top: 1px;
}

/* Message body */
.pip-broadcast-body {
    flex: 1;
}

.pip-broadcast-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    opacity: 0.7;
    display: block;
    margin-bottom: 0.2rem;
}

.pip-broadcast-msg {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    line-height: 1.5;
    display: block;
}

/* Timestamp / source tag — top right */
.pip-broadcast-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.5;
    flex-shrink: 0;
    padding-top: 3px;
}

/* Colour variants */
.pip-broadcast-nominal {
    background-color: #040d02;
    border: 1px solid var(--green-border);
    border-left: 3px solid var(--green-primary);
    color: var(--green-primary);
}

.pip-broadcast-caution {
    background-color: #0a0600;
    border: 1px solid var(--amber-dim);
    border-left: 3px solid var(--amber);
    color: var(--amber);
}

.pip-broadcast-critical {
    background-color: #0a0100;
    border: 1px solid var(--red-dim);
    border-left: 3px solid var(--red-alert);
    color: var(--red-alert);
    animation: pulse-alert 2.2s infinite;
}

/* ── DIVIDERS ───────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid var(--green-dim) !important;
    box-shadow: 0 0 6px rgba(57,255,20,0.2) !important;
    margin: 1.2rem 0 !important;
}

/* ── SLIDERS ────────────────────────────────────────── */
[data-testid="stSlider"] div[data-baseweb="slider"] div {
    background-color: var(--green-primary) !important;
}

/* ── EXPANDERS ──────────────────────────────────────── */
[data-testid="stExpander"] {
    border: var(--border-main) !important;
    border-radius: 0px !important;
    background-color: var(--bg-panel) !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--text-primary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── COLUMNS / LAYOUT ───────────────────────────────── */
[data-testid="column"] {
    background-color: transparent !important;
}

/* ── FILE UPLOADER ──────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 1px dashed var(--green-border) !important;
    border-radius: 0px !important;
    background-color: var(--bg-panel) !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── SPINNER ────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--green-primary) !important;
}

/* ── PROGRESS BAR ───────────────────────────────────── */
.stProgress > div > div > div {
    background-color: var(--green-primary) !important;
    box-shadow: var(--glow-subtle) !important;
    border-radius: 0px !important;
}

.stProgress > div > div {
    background-color: var(--green-muted) !important;
    border-radius: 0px !important;
}

/* ── PYPLOT FIGURE CONTAINERS ───────────────────────── */
[data-testid="stImage"] img,
.stPlotlyChart,
[data-testid="stPyplotContainer"] {
    border: var(--border-main) !important;
    border-top: 2px solid var(--green-border) !important;
    box-shadow: 0 0 20px rgba(57,255,20,0.1),
                inset 0 0 30px rgba(57,255,20,0.02) !important;
    background-color: var(--bg-panel) !important;
}

/* Chart caption label below figures */
[data-testid="stPyplotContainer"] + div small,
[data-testid="caption"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.6rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

/* ── ANIMATIONS ─────────────────────────────────────── */

/* Blink — status dots, 2-state hard cut */
@keyframes blink {
    0%, 49% { opacity: 1; }
    50%, 100% { opacity: 0; }
}

/* Pulse-alert — critical panel box-shadow breathe */
@keyframes pulse-alert {
    0%, 100% {
        box-shadow: 0 0 4px rgba(255,60,0,0.2),
                    inset 0 0 8px rgba(255,60,0,0.03);
    }
    50% {
        box-shadow: 0 0 22px rgba(255,60,0,0.65),
                    inset 0 0 14px rgba(255,60,0,0.08);
    }
}

/* Pulse-amber — caution panel breathe */
@keyframes pulse-amber {
    0%, 100% {
        box-shadow: 0 0 4px rgba(255,176,0,0.15),
                    inset 0 0 8px rgba(255,176,0,0.02);
    }
    50% {
        box-shadow: 0 0 18px rgba(255,176,0,0.5),
                    inset 0 0 12px rgba(255,176,0,0.06);
    }
}

/* Flicker — CRT phosphor title, very rare drop */
@keyframes flicker {
    0%,  72%,  74%,  97%, 100% { opacity: 1;    }
    73%                        { opacity: 0.82;  }
    98%                        { opacity: 0.91;  }
}

/* Glow-pulse — soft green glow breathe on KPI values */
@keyframes glow-pulse {
    0%, 100% { text-shadow: 0 0 6px #39ff14, 0 0 14px rgba(57,255,20,0.25); }
    50%      { text-shadow: 0 0 12px #39ff14, 0 0 30px rgba(57,255,20,0.5); }
}

/* Phosphor sweep — single bright line descending over the screen */
@keyframes sweep {
    0%   { top: -4px;   opacity: 0.6; }
    8%   { opacity: 0.6; }
    92%  { opacity: 0.3; }
    100% { top: 100vh;  opacity: 0; }
}

/* Fade-in-up — component entry animation */
@keyframes fade-in-up {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Slide-in-left — sidebar entry */
@keyframes slide-in-left {
    from {
        opacity: 0;
        transform: translateX(-12px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Type-cursor — blinking block cursor for boot lines */
@keyframes type-cursor {
    0%, 49% { border-right-color: var(--green-primary); }
    50%, 100% { border-right-color: transparent; }
}

/* ── PHOSPHOR SWEEP LINE ─────────────────────────────── */
/* A single bright horizontal line that descends the screen
   every ~14s, simulating a CRT refresh sweep */
.stApp::before {
    content: "";
    position: fixed;
    left: 0;
    width: 100vw;
    height: 3px;
    background: linear-gradient(
        to bottom,
        transparent 0%,
        rgba(57,255,20,0.18) 40%,
        rgba(57,255,20,0.35) 50%,
        rgba(57,255,20,0.18) 60%,
        transparent 100%
    );
    pointer-events: none;
    z-index: 9997;
    animation: sweep 14s linear infinite;
}

/* ── ENTRY ANIMATIONS — component fade-in on load ────── */

/* Masthead fades in first */
.pip-masthead {
    animation: fade-in-up 0.6s ease both;
}

/* Broadcast panel slightly delayed */
.pip-broadcast {
    animation: fade-in-up 0.5s ease both;
}

/* KPI panels staggered */
.pip-kpi-panel {
    animation: fade-in-up 0.55s ease both;
}

/* Section headers */
.pip-section-header {
    animation: fade-in-up 0.4s ease both;
}

/* Sidebar slides in from left */
[data-testid="stSidebar"] > div:first-child {
    animation: slide-in-left 0.5s ease both;
}

/* Scan dividers fade in */
.pip-scan-divider {
    animation: fade-in-up 0.4s ease both;
}

/* ── KPI VALUE GLOW PULSE ───────────────────────────── */
/* The primary value readout on green KPI cards breathes */
.pip-kpi-green .pip-kpi-value {
    animation: glow-pulse 3.5s ease-in-out infinite;
}

/* ── SECTION HEADER BAR PULSE ───────────────────────── */
.pip-section-header-bar {
    animation: glow-pulse 4s ease-in-out infinite;
}

/* ── BROADCAST PANEL CAUTION PULSE ─────────────────── */
.pip-broadcast-caution {
    animation: pulse-amber 3s ease-in-out infinite;
}

/* ── HOVER LIFT — panels & charts ───────────────────── */
.pip-kpi-panel,
[data-testid="stPyplotContainer"],
[data-testid="stDataFrame"] {
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.pip-kpi-panel:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(57,255,20,0.12),
                inset 0 0 18px rgba(57,255,20,0.05) !important;
}

[data-testid="stPyplotContainer"]:hover {
    box-shadow: 0 4px 28px rgba(57,255,20,0.14),
                inset 0 0 30px rgba(57,255,20,0.03) !important;
}

/* ── SELECTBOX FOCUS GLOW ANIMATION ─────────────────── */
@keyframes input-focus-glow {
    from { box-shadow: 0 0 0px rgba(57,255,20,0); }
    to   { box-shadow: 0 0 10px rgba(57,255,20,0.35); }
}

.stSelectbox > div > div:focus-within {
    animation: input-focus-glow 0.15s ease forwards !important;
}

/* ── FOOTER ─────────────────────────────────────────── */
.pip-footer {
    margin-top: 2.5rem;
    border-top: 1px solid var(--green-dim);
    padding-top: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
    animation: fade-in-up 0.6s ease both;
}

.pip-footer-left {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    line-height: 1.7;
}

.pip-footer-right {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    text-align: right;
    line-height: 1.7;
}

.pip-footer-right span {
    color: var(--text-dim);
}

/* ── BOOT SEQUENCE PANEL ────────────────────────────── */
.pip-boot {
    background-color: var(--bg-panel);
    border: var(--border-dim);
    border-left: 2px solid var(--green-dim);
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    animation: fade-in-up 0.4s ease both;
}

.pip-boot-line {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    line-height: 1.8;
    display: block;
}

.pip-boot-line span {
    color: var(--green-primary);
}

.pip-boot-line.ok span  { color: var(--green-primary); }
.pip-boot-line.err span { color: var(--red-alert); }
.pip-boot-line.wrn span { color: var(--amber); }

/* Blinking cursor at end of last boot line */
.pip-boot-cursor {
    display: inline-block;
    width: 7px;
    height: 0.75em;
    background: var(--green-primary);
    vertical-align: middle;
    margin-left: 3px;
    animation: blink 1s step-end infinite;
}

/* ── HELPER CLASSES (used by st.markdown HTML blocks) ── */

/* Blinking status dot */
.pip-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--green-primary);
    border-radius: 50%;
    box-shadow: var(--glow-green);
    animation: blink 1.4s step-end infinite;
    margin-right: 6px;
    vertical-align: middle;
}

/* Section header block */
.pip-header {
    font-family: 'VT323', monospace;
    font-size: 1.4rem;
    color: var(--green-primary);
    text-shadow: var(--glow-green);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-top: var(--border-main);
    border-bottom: var(--border-main);
    padding: 0.35rem 0.6rem;
    margin: 1.2rem 0 0.6rem 0;
    background-color: var(--green-muted);
}

/* Terminal readout panel */
.pip-panel {
    background-color: var(--bg-panel);
    border: var(--border-main);
    padding: 0.8rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-primary);
    margin-bottom: 0.8rem;
    box-shadow: inset 0 0 10px rgba(57,255,20,0.04);
}

/* Amber warning panel */
.pip-warn {
    background-color: #0d0800;
    border: 1px solid var(--amber);
    padding: 0.7rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--amber);
    margin-bottom: 0.8rem;
}

/* Critical alert panel */
.pip-critical {
    background-color: #0d0200;
    border: 1px solid var(--red-alert);
    padding: 0.7rem 1rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    color: var(--red-alert);
    margin-bottom: 0.8rem;
    animation: pulse-alert 2s infinite;
}

/* Inline label tag */
.pip-tag {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border: 1px solid var(--green-dim);
    padding: 1px 5px;
    margin-right: 6px;
}

/* Main title flicker */
.pip-title {
    font-family: 'VT323', monospace;
    font-size: 3.2rem;
    color: var(--green-primary);
    text-shadow: var(--glow-green);
    text-transform: uppercase;
    letter-spacing: 0.18em;
    animation: flicker 8s infinite;
    line-height: 1.1;
}

/* Subtitle / descriptor line */
.pip-subtitle {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-top: 0.2rem;
}

/* Horizontal rule styled as scan line */
.pip-divider {
    border: none;
    border-top: 1px solid var(--green-dim);
    box-shadow: 0 0 8px rgba(57,255,20,0.25);
    margin: 1rem 0;
}

</style>
""", unsafe_allow_html=True)


# =========================
# 2. Load Saved Artifacts
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/lstm_autoencoder.keras")

@st.cache_resource
def load_scalers():
    return joblib.load("models/scalers.pkl")


@st.cache_resource
def load_config():
    with open("models/model_config.json") as f:
        return json.load(f)

@st.cache_resource
def load_engine_thresholds():
    return joblib.load("models/engine_thresholds.pkl")

engine_thresholds = load_engine_thresholds()

model = load_model()
scalers = load_scalers()
config = load_config()

selected_sensors = config["selected_sensors"]
seq_length = config["seq_length"]


# =========================
# PHASE 8: PAGE MASTHEAD
# =========================

st.markdown("""
<div class="pip-masthead">
    <div class="pip-masthead-eyebrow">
        <span class="pip-dot"></span>
        Vault-Tec Industrial Monitoring Suite &nbsp;&#124;&nbsp; Rev 4.7.1
    </div>
    <span class="pip-masthead-title">
        Nuclear Auxiliary Cooling<br>Pump Monitoring System
    </span>
    <div class="pip-masthead-subtitle">
        <span class="pip-masthead-tag">
            <span class="pip-dot" style="width:5px;height:5px;"></span>
            SIMULATION MODE
        </span>
        <span class="pip-masthead-tag">UNIT 7 &nbsp;&#124;&nbsp; PRIMARY LOOP</span>
        <span class="pip-masthead-tag">LSTM AUTOENCODER &nbsp;&#124;&nbsp; NASA CMAPSS</span>
        <span class="pip-masthead-tag">FLEET MONITORING ACTIVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pip-panel" style="margin-bottom:0.6rem; padding: 0.55rem 1rem;">
    <span style="font-size:0.75rem; color:var(--text-dim); letter-spacing:0.06em;">
    This dashboard simulates fleet-wide health monitoring of auxiliary cooling pumps
    in a nuclear power facility using LSTM-based anomaly detection trained on
    NASA CMAPSS turbofan degradation data.
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="pip-scan-divider"><span class="pip-scan-divider-label">// begin telemetry</span></div>', unsafe_allow_html=True)

# =========================
# PHASE 9: BOOT SEQUENCE
# =========================

st.markdown(f"""
<div class="pip-boot">
    <span class="pip-boot-line ok">VAULT-TEC NACP MONITORING SUITE <span>v4.7.1</span> &nbsp;&#124;&nbsp; INITIALISING...</span>
    <span class="pip-boot-line ok">LSTM AUTOENCODER &nbsp;&bull;&nbsp; <span>LOADED</span> &nbsp;&#124;&nbsp; NASA CMAPSS FD001 WEIGHTS</span>
    <span class="pip-boot-line ok">SENSOR SCALERS &nbsp;&bull;&nbsp; <span>LOADED</span> &nbsp;&#124;&nbsp; {len(selected_sensors)} CHANNELS CALIBRATED</span>
    <span class="pip-boot-line ok">ENGINE THRESHOLDS &nbsp;&bull;&nbsp; <span>LOADED</span> &nbsp;&#124;&nbsp; PER-UNIT ADAPTIVE</span>
    <span class="pip-boot-line ok">SEQUENCE BUILDER &nbsp;&bull;&nbsp; <span>READY</span> &nbsp;&#124;&nbsp; WINDOW = {seq_length} CYCLES</span>
    <span class="pip-boot-line ok">ANOMALY DETECTION &nbsp;&bull;&nbsp; <span>ACTIVE</span> &nbsp;&#124;&nbsp; RECONSTRUCTION ERROR METHOD</span>
    <span class="pip-boot-line ok">ALL SYSTEMS &nbsp;&bull;&nbsp; <span>NOMINAL</span> &nbsp;&#124;&nbsp; MONITORING ONLINE<span class="pip-boot-cursor"></span></span>
</div>
""", unsafe_allow_html=True)

# Load Data
df = load_processed_data()

for sensor in selected_sensors:
    df[sensor] = scalers[sensor].transform(
        df[sensor].values.reshape(-1, 1)
    )


X_sequences, engine_ids, end_cycles = create_sequences(
    df,
    sensors=selected_sensors,
    seq_length=seq_length
)

# Reconstruction Error
reconstruction_error = compute_reconstruction_error(model, X_sequences)

# Thresholding
is_anomaly = np.array([
    reconstruction_error[i] > engine_thresholds[int(engine_ids[i])]
    for i in range(len(reconstruction_error))
])


# Consecutive Detection
consecutive_flags = apply_consecutive_logic(
    is_anomaly,
    engine_ids
)


# Lead Time
lead_times = compute_lead_times(
    engine_ids,
    end_cycles,
    consecutive_flags
)

lead_df = pd.DataFrame({
    "engine_id": list(lead_times.keys()),
    "lead_time": list(lead_times.values())
})

# =========================
# PHASE 2: SIDEBAR PANEL
# Styled instrument panel with live fleet telemetry
# =========================

_total_pumps      = len(lead_df)
_avg_lead         = lead_df["lead_time"].mean()
_low_lead_count   = int((lead_df["lead_time"] < 50).sum())
_min_lead         = int(lead_df["lead_time"].min())
_max_lead         = int(lead_df["lead_time"].max())

# Derive overall fleet status for the badge
if _low_lead_count == 0:
    _fleet_status       = "ALL SYSTEMS NOMINAL"
    _fleet_status_color = "var(--green-primary)"
    _fleet_dot_color    = "var(--green-primary)"
elif _low_lead_count <= 2:
    _fleet_status       = "CAUTION — REVIEW REQ."
    _fleet_status_color = "var(--amber)"
    _fleet_dot_color    = "var(--amber)"
else:
    _fleet_status       = "ALERT — DEGRADATION"
    _fleet_status_color = "var(--red-alert)"
    _fleet_dot_color    = "var(--red-alert)"

_low_lead_class = "red" if _low_lead_count > 2 else ("amber" if _low_lead_count > 0 else "")

st.sidebar.markdown(f"""
<div class="sb-facility-plate">
    <div class="sb-facility-name">&#9762; NACP-7</div>
    <div class="sb-facility-sub">Nuclear Aux. Cooling System</div>
    <div class="sb-facility-sub" style="margin-top:0.3rem; color:#0f4004;">
        Unit 7 &nbsp;&#124;&nbsp; Primary Loop &nbsp;&#124;&nbsp; Sector D
    </div>
    <div class="sb-status-badge" style="color:{_fleet_status_color}; border-color:{_fleet_dot_color};">
        <span class="pip-dot" style="background:{_fleet_dot_color}; box-shadow: 0 0 6px {_fleet_dot_color};"></span>
        {_fleet_status}
    </div>
</div>

<div class="sb-section-label">&#9492;&#9472; Fleet Telemetry</div>

<div class="pip-panel" style="padding: 0.5rem 0.7rem; margin-bottom: 0.5rem;">
    <div class="sb-stat-row">
        <span class="sb-stat-label">Monitored Pumps</span>
        <span class="sb-stat-value">{_total_pumps}</span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-label">Avg Lead Time</span>
        <span class="sb-stat-value">{_avg_lead:.1f} <span style="font-size:0.65rem; color:var(--text-dim);">CYC</span></span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-label">Min Lead Time</span>
        <span class="sb-stat-value {'amber' if _min_lead < 50 else ''}">{_min_lead} <span style="font-size:0.65rem; color:var(--text-dim);">CYC</span></span>
    </div>
    <div class="sb-stat-row">
        <span class="sb-stat-label">Max Lead Time</span>
        <span class="sb-stat-value">{_max_lead} <span style="font-size:0.65rem; color:var(--text-dim);">CYC</span></span>
    </div>
    <div class="sb-stat-row" style="border-bottom:none;">
        <span class="sb-stat-label">Low Lead (&lt;50)</span>
        <span class="sb-stat-value {_low_lead_class}">{_low_lead_count}</span>
    </div>
</div>

<div class="sb-section-label">&#9492;&#9472; Model Configuration</div>

<div class="sb-model-block">
    <div class="sb-model-line">Architecture &nbsp;<span>LSTM Autoencoder</span></div>
    <div class="sb-model-line">Dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>NASA CMAPSS FD001</span></div>
    <div class="sb-model-line">Seq Length &nbsp;&nbsp;<span>{seq_length} cycles</span></div>
    <div class="sb-model-line">Sensors &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span>{len(selected_sensors)} channels</span></div>
    <div class="sb-model-line">Threshold &nbsp;&nbsp;&nbsp;<span>Per-engine adaptive</span></div>
</div>

<div class="sb-section-label">&#9492;&#9472; Operator Controls</div>
""", unsafe_allow_html=True)


# ---- Phase 7: Fleet Status Broadcast ----
# Dynamically driven by the same threshold logic as the KPI panels
if _low_lead_count == 0:
    _bc_class   = "pip-broadcast-nominal"
    _bc_code    = "&#9679; SYS"
    _bc_title   = "SYSTEM STATUS BROADCAST // FLEET HEALTH"
    _bc_msg     = (
        f"All {_total_pumps} monitored pumps operating within nominal parameters. "
        f"Fleet average lead time: <strong>{_avg_lead:.1f}</strong> cycles. "
        f"No units approaching service threshold. Continuous monitoring active."
    )
elif _low_lead_count <= 2:
    _bc_class   = "pip-broadcast-caution"
    _bc_code    = "&#9888; CAU"
    _bc_title   = "CAUTION ADVISORY // FLEET HEALTH MONITOR"
    _bc_msg     = (
        f"<strong>{_low_lead_count}</strong> pump(s) approaching service threshold "
        f"(&lt;50 cycles remaining). Fleet average lead time: "
        f"<strong>{_avg_lead:.1f}</strong> cycles. "
        f"Operator review of flagged units is recommended."
    )
else:
    _bc_class   = "pip-broadcast-critical"
    _bc_code    = "&#9888; ALT"
    _bc_title   = "CRITICAL ALERT // FLEET DEGRADATION DETECTED"
    _bc_msg     = (
        f"<strong>{_low_lead_count}</strong> pump(s) below service threshold "
        f"(&lt;50 cycles remaining). Fleet average lead time: "
        f"<strong>{_avg_lead:.1f}</strong> cycles. "
        f"Immediate inspection of flagged units required. Do not defer maintenance."
    )

st.markdown(f"""
<div class="pip-broadcast {_bc_class}">
    <div class="pip-broadcast-code">{_bc_code}</div>
    <div class="pip-broadcast-body">
        <span class="pip-broadcast-title">{_bc_title}</span>
        <span class="pip-broadcast-msg">{_bc_msg}</span>
    </div>
    <div class="pip-broadcast-meta">NACP-7 &nbsp;&#124;&nbsp; UNIT 7<br>PRIMARY LOOP<br>AUTO-GENERATED</div>
</div>
""", unsafe_allow_html=True)

# ---- Fleet Summary ----
st.markdown("""
<div class="pip-section-header">
    <span class="pip-section-header-bar"></span>
    <span class="pip-section-header-text">Fleet Health Summary</span>
    <span class="pip-section-header-tag">SEC-01</span>
</div>
""", unsafe_allow_html=True)

# ---- Phase 3: Custom KPI Readout Panels ----
# Compute values and derive colour states from thresholds
_kpi_total        = len(lead_df)
_kpi_avg_lead     = lead_df["lead_time"].mean()
_kpi_low_lead     = int((lead_df["lead_time"] < 50).sum())

# Colour logic: avg lead time
_avg_colour = "pip-kpi-green" if _kpi_avg_lead >= 80 else ("pip-kpi-amber" if _kpi_avg_lead >= 50 else "pip-kpi-red")
_avg_status = "NOMINAL" if _kpi_avg_lead >= 80 else ("MONITOR" if _kpi_avg_lead >= 50 else "CRITICAL")
_avg_dot    = "#39ff14" if _kpi_avg_lead >= 80 else ("#ffb000" if _kpi_avg_lead >= 50 else "#ff3c00")

# Colour logic: low lead time count
_low_colour = "pip-kpi-green" if _kpi_low_lead == 0 else ("pip-kpi-amber" if _kpi_low_lead <= 2 else "pip-kpi-red")
_low_status = "NOMINAL" if _kpi_low_lead == 0 else ("CAUTION" if _kpi_low_lead <= 2 else "ALERT")
_low_dot    = "#39ff14" if _kpi_low_lead == 0 else ("#ffb000" if _kpi_low_lead <= 2 else "#ff3c00")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="pip-kpi-panel pip-kpi-green">
        <span class="pip-kpi-label">// CH-01 &nbsp;&bull;&nbsp; Fleet Size</span>
        <span class="pip-kpi-value">{_kpi_total}<span class="pip-kpi-unit">units</span></span>
        <div class="pip-kpi-status">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                         background:#39ff14; box-shadow: 0 0 5px #39ff14;
                         animation: blink 1.4s step-end infinite;"></span>
            MONITORING ACTIVE
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="pip-kpi-panel {_avg_colour}">
        <span class="pip-kpi-label">// CH-02 &nbsp;&bull;&nbsp; Avg Lead Time</span>
        <span class="pip-kpi-value">{_kpi_avg_lead:.1f}<span class="pip-kpi-unit">cycles</span></span>
        <div class="pip-kpi-status">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                         background:{_avg_dot}; box-shadow: 0 0 5px {_avg_dot};
                         animation: blink 1.4s step-end infinite;"></span>
            {_avg_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="pip-kpi-panel {_low_colour}">
        <span class="pip-kpi-label">// CH-03 &nbsp;&bull;&nbsp; Low Lead Pumps &lt;50</span>
        <span class="pip-kpi-value">{_kpi_low_lead}<span class="pip-kpi-unit">pumps</span></span>
        <div class="pip-kpi-status">
            <span style="display:inline-block; width:6px; height:6px; border-radius:50%;
                         background:{_low_dot}; box-shadow: 0 0 5px {_low_dot};
                         animation: blink 1.4s step-end infinite;"></span>
            {_low_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Lead Time Chart ----
st.markdown('<div class="pip-scan-divider"><span class="pip-scan-divider-label">// fleet analysis</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="pip-section-header">
    <span class="pip-section-header-bar"></span>
    <span class="pip-section-header-text">Per-Pump Lead Time</span>
    <span class="pip-section-header-tag">SEC-02</span>
</div>
""", unsafe_allow_html=True)

# ── Phase 4: Pip-Boy Matplotlib theme ───────────────────
_BG       = "#050a03"     # figure & axes background
_GREEN    = "#39ff14"     # primary trace / bar colour
_GREEN_DM = "#1f5c0a"     # dim green for grid lines
_GREEN_BD = "#2a7a0e"     # border / spine colour
_AMBER    = "#ffb000"     # threshold / mean line
_TEXT_DM  = "#1e7a06"     # axis label & tick colour
_FONT     = "monospace"   # tick & label font family

fig, ax = plt.subplots(figsize=(12, 4))

# ── Figure & axes backgrounds
fig.patch.set_facecolor(_BG)
ax.set_facecolor(_BG)

# ── Bar colours — colour each bar individually by threshold
_bar_colours = [
    "#ff3c00" if v < 30 else ("#ffb000" if v < 50 else _GREEN)
    for v in lead_df["lead_time"]
]
bars = ax.bar(
    lead_df["engine_id"],
    lead_df["lead_time"],
    color=_bar_colours,
    edgecolor="#0d2b05",
    linewidth=0.6,
    zorder=3
)

# ── Mean threshold line
ax.axhline(
    lead_df["lead_time"].mean(),
    linestyle="--",
    color=_AMBER,
    linewidth=1.2,
    alpha=0.85,
    zorder=4,
    label=f"FLEET AVG  {lead_df['lead_time'].mean():.1f} CYC"
)

# ── Danger threshold line at 50 cycles
ax.axhline(
    50,
    linestyle=":",
    color="#ff3c00",
    linewidth=0.9,
    alpha=0.6,
    zorder=4,
    label="LOW LEAD THRESHOLD  50 CYC"
)

# ── Grid
ax.grid(
    True,
    axis="y",
    color=_GREEN_DM,
    linestyle="--",
    linewidth=0.5,
    alpha=0.5,
    zorder=0
)
ax.set_axisbelow(True)

# ── Spines
for spine in ax.spines.values():
    spine.set_edgecolor(_GREEN_BD)
    spine.set_linewidth(0.8)

# ── Axis labels
ax.set_xlabel(
    "// PUMP ID",
    color=_TEXT_DM,
    fontfamily=_FONT,
    fontsize=8,
    labelpad=8
)
ax.set_ylabel(
    "LEAD TIME (CYCLES)",
    color=_TEXT_DM,
    fontfamily=_FONT,
    fontsize=8,
    labelpad=8
)

# ── Tick params
ax.tick_params(
    colors=_TEXT_DM,
    labelsize=7,
    length=3,
    width=0.6
)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontfamily(_FONT)

# ── Legend
leg = ax.legend(
    frameon=True,
    facecolor="#070f04",
    edgecolor=_GREEN_BD,
    fontsize=7,
    labelcolor=_TEXT_DM,
    loc="upper right"
)
for txt in leg.get_texts():
    txt.set_fontfamily(_FONT)

# ── Title
ax.set_title(
    "[ FLEET LEAD TIME ANALYSIS — ALL PUMPS ]",
    color=_GREEN,
    fontfamily=_FONT,
    fontsize=9,
    fontweight="bold",
    pad=10,
    loc="left"
)

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---- Phase 5: Fleet Data Table ----
st.markdown(
    '<span class="pip-table-label">&#9492;&#9472; RAW TELEMETRY READOUT &nbsp;&bull;&nbsp; '
    'PER-PUMP LEAD TIME REGISTER</span>',
    unsafe_allow_html=True
)

# Build display dataframe
_display_df = lead_df.copy()
_display_df.columns = ["PUMP ID", "LEAD TIME (CYC)"]
_display_df["STATUS"] = _display_df["LEAD TIME (CYC)"].apply(
    lambda v: "&#9888; CRITICAL" if v < 30 else ("&#9679; CAUTION" if v < 50 else "&#10004; NOMINAL")
)

# ── Build pure HTML table — version-independent, guaranteed to render
def _row_to_html(row):
    v = row["LEAD TIME (CYC)"]
    if v < 30:
        row_bg  = "#0d0200"
        val_col = "#ff3c00"
        sta_col = "#ff3c00"
    elif v < 50:
        row_bg  = "#0d0800"
        val_col = "#ffb000"
        sta_col = "#ffb000"
    else:
        row_bg  = "#050a03"
        val_col = "#39ff14"
        sta_col = "#39ff14"
    return (
        f'<tr style="background-color:{row_bg}; border-bottom:1px solid #0d2b05;">'
        f'<td style="color:#1e7a06; padding:0.38rem 0.9rem;">{int(row["PUMP ID"])}</td>'
        f'<td style="color:{val_col}; font-weight:bold; padding:0.38rem 0.9rem;">{int(v)}</td>'
        f'<td style="color:{sta_col}; padding:0.38rem 0.9rem;">{row["STATUS"]}</td>'
        f'</tr>'
    )

_rows_html = "\n".join(_display_df.apply(_row_to_html, axis=1).tolist())


# ── Row height ~34px × 10 visible rows = 340px scroll window
_TABLE_COMMON = (
    "width:100%; border-collapse:collapse; table-layout:fixed;"
    "font-family:'Share Tech Mono',monospace; font-size:0.78rem;"
)
_COL_WIDTHS = (
    '<colgroup>'
    '<col style="width:28%;">'
    '<col style="width:32%;">'
    '<col style="width:40%;">'
    '</colgroup>'
)
_TH_STYLE = (
    "color:#1e7a06; font-weight:normal; text-transform:uppercase;"
    "letter-spacing:0.14em; padding:0.45rem 0.9rem; text-align:left;"
)

_table_html = f"""
<div style="border:1px solid #2a7a0e; border-top:2px solid #2a7a0e;
            box-shadow:0 0 18px rgba(57,255,20,0.07);
            margin-bottom:1rem; overflow:hidden;">

  <!-- FROZEN HEADER — sits outside the scroll container -->
  <div style="background:#040802; border-bottom:1px solid #2a7a0e;">
    <table style="{_TABLE_COMMON}">
      {_COL_WIDTHS}
      <thead>
        <tr>
          <th style="{_TH_STYLE}">Pump ID</th>
          <th style="{_TH_STYLE}">Lead Time (Cyc)</th>
          <th style="{_TH_STYLE}">Status</th>
        </tr>
      </thead>
    </table>
  </div>

  <!-- SCROLLABLE BODY — 10 rows visible (~340px), hover-scroll enabled -->
  <div style="overflow-y:auto; max-height:340px;
              scrollbar-width:thin;
              scrollbar-color:#1f5c0a #050a03;">
    <table style="{_TABLE_COMMON} background:#070f04;">
      {_COL_WIDTHS}
      <tbody>
{_rows_html}
      </tbody>
    </table>
  </div>

</div>
"""

st.markdown(_table_html, unsafe_allow_html=True)

# ---- Pump Drilldown ----
st.markdown('<div class="pip-scan-divider"><span class="pip-scan-divider-label">// unit inspection</span></div>', unsafe_allow_html=True)
st.markdown("""
<div class="pip-section-header">
    <span class="pip-section-header-bar"></span>
    <span class="pip-section-header-text">Individual Pump Inspection</span>
    <span class="pip-section-header-tag">SEC-03</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="pip-controls-header">'
    '&#9492;&#9472; OPERATOR CONTROLS &nbsp;&bull;&nbsp; PUMP SELECTION &amp; SENSOR CHANNEL'
    '</div>',
    unsafe_allow_html=True
)

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    st.markdown(
        '<div class="pip-control-group" data-channel="// SEL-01 — Target Unit">'
        '<span style="font-family:\'Share Tech Mono\',monospace; font-size:0.6rem; '
        'color:var(--text-dim); text-transform:uppercase; letter-spacing:0.16em;">'
        'Select Pump ID</span></div>',
        unsafe_allow_html=True
    )
    selected_engine = st.selectbox(
        "Select Pump ID", lead_df["engine_id"], label_visibility="collapsed"
    )

with col_sel2:
    st.markdown(
        '<div class="pip-control-group" data-channel="// SEL-02 — Sensor Channel">'
        '<span style="font-family:\'Share Tech Mono\',monospace; font-size:0.6rem; '
        'color:var(--text-dim); text-transform:uppercase; letter-spacing:0.16em;">'
        'Select Sensor</span></div>',
        unsafe_allow_html=True
    )
    sensor_to_plot = st.selectbox(
        "Select Sensor", selected_sensors, label_visibility="collapsed"
    )

engine_data = df[df["engine_id"] == selected_engine]

# ---- Phase 7: Per-Pump Status Alert ----
_pump_lead = int(lead_df.loc[lead_df["engine_id"] == selected_engine, "lead_time"].values[0])

if _pump_lead < 30:
    _pump_bc_class = "pip-broadcast-critical"
    _pump_bc_code  = "&#9888; ALT"
    _pump_bc_title = f"CRITICAL — PUMP {selected_engine} // IMMEDIATE ACTION REQUIRED"
    _pump_bc_msg   = (
        f"Pump <strong>{selected_engine}</strong> has only "
        f"<strong>{_pump_lead}</strong> cycles of estimated useful life remaining. "
        f"Unit has breached the critical service threshold (&lt;30 cycles). "
        f"Schedule immediate maintenance."
    )
elif _pump_lead < 50:
    _pump_bc_class = "pip-broadcast-caution"
    _pump_bc_code  = "&#9888; CAU"
    _pump_bc_title = f"CAUTION — PUMP {selected_engine} // REVIEW RECOMMENDED"
    _pump_bc_msg   = (
        f"Pump <strong>{selected_engine}</strong> has "
        f"<strong>{_pump_lead}</strong> cycles of estimated useful life remaining. "
        f"Unit is approaching the service threshold (&lt;50 cycles). "
        f"Schedule maintenance at next available window."
    )
else:
    _pump_bc_class = "pip-broadcast-nominal"
    _pump_bc_code  = "&#9679; NOM"
    _pump_bc_title = f"NOMINAL — PUMP {selected_engine} // OPERATING WITHIN PARAMETERS"
    _pump_bc_msg   = (
        f"Pump <strong>{selected_engine}</strong> has "
        f"<strong>{_pump_lead}</strong> cycles of estimated useful life remaining. "
        f"No anomalies detected. Continue standard monitoring interval."
    )

st.markdown(f"""
<div class="pip-broadcast {_pump_bc_class}">
    <div class="pip-broadcast-code">{_pump_bc_code}</div>
    <div class="pip-broadcast-body">
        <span class="pip-broadcast-title">{_pump_bc_title}</span>
        <span class="pip-broadcast-msg">{_pump_bc_msg}</span>
    </div>
    <div class="pip-broadcast-meta">PUMP ID: {selected_engine}<br>RUL EST.<br>LSTM MODEL</div>
</div>
""", unsafe_allow_html=True)

# ── Phase 4: Pip-Boy Matplotlib theme ───────────────────
fig2, ax2 = plt.subplots(figsize=(12, 4))

# ── Figure & axes backgrounds
fig2.patch.set_facecolor(_BG)
ax2.set_facecolor(_BG)

# ── Sensor trace — main phosphor green line
ax2.plot(
    engine_data["cycle"],
    engine_data[sensor_to_plot],
    color=_GREEN,
    linewidth=1.4,
    alpha=0.92,
    zorder=3,
    solid_capstyle="round"
)

# ── Subtle fill under the line for depth
ax2.fill_between(
    engine_data["cycle"],
    engine_data[sensor_to_plot],
    alpha=0.06,
    color=_GREEN,
    zorder=2
)

# ── Grid
ax2.grid(
    True,
    color=_GREEN_DM,
    linestyle="--",
    linewidth=0.5,
    alpha=0.45,
    zorder=0
)
ax2.set_axisbelow(True)

# ── Spines
for spine in ax2.spines.values():
    spine.set_edgecolor(_GREEN_BD)
    spine.set_linewidth(0.8)

# ── Axis labels
ax2.set_xlabel(
    "// OPERATIONAL CYCLE",
    color=_TEXT_DM,
    fontfamily=_FONT,
    fontsize=8,
    labelpad=8
)
ax2.set_ylabel(
    f"SENSOR: {sensor_to_plot.upper()}",
    color=_TEXT_DM,
    fontfamily=_FONT,
    fontsize=8,
    labelpad=8
)

# ── Tick params
ax2.tick_params(
    colors=_TEXT_DM,
    labelsize=7,
    length=3,
    width=0.6
)
for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
    lbl.set_fontfamily(_FONT)

# ── Title
ax2.set_title(
    f"[ PUMP {selected_engine} — SENSOR TREND ANALYSIS ]",
    color=_GREEN,
    fontfamily=_FONT,
    fontsize=9,
    fontweight="bold",
    pad=10,
    loc="left"
)

fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)

# =========================
# PHASE 9: FOOTER
# =========================

st.markdown(f"""
<div class="pip-footer">
    <div class="pip-footer-left">
        &#9762; &nbsp;NACP-7 &nbsp;&#124;&nbsp; Nuclear Auxiliary Cooling Pump Monitoring System<br>
        Vault-Tec Industrial Suite &nbsp;&#124;&nbsp; Rev 4.7.1 &nbsp;&#124;&nbsp; Unit 7 &nbsp;&#124;&nbsp; Primary Loop &nbsp;&#124;&nbsp; Sector D<br>
        Model: LSTM Autoencoder &nbsp;&#124;&nbsp; Dataset: NASA CMAPSS FD001 &nbsp;&#124;&nbsp; Seq Length: {seq_length} Cycles
    </div>
    <div class="pip-footer-right">
        <span>&#9679; SYSTEM ONLINE</span><br>
        {len(selected_sensors)} Sensor Channels &nbsp;&#124;&nbsp; Per-Engine Adaptive Thresholds<br>
        Simulation Mode &nbsp;&#124;&nbsp; Not For Operational Use
    </div>
</div>
""", unsafe_allow_html=True)