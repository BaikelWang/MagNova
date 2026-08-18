"""Theming and presentational components for the MagNova Streamlit demo.

Frontend only. No backend logic lives here.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = Path(os.getenv("MAGNOVA_REPORTS_DIR", PROJECT_ROOT / "reports"))
STREAMING_REPORT = REPORTS_DIR / "inference_baseline_v2_streaming.json"
SCHEMA_REPORT = REPORTS_DIR / "v61_plain500_slot_verified_schema_dev_v4.json"
RAG_DEV_REPORT = REPORTS_DIR / "v2_production_rag_answer_dev_recheck_20260812.json"

PAGES = ("对话", "候选", "溯源", "指标", "系统")

NAV_ITEMS = (
    ("对话", "💬"),
    ("候选", "🧪"),
    ("溯源", "🧾"),
    ("指标", "📊"),
    ("系统", "⚙️"),
)

WORKFLOW_STEPS = (
    ("讨论", "科研问答"),
    ("证据", "论文 RAG"),
    ("需求", "MaterialRequest"),
    ("确认", "人工闸门"),
    ("候选", "生成筛选"),
)

PIPELINE_STEPS = (
    "生成晶体结构",
    "结构合法性检查",
    "MatterSim 预弛豫",
    "SACGT 性质预测",
    "可信门控与排序",
)

EXAMPLES = (
    ("论文事实", "单层 CrI3 的居里温度是多少，与块体相比如何？"),
    ("方法原理", "MatterGen 如何生成材料？"),
    ("多轮追问", "那它如何处理周期边界？"),
    ("知识边界", "唐代诗人李白的生平是什么？"),
    ("材料设计", "请设计带隙大于 1.5 eV 且不含稀土元素的材料"),
    ("需求澄清", "请帮我设计高磁矩密度材料"),
)

DEMO_SCRIPT: tuple[tuple[str, str], ...] = (
    ("prompt", "单层 CrI3 的居里温度是多少，与块体相比如何？"),
    ("prompt", "MatterGen 如何生成材料？"),
    ("prompt", "那它如何处理周期边界？"),
    ("prompt", "请设计带隙大于 1.5 eV 且不含稀土元素的材料"),
    ("confirm", "approve"),
)

_DARK = {
    "bg": "#0a1512",
    "sidebar": "#081310",
    "surface": "#0f201c",
    "surface-2": "#132923",
    "border": "rgba(94,210,184,.16)",
    "ink": "#e6f5f0",
    "muted": "#8aa9a0",
    "accent": "#3fbf9f",
    "accent-soft": "rgba(63,191,159,.14)",
    "rail": "rgba(148,163,184,.26)",
    "user-bubble": "rgba(63,191,159,.10)",
}

_LIGHT = {
    "bg": "#f7faf9",
    "sidebar": "#ffffff",
    "surface": "#ffffff",
    "surface-2": "#f2f7f5",
    "border": "rgba(19,93,78,.14)",
    "ink": "#122a25",
    "muted": "#638077",
    "accent": "#0c8068",
    "accent-soft": "rgba(12,128,104,.10)",
    "rail": "rgba(100,116,139,.32)",
    "user-bubble": "rgba(12,128,104,.07)",
}

_BASE_CSS = """
html{font-size:17px}
.block-container{max-width:900px;padding-top:2rem;padding-bottom:7rem}
.stApp{background:var(--bg)}
.stApp,.stApp p,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--ink)}
.stApp p,.stApp li{font-size:1rem;line-height:1.7}
[data-testid="stCaptionContainer"] p{font-size:.875rem;color:var(--muted)}
[data-testid="stSidebar"]{background:var(--sidebar);border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:var(--ink)}
[data-testid="stSidebarCollapsedControl"] svg{color:var(--ink)}

div[data-testid="stButton"]>button{border-radius:10px;min-height:2.5rem;font-size:.9375rem;font-weight:600}
div[data-testid="stMetric"]{background:var(--surface-2);border:1px solid var(--border);border-radius:16px;padding:.75rem 1rem}
div[data-testid="stMetric"] label p{color:var(--muted)!important;font-size:.8125rem}
div[data-testid="stExpander"]{border:1px solid var(--border)!important;border-radius:16px!important;background:var(--surface)!important}
div[data-testid="stExpander"] summary p{font-size:.9375rem;font-weight:600;color:var(--ink)}
div[data-testid="stChatMessage"]{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:.5rem 1rem;margin-bottom:.875rem}
div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:var(--user-bubble);border-color:var(--accent-soft)}
[data-testid="stChatInput"]{border-radius:12px}
[data-testid="stToolbar"],[data-testid="stToolbarActions"],.stAppDeployButton{display:none!important}
header[data-testid="stHeader"]{background:transparent}
.mn-funnel-note{font-size:.875rem;line-height:1.65;color:var(--muted);background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:.875rem 1rem;margin:.75rem 0 1rem}

/* sidebar brand + navigation */
.mn-brand{padding:.25rem 0 1rem}
.mn-brand-name{font-size:1.25rem;font-weight:700;letter-spacing:-.01em;line-height:1.2}
.mn-brand-sub{font-size:.8125rem;color:var(--muted);margin-top:.125rem}
.mn-nav-label{font-size:.6875rem;font-weight:700;letter-spacing:.12em;color:var(--muted);margin:.25rem 0 .5rem}
[data-testid="stSidebar"] div[data-testid="stButton"]>button{
  justify-content:flex-start;text-align:left;min-height:2.75rem;padding:.5rem .875rem;
  font-size:1rem;font-weight:600;border:1px solid transparent;background:transparent;transition:background .12s ease
}
[class*="st-key-mn-nav-off-"] div[data-testid="stButton"]>button:hover{background:var(--surface-2)}
[class*="st-key-mn-nav-on-"] div[data-testid="stButton"]>button{background:var(--accent);border-color:var(--accent)}
[class*="st-key-mn-nav-on-"] div[data-testid="stButton"]>button p{color:#ffffff!important}
[class*="st-key-mn-nav-on-"] div[data-testid="stButton"]>button:hover{background:var(--accent)}
[class*="st-key-mn-new-"] div[data-testid="stButton"]>button{
  justify-content:center;background:var(--surface-2);border:1px solid var(--border);margin-bottom:.5rem
}
[class*="st-key-mn-new-"] div[data-testid="stButton"]>button:hover{border-color:var(--accent)}
[class*="st-key-mn-conv-"] div[data-testid="stButton"]>button{
  min-height:2.375rem;font-size:.875rem;font-weight:500;border-left:2px solid transparent;border-radius:8px
}
[class*="st-key-mn-conv-"] div[data-testid="stButton"]>button p{
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--muted)!important
}
[class*="st-key-mn-conv-off-"] div[data-testid="stButton"]>button:hover{background:var(--surface-2)}
[class*="st-key-mn-conv-on-"] div[data-testid="stButton"]>button{
  background:var(--accent-soft);border-left-color:var(--accent)
}
[class*="st-key-mn-conv-on-"] div[data-testid="stButton"]>button p{color:var(--ink)!important;font-weight:600}

.mn-head{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:0 0 1.75rem}
.mn-head-eyebrow{color:var(--accent);font-size:.75rem;font-weight:700;letter-spacing:.14em;margin-bottom:.25rem}
.mn-head-title{font-size:1.625rem;font-weight:700;letter-spacing:-.02em;line-height:1.3;margin:0}
.mn-badge{padding:.4375rem .875rem;border-radius:999px;font-size:.75rem;font-weight:700;letter-spacing:.04em;white-space:nowrap;flex-shrink:0}
.mn-badge-mock{border:1px dashed var(--rail);color:var(--muted)}
.mn-badge-real{border:1px solid var(--accent);color:var(--accent);background:var(--accent-soft)}

.mn-steps{display:flex;margin:0 0 2rem}
.mn-step{flex:1;min-width:0;text-align:center}
.mn-step-head{display:flex;align-items:center;height:28px}
.mn-step-rail{flex:1;height:2px;background:var(--rail)}
.mn-step-rail.on{background:var(--accent)}
.mn-step-rail.off{visibility:hidden}
.mn-step-dot{
  width:28px;height:28px;border-radius:999px;border:2px solid var(--rail);background:var(--surface);
  flex-shrink:0;box-sizing:border-box;display:grid;place-items:center;
  font-size:.8125rem;font-weight:700;color:var(--muted);line-height:1
}
.mn-step.done .mn-step-dot,.mn-step.now .mn-step-dot{background:var(--accent);border-color:var(--accent);color:#ffffff}
.mn-step.now .mn-step-dot{box-shadow:0 0 0 5px var(--accent-soft)}
.mn-step-name{font-size:.9375rem;font-weight:600;color:var(--muted);margin-top:.625rem;line-height:1.25}
.mn-step.done .mn-step-name,.mn-step.now .mn-step-name{color:var(--ink)}
.mn-step-sub{font-size:.8125rem;color:var(--muted);margin-top:.1875rem;line-height:1.3}

.mn-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.5rem}
.mn-card-title{font-size:1.125rem;font-weight:700;margin:0 0 .5rem}
.mn-card-note{font-size:1rem;color:var(--muted);line-height:1.7;margin:0}
.mn-flow-label{font-size:.75rem;font-weight:700;letter-spacing:.1em;color:var(--muted);margin:1.5rem 0 .75rem}
.mn-flow{display:flex;align-items:stretch;flex-wrap:wrap;gap:.375rem}
.mn-flow-node{flex:1;min-width:96px;background:var(--surface-2);border-radius:10px;padding:.75rem .5rem;text-align:center;font-size:.8125rem;color:var(--muted);line-height:1.45}
.mn-flow-node b{display:block;color:var(--ink);font-size:.875rem;font-weight:600;margin-bottom:.125rem}
.mn-flow-arrow{display:grid;place-items:center;color:var(--rail);font-size:1rem;flex-shrink:0}
.mn-scoregrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.625rem;margin:1.25rem 0 .25rem}
.mn-scorecell{background:var(--surface-2);border:1px solid var(--border);border-radius:12px;padding:.875rem .625rem;text-align:center}
.mn-scoreval{font-size:1.375rem;font-weight:700;color:var(--accent);line-height:1.2;letter-spacing:-.02em}
.mn-scorelab{font-size:.75rem;color:var(--muted);margin-top:.375rem;line-height:1.35}
.mn-scorenote{font-size:.75rem;color:var(--muted);margin:.5rem 0 0;line-height:1.45}
.mn-hero-3d{margin:0 0 1rem}
.mn-hero-3d-title{font-size:1.0625rem;font-weight:700;margin:0 0 .25rem}
.mn-hero-3d-sub{font-size:.8125rem;color:var(--muted);margin:0 0 .75rem}

.mn-ev{border:1px solid var(--border);border-radius:12px;padding:1rem 1.125rem;margin-bottom:.75rem;background:var(--surface-2)}
.mn-ev-top{display:flex;justify-content:space-between;align-items:baseline;gap:.75rem;margin-bottom:.5rem}
.mn-ev-src{font-size:.9375rem;font-weight:600;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mn-ev-meta{font-size:.8125rem;color:var(--muted);white-space:nowrap;flex-shrink:0}
.mn-ev-bar{height:4px;border-radius:999px;background:var(--rail);overflow:hidden;margin-bottom:.75rem}
.mn-ev-fill{height:100%;background:var(--accent)}
.mn-ev-text{font-size:.9375rem;line-height:1.7;color:var(--muted)}

.mn-cand{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.25rem;text-align:center}
.mn-cand-formula{font-size:1.5rem;font-weight:700;letter-spacing:.01em;margin:0 0 1rem;color:var(--ink)}
.mn-ring{width:72px;height:72px;border-radius:999px;margin:0 auto 1rem;position:relative;background:conic-gradient(var(--accent) calc(var(--pct)*1%),var(--rail) 0)}
.mn-ring::after{content:attr(data-label);position:absolute;inset:8px;border-radius:999px;background:var(--surface);display:grid;place-items:center;font-size:.875rem;font-weight:700;color:var(--ink)}
.mn-cand-prop{font-size:.875rem;color:var(--muted);line-height:1.65;min-height:2.75rem}
.mn-tag{display:inline-block;padding:.1875rem .5625rem;border-radius:999px;font-size:.75rem;font-weight:600;margin:.3125rem .1875rem 0}
.mn-tag-ok{background:var(--accent-soft);color:var(--accent)}
.mn-tag-warn{background:rgba(217,119,6,.15);color:#b45309}
.mn-tag-risk{background:rgba(225,29,72,.15);color:#be123c}
.mn-cand-id{font-size:.75rem;color:var(--muted);margin-top:.75rem}

.mn-gate{border:1px solid var(--accent);border-radius:16px;background:var(--accent-soft);padding:1.25rem 1.375rem;margin:.5rem 0 1rem}
.mn-gate-title{font-size:1.0625rem;font-weight:700;margin:0 0 .3125rem}
.mn-gate-note{font-size:.9375rem;color:var(--muted);margin:0;line-height:1.65}
.mn-mode-mock{border:1px dashed var(--rail);border-radius:12px;padding:1.125rem;font-size:.9375rem;line-height:1.7;color:var(--muted)}
.mn-mode-real{border:1px solid #e11d48;border-radius:12px;padding:1.125rem;font-size:.9375rem;line-height:1.7;background:rgba(225,29,72,.08)}
.mn-mode-mock b,.mn-mode-real b{display:block;font-size:1.0625rem;color:var(--ink);margin-bottom:.375rem}

.mn-foot{font-size:.8125rem;line-height:1.65;color:var(--muted);border-top:1px solid var(--border);margin-top:2.5rem;padding-top:1rem}

.mn-table-wrap{border:1px solid var(--border);border-radius:12px;overflow:hidden;margin:.5rem 0 1rem;background:var(--surface)}
.mn-table{width:100%;border-collapse:collapse;font-size:.9375rem}
.mn-table th,.mn-table td{padding:.625rem .875rem;text-align:left;border-bottom:1px solid var(--border)}
.mn-table th{background:var(--surface-2);color:var(--muted);font-weight:600;font-size:.8125rem}
.mn-table td{background:var(--surface);color:var(--ink)}
.mn-table tbody tr:last-child td{border-bottom:none}
.mn-table tbody tr:hover td{background:var(--surface-2)}
.mn-table td.mn-td-bar{min-width:150px;white-space:nowrap}
.mn-cellbar{display:inline-block;vertical-align:middle;width:84px;height:6px;border-radius:999px;background:var(--rail);overflow:hidden;margin-right:.5rem}
.mn-cellbar-fill{height:100%;background:var(--accent)}
.mn-cellnum{font-variant-numeric:tabular-nums;font-size:.875rem;color:var(--muted)}

@media (max-width:720px){
  .mn-scoregrid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .mn-step-sub{display:none}
  .mn-head-title{font-size:1.25rem}
  .mn-flow-node{min-width:72px;font-size:.75rem}
}
"""

# Streamlit 1.61 uses [data-testid="stMain"], not section.main — scope under .stApp.
_DARK_WIDGET_CSS = """
.stApp [data-testid="stBottom"],.stApp [data-testid="stBottomBlockContainer"],
.stApp [data-testid="stBottomBlockContainer"]>div,.stApp [data-testid="stBottom"]>div{
  background:var(--bg)!important;box-shadow:none!important
}
.stApp [data-testid="stVerticalBlockBorderWrapper"]{
  background:var(--surface)!important;border-color:var(--border)!important
}
.stApp div[data-testid="stButton"]>button,
.stApp div[data-testid="stButtonGroup"] button,
.stApp div[data-baseweb="button-group"] button,
.stApp button[data-testid="stBaseButton-secondary"],
.stApp button[data-testid="stBaseButton-primary"]{
  background:var(--surface-2)!important;color:var(--ink)!important;border:1px solid var(--border)!important
}
.stApp div[data-testid="stButton"]>button p,
.stApp div[data-testid="stButtonGroup"] button p,
.stApp div[data-baseweb="button-group"] button p,
.stApp button[data-testid="stBaseButton-secondary"] p,
.stApp button[data-testid="stBaseButton-primary"] p{color:var(--ink)!important}
.stApp div[data-testid="stButtonGroup"] button[aria-pressed="true"],
.stApp div[data-baseweb="button-group"] button[aria-pressed="true"]{
  background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important
}
.stApp div[data-testid="stButtonGroup"] button[aria-pressed="true"] p,
.stApp div[data-baseweb="button-group"] button[aria-pressed="true"] p{color:#fff!important}
.stApp div[data-testid="stButtonGroup"] button:hover:not([aria-pressed="true"]),
.stApp div[data-baseweb="button-group"] button:hover:not([aria-pressed="true"]){
  background:var(--surface)!important;border-color:var(--accent)!important;color:var(--ink)!important
}
.stApp div[data-testid="stSegmentedControl"] button,
.stApp [data-testid="stSegmentedControl"] button{
  background:var(--surface-2)!important;color:var(--muted)!important;border-color:var(--border)!important
}
.stApp div[data-testid="stSegmentedControl"] button[aria-selected="true"],
.stApp [data-testid="stSegmentedControl"] button[aria-selected="true"]{
  background:var(--accent)!important;color:#fff!important;border-color:var(--accent)!important
}
.stApp button[data-testid="stBaseButton-primary"]{
  background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important
}
.stApp button[data-testid="stBaseButton-primary"] p{color:#fff!important}
.stApp [data-testid="stMetric"]{background:var(--surface-2)!important;border-color:var(--border)!important}
.stApp [data-testid="stMetricLabel"] p{color:var(--muted)!important}
.stApp [data-testid="stMetricValue"]{color:var(--ink)!important}
.stApp [data-testid="stMetricDelta"]{color:var(--muted)!important}
.stApp pre,.stApp code,.stApp [data-testid="stCode"]>pre{
  background:var(--surface-2)!important;color:var(--ink)!important;border:1px solid var(--border)!important
}
.stApp .react-json-view,.stApp [class*="Json"]{
  background:var(--surface-2)!important;color:var(--ink)!important
}
/* Keep error/warning/info visually distinct instead of flattening them to --ink. */
.stApp [data-testid="stAlertContainer"]{
  background:var(--surface-2)!important;color:var(--ink)!important;
  border:1px solid var(--border)!important;border-radius:12px!important
}
.stApp [data-testid="stAlertContainer"] p,
.stApp [data-testid="stAlertContainer"] li{color:var(--ink)!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){
  background:rgba(248,113,113,.13)!important;border-color:rgba(248,113,113,.45)!important
}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){
  background:rgba(251,191,36,.13)!important;border-color:rgba(251,191,36,.45)!important
}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]){
  background:rgba(125,211,252,.11)!important;border-color:rgba(125,211,252,.38)!important
}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){
  background:var(--accent-soft)!important;border-color:var(--accent)!important
}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) [data-testid="stAlertDynamicIcon"] svg{color:#f87171!important;fill:#f87171!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertDynamicIcon"] svg{color:#fbbf24!important;fill:#fbbf24!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertDynamicIcon"] svg{color:#7dd3fc!important;fill:#7dd3fc!important}
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertDynamicIcon"] svg{color:var(--accent)!important;fill:var(--accent)!important}
.stApp [data-testid="stDataFrame"],.stApp [data-testid="stDataFrame"]>div{
  background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:12px!important
}
.stApp [data-testid="stDataFrame"] [role="gridcell"],
.stApp [data-testid="stDataFrame"] [role="columnheader"]{
  background:var(--surface-2)!important;color:var(--ink)!important
}
.stApp [data-testid="stChatInput"],
.stApp [data-testid="stChatInput"]>div,
.stApp [data-testid="stChatInput"]>div>div,
.stApp [data-testid="stChatInput"] [data-testid="stChatInputTextArea"],
.stApp [data-testid="stChatInput"] [data-baseweb="textarea"]{
  background:var(--surface)!important;border-color:var(--border)!important;box-shadow:none!important
}
.stApp [data-testid="stChatInput"] textarea,.stApp [data-testid="stChatInput"] textarea:focus{
  color:var(--ink)!important;background:transparent!important;caret-color:var(--accent)
}
.stApp [data-testid="stChatInput"] textarea::placeholder{color:var(--muted)!important;opacity:1}
.stApp [data-testid="stSidebar"] div[data-testid="stButton"]>button p{color:var(--ink)!important}
.stApp [class*="st-key-mn-new-"] div[data-testid="stButton"]>button{
  background:var(--surface-2)!important;border:1px solid var(--border)!important
}
.stApp [class*="st-key-mn-new-"] div[data-testid="stButton"]>button p{color:var(--ink)!important}
.stApp [class*="st-key-mn-nav-on-"] div[data-testid="stButton"]>button p{color:#ffffff!important}
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] label span,
.stApp [data-testid="stSidebar"] [data-testid="stToggle"] label p{color:var(--ink)!important}
.stApp div[data-baseweb="select"]>div{background:var(--surface)!important;border-color:var(--border)!important}
.stApp div[data-baseweb="select"] span{color:var(--ink)!important}
.stApp [data-testid="stMarkdownContainer"] code{
  background:var(--surface-2)!important;color:var(--accent)!important
}
.stApp hr{border-color:var(--border)!important}
.stApp [data-testid="stIFrame"],.stApp iframe{background:var(--surface-2)!important;border:1px solid var(--border)!important;border-radius:12px!important}
.stApp [data-testid="stVegaLiteChart"],.stApp .vega-embed{background:var(--surface-2)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:.5rem!important}
.stApp .react-json-view .object-key,.stApp .react-json-view .string-value{color:var(--ink)!important}
.stApp .react-json-view .number-value,.stApp .react-json-view .boolean-value{color:var(--accent)!important}
.stApp .react-json-view .object-key-val,.stApp .react-json-view .variable-row{border-color:var(--border)!important}
.stApp div[data-testid="stExpander"] summary span{color:var(--ink)!important}
.stApp div[data-testid="stExpander"] summary,
.stApp div[data-testid="stExpander"] details[open]>summary{
  background:var(--surface-2)!important;color:var(--ink)!important
}
.stApp div[data-testid="stExpander"] summary p,
.stApp div[data-testid="stExpander"] summary span,
.stApp div[data-testid="stExpander"] summary svg,
.stApp div[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"]{
  color:var(--ink)!important;fill:var(--ink)!important
}
.stApp div[data-testid="stExpander"] [data-testid="stExpanderDetails"]{
  background:var(--surface)!important;border-top:1px solid var(--border)!important
}
.stApp [data-testid="stSelectbox"] label p,.stApp [data-testid="stSelectbox"] span{color:var(--ink)!important}
"""


def esc(value: Any) -> str:
    """Paper text and CIF-derived fields may contain <, > or &."""
    return html.escape(str(value), quote=True)


def is_dark_mode() -> bool:
    return bool(
        st.session_state.get("mn_pres_mode")
        or st.session_state.get("mn_dark_mode")
    )


def theme_css(dark: bool) -> str:
    palette = _DARK if dark else _LIGHT
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in palette.items()) + "}"
    if dark:
        return root + _BASE_CSS + _DARK_WIDGET_CSS
    return root + _BASE_CSS


def inject_css(dark: bool) -> None:
    css = theme_css(dark)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_css_footer(dark: bool) -> None:
    """Re-inject at page bottom so Streamlit widget styles do not win."""
    if not dark:
        return
    st.markdown(
        f"<style id='mn-dark-footer'>{_DARK_WIDGET_CSS}</style>",
        unsafe_allow_html=True,
    )


def render_header(*, backend_mode: str) -> None:
    if backend_mode == "mock":
        badge = '<span class="mn-badge mn-badge-mock">MOCK · 演示候选</span>'
    else:
        badge = '<span class="mn-badge mn-badge-real">REAL · SACGT 闭环</span>'
    st.markdown(
        f"""
<div class="mn-head">
  <div>
    <div class="mn-head-eyebrow">MAGNOVA</div>
    <div class="mn-head-title">从科研讨论，到可执行的材料需求</div>
  </div>
  {badge}
</div>
""",
        unsafe_allow_html=True,
    )


def render_progress(active: int) -> None:
    """Read-only stage indicator: done steps show a check, upcoming ones their number."""
    last = len(WORKFLOW_STEPS) - 1
    blocks = []
    for index, (name, sub) in enumerate(WORKFLOW_STEPS):
        if index < active:
            state, marker = "done", "✓"
        elif index == active:
            state, marker = "now", str(index + 1)
        else:
            state, marker = "", str(index + 1)
        left = "off" if index == 0 else ("on" if index <= active else "")
        right = "off" if index == last else ("on" if index < active else "")
        blocks.append(
            f'<div class="mn-step {state}">'
            f'<div class="mn-step-head">'
            f'<div class="mn-step-rail {left}"></div>'
            f'<div class="mn-step-dot">{marker}</div>'
            f'<div class="mn-step-rail {right}"></div>'
            f"</div>"
            f'<div class="mn-step-name">{name}</div>'
            f'<div class="mn-step-sub">{sub}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="mn-steps">{"".join(blocks)}</div>', unsafe_allow_html=True)


_FLOW_NODES = (
    ("BGE + RAG", "论文证据"),
    ("MiniMind", "需求解析"),
    ("LangGraph", "流程编排"),
    ("MatterGen", "结构生成"),
    ("SACGT", "性质预测"),
)


def _welcome_scorecard_html(bundle: dict[str, Any]) -> str:
    streaming = bundle.get("streaming") or {}
    schema = bundle.get("schema") or {}
    rag = bundle.get("rag") or {}
    levels = streaming.get("levels") or []
    first = levels[0] if levels else {}
    summary = schema.get("summary") or {}
    rag_metrics = rag.get("metrics") or rag.get("summary") or {}

    ttft = (first.get("ttft_seconds") or {}).get("mean")
    contract = summary.get("contract_exact_rate")
    refusal = rag_metrics.get("no_answer_refusal_accuracy")
    forbidden = rag_metrics.get("forbidden_claim_rate")

    cells = [
        (f"{ttft:.3f}s" if ttft is not None else "—", "首字响应 TTFT"),
        (
            f"{float(contract) * 100:.1f}%" if contract is not None else "—",
            "契约精确率",
        ),
        (
            f"{float(refusal) * 100:.0f}%" if refusal is not None else "—",
            "无证据拒答",
        ),
        (
            f"{float(forbidden) * 100:.1f}%" if forbidden is not None else "—",
            "违规断言率",
        ),
    ]
    grid = "".join(
        f'<div class="mn-scorecell"><div class="mn-scoreval">{val}</div>'
        f'<div class="mn-scorelab">{lab}</div></div>'
        for val, lab in cells
    )
    return (
        f'<div class="mn-scoregrid">{grid}</div>'
        f'<p class="mn-scorenote">实测指标来自 reports/ · 开发/评测集 · 答辩时可切「指标」页看明细</p>'
    )


def render_welcome() -> None:
    bundle = load_metrics()
    scorecard = _welcome_scorecard_html(bundle) if bundle else ""
    nodes = '<span class="mn-flow-arrow">→</span>'.join(
        f'<span class="mn-flow-node"><b>{name}</b>{role}</span>' for name, role in _FLOW_NODES
    )
    st.markdown(
        f"""
<div class="mn-card">
  <div class="mn-card-title">MagNova 科研智能体</div>
  <p class="mn-card-note">从论文讨论到结构化材料需求，经人工确认后进入生成-预测-筛选闭环。点下方示例，或直接输入问题。</p>
  {scorecard}
  <div class="mn-flow-label">系统组成（示意）</div>
  <div class="mn-flow">{nodes}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        return
    with st.expander(f"论文证据 · {len(evidence)} 条", expanded=True):
        for index, item in enumerate(evidence, start=1):
            score = float(item.get("score") or 0)
            pct = max(0.0, min(100.0, score * 100))
            page = f" · p.{item['page']}" if item.get("page") else ""
            text = re.sub(r"\s+", " ", str(item.get("content", ""))).strip()
            preview = text[:170] + ("…" if len(text) > 170 else "")
            st.markdown(
                f"""
<div class="mn-ev">
  <div class="mn-ev-top">
    <div class="mn-ev-src">{index}. {esc(item.get("source", "unknown"))}</div>
    <div class="mn-ev-meta">{esc(page.lstrip(" ·")) or "—"} · 相关度 {score:.0%}</div>
  </div>
  <div class="mn-ev-bar"><div class="mn-ev-fill" style="width:{pct:.1f}%"></div></div>
  <div class="mn-ev-text">{esc(preview)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if len(text) > 170:
                with st.expander("全文", expanded=False):
                    st.write(text)


def _tags(item: dict[str, Any], backend_mode: str) -> str:
    tags: list[tuple[str, str]] = []
    score = float(item.get("score") or 0)
    if backend_mode == "mock":
        tags.append(("Mock", "warn"))
    if "探索" in str(item.get("rationale") or "") or "exploration" in str(
        item.get("rationale") or ""
    ).lower():
        tags.append(("探索队列", "warn"))
    if score >= 0.85:
        tags.append(("高置信", "ok"))
    elif score < 0.5:
        tags.append(("待验证", "risk"))
    if not tags:
        tags.append(("AI 筛选", "ok"))
    return "".join(
        f'<span class="mn-tag mn-tag-{kind}">{esc(label)}</span>' for label, kind in tags
    )


def render_candidates(
    candidates: list[dict[str, Any]],
    *,
    backend_mode: str,
    top_n: int = 3,
) -> None:
    if not candidates:
        return
    ranked = sorted(candidates, key=lambda x: float(x.get("score") or 0), reverse=True)
    top, rest = ranked[:top_n], ranked[top_n:]

    for column, item in zip(st.columns(len(top)), top):
        score = float(item.get("score") or 0)
        props = item.get("predicted_properties") or {}
        summary = " · ".join(
            f"{name} {value:.3f}" if isinstance(value, (int, float)) else f"{name} {value}"
            for name, value in list(props.items())[:2]
        ) or "预测性质待补充"
        with column:
            st.markdown(
                f"""
<div class="mn-cand">
  <div class="mn-cand-formula">{esc(item.get("formula", "?"))}</div>
  <div class="mn-ring" style="--pct:{max(0.0, min(100.0, score * 100)):.1f}" data-label="{score:.2f}"></div>
  <div class="mn-cand-prop">{esc(summary)}</div>
  <div>{_tags(item, backend_mode)}</div>
  <div class="mn-cand-id">{esc(item.get("candidate_id", "—"))}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    if rest:
        with st.expander(f"其余 {len(rest)} 个候选"):
            render_data_table(
                pd.DataFrame(
                    [
                        {
                            "候选材料": item.get("formula"),
                            "综合分数": float(item.get("score") or 0),
                            "候选 ID": item.get("candidate_id"),
                        }
                        for item in rest
                    ]
                ),
                bar_columns=("综合分数",),
            )


def render_request_summary(request: dict[str, Any], format_constraint: Callable[[dict], str]) -> None:
    for item in request.get("target_properties", []):
        st.markdown(f"- `{format_constraint(item)}`")
    composition = request.get("composition", {})
    details = []
    if composition.get("required_elements"):
        details.append("必须包含 " + "、".join(composition["required_elements"]))
    if composition.get("excluded_elements"):
        details.append("排除 " + "、".join(composition["excluded_elements"]))
    if composition.get("exclude_rare_earth"):
        details.append("不含稀土元素")
    if request.get("material_family"):
        details.append(f"类别 {request['material_family']}")
    if details:
        st.caption(" · ".join(details))


def render_gate_banner() -> None:
    st.markdown(
        """
<div class="mn-gate">
  <div class="mn-gate-title">需求已结构化，等待人工确认</div>
  <p class="mn-gate-note">这是 Human-in-the-loop 闸门：未经确认，系统不会调用任何材料计算工具。</p>
</div>
""",
        unsafe_allow_html=True,
    )


@st.dialog("人工确认闸门", width="large")
def confirmation_dialog(
    request: dict[str, Any],
    *,
    backend_mode: str,
    backend_label: str,
    format_constraint: Callable[[dict], str],
) -> None:
    st.caption("核对结构化需求与执行模式后再提交。模型不会自动调用材料计算工具。")
    left, right = st.columns(2)
    with left:
        st.markdown("**MaterialRequest**")
        render_request_summary(request, format_constraint)
        with st.expander("完整 JSON"):
            st.json(request)
    with right:
        if backend_mode == "mock":
            st.markdown(
                '<div class="mn-mode-mock"><b>Mock 演示模式</b>'
                "候选来自 Mock Pipeline，仅验证接口与交互流程，不可当作科研预测。</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="mn-mode-real"><b>Real 生产模式</b>'
                f"将调用 {backend_label}。候选来自 SACGT AI 筛选闭环，仍非 DFT / 实验验证。</div>",
                unsafe_allow_html=True,
            )
        st.markdown("**提交后执行**")
        st.caption(" → ".join(PIPELINE_STEPS))

    approve, reject = st.columns(2)
    if approve.button("确认提交", type="primary", width="stretch"):
        st.session_state.mn_gate_decision = "approve"
        st.session_state.mn_gate_open = False
        st.rerun()
    if reject.button("取消任务", width="stretch"):
        st.session_state.mn_gate_decision = "reject"
        st.session_state.mn_gate_open = False
        st.rerun()


def stream_text_chunks(text: str, *, chunk_size: int = 2, delay: float = 0.01) -> Iterator[str]:
    if not text:
        yield ""
        return
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]
        if delay:
            time.sleep(delay)


def run_with_pipeline_status(
    invoke_fn: Callable[[], dict[str, Any]],
    *,
    backend_mode: str,
) -> dict[str, Any]:
    """Run a blocking backend call while ticking through pipeline stage labels."""
    with st.status("MagNova 流水线执行中", expanded=True) as status:
        holder: dict[str, Any] = {}
        failure: dict[str, Exception] = {}

        def worker() -> None:
            try:
                holder["result"] = invoke_fn()
            except Exception as exc:
                failure["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        step = 0
        pace = 0.5 if backend_mode == "mock" else 0.9
        while thread.is_alive() and step < len(PIPELINE_STEPS):
            st.write(f"✓ {PIPELINE_STEPS[step]}")
            step += 1
            time.sleep(pace)
        thread.join()
        while step < len(PIPELINE_STEPS):
            st.write(f"✓ {PIPELINE_STEPS[step]}")
            step += 1

        if "error" in failure:
            status.update(label="执行失败", state="error")
            raise failure["error"]
        status.update(label="流水线完成", state="complete")
        return holder["result"]


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for key, path in (
        ("streaming", STREAMING_REPORT),
        ("schema", SCHEMA_REPORT),
        ("rag", RAG_DEV_REPORT),
    ):
        if path.exists():
            bundle[key] = json.loads(path.read_text(encoding="utf-8"))
    return bundle


def _table_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"
    return esc(value)


def _percent_cell(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "—"


def _bar_cell(value: Any) -> str:
    try:
        ratio = max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return "—"
    return (
        f'<span class="mn-cellbar"><span class="mn-cellbar-fill" '
        f'style="width:{ratio * 100:.1f}%"></span></span>'
        f'<span class="mn-cellnum">{ratio:.2f}</span>'
    )


def render_data_table(
    frame: pd.DataFrame,
    *,
    bar_columns: tuple[str, ...] = (),
    percent_columns: tuple[str, ...] = (),
) -> None:
    """Theme-aware table; avoids Glide Data Grid which ignores custom dark CSS."""
    headers = "".join(f"<th>{esc(column)}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in bar_columns:
                cells.append(f'<td class="mn-td-bar">{_bar_cell(value)}</td>')
            elif column in percent_columns:
                cells.append(f"<td>{_percent_cell(value)}</td>")
            else:
                cells.append(f"<td>{_table_cell(value)}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<div class="mn-table-wrap"><table class="mn-table">'
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def render_metrics_page() -> None:
    bundle = load_metrics()
    if not bundle:
        st.warning("未找到 reports/ 下的指标文件。")
        return

    streaming = bundle.get("streaming") or {}
    schema = bundle.get("schema") or {}
    rag = bundle.get("rag") or {}
    levels = streaming.get("levels") or []
    first = levels[0] if levels else {}
    summary = schema.get("summary") or {}
    rag_metrics = rag.get("metrics") or rag.get("summary") or {}

    ttft = (first.get("ttft_seconds") or {}).get("mean")
    tps = first.get("mean_request_generation_tokens_per_second")
    schema_samples = summary.get("samples")
    rag_cases = rag_metrics.get("num_cases")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TTFT 均值", f"{ttft:.3f}s" if ttft is not None else "—")
    c2.metric("生成吞吐", f"{tps:.0f} tok/s" if tps is not None else "—")
    contract = summary.get("contract_exact_rate")
    c3.metric(
        "契约精确率",
        f"{float(contract) * 100:.1f}%" if contract is not None else "—",
        help=f"MaterialRequest 契约评测 · {schema_samples or '?'} 样本",
    )
    refusal = rag_metrics.get("no_answer_refusal_accuracy")
    c4.metric(
        "无证据拒答",
        f"{float(refusal) * 100:.0f}%" if refusal is not None else "—",
        help=f"RAG 开发集 · {rag_cases or '?'} 例 · 证据不足时不编造",
    )

    st.markdown("**RAG 安全与引用**")
    r1, r2, r3, r4 = st.columns(4)
    decision = rag_metrics.get("answer_decision_accuracy")
    r1.metric(
        "该答/该拒判定",
        f"{float(decision) * 100:.0f}%" if decision is not None else "—",
    )
    condense = rag_metrics.get("query_condensation_accuracy")
    r2.metric(
        "多轮指代改写",
        f"{float(condense) * 100:.0f}%" if condense is not None else "—",
    )
    forbidden = rag_metrics.get("forbidden_claim_rate")
    r3.metric(
        "违规断言率",
        f"{float(forbidden) * 100:.2f}%" if forbidden is not None else "—",
    )
    cite_prec = rag_metrics.get("macro_citation_precision")
    r4.metric(
        "引用精确率",
        f"{float(cite_prec) * 100:.1f}%" if cite_prec is not None else "—",
    )

    grounded = rag_metrics.get("grounded_proxy_pass_rate")
    st.markdown("**严格 grounded 代理（开发集）**")
    st.caption(
        "要求答案中每条事实均能在检索证据中对齐；口径最严，"
        f"与上方「拒答/引用」指标互补。"
        f" 样本 {rag_cases or '?'} 例。"
    )
    if grounded is not None:
        st.metric("grounded_proxy_pass_rate", f"{float(grounded) * 100:.1f}%")
    else:
        st.caption("无 grounded 代理指标。")

    st.markdown("**推理基线（streaming）**")
    if levels:
        render_data_table(
            pd.DataFrame(
                [
                    {
                        "并发": item.get("concurrency"),
                        "成功率": 1 - float(item.get("error_rate") or 0),
                        "TTFT p50 (s)": (item.get("ttft_seconds") or {}).get("p50"),
                        "E2E p50 (s)": (item.get("e2e_seconds") or {}).get("p50"),
                        "吞吐 tok/s": item.get("aggregate_output_tokens_per_second"),
                    }
                    for item in levels
                ]
            ),
            percent_columns=("成功率",),
        )
    else:
        st.caption("无 streaming 级别数据。")

    st.markdown("**MaterialRequest 契约评测**")
    if summary:
        st.json(
            {
                "samples": summary.get("samples"),
                "valid_rate": summary.get("valid_rate"),
                "contract_exact_rate": summary.get("contract_exact_rate"),
                "field_accuracy": summary.get("field_accuracy"),
            }
        )
    else:
        st.caption("无 schema 报告。")

    st.caption(
        f"来源：{STREAMING_REPORT.name} · {SCHEMA_REPORT.name} · {RAG_DEV_REPORT.name}"
    )


def candidate_report_markdown(
    outcome: dict[str, Any],
    *,
    backend_mode: str,
    thread_id: str,
) -> str:
    lines = [
        "# MagNova 候选报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 会话 ID：`{thread_id}`",
        f"- 后端模式：`{backend_mode}`",
        f"- 任务状态：`{outcome.get('status', 'unknown')}`",
        "",
        "## 候选",
        "",
    ]
    for index, item in enumerate(outcome.get("candidates") or [], start=1):
        lines += [
            f"### {index}. {item.get('formula', '?')}",
            f"- 分数：`{item.get('score')}`",
            f"- ID：`{item.get('candidate_id')}`",
            f"- 预测性质：{item.get('predicted_properties') or {}}",
            f"- 说明：{item.get('rationale') or '—'}",
            "",
        ]
    lines += ["---", "声明：候选来自 MagNova 演示系统，需经 DFT / 实验验证。"]
    return "\n".join(lines)


def render_export_button(
    outcome: dict[str, Any] | None,
    *,
    backend_mode: str,
    thread_id: str,
) -> None:
    if not outcome or not outcome.get("candidates"):
        return
    st.download_button(
        "导出 Markdown 报告",
        data=candidate_report_markdown(
            outcome, backend_mode=backend_mode, thread_id=thread_id
        ).encode("utf-8"),
        file_name=f"magnova_candidates_{thread_id[:8]}.md",
        mime="text/markdown",
    )


def next_demo_action() -> tuple[str, str] | None:
    if not st.session_state.get("mn_demo_on"):
        return None
    index = int(st.session_state.get("mn_demo_step", 0))
    if index >= len(DEMO_SCRIPT):
        st.session_state.mn_demo_on = False
        st.session_state.mn_demo_step = 0
        return None
    return DEMO_SCRIPT[index]


def advance_demo() -> None:
    st.session_state.mn_demo_step = int(st.session_state.get("mn_demo_step", 0)) + 1
