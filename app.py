import streamlit as st
import streamlit.components.v1 as components
import importlib.util
import sys

# Page Config
st.set_page_config(
    page_title="ARPaD++ | Nebula Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Styles
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("styles/main.css")

# GLOBAL BACKGROUND COMPONENT
# This embeds the Three.js starfield. 
# The CSS in styles/main.css positions this iframe as fixed/background.
# GLOBAL BACKGROUND COMPONENT
# We use st.markdown to inject the Three.js script directly into the DOM.
# This avoids iframe sandbox issues and allows proper full-screen rendering.
try:
    with open("assets/three_scene.html", "r") as f:
        three_js_html = f.read()
    # Inject as a fixed div at the bottom of the DOM
    st.markdown(f"""
    <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; pointer-events: none;">
        {three_js_html}
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    pass

# Session State for Navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def nav_to(page_name):
    st.session_state.page = page_name

# Custom Navbar (Floating Pill Design)
st.markdown('<div class="navbar-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Centered Nav Pill
    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("Home", use_container_width=True, type="primary" if st.session_state.page == 'Home' else "secondary"):
            nav_to("Home")
    with nav_cols[1]:
        if st.button("LLM Detector", use_container_width=True, type="primary" if st.session_state.page == 'LLM Detector' else "secondary"):
            nav_to("LLM Detector")
    with nav_cols[2]:
        if st.button("About", use_container_width=True, type="primary" if st.session_state.page == 'About' else "secondary"):
            nav_to("About")
st.markdown('</div>', unsafe_allow_html=True)

# Spacing
# st.markdown("<br>", unsafe_allow_html=True)

# Router
if st.session_state.page == 'Home':
    spec = importlib.util.spec_from_file_location("home", "pages/home.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["home"] = module
    spec.loader.exec_module(module)
    module.app()

elif st.session_state.page == 'LLM Detector':
    spec = importlib.util.spec_from_file_location("llm_detector", "pages/llm_detector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["llm_detector"] = module
    spec.loader.exec_module(module)
    module.app()

elif st.session_state.page == 'About':
    spec = importlib.util.spec_from_file_location("about", "pages/about.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["about"] = module
    spec.loader.exec_module(module)
    module.app()
