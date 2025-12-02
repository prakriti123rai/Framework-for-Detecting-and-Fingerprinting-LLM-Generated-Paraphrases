import streamlit as st
import streamlit.components.v1 as components

def app():
    # Hero Section
    # 3D Background is now global in app.py
    
    st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0; position: relative; z-index: 2;">
    <h1 style="font-size: 5rem; margin-bottom: 0.5rem; text-shadow: 0 0 50px rgba(0, 242, 255, 0.5);">ARPaD++</h1>
    <p style="font-size: 1.5rem; color: #8b9bb4; letter-spacing: 0.1em;">Discriminating Human Creativity from Algorithmic Generation</p>
    <p style="font-size: 1.2rem; color: #aaa; max-width: 600px; margin: 0.5rem auto 1rem auto;">
        Unveiling the invisible fingerprints of artificial intelligence with <br>
        <span style="color: #00f2ff;">Multi-LLM Paraphrase Detection</span>.
    </p>
</div>
""", unsafe_allow_html=True)

    # Navigation Button (Native Streamlit for functionality)
    b1, b2, b3 = st.columns([1, 1, 1])
    with b2:
        if st.button("Start Analysis", type="primary", use_container_width=True):
            st.session_state.page = 'LLM Detector'
            st.rerun()

    # Feature Cards
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
<div class="glass-panel animate-float" style="animation-delay: 0s;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">🧬</div>
    <h3 style="color: #00f2ff;">Fingerprint Explorer</h3>
    <p style="color: #8b9bb4; line-height: 1.6;">
        Identify the unique stylistic signatures of specific LLM architectures (GPT-4, Claude, Llama) with high-precision heatmap analysis.
    </p>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="glass-panel animate-float" style="animation-delay: 1s;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">↔️</div>
    <h3 style="color: #bc13fe;">Semantic Shift</h3>
    <p style="color: #8b9bb4; line-height: 1.6;">
        Visualize how meaning drifts during paraphrasing. Our advanced vector analysis detects subtle semantic distortions.
    </p>
</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="glass-panel animate-float" style="animation-delay: 2s;">
    <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
    <h3 style="color: #00f2ff;">Pattern Analysis</h3>
    <p style="color: #8b9bb4; line-height: 1.6;">
        Deep n-gram analysis reveals the statistical improbabilities that characterize machine-generated text.
    </p>
</div>
""", unsafe_allow_html=True)
