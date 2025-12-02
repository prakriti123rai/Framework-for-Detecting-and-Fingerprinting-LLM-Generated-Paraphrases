import streamlit as st

def app():
    st.markdown("""
        <div style="text-align: center; padding: 3rem 0;">
            <h1>About The Project</h1>
            <p style="color: #8b9bb4; max-width: 600px; margin: 0 auto;">
                ARPaD++ represents the convergence of academic research and advanced software engineering.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main description block (full-width glass panel)
    st.markdown(
        """
<div class="glass-panel animate-float">
    <h3 style="color: #00f2ff;">Mission</h3>
    <p style="color: #8b9bb4; line-height: 1.8;">
        As Large Language Models become ubiquitous, distinguishing between human creativity and 
        machine generation becomes critical for academic integrity and information authenticity. 
        <b>ARPaD++</b> (Advanced Recursive Pattern Detection) provides a robust, interpretable 
        framework for this verification.
    </p>
    <br>
    <h3 style="color: #bc13fe;">Methodology</h3>
    <p style="color: #8b9bb4; line-height: 1.8;">
        Unlike simple classifiers, ARPaD++ employs a multi-faceted approach combining:
        <ul style="list-style-type: none; padding-left: 0;">
            <li style="margin-bottom: 0.5rem;">🔹 <b>Stylometric Fingerprinting</b></li>
            <li style="margin-bottom: 0.5rem;">🔹 <b>Semantic Vector Analysis</b></li>
            <li style="margin-bottom: 0.5rem;">🔹 <b>Statistical Rarity Metrics</b></li>
        </ul>
    </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Team cards section (3 researchers + 1 guide)
    st.markdown(
        "<h2 style='text-align:center; margin: 2rem 0 1.5rem 0;'>Research Team & Guide</h2>",
        unsafe_allow_html=True,
    )

    # First row: centered guide card
    g_left, g_center, g_right = st.columns([1, 1.2, 1])

    with g_center:
        st.markdown(
            """
<div class="glass-panel" style="padding: 1.25rem;">
    <p style="font-size:0.75rem; letter-spacing:0.12em; color:#00f2ff; margin-bottom:0.5rem;">PROJECT GUIDE</p>
    <h3 style="margin:0 0 0.4rem 0;">Prof. Devendra K. Tayal</h3>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>Professor</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b> Department of Computer Science and Engineering</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>dev_tayal2001@yahoo.com</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    # Spacer between rows
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Second row: three researcher cards
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(
            """
<div class="glass-panel" style="padding: 1.25rem;">
    <p style="font-size:0.75rem; letter-spacing:0.12em; color:#bc13fe; margin-bottom:0.5rem;">RESEARCHER</p>
    <h3 style="margin:0 0 0.4rem 0;">Prakriti Rai</h3>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>BTech CSE-II</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>Department of Computer Science and Engineering</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b> prakriti138btcse22@igdtuw.ac.in</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with r2:
        st.markdown(
            """
<div class="glass-panel" style="padding: 1.25rem;">
    <p style="font-size:0.75rem; letter-spacing:0.12em; color:#bc13fe; margin-bottom:0.5rem;">RESEARCHER</p>
    <h3 style="margin:0 0 0.4rem 0;">Natasha Sethi</h3>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>BTech CSE-II</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b> Department of Computer Science and Engineering</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>natasha118btcse22@igdtuw.ac.in</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            """
<div class="glass-panel" style="padding: 1.25rem;">
    <p style="font-size:0.75rem; letter-spacing:0.12em; color:#bc13fe; margin-bottom:0.5rem;">RESEARCHER</p>
    <h3 style="margin:0 0 0.4rem 0;">Lakshita Verma</h3>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>BTech CSE-II</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>Department of Computer Science and Engineering</p>
    <p style="color:#8b9bb4; margin:0.1rem 0;"><b></b>lakshita97btcse22@igdtuw.ac.in</p>
</div>
            """,
            unsafe_allow_html=True,
        )
