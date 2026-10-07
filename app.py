"""ReviewShield — AI-Powered Review Authenticity Checker.

Professional consumer-facing AI product application.
Frontend powered by Streamlit; Backend powered by verified TF-IDF + Multinomial Naive Bayes.
"""

import os
import json
import time
import datetime
import streamlit as st
import pandas as pd

# Import backend inference engine
from src.predict import predict_review, is_model_trained

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="ReviewShield | Shop Smarter. Trust Reviews with Confidence.",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# CENTRALIZED DESIGN SYSTEM (CSS)
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Hide Streamlit default chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Layout Wrapper */
    .site-wrap {
        max-width: 1140px;
        margin: 0 auto;
        padding: 0 16px;
    }
    
    /* Navigation Bar */
    .top-nav {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 14px 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
        margin-bottom: 24px;
    }
    
    .brand-logo-text {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .brand-accent {
        color: #2563eb;
    }
    
    /* Hero Section */
    .hero-box {
        background: radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.08) 0%, transparent 40%),
                    #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 56px 48px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px -10px rgba(15, 23, 42, 0.05);
    }
    
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        font-size: 0.8rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 100px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 20px;
    }
    
    .hero-h1 {
        font-size: 3.1rem;
        font-weight: 800;
        line-height: 1.15;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin-bottom: 20px;
    }
    
    .hero-p {
        font-size: 1.18rem;
        line-height: 1.65;
        color: #475569;
        max-width: 650px;
        margin-bottom: 32px;
    }
    
    /* Product Mockup Card */
    .mockup-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.12);
    }
    
    .mockup-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 14px;
        margin-bottom: 16px;
    }
    
    /* Feature Cards */
    .bento-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 28px 24px;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.02);
    }
    .bento-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.08);
        border-color: #cbd5e1;
    }
    
    .bento-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-bottom: 20px;
        background: #f1f5f9;
    }
    
    .bento-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 10px;
    }
    
    .bento-desc {
        font-size: 0.94rem;
        line-height: 1.6;
        color: #64748b;
        margin: 0;
    }
    
    /* How It Works Steps */
    .step-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 26px;
        position: relative;
        height: 100%;
    }
    
    .step-badge {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 800;
        color: #2563eb;
        background: #eff6ff;
        padding: 4px 12px;
        border-radius: 100px;
        margin-bottom: 14px;
    }
    
    /* Result Banners */
    .result-box-genuine {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-left: 6px solid #16a34a;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    
    .result-box-fake {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 1px solid #fca5a5;
        border-left: 6px solid #e11d48;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    
    .result-heading-genuine {
        color: #166534;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 6px 0;
    }
    
    .result-heading-fake {
        color: #9f1239;
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0 0 6px 0;
    }
    
    .stat-metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    
    .stat-metric-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
    }
    
    .stat-metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
    }
    
    /* Footer */
    .footer-section {
        margin-top: 80px;
        padding: 48px 0 32px 0;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
    }
    
    .footer-col-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 14px;
    }
    
    .footer-link-text {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 2.1;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
def init_session_state():
    """Initializes persistent application session variables."""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "home"  # home, features, how_it_works, about, login, signup, dashboard, analyze, compare, insights, history, saved, settings, profile
    if "history" not in st.session_state:
        st.session_state.history = []
    if "saved_reviews" not in st.session_state:
        st.session_state.saved_reviews = []
    if "active_analysis" not in st.session_state:
        st.session_state.active_analysis = None
    if "active_review_text" not in st.session_state:
        st.session_state.active_review_text = ""
    if "show_signals" not in st.session_state:
        st.session_state.show_signals = True
    if "show_likelihood" not in st.session_state:
        st.session_state.show_likelihood = True
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"

init_session_state()


# ==========================================
# NAVIGATION BAR COMPONENT
# ==========================================
def render_navbar():
    """Renders the responsive top navigation bar for both public and authenticated views."""
    is_auth = st.session_state.user is not None
    current_page = st.session_state.current_page

    col_logo, col_nav = st.columns([1.6, 4.2])

    with col_logo:
        # Brand logo button returns to home (public) or dashboard (authenticated)
        brand_target = "dashboard" if is_auth else "home"
        if st.button("🛡️ ReviewShield", key="brand_nav_btn", type="tertiary"):
            st.session_state.current_page = brand_target
            st.rerun()

    with col_nav:
        if not is_auth:
            # Public Navigation Tabs
            n1, n2, n3, n4, n5, n6 = st.columns([1, 1.1, 1.2, 1, 1.1, 1.3])
            with n1:
                style_home = "primary" if current_page == "home" else "tertiary"
                if st.button("Home", key="nav_pub_home", type=style_home):
                    st.session_state.current_page = "home"
                    st.rerun()
            with n2:
                style_feat = "primary" if current_page == "features" else "tertiary"
                if st.button("Features", key="nav_pub_features", type=style_feat):
                    st.session_state.current_page = "features"
                    st.rerun()
            with n3:
                style_how = "primary" if current_page == "how_it_works" else "tertiary"
                if st.button("How It Works", key="nav_pub_how", type=style_how):
                    st.session_state.current_page = "how_it_works"
                    st.rerun()
            with n4:
                style_about = "primary" if current_page == "about" else "tertiary"
                if st.button("About", key="nav_pub_about", type=style_about):
                    st.session_state.current_page = "about"
                    st.rerun()
            with n5:
                if st.button("Log In", key="nav_pub_login", type="secondary"):
                    st.session_state.current_page = "login"
                    st.rerun()
            with n6:
                if st.button("Get Started", key="nav_pub_get_started", type="primary"):
                    st.session_state.current_page = "login"
                    st.rerun()
        else:
            # Authenticated SaaS Navigation Tabs
            u1, u2, u3, u4, u5, u6, u7 = st.columns([1.1, 1.4, 1.3, 1, 1, 1, 1.1])
            with u1:
                dash_style = "primary" if current_page == "dashboard" else "tertiary"
                if st.button("Dashboard", key="nav_auth_dash", type=dash_style):
                    st.session_state.current_page = "dashboard"
                    st.rerun()
            with u2:
                ana_style = "primary" if current_page == "analyze" else "tertiary"
                if st.button("Analyze Review", key="nav_auth_ana", type=ana_style):
                    st.session_state.current_page = "analyze"
                    st.rerun()
            with u3:
                cmp_style = "primary" if current_page == "compare" else "tertiary"
                if st.button("Compare", key="nav_auth_cmp", type=cmp_style):
                    st.session_state.current_page = "compare"
                    st.rerun()
            with u4:
                hist_style = "primary" if current_page == "history" else "tertiary"
                if st.button("History", key="nav_auth_hist", type=hist_style):
                    st.session_state.current_page = "history"
                    st.rerun()
            with u5:
                saved_style = "primary" if current_page == "saved" else "tertiary"
                if st.button("Saved", key="nav_auth_saved", type=saved_style):
                    st.session_state.current_page = "saved"
                    st.rerun()
            with u6:
                set_style = "primary" if current_page == "settings" else "tertiary"
                if st.button("Settings", key="nav_auth_set", type=set_style):
                    st.session_state.current_page = "settings"
                    st.rerun()
            with u7:
                if st.button("Log Out", key="nav_auth_logout", type="secondary"):
                    st.session_state.user = None
                    st.session_state.current_page = "home"
                    st.session_state.active_analysis = None
                    st.rerun()

    st.markdown("<hr style='margin: 8px 0 28px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)


# ==========================================
# FOOTER COMPONENT
# ==========================================
def render_footer():
    """Renders the comprehensive professional footer."""
    st.markdown("""
    <div class="footer-section">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 32px; margin-bottom: 32px;">
            <div style="max-width: 340px;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a; margin-bottom: 10px;">
                    🛡️ ReviewShield
                </div>
                <div style="line-height: 1.6; font-size: 0.9rem; color: #64748b;">
                    AI-powered review authenticity analysis for smarter online shopping. Identify suspicious language patterns before you buy.
                </div>
                <div style="margin-top: 14px; font-size: 0.8rem; font-weight: 600; color: #94a3b8;">
                    TAGLINE: "Shop Smarter. Trust Reviews with Confidence."
                </div>
            </div>
            <div>
                <div class="footer-col-title">Product</div>
                <div class="footer-link-text">
                    • AI Review Detection<br>
                    • Likelihood Breakdown<br>
                    • Language Signals<br>
                    • Review Comparison
                </div>
            </div>
            <div>
                <div class="footer-col-title">Resources</div>
                <div class="footer-link-text">
                    • How It Works<br>
                    • Frequently Asked Questions<br>
                    • Trust Guidelines<br>
                    • Language Insights
                </div>
            </div>
            <div>
                <div class="footer-col-title">About</div>
                <div class="footer-link-text">
                    • Machine Learning Engine<br>
                    • Privacy Architecture<br>
                    • Limitations & Scope<br>
                    • Academic Project (2026)
                </div>
            </div>
        </div>
        <div style="border-top: 1px solid #f1f5f9; padding-top: 20px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #94a3b8;">
            <div>&copy; 2026 ReviewShield. Built for Academic Demonstration.</div>
            <div>Privacy-Friendly &bull; No Tracking &bull; Consumer AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PUBLIC PAGE 1: RICH LANDING PAGE
# ==========================================
def render_home():
    """Renders the modern, visually impressive public landing page."""
    # 1. Hero Section
    h_left, h_right = st.columns([1.3, 1.1])
    with h_left:
        st.markdown("""
        <div class="hero-tag">✨ AI-Powered Review Intelligence</div>
        <div class="hero-h1">Not Every Review Tells the Whole Story.</div>
        <div class="hero-p">
            ReviewShield uses AI-powered language analysis to help you identify reviews that may be fake, 
            computer-generated, or unusually suspicious before they influence your buying decision.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.3, 1.4])
        with c1:
            if st.button("⚡ Analyze a Review", key="hero_primary_cta", type="primary", use_container_width=True):
                st.session_state.current_page = "login"
                st.rerun()
        with c2:
            if st.button("Explore How It Works", key="hero_sec_cta", type="secondary", use_container_width=True):
                st.session_state.current_page = "how_it_works"
                st.rerun()

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        # Trust Indicators Row
        st.markdown("""
        <div style="display: flex; gap: 20px; color: #64748b; font-size: 0.88rem; font-weight: 600;">
            <span>🛡️ AI-Powered</span>
            <span>⚡ Instant Analysis</span>
            <span>🔍 Explainable</span>
            <span>🔒 Privacy-Friendly</span>
        </div>
        """, unsafe_allow_html=True)

    with h_right:
        # Hero Interactive Visual Mockup
        st.markdown("""
        <div class="mockup-card">
            <div class="mockup-header">
                <span style="font-weight: 700; font-size: 0.88rem; color: #64748b;">CUSTOMER REVIEW MOCKUP</span>
                <span style="color: #f59e0b; font-weight: 700; font-size: 0.88rem;">★★★★★ 5.0</span>
            </div>
            <div style="background: #f8fafc; border-radius: 10px; padding: 14px; font-size: 0.95rem; color: #334155; margin-bottom: 16px; line-height: 1.5;">
                "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty and goes great with the other items."
            </div>
            <div style="display: flex; align-items: center; justify-content: center; color: #2563eb; font-size: 0.85rem; font-weight: 700; gap: 6px; margin-bottom: 16px;">
                <span>↓ AI SCANNING INTERFACE ↓</span>
            </div>
            <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 12px; padding: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="color: #9f1239; font-weight: 800; font-size: 1.05rem;">⚠️ Likely Fake Review</span>
                    <span style="background: #ffe4e6; color: #e11d48; font-weight: 700; font-size: 0.8rem; padding: 2px 8px; border-radius: 6px;">79% Likelihood</span>
                </div>
                <div style="font-size: 0.85rem; color: #475569;">
                    Language patterns strongly align with automated generative review templates.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 2. Problem Section
    st.markdown("## Online Reviews Can Be Hard to Trust.")
    st.write("E-commerce platforms are flooded with biased, incentivized, or machine-generated feedback. Star ratings alone no longer guarantee genuine satisfaction.")

    prob_left, prob_right = st.columns([1.1, 1.4])
    with prob_left:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 20px; padding: 32px; color: #ffffff; height: 100%;">
            <div style="font-size: 2.2rem; margin-bottom: 16px;">🛒 📦</div>
            <h3 style="color: #ffffff; margin-bottom: 12px;">The E-Commerce Trust Dilemma</h3>
            <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.6;">
                Dishonest sellers and automated bot syndicates post fake 5-star ratings to artificially inflate product rankings. 
                Consumers end up spending money on sub-par products based on manufactured enthusiasm.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with prob_right:
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("""
            <div class="bento-card">
                <div style="color: #e11d48; font-size: 1.3rem; margin-bottom: 12px;">⚡</div>
                <div class="bento-title">Too Positive</div>
                <p class="bento-desc">Unrealistic perfection with continuous superlatives and zero balanced critique.</p>
            </div>
            """, unsafe_allow_html=True)
        with p2:
            st.markdown("""
            <div class="bento-card">
                <div style="color: #f59e0b; font-size: 1.3rem; margin-bottom: 12px;">📄</div>
                <div class="bento-title">Too Generic</div>
                <p class="bento-desc">Vague praise lacking specific product details, specs, or real day-to-day context.</p>
            </div>
            """, unsafe_allow_html=True)
        with p3:
            st.markdown("""
            <div class="bento-card">
                <div style="color: #6366f1; font-size: 1.3rem; margin-bottom: 12px;">📢</div>
                <div class="bento-title">Too Promotional</div>
                <p class="bento-desc">Aggressive urgency urging readers to buy now with excessive punctuation stress.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 3. Interactive Live Demo Section
    st.markdown("## Try ReviewShield")
    st.write("Test our live machine learning model directly below. Enter any review or select a sample to see instant analysis.")

    demo_c1, demo_c2, demo_c3 = st.columns([1.5, 1.5, 2])
    with demo_c1:
        if st.button("Sample 1: Promotional Hype", key="landing_sample1", type="tertiary"):
            st.session_state.active_review_text = "MUST BUY!!! This is hands down the best product ever made in the entire world! 100% miracle item, changed my life forever. Buy this now, five stars!!!"
            st.rerun()
    with demo_c2:
        if st.button("Sample 2: Authentic Review", key="landing_sample2", type="tertiary"):
            st.session_state.active_review_text = "I bought this kettle three weeks ago to replace an older Hamilton Beach model. It works well for daily tea, though the lid latch feels a bit stiff. Easy to rinse."
            st.rerun()
    with demo_c3:
        if st.button("Sample 3: Generative Style", key="landing_sample3", type="tertiary"):
            st.session_state.active_review_text = "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty and goes great with the other items."
            st.rerun()

    demo_text = st.text_area(
        label="Test Review Text",
        value=st.session_state.active_review_text,
        placeholder="Paste any customer product review here to test the model...",
        height=110,
        label_visibility="collapsed"
    )

    if st.button("⚡ Analyze Example (Live Model)", key="landing_demo_analyze_btn", type="primary"):
        if not demo_text.strip():
            st.warning("Please enter or select a review to analyze.")
        else:
            with st.spinner("Analyzing text patterns..."):
                demo_res = predict_review(demo_text)
                if demo_res.get("status") == "success":
                    d_is_fake = demo_res["is_fake"]
                    d_conf = demo_res["confidence"]
                    d_probs = demo_res.get("class_probabilities", {})

                    if d_is_fake:
                        st.markdown(f"""
                        <div class="result-box-fake">
                            <div class="result-heading-fake">⚠️ Likely Fake Review</div>
                            <div style="font-size: 0.95rem; color: #475569;">
                                Model Likelihood: <strong>Fake: {d_probs.get('Fake', 0)}%</strong> &bull; Genuine: {d_probs.get('Genuine', 0)}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box-genuine">
                            <div class="result-heading-genuine">✅ Likely Genuine Review</div>
                            <div style="font-size: 0.95rem; color: #475569;">
                                Model Likelihood: <strong>Genuine: {d_probs.get('Genuine', 0)}%</strong> &bull; Fake: {d_probs.get('Fake', 0)}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 4. Solution & Workflow
    st.markdown("## Meet ReviewShield")
    st.write("ReviewShield analyzes the language of a product review and compares its patterns with what the trained model has learned from labeled fake and genuine reviews.")

    st.markdown("""
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 16px; margin: 24px 0;">
        <div style="flex: 1; min-width: 180px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; text-align: center;">
            <div style="font-size: 1.4rem; margin-bottom: 6px;">📝</div>
            <div style="font-weight: 700; font-size: 0.95rem;">1. Raw Review</div>
            <div style="color: #64748b; font-size: 0.82rem;">Text intake</div>
        </div>
        <div style="display: flex; align-items: center; color: #94a3b8; font-weight: 700;">→</div>
        <div style="flex: 1; min-width: 180px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; text-align: center;">
            <div style="font-size: 1.4rem; margin-bottom: 6px;">🔍</div>
            <div style="font-weight: 700; font-size: 0.95rem;">2. NLP Preprocessing</div>
            <div style="color: #64748b; font-size: 0.82rem;">Cleaning & tokens</div>
        </div>
        <div style="display: flex; align-items: center; color: #94a3b8; font-weight: 700;">→</div>
        <div style="flex: 1; min-width: 180px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; text-align: center;">
            <div style="font-size: 1.4rem; margin-bottom: 6px;">🧠</div>
            <div style="font-weight: 700; font-size: 0.95rem;">3. TF-IDF & MNB</div>
            <div style="color: #64748b; font-size: 0.82rem;">Pattern evaluation</div>
        </div>
        <div style="display: flex; align-items: center; color: #94a3b8; font-weight: 700;">→</div>
        <div style="flex: 1; min-width: 180px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 18px; text-align: center;">
            <div style="font-size: 1.4rem; margin-bottom: 6px;">📊</div>
            <div style="font-weight: 700; font-size: 0.95rem;">4. Likelihood Insights</div>
            <div style="color: #64748b; font-size: 0.82rem;">Decision support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 5. Use Cases Section
    st.markdown("## Who Can Use ReviewShield?")
    st.write("Empowering consumers, shoppers, and researchers with AI text transparency.")

    u1, u2, u3, u4 = st.columns(4)
    with u1:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-title">Online Shoppers</div>
            <p class="bento-desc">Inspect high-ticket or unfamiliar brand reviews before checkout to avoid counterfeit regrets.</p>
        </div>
        """, unsafe_allow_html=True)
    with u2:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-title">Conscious Consumers</div>
            <p class="bento-desc">Gain independent clarity beyond inflated 5-star ratings and paid influencer blurbs.</p>
        </div>
        """, unsafe_allow_html=True)
    with u3:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-title">Students & Researchers</div>
            <p class="bento-desc">Explore practical applications of Natural Language Processing and Bayes classification.</p>
        </div>
        """, unsafe_allow_html=True)
    with u4:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-title">E-Commerce Platforms</div>
            <p class="bento-desc">Automated review moderation at scale (future roadmap enhancement).</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 6. FAQ Section
    st.markdown("## Frequently Asked Questions")
    with st.expander("What does ReviewShield detect?"):
        st.write("ReviewShield analyzes writing structure, vocabulary distributions, and n-gram patterns to distinguish authentic customer feedback from computer-generated, bot-written, or deceptive promotional reviews.")
    with st.expander("How does the AI analyze a review?"):
        st.write("The review undergoes NLP cleaning (removing URLs, HTML, punctuation, and stopwords) and is transformed into TF-IDF n-gram vectors. Our trained Multinomial Naive Bayes model evaluates word frequencies against learned authenticity patterns.")
    with st.expander("Does a 'Likely Fake' result mean the review is 100% fraudulent?"):
        st.write("No. ReviewShield predicts statistical likelihood based on language patterns. It does not provide absolute legal proof, which is why ratings are clearly labeled as 'Likely Fake' or 'Likely Genuine'.")
    with st.expander("What is the likelihood score?"):
        st.write("The likelihood score represents the calibrated posterior probability calculated by the model (e.g., Fake: 85%, Genuine: 15%), reflecting its statistical confidence.")
    with st.expander("Can ReviewShield analyze any product review?"):
        st.write("Yes! You can paste customer reviews for electronics, home appliances, clothing, books, tools, and general consumer goods.")
    with st.expander("Does ReviewShield store my submitted reviews permanently?"):
        st.write("No. ReviewShield is privacy-centric. Submitted reviews are only kept in your temporary browser session state for your immediate dashboard history.")
    with st.expander("Can ReviewShield analyze an entire Amazon product page at once?"):
        st.write("Currently, ReviewShield analyzes individual review texts. Batch product URL scraping is reserved for future platform API integrations.")

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 7. Subtle Model Information (Behind the Technology)
    st.markdown("### Behind the Technology")
    st.write("ReviewShield is powered by a rigorously validated Natural Language Processing engine.")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown("""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Test Set Accuracy</div>
            <div class="stat-metric-value">89.37%</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Evaluated on 8,083 unseen reviews</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown("""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Fake F1-Score</div>
            <div class="stat-metric-value">0.8934</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Balanced precision & recall</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown("""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Training Benchmark</div>
            <div class="stat-metric-value">40,432</div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Verified consumer reviews</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

    # 8. Final CTA Card
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); border-radius: 20px; padding: 44px; text-align: center; color: #ffffff;">
        <h2 style="color: #ffffff; margin-bottom: 12px; font-size: 2.2rem;">Before You Trust the Review, Check It.</h2>
        <p style="color: #dbeafe; font-size: 1.1rem; max-width: 600px; margin: 0 auto 28px auto;">
            Give your next product review a second look with ReviewShield's AI analysis engine.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_btn_l, col_btn_center, col_btn_r = st.columns([1.5, 1.2, 1.5])
    with col_btn_center:
        if st.button("Analyze a Review Now", key="landing_final_cta_btn", type="primary", use_container_width=True):
            st.session_state.current_page = "login"
            st.rerun()


# ==========================================
# PUBLIC PAGE 2: FEATURES SHOWCASE
# ==========================================
def render_features():
    """Renders the comprehensive Features page."""
    st.markdown("## Powerful Features Built for Consumer Trust")
    st.write("Everything you need to inspect online customer feedback with speed and clarity.")

    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">🛡️</div>
            <div class="bento-title">1. AI Review Detection</div>
            <p class="bento-desc">
                Paste any customer review and receive an instantaneous Likelihood assessment (Likely Genuine or Likely Fake) 
                derived from a machine learning model trained on tens of thousands of real reviews.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">📊</div>
            <div class="bento-title">2. Likelihood Breakdown</div>
            <p class="bento-desc">
                View calibrated statistical probabilities for both classes (e.g., Fake: 85%, Genuine: 15%) 
                so you can understand how strongly the language aligns with generative patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    f3, f4 = st.columns(2)
    with f3:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">🔍</div>
            <div class="bento-title">3. Contextual Language Signals</div>
            <p class="bento-desc">
                Inspect auxiliary surface markers including word counts, character density, exclamation stress, 
                all-caps emphasis words, and detected promotional hype keywords.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">⚖️</div>
            <div class="bento-title">4. Side-by-Side Review Comparison</div>
            <p class="bento-desc">
                Compare two reviews side-by-side to determine which review displays more artificial or promotional 
                linguistic signals relative to the other.
            </p>
        </div>
        """, unsafe_allow_html=True)

    f5, f6 = st.columns(2)
    with f5:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">🕒</div>
            <div class="bento-title">5. Analysis Session History</div>
            <p class="bento-desc">
                Review, filter, and search all your analyzed reviews during your current session. Export reports 
                or clear history with a single click.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with f6:
        st.markdown("""
        <div class="bento-card">
            <div class="bento-icon">💾</div>
            <div class="bento-title">6. Saved Reviews & Bookmarks</div>
            <p class="bento-desc">
                Save critical reviews that warrant further inspection or comparison later during your session.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="features_back_home", type="secondary"):
        st.session_state.current_page = "home"
        st.rerun()


# ==========================================
# PUBLIC PAGE 3: HOW IT WORKS
# ==========================================
def render_how_it_works():
    """Renders the detailed 4-step How It Works guide."""
    st.markdown("## How ReviewShield Works")
    st.write("Understand the 4-step journey from raw review text to actionable decision intelligence.")

    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin: 28px 0;">
        <div class="step-card">
            <div class="step-badge">STEP 1</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Paste a Review</div>
            <p style="color: #64748b; font-size: 0.92rem; line-height: 1.6;">
                Copy any customer review from Amazon, Flipkart, or any online merchant and paste it directly into ReviewShield.
            </p>
        </div>
        <div class="step-card">
            <div class="step-badge">STEP 2</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Clean & Tokenize</div>
            <p style="color: #64748b; font-size: 0.92rem; line-height: 1.6;">
                The text is normalized by removing web URLs, HTML tags, punctuation artifacts, and common stopwords.
            </p>
        </div>
        <div class="step-card">
            <div class="step-badge">STEP 3</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">AI Classification</div>
            <p style="color: #64748b; font-size: 0.92rem; line-height: 1.6;">
                TF-IDF statistical n-grams are evaluated against our trained Multinomial Naive Bayes model.
            </p>
        </div>
        <div class="step-card">
            <div class="step-badge">STEP 4</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;">Actionable Results</div>
            <p style="color: #64748b; font-size: 0.92rem; line-height: 1.6;">
                ReviewShield delivers a clear Likely Genuine or Likely Fake verdict alongside detailed likelihood percentages.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Try It Out Now", key="how_try_now", type="primary"):
        st.session_state.current_page = "login"
        st.rerun()


# ==========================================
# PUBLIC PAGE 4: ABOUT REVIEWSHIELD
# ==========================================
def render_about():
    """Renders the consumer-friendly About ReviewShield page."""
    st.markdown("## About ReviewShield")
    st.write("Building transparency and trust in online consumer feedback through accessible AI.")

    st.markdown("""
    ### What Problem Does ReviewShield Solve?
    E-commerce platforms rely heavily on customer feedback to build trust. However, unscrupulous sellers, competitors, 
    and automated text bots routinely post computer-generated reviews to distort product reputations. ReviewShield was built 
    to provide shoppers with an independent AI assistant to evaluate whether review language looks organic or manufactured.

    ### Who Is ReviewShield For?
    - **Smart Shoppers:** Consumers wanting an extra layer of confidence before buying.
    - **Researchers & Students:** Academics exploring text classification using Bayes theorem.
    - **Online Sellers:** Merchants verifying genuine feedback versus malicious competitor attacks.

    ### Privacy Architecture
    ReviewShield operates without collecting personal browsing history, tracking user profiles, or storing reviews 
    permanently. Text analyses exist only in your current session.

    ### Practical Limitations
    - ReviewShield evaluates **linguistic patterns**, not real-world seller identity.
    - Extreme sarcasm or highly subtle irony may occasionally present ambiguous probability distributions.
    - Predictions represent **statistical likelihoods** rather than definitive legal proof.

    ### Technology Foundation
    - **Model:** Multinomial Naive Bayes Classifier ($\alpha = 1.0$)
    - **Feature Extraction:** TF-IDF Vectorizer (25,000 features, unigrams & bigrams)
    - **Training Data:** 40,432 verified balanced reviews spanning 10 product categories.
    """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="about_back_home", type="secondary"):
        st.session_state.current_page = "home"
        st.rerun()


# ==========================================
# PUBLIC PAGE 5: LOGIN
# ==========================================
def render_login():
    """Renders the split-screen login page with 1-click Demo User access."""
    col_l, col_r = st.columns([1.2, 1])

    with col_l:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 20px; padding: 36px; color: #ffffff;">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">🛡️</div>
            <h2 style="color: #ffffff; margin-bottom: 10px;">Your reviews. Your decisions. More confidence.</h2>
            <p style="color: #94a3b8; font-size: 1.0rem; line-height: 1.6; margin-bottom: 24px;">
                Sign in to access your personal dashboard, review comparison tools, and session analysis history.
            </p>
            <div style="font-size: 0.9rem; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 16px;">
                💡 <strong>Evaluation Note:</strong> Click <strong>"Continue as Demo User"</strong> to test immediately with zero setup.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 26px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
            <h3 style="margin: 0 0 4px 0; color: #0f172a;">Welcome back</h3>
            <p style="color: #64748b; font-size: 0.92rem; margin: 0 0 20px 0;">Sign in to your ReviewShield account.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form_comp"):
            l_email = st.text_input("Email address", value="alex.morgan@example.com")
            l_pwd = st.text_input("Password", type="password", value="password123")
            remember = st.checkbox("Remember this session", value=True)
            submit_l = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit_l:
                if not l_email.strip():
                    st.error("Please enter a valid email.")
                else:
                    user_disp = l_email.split("@")[0].replace(".", " ").title()
                    st.session_state.user = {
                        "name": user_disp,
                        "email": l_email.strip(),
                        "role": "Verified Consumer"
                    }
                    st.session_state.current_page = "dashboard"
                    st.rerun()

        st.markdown("<div style='text-align: center; margin: 12px 0; color: #94a3b8; font-size: 0.85rem;'>— OR FOR DEMONSTRATION —</div>", unsafe_allow_html=True)

        if st.button("⚡ Continue as Demo User (1-Click)", key="login_demo_btn", type="secondary", use_container_width=True):
            st.session_state.user = {
                "name": "Alex Morgan",
                "email": "alex.morgan@example.com",
                "role": "Demo Consumer"
            }
            st.session_state.current_page = "dashboard"
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Create an account", key="login_go_signup", type="tertiary"):
                st.session_state.current_page = "signup"
                st.rerun()
        with col_c2:
            if st.button("← Back to Home", key="login_go_home", type="tertiary"):
                st.session_state.current_page = "home"
                st.rerun()


# ==========================================
# PUBLIC PAGE 6: SIGN UP
# ==========================================
def render_signup():
    """Renders the account registration page."""
    c_l, c_mid, c_r = st.columns([1, 1.8, 1])
    with c_mid:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 30px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-top: 10px;">
            <h2 style="margin: 0 0 6px 0; color: #0f172a;">Create your account</h2>
            <p style="color: #64748b; font-size: 0.94rem; margin: 0 0 20px 0;">Get started with ReviewShield to check product reviews.</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("signup_form_comp"):
            s_name = st.text_input("Full Name", placeholder="e.g. Jordan Lee")
            s_email = st.text_input("Email address", placeholder="jordan@example.com")
            s_pwd1 = st.text_input("Password", type="password", placeholder="Create password")
            s_pwd2 = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
            submit_s = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submit_s:
                if not s_name.strip() or not s_email.strip():
                    st.error("Please provide both name and email.")
                elif s_pwd1 != s_pwd2:
                    st.error("Passwords do not match.")
                else:
                    st.session_state.user = {
                        "name": s_name.strip(),
                        "email": s_email.strip(),
                        "role": "Consumer Account"
                    }
                    st.session_state.current_page = "dashboard"
                    st.rerun()

        st.markdown("<div style='text-align: center; margin-top: 16px;'>", unsafe_allow_html=True)
        if st.button("Already have an account? Sign in", key="signup_back_login", type="tertiary"):
            st.session_state.current_page = "login"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# AUTH PAGE 7: USER DASHBOARD
# ==========================================
def render_dashboard():
    """Renders the main authenticated consumer dashboard."""
    user = st.session_state.user or {"name": "Alex Morgan", "email": "alex@example.com"}
    name = user.get("name", "User")

    st.markdown(f"## Good morning, {name} 👋")
    st.write("Ready to check your next product review? Analyze reviews, inspect linguistic signals, and compare products.")

    # Primary Action Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%); border-radius: 18px; padding: 32px; color: #ffffff; margin-bottom: 24px;">
        <h3 style="color: #ffffff; margin-bottom: 8px;">Verify Customer Reviews with AI</h3>
        <p style="color: #dbeafe; font-size: 1.05rem; margin-bottom: 20px; max-width: 600px;">
            Paste any customer review to detect automated patterns, promotional bias, and authenticity likelihood.
        </p>
    </div>
    """, unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1.3, 3])
    with d_col1:
        if st.button("⚡ Analyze a Review", key="dash_btn_analyze", type="primary", use_container_width=True):
            st.session_state.current_page = "analyze"
            st.rerun()
    with d_col2:
        if st.button("⚖️ Compare Two Reviews", key="dash_btn_compare", type="secondary"):
            st.session_state.current_page = "compare"
            st.rerun()

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # Session Activity Statistics
    st.markdown("### Your Activity")
    history = st.session_state.history
    saved = st.session_state.saved_reviews

    total_n = len(history)
    gen_n = sum(1 for x in history if not x.get("is_fake"))
    fake_n = sum(1 for x in history if x.get("is_fake"))
    saved_n = len(saved)

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Reviews Analyzed</div>
            <div class="stat-metric-value">{total_n}</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Likely Genuine</div>
            <div class="stat-metric-value" style="color: #16a34a;">{gen_n}</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Likely Fake</div>
            <div class="stat-metric-value" style="color: #e11d48;">{fake_n}</div>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-metric-card">
            <div class="stat-metric-label">Saved Bookmarks</div>
            <div class="stat-metric-value" style="color: #2563eb;">{saved_n}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # Recent Analyses Feed
    st.markdown("### Recent Analyses")
    if not history:
        st.markdown("""
        <div style="background: #ffffff; border: 1px dashed #cbd5e1; border-radius: 14px; padding: 36px; text-align: center; color: #64748b;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📋</div>
            <div style="font-weight: 700; font-size: 1.05rem; color: #0f172a; margin-bottom: 4px;">No reviews analyzed yet in this session.</div>
            <div style="font-size: 0.92rem;">Click "Analyze a Review" above to check your first product review.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, item in enumerate(reversed(history[-4:])):
            excerpt = item["text"][:130] + ("..." if len(item["text"]) > 130 else "")
            is_fake = item["is_fake"]
            bg_badge = "#fff1f2" if is_fake else "#f0fdf4"
            col_badge = "#e11d48" if is_fake else "#16a34a"
            txt_badge = "Likely Fake" if is_fake else "Likely Genuine"

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div style="max-width: 80%;">
                    <div style="font-size: 0.94rem; color: #1e293b; font-weight: 500; margin-bottom: 6px;">"{excerpt}"</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Analyzed at {item['time']}</div>
                </div>
                <div>
                    <span style="background: {bg_badge}; color: {col_badge}; font-weight: 700; font-size: 0.82rem; padding: 4px 10px; border-radius: 6px;">
                        {txt_badge} ({item['confidence']}%)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ==========================================
# AUTH PAGE 8: ANALYZE REVIEW (CORE FEATURE)
# ==========================================
def render_analyze():
    """Renders the main Review Analysis interface and live result."""
    st.markdown("## Analyze a Product Review")
    st.write("Paste a customer product review below and we'll check for patterns associated with potentially fake reviews.")

    # Sample Quick Buttons
    st.markdown("<span style='font-size: 0.82rem; font-weight: 700; color: #64748b; letter-spacing: 0.04em;'>QUICK SAMPLES FOR TESTING:</span>", unsafe_allow_html=True)
    s_col1, s_col2, s_col3 = st.columns([1.5, 1.5, 2])
    with s_col1:
        if st.button("Sample 1: Computer-Generated", key="btn_s1_ana", type="tertiary"):
            st.session_state.active_review_text = "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty and goes great with the other items."
            st.rerun()
    with s_col2:
        if st.button("Sample 2: Authentic Review", key="btn_s2_ana", type="tertiary"):
            st.session_state.active_review_text = "I bought this kettle three weeks ago to replace an older Hamilton Beach model. It works well for daily tea, though the lid latch feels a bit stiff. Easy to rinse."
            st.rerun()
    with s_col3:
        if st.button("Sample 3: Promotional Review", key="btn_s3_ana", type="tertiary"):
            st.session_state.active_review_text = "MUST BUY!!! This is hands down the best product ever made in the entire world! 100% miracle item, changed my life forever. Buy this now, five stars!!!"
            st.rerun()

    # Text Input with Counter
    review_input = st.text_area(
        label="Customer Review Text",
        value=st.session_state.active_review_text,
        placeholder="Paste your product review text here (e.g., 'I purchased these wireless earbuds two weeks ago. Sound quality is clear...')...",
        height=140,
        label_visibility="collapsed"
    )

    # Character and Word Counts
    char_len = len(review_input)
    word_len = len(review_input.split())
    st.caption(f"Review Length: **{word_len} words** ({char_len} characters)")

    b_exec, b_clr = st.columns([1.4, 4])
    with b_exec:
        run_analysis = st.button("⚡ Analyze Review", key="btn_run_main_analysis", type="primary", use_container_width=True)
    with b_clr:
        if st.button("Clear Text", key="btn_clr_ana_text", type="tertiary"):
            st.session_state.active_review_text = ""
            st.session_state.active_analysis = None
            st.rerun()

    # Execution Handling
    if run_analysis:
        if not review_input.strip():
            st.warning("Please enter or paste a customer review to analyze.")
        else:
            with st.spinner("Analyzing text patterns with ReviewShield AI..."):
                res = predict_review(review_input)
                if res.get("status") == "success":
                    st.session_state.active_analysis = res
                    # Save to session history
                    st.session_state.history.append({
                        "text": review_input.strip(),
                        "is_fake": res["is_fake"],
                        "label": res["prediction_label"],
                        "confidence": res["confidence"],
                        "class_probabilities": res.get("class_probabilities", {}),
                        "heuristics": res.get("heuristics", {}),
                        "time": datetime.datetime.now().strftime("%b %d, %H:%M")
                    })

    # Render Active Analysis Result
    active = st.session_state.active_analysis
    if active and active.get("status") == "success":
        st.markdown("<hr style='margin: 32px 0 24px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
        st.markdown("### Analysis Result")

        is_fake = active["is_fake"]
        probs = active.get("class_probabilities", {})
        heuristics = active.get("heuristics", {})

        # Primary Verdict Banner
        if is_fake:
            st.markdown(f"""
            <div class="result-box-fake">
                <div class="result-heading-fake">⚠️ Likely Fake Review</div>
                <div style="font-size: 0.98rem; color: #475569; margin-top: 4px;">
                    Our AI detected phrasing patterns, vocabulary density, and structure characteristic of automated or promotional reviews.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box-genuine">
                <div class="result-heading-genuine">✅ Likely Genuine Review</div>
                <div style="font-size: 0.98rem; color: #475569; margin-top: 4px;">
                    This review exhibits the organic sentence length, vocabulary variance, and descriptive specificity typical of authentic feedback.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Model Likelihood Breakdown
        st.markdown("#### Model Likelihood")
        fake_p = probs.get("Fake", 0.0)
        gen_p = probs.get("Genuine", 0.0)

        col_lp1, col_lp2 = st.columns(2)
        with col_lp1:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 8px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Fake Likelihood</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #e11d48;">{fake_p}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(fake_p / 100.0, 0.0), 1.0))
        with col_lp2:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 8px;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Genuine Likelihood</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #16a34a;">{gen_p}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(gen_p / 100.0, 0.0), 1.0))

        # Language Signals
        if st.session_state.show_signals:
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            st.markdown("#### Language Signals")
            sig1, sig2, sig3, sig4 = st.columns(4)
            with sig1:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-label">Word Count</div>
                    <div class="stat-metric-value">{heuristics.get('word_count', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with sig2:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-label">Characters</div>
                    <div class="stat-metric-value">{heuristics.get('char_count', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with sig3:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-label">Exclamations</div>
                    <div class="stat-metric-value">{heuristics.get('exclamation_count', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            with sig4:
                st.markdown(f"""
                <div class="stat-metric-card">
                    <div class="stat-metric-label">All-Caps Words</div>
                    <div class="stat-metric-value">{heuristics.get('uppercase_words', 0)}</div>
                </div>
                """, unsafe_allow_html=True)

            patterns = heuristics.get("suspicious_patterns", [])
            if patterns:
                st.caption(f"Detected promotional patterns: **{', '.join(patterns)}**")

            st.markdown("""
            <div style="background: #f1f5f9; border-left: 4px solid #94a3b8; border-radius: 6px; padding: 12px 16px; font-size: 0.85rem; color: #475569; margin-top: 14px;">
                <strong>Language Signals:</strong> These indicators provide additional reading context and do not independently determine the final classification.
            </div>
            """, unsafe_allow_html=True)

        # Why This Result Context
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Why this result?")
        if is_fake:
            st.write(
                "The model found vocabulary combinations and syntactic structures that correlate with "
                "computer-generated and promotional review templates in its verified training data."
            )
        else:
            st.write(
                "The review displays natural descriptive variance, realistic product specifics, and linguistic "
                "patterns typical of authentic customer feedback."
            )

        # Action Buttons (Save, Download Report, Reset)
        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        act_col1, act_col2, act_col3 = st.columns([1.5, 1.8, 2])
        
        with act_col1:
            if st.button("💾 Bookmark Review", key="save_current_review_btn", type="secondary"):
                st.session_state.saved_reviews.append({
                    "text": review_input.strip(),
                    "is_fake": is_fake,
                    "label": active["prediction_label"],
                    "confidence": active["confidence"],
                    "time": datetime.datetime.now().strftime("%b %d, %H:%M")
                })
                st.toast("Review saved to your bookmarks!")

        with act_col2:
            # Downloadable analysis report
            report_body = (
                f"REVIEWSHIELD ANALYSIS REPORT\n"
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"-----------------------------------------\n"
                f"Prediction: {active['prediction_label']}\n"
                f"Model Confidence: {active['confidence']}%\n"
                f"Fake Likelihood: {fake_p}%\n"
                f"Genuine Likelihood: {gen_p}%\n"
                f"Word Count: {heuristics.get('word_count', 0)}\n"
                f"Character Count: {heuristics.get('char_count', 0)}\n"
                f"-----------------------------------------\n"
                f"REVIEW TEXT:\n{review_input.strip()}\n"
            )
            st.download_button(
                label="📥 Download Report",
                data=report_body,
                file_name=f"reviewshield_report_{int(time.time())}.txt",
                mime="text/plain",
                type="secondary"
            )

        with act_col3:
            if st.button("← Analyze Another Review", key="analyze_another_reset", type="tertiary"):
                st.session_state.active_analysis = None
                st.session_state.active_review_text = ""
                st.rerun()


# ==========================================
# AUTH PAGE 9: COMPARE REVIEWS
# ==========================================
def render_compare():
    """Renders the side-by-side Review Comparison feature."""
    st.markdown("## Compare Two Reviews")
    st.write("Compare the language patterns and authenticity likelihoods of two customer reviews side-by-side.")

    c_col_a, c_col_b = st.columns(2)
    with c_col_a:
        st.markdown("#### Review A")
        text_a = st.text_area(
            "Review A Text",
            value="Love this! Well made, sturdy, and very comfortable. I love it! Very pretty and goes great with the other items.",
            height=130,
            key="compare_text_a"
        )
    with c_col_b:
        st.markdown("#### Review B")
        text_b = st.text_area(
            "Review B Text",
            value="I bought this kettle three weeks ago to replace an older Hamilton Beach model. It works well for daily tea, though the lid latch feels a bit stiff. Easy to rinse.",
            height=130,
            key="compare_text_b"
        )

    if st.button("⚖️ Compare Reviews", key="btn_run_compare", type="primary"):
        if not text_a.strip() or not text_b.strip():
            st.warning("Please enter text for both Review A and Review B.")
        else:
            with st.spinner("Analyzing and comparing both reviews..."):
                res_a = predict_review(text_a)
                res_b = predict_review(text_b)

                st.markdown("<hr style='margin: 24px 0;'>", unsafe_allow_html=True)
                st.markdown("### Comparison Results")

                r1, r2 = st.columns(2)
                with r1:
                    st.markdown("#### Review A Results")
                    lbl_a = "Likely Fake" if res_a["is_fake"] else "Likely Genuine"
                    col_a = "#e11d48" if res_a["is_fake"] else "#16a34a"
                    st.markdown(f"<span style='color: {col_a}; font-weight: 800; font-size: 1.2rem;'>{lbl_a} ({res_a['confidence']}%)</span>", unsafe_allow_html=True)
                    st.write(f"- **Fake Likelihood:** `{res_a.get('class_probabilities', {}).get('Fake', 0)}%`")
                    st.write(f"- **Genuine Likelihood:** `{res_a.get('class_probabilities', {}).get('Genuine', 0)}%`")
                    st.write(f"- **Word Count:** {res_a.get('heuristics', {}).get('word_count', 0)} words")
                with r2:
                    st.markdown("#### Review B Results")
                    lbl_b = "Likely Fake" if res_b["is_fake"] else "Likely Genuine"
                    col_b = "#e11d48" if res_b["is_fake"] else "#16a34a"
                    st.markdown(f"<span style='color: {col_b}; font-weight: 800; font-size: 1.2rem;'>{lbl_b} ({res_b['confidence']}%)</span>", unsafe_allow_html=True)
                    st.write(f"- **Fake Likelihood:** `{res_b.get('class_probabilities', {}).get('Fake', 0)}%`")
                    st.write(f"- **Genuine Likelihood:** `{res_b.get('class_probabilities', {}).get('Genuine', 0)}%`")
                    st.write(f"- **Word Count:** {res_b.get('heuristics', {}).get('word_count', 0)} words")

                # Verdict Box
                p_fake_a = res_a.get('class_probabilities', {}).get('Fake', 0)
                p_fake_b = res_b.get('class_probabilities', {}).get('Fake', 0)

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                if p_fake_a > p_fake_b:
                    st.info(f"**Comparative Finding:** Review A displays a higher model likelihood of being fake ({p_fake_a}% vs {p_fake_b}%).")
                elif p_fake_b > p_fake_a:
                    st.info(f"**Comparative Finding:** Review B displays a higher model likelihood of being fake ({p_fake_b}% vs {p_fake_a}%).")
                else:
                    st.info("Both reviews show equivalent likelihood distributions.")


# ==========================================
# AUTH PAGE 10: REVIEW INSIGHTS
# ==========================================
def render_insights():
    """Renders detailed linguistic breakdown and indicators."""
    st.markdown("## Review Language Insights")
    st.write("Examine the nuanced linguistic signals extracted from your latest analyzed review.")

    active = st.session_state.active_analysis
    if not active:
        st.info("No active review analysis found. Analyze a review first to inspect its detailed language profile.")
        if st.button("⚡ Go to Review Analyzer", key="insights_go_analyze", type="primary"):
            st.session_state.current_page = "analyze"
            st.rerun()
    else:
        heuristics = active.get("heuristics", {})
        st.markdown(f"#### Active Verdict: **{active.get('prediction_label')}** ({active.get('confidence')}%)")

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-title">Text Specificity & Detail</div>
                <p class="bento-desc">Evaluates word volume, structural depth, and detailed product terminology.</p>
            </div>
            """, unsafe_allow_html=True)
            words = heuristics.get('word_count', 0)
            score_spec = min(max(words / 40.0, 0.1), 1.0)
            st.progress(score_spec)
            st.caption(f"Linguistic Indicator: {words} words in review.")

        with col_in2:
            st.markdown("""
            <div class="bento-card">
                <div class="bento-title">Emotional & Punctuation Stress</div>
                <p class="bento-desc">Measures frequency of exclamation marks, question symbols, and all-caps emphasis.</p>
            </div>
            """, unsafe_allow_html=True)
            excls = heuristics.get('exclamation_count', 0)
            score_emo = min(max(excls / 5.0, 0.0), 1.0)
            st.progress(score_emo)
            st.caption(f"Linguistic Indicator: {excls} emphasis symbols detected.")


# ==========================================
# AUTH PAGE 11: ANALYSIS HISTORY
# ==========================================
def render_history():
    """Renders the user's session history with search and filter controls."""
    st.markdown("## Analysis History")
    st.write("Browse and manage reviews analyzed during your current session.")

    history = st.session_state.history
    if not history:
        st.info("No analysis history recorded in this session yet.")
        if st.button("⚡ Analyze Your First Review", key="hist_first_run", type="primary"):
            st.session_state.current_page = "analyze"
            st.rerun()
    else:
        top_h1, top_h2, top_h3 = st.columns([2, 1.2, 1])
        with top_h1:
            search_query = st.text_input("Search history text", placeholder="Type keywords...", label_visibility="collapsed")
        with top_h2:
            filter_choice = st.selectbox("Filter", ["All Reviews", "Likely Fake Only", "Likely Genuine Only"], label_visibility="collapsed")
        with top_h3:
            if st.button("Clear History", key="btn_clear_hist_top", type="secondary", use_container_width=True):
                st.session_state.history = []
                st.session_state.active_analysis = None
                st.rerun()

        # Filter list
        filtered = history
        if filter_choice == "Likely Fake Only":
            filtered = [x for x in filtered if x.get("is_fake") is True]
        elif filter_choice == "Likely Genuine Only":
            filtered = [x for x in filtered if x.get("is_fake") is False]

        if search_query.strip():
            filtered = [x for x in filtered if search_query.lower() in x.get("text", "").lower()]

        st.caption(f"Showing **{len(filtered)}** of {len(history)} analyses.")

        for idx, item in enumerate(reversed(filtered)):
            is_fake = item["is_fake"]
            badge_bg = "#fff1f2" if is_fake else "#f0fdf4"
            badge_col = "#e11d48" if is_fake else "#16a34a"
            badge_txt = "Likely Fake" if is_fake else "Likely Genuine"
            probs = item.get("class_probabilities", {})

            with st.expander(f"Analysis #{len(filtered) - idx}: {badge_txt} ({item['confidence']}%) &bull; {item['time']}"):
                st.markdown(f"""
                <div style="background: {badge_bg}; border-left: 4px solid {badge_col}; padding: 10px 14px; border-radius: 6px; margin-bottom: 10px;">
                    <strong style="color: {badge_col};">{badge_txt}</strong> &bull; Model Confidence: {item['confidence']}%
                    <span style="float: right; color: #64748b; font-size: 0.82rem;">Fake: {probs.get('Fake', 0)}% | Genuine: {probs.get('Genuine', 0)}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"**Review Content:**")
                st.info(f"\"{item['text']}\"")


# ==========================================
# AUTH PAGE 12: SAVED REVIEWS
# ==========================================
def render_saved():
    """Renders bookmarked reviews saved during the current session."""
    st.markdown("## Saved Reviews")
    st.write("Bookmarks of critical reviews saved for later reference during your session.")

    saved = st.session_state.saved_reviews
    if not saved:
        st.info("No saved reviews yet. Use the '💾 Bookmark Review' button on any analysis result to bookmark it here.")
    else:
        st.caption(f"Showing **{len(saved)}** bookmarked reviews.")
        for idx, item in enumerate(reversed(saved)):
            is_fake = item["is_fake"]
            col_b = "#e11d48" if is_fake else "#16a34a"
            bg_b = "#fff1f2" if is_fake else "#f0fdf4"
            txt_b = "Likely Fake" if is_fake else "Likely Genuine"

            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="background: {bg_b}; color: {col_b}; font-weight: 700; font-size: 0.82rem; padding: 4px 10px; border-radius: 6px;">
                        {txt_b} ({item['confidence']}%)
                    </span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">Saved at {item['time']}</span>
                </div>
                <div style="font-size: 0.95rem; color: #1e293b; line-height: 1.5;">"{item['text']}"</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear Saved Bookmarks", key="clear_all_saved_btn", type="secondary"):
            st.session_state.saved_reviews = []
            st.rerun()


# ==========================================
# AUTH PAGE 13: SETTINGS
# ==========================================
def render_settings():
    """Renders user preferences, session management, and optional technical specs."""
    user = st.session_state.user or {"name": "Alex Morgan", "email": "alex@example.com"}

    st.markdown("## Settings & Preferences")
    st.write("Configure your ReviewShield preferences and session options.")

    # Account Card
    st.markdown("### Profile Information")
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 24px;">
        <div style="font-size: 0.8rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Active Session Account</div>
        <div style="font-size: 1.3rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{user.get('name', 'User')}</div>
        <div style="color: #64748b; font-size: 0.92rem;">{user.get('email', 'alex@example.com')} &bull; <span style="color: #2563eb; font-weight: 600;">Authenticated</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Preferences
    st.markdown("### Analysis Display Preferences")
    st.markdown("""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 24px;">""", unsafe_allow_html=True)
    c_sig = st.checkbox("Show Language Signals on Analysis Results", value=st.session_state.show_signals)
    c_lik = st.checkbox("Show Model Likelihood Probability Bars", value=st.session_state.show_likelihood)
    st.session_state.show_signals = c_sig
    st.session_state.show_likelihood = c_lik
    st.markdown("</div>", unsafe_allow_html=True)

    # Privacy & Data
    st.markdown("### Privacy & Session Data")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🗑️ Clear Analysis History", key="set_clear_hist", type="secondary", use_container_width=True):
            st.session_state.history = []
            st.session_state.active_analysis = None
            st.toast("Analysis history cleared.")
    with col_c2:
        if st.button("🗑️ Clear Saved Bookmarks", key="set_clear_saved", type="secondary", use_container_width=True):
            st.session_state.saved_reviews = []
            st.toast("Saved bookmarks cleared.")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Optional Technical Drawer
    with st.expander("ℹ️ AI Engine Specifications (Demonstration Reference)"):
        st.write("**Model Specifications:**")
        st.write("- **Classification Algorithm:** Multinomial Naive Bayes (MNB) with Laplace smoothing")
        st.write("- **Feature Extractor:** TF-IDF Vectorizer (25,000 unigrams & bigrams)")
        st.write("- **Verified Test Accuracy:** **89.37%** (Evaluated on 8,083 unseen samples)")
        st.write("- **Verified Fake F1-Score:** **0.8934** (Precision: 89.61%, Recall: 89.06%)")

    st.markdown("<hr style='margin: 28px 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)

    if st.button("Sign Out of Session", key="set_sign_out", type="primary"):
        st.session_state.user = None
        st.session_state.current_page = "home"
        st.session_state.active_analysis = None
        st.rerun()


# ==========================================
# AUTH PAGE 14: PROFILE
# ==========================================
def render_profile():
    """Renders the user profile overview."""
    user = st.session_state.user or {"name": "Alex Morgan", "email": "alex@example.com"}
    st.markdown(f"## Consumer Profile")
    st.write(f"Account management for **{user.get('name')}**.")

    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 28px; max-width: 600px;">
        <div style="font-size: 2rem; margin-bottom: 8px;">👤</div>
        <div style="font-size: 1.4rem; font-weight: 800; color: #0f172a;">{user.get('name')}</div>
        <div style="color: #64748b; font-size: 0.95rem; margin-bottom: 16px;">{user.get('email')}</div>
        <div style="border-top: 1px solid #f1f5f9; padding-top: 14px; font-size: 0.88rem; color: #475569;">
            Status: <span style="color: #16a34a; font-weight: 700;">● Active Consumer Session</span><br>
            Role: Demo Evaluator Account
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MASTER APPLICATION ROUTER
# ==========================================
def main():
    """Top-level master router handling authentication state and page navigation."""
    st.markdown("<div class='site-wrap'>", unsafe_allow_html=True)
    render_navbar()

    current_page = st.session_state.current_page
    user = st.session_state.user
    is_auth = user is not None

    # Protect internal routes from unauthenticated access
    auth_routes = ["dashboard", "analyze", "compare", "insights", "history", "saved", "settings", "profile"]
    if current_page in auth_routes and not is_auth:
        render_login()
    elif current_page == "home":
        render_home()
    elif current_page == "features":
        render_features()
    elif current_page == "how_it_works":
        render_how_it_works()
    elif current_page == "about":
        render_about()
    elif current_page == "login":
        render_login()
    elif current_page == "signup":
        render_signup()
    elif current_page == "dashboard":
        render_dashboard()
    elif current_page == "analyze":
        render_analyze()
    elif current_page == "compare":
        render_compare()
    elif current_page == "insights":
        render_insights()
    elif current_page == "history":
        render_history()
    elif current_page == "saved":
        render_saved()
    elif current_page == "settings":
        render_settings()
    elif current_page == "profile":
        render_profile()
    else:
        render_home()

    render_footer()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
