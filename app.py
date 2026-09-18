import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

# 1. Boot up the smart AI brain
@st.cache_resource
def load_ai_brain():
    # This is a super-smart mini AI trained to read human emotions instantly
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

classifier = load_ai_brain()

# 2. Design the VIP Corporate Dashboard look
st.set_page_config(page_title="Money-Rescue Engine", layout="wide")

st.title("💰 The AI Money-Rescue Engine")
st.markdown("### *Winning Hackathon Entry: Active Business Risk Resolution Platform*")
st.write("---")

# Make tabs to look super organized
tab1, tab2 = st.tabs(["🚀 Instant Live Scanner", "📊 Bulk Revenue Risk Auditor"])

# --- TAB 1: LIVE SCANNER ---
with tab1:
    st.subheader("💬 Test a Live Customer Message")
    user_input = st.text_area("Paste customer message here:", "The delivery is 5 days late and customer service isn't replying! Cancel my order.")
    
    if st.button("Analyze & Auto-Fix 🧠"):
        if user_input.strip() != "":
            res = classifier(user_input)[0]
            text_lower = user_input.lower()
            
            # Category checking
            if any(w in text_lower for w in ["ship", "delivery", "late", "arrive", "wait"]):
                category = "🚚 Logistics & Shipping Delay"
            elif any(w in text_lower for w in ["broke", "quality", "cheap", "damaged", "fail"]):
                category = "🛠️ Product Quality Defect"
            else:
                category = "❓ General Operational Friction"
                
            if res['label'] == "NEGATIVE":
                st.error(f"🛑 CRITICAL ALERTS: Customer is Angry! (Confidence: {res['score']:.1%})")
                st.info(f"📍 **Detected Root Cause:** {category}")
                
                # Dynamic Response Generator (The award winner feature!)
                st.subheader("✨ Generated Instant Recovery Email:")
                st.code(f"Subject: Important update regarding your experience\n\nHi there,\n\nWe noticed you faced a {category.split(' ')[1]} issue. We are incredibly sorry. Our team is prioritizing this, and we have credited a credit/refund back to your account.\n\nBest,\nCustomer Care Team")
            else:
                st.success(f"🎉 Customer is Happy! (Confidence: {res['score']:.1%})")

# --- TAB 2: BULK SPREADSHEET AUDITOR ---
with tab2:
    st.subheader("📁 Upload Corporate Review Sheet")
    st.write("Drop a CSV or Excel sheet with customer feedback and transaction amounts.")
    
    uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.write("📋 Data Preview:", df.head(3))
        
        # Find matching columns automatically
        txt_col = next((c for c in df.columns if c.lower() in ['review', 'text', 'message', 'comment']), None)
        val_col = next((c for c in df.columns if c.lower() in ['money', 'value', 'price', 'spending', 'cost']), None)
        
        if not txt_col or not val_col:
            st.error("Make sure your file has a column for text (e.g., 'Review') and numbers (e.g., 'Money')!")
        else:
            if st.button("Run Financial Risk Audit ⚡"):
                with st.spinner("AI is calculating financial impacts..."):
                    
                    labels, categories = [], []
                    for text in df[txt_col]:
                        t_str = str(text).lower()
                        res = classifier(str(text))[0]
                        labels.append(res['label'])
                        
                        if any(w in t_str for w in ["ship", "delivery", "late", "arrive"]):
                            categories.append("Shipping Delay")
                        elif any(w in t_str for w in ["broke", "quality", "cheap", "damaged"]):
                            categories.append("Quality Issue")
                        else:
                            categories.append("General Feedback")
                            
                    df['AI_Sentiment'] = labels
                    df['Root_Cause'] = categories
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce').fillna(0)
                    
                    # Compute total money at risk
                    angry_df = df[df['AI_Sentiment'] == 'NEGATIVE']
                    total_at_risk = angry_df[val_col].sum()
                    
                    # Big Hero Widget
                    st.metric(label="🚨 TOTAL CORPORATE REVENUE AT IMMEDIATE RISK", value=f"${total_at_risk:,.2f}")
                    
                    # Side-by-Side Charts
                    c1, c2 = st.columns(2)
                    with c1:
                        fig1 = px.pie(df, names='AI_Sentiment', title='Customer Vibe Breakdown',
                                     color='AI_Sentiment', color_discrete_map={'POSITIVE':'#00CC96','NEGATIVE':'#EF553B'})
                        st.plotly_chart(fig1, use_container_width=True)
                    with c2:
                        if not angry_df.empty:
                            fig2 = px.bar(angry_df.groupby('Root_Cause')[val_col].sum().reset_index(), 
                                         x='Root_Cause', y=val_col, title='Financial Risk by Problem Area',
                                         color='Root_Cause', color_discrete_sequence=px.colors.sequential.Reds_r)
                            st.plotly_chart(fig2, use_container_width=True)
                    
                    st.subheader("⚠️ High-Priority Rescue List")
                    st.dataframe(angry_df[[txt_col, val_col, 'Root_Cause']].sort_values(by=val_col, ascending=False))
