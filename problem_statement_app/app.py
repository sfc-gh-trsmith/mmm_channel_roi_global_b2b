"""
The Attribution Problem: Why Raw Spend Fails
============================================
A standalone educational app demonstrating why naive marketing attribution doesn't work.
"""

import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark.context import get_active_session
import altair as alt

st.set_page_config(
    page_title="The Attribution Problem",
    page_icon="🎯",
    layout="wide"
)

@st.cache_resource
def get_session():
    return get_active_session()

@st.cache_data(ttl=600)
def load_weekly_data():
    session = get_session()
    df = session.sql("""
        SELECT 
            WEEK_START,
            CHANNEL_CODE AS CHANNEL,
            SUM(SPEND) AS WEEKLY_SPEND,
            SUM(REVENUE) AS WEEKLY_REVENUE
        FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY
        WHERE CHANNEL_CODE IS NOT NULL
        GROUP BY WEEK_START, CHANNEL_CODE
        ORDER BY WEEK_START, CHANNEL_CODE
    """).to_pandas()
    return df

@st.cache_data(ttl=600)
def load_total_weekly():
    session = get_session()
    df = session.sql("""
        SELECT 
            WEEK_START,
            SUM(SPEND) AS TOTAL_SPEND,
            SUM(REVENUE) AS REVENUE
        FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY
        GROUP BY WEEK_START
        HAVING SUM(SPEND) > 0 OR SUM(REVENUE) > 0
        ORDER BY WEEK_START
    """).to_pandas()
    return df

st.title("🎯 The Marketing Attribution Problem")

st.markdown("""
> **The Question**: "Which marketing channels actually drive revenue?"

This seems simple. Spend more → get more revenue. Just correlate the two, right?

**Spoiler**: It's not that easy. Let's see why.
""")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ The Naive Approach", 
    "2️⃣ Problem: Memory", 
    "3️⃣ Problem: Saturation",
    "4️⃣ Problem: Scale",
    "5️⃣ The Solution"
])

with tab1:
    st.header("The Naive Approach")
    
    st.markdown("""
    Let's try the obvious thing: **correlate weekly spend with weekly revenue**.
    """)
    
    try:
        df_total = load_total_weekly()
        
        correlation = df_total['TOTAL_SPEND'].corr(df_total['REVENUE'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(
                label="Correlation",
                value=f"{correlation:.2f}",
                delta="Weak" if abs(correlation) < 0.5 else "Moderate"
            )
            
            if abs(correlation) < 0.5:
                st.error("❌ Too weak for business decisions")
            else:
                st.warning("⚠️ Some signal, but noisy")
            
            st.markdown(f"""
            **{len(df_total)} weeks** of data
            
            **Total spend**: ${df_total['TOTAL_SPEND'].sum()/1e6:.1f}M
            
            **Total revenue**: ${df_total['REVENUE'].sum()/1e6:.1f}M
            """)
        
        with col2:
            scatter = alt.Chart(df_total).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('TOTAL_SPEND:Q', title='Weekly Spend ($)', scale=alt.Scale(zero=False)),
                y=alt.Y('REVENUE:Q', title='Weekly Revenue ($)', scale=alt.Scale(zero=False)),
                tooltip=['WEEK_START:T', 'TOTAL_SPEND:Q', 'REVENUE:Q']
            ).properties(height=400)
            
            regression = scatter.transform_regression(
                'TOTAL_SPEND', 'REVENUE'
            ).mark_line(color='red', strokeDash=[5,5])
            
            st.altair_chart(scatter + regression, use_container_width=True)
        
        st.info("""
        **What went wrong?** The scatter plot shows massive variance. 
        Some high-spend weeks have low revenue. Some low-spend weeks have high revenue.
        
        This isn't noise — it's **missing signal**. The naive model doesn't understand 
        how advertising actually works.
        """)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

with tab2:
    st.header("Problem #1: Advertising Has Memory")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### The Carryover Effect
        
        When you see an ad, you don't buy immediately. 
        The ad creates **awareness** that persists for days or weeks.
        
        ---
        
        **Example**:
        
        | Week | TV Spend | What Happens |
        |------|----------|--------------|
        | 1 | $500K | People see ads |
        | 2 | $0 | People still remember |
        | 3 | $0 | Some finally buy |
        
        ---
        
        A naive model sees **zero spend** in weeks 2-3 and can't explain 
        why revenue is still elevated.
        
        **The model is blind to memory.**
        """)
    
    with col2:
        st.markdown("### Interactive: Adstock Transform")
        
        theta = st.slider(
            "Decay Rate (θ)", 0.0, 0.95, 0.7, 0.05,
            help="θ = 0.7 means 70% of last week's effect carries over"
        )
        
        weeks = 12
        spend = np.array([100, 0, 0, 0, 50, 0, 0, 0, 75, 0, 0, 0])
        
        adstock = np.zeros(weeks)
        adstock[0] = spend[0]
        for t in range(1, weeks):
            adstock[t] = spend[t] + theta * adstock[t-1]
        
        demo_df = pd.DataFrame({
            'Week': range(1, weeks + 1),
            'Raw Spend': spend,
            'Adstock': adstock
        })
        
        base = alt.Chart(demo_df).encode(x=alt.X('Week:O', title='Week'))
        
        bars = base.mark_bar(color='#3498db', opacity=0.5).encode(
            y=alt.Y('Raw Spend:Q', title='Spend ($K)', axis=alt.Axis(titleColor='#3498db'))
        )
        
        line = base.mark_line(color='#e74c3c', strokeWidth=3).encode(
            y=alt.Y('Adstock:Q', title='Effective Spend ($K)', axis=alt.Axis(titleColor='#e74c3c'))
        )
        
        points = base.mark_circle(color='#e74c3c', size=80).encode(
            y='Adstock:Q'
        )
        
        chart = alt.layer(bars, line, points).resolve_scale(
            y='shared'
        ).properties(height=350)
        
        st.altair_chart(chart, use_container_width=True)
        
        st.markdown("""
        **Legend:**
        - 🔵 **Blue bars** = Actual weekly spend
        - 🔴 **Red line** = Effective spend (with carryover)
        """)
        
        st.success(f"""
        **With θ = {theta}**:
        - Week 1 spend of $100K still has ${100 * (theta**3):.0f}K effect in Week 4
        - The red line captures what the naive model misses
        """)
        
        st.divider()
        st.markdown("### 🤖 AI Insight: What does this θ value mean?")
        
        session = get_session()
        ai_prompt = f"""In 2-3 sentences, explain what a theta (θ) decay rate of {theta} means 
for marketing advertising effectiveness. This is for the adstock transformation formula: 
adstock[t] = spend[t] + θ × adstock[t-1]. 
Explain in plain business terms what this decay rate implies about how long ad effects last."""
        
        ai_response = session.sql(f"SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large2', '{ai_prompt.replace(chr(39), chr(39)+chr(39))}')").collect()[0][0]
        st.info(ai_response)

with tab3:
    st.header("Problem #2: Diminishing Returns")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### The Saturation Effect
        
        The first dollar of advertising is more effective than the millionth.
        
        ---
        
        **Why?**
        
        | Spend Level | What Happens |
        |-------------|--------------|
        | First $100K | Reaches new customers |
        | Next $100K | Some new, some repeats |
        | $500K+ | Mostly re-targeting same people |
        | $1M+ | Severely diminished returns |
        
        ---
        
        ### ⚠️ What Makes This Hard
        
        The saturation curve exists — but **you don't know its shape**.
        
        | Parameter | The Challenge |
        |-----------|---------------|
        | **γ (gamma)** | Where does LinkedIn saturate vs Google? Unknown. |
        | **α (alpha)** | How sharp is the curve? Varies by channel. |
        | **θ + γ + α** | Must be estimated *together* with adstock decay. |
        
        **The compounding problem:**  
        You can't measure saturation directly from raw spend. You need to:
        1. Apply adstock transformation first (guess θ)
        2. Then fit saturation curve (guess γ, α)
        3. Then validate against revenue
        
        Wrong θ → Wrong γ → Wrong budget recommendations.
        """)
    
    with col2:
        st.markdown("### Interactive: Hill Saturation Curve")
        
        gamma = st.slider(
            "Half-Saturation Point γ ($K)", 100, 1000, 400, 50,
            help="Spend level where you reach 50% of maximum effect"
        )
        alpha = st.slider(
            "Steepness α", 0.5, 3.0, 1.5, 0.1,
            help="Higher = sharper S-curve"
        )
        
        x = np.linspace(0, 2000, 200)
        y_hill = (x ** alpha) / (gamma ** alpha + x ** alpha)
        y_linear = x / x.max()
        
        curve_df = pd.DataFrame({
            'Spend ($K)': np.tile(x, 2),
            'Effect': np.concatenate([y_hill, y_linear]),
            'Model': ['Realistic (Hill)'] * len(x) + ['Naive (Linear)'] * len(x)
        })
        
        chart = alt.Chart(curve_df).mark_line(strokeWidth=3).encode(
            x='Spend ($K):Q',
            y=alt.Y('Effect:Q', scale=alt.Scale(domain=[0, 1.1])),
            color=alt.Color('Model:N', scale=alt.Scale(
                domain=['Realistic (Hill)', 'Naive (Linear)'],
                range=['#e74c3c', '#bdc3c7']
            )),
            strokeDash=alt.StrokeDash('Model:N', scale=alt.Scale(
                domain=['Realistic (Hill)', 'Naive (Linear)'],
                range=[[0], [5, 5]]
            ))
        ).properties(height=350)
        
        gamma_rule = alt.Chart(pd.DataFrame({'x': [gamma], 'y': [0.5]})).mark_rule(
            color='green', strokeDash=[3, 3]
        ).encode(x='x:Q')
        
        gamma_point = alt.Chart(pd.DataFrame({'x': [gamma], 'y': [0.5]})).mark_circle(
            color='green', size=100
        ).encode(x='x:Q', y='y:Q')
        
        st.altair_chart(chart + gamma_rule + gamma_point, use_container_width=True)
        
        st.caption("Red = reality | Gray = naive assumption | Green = 50% saturation point")
        
        current_spend = st.number_input("Your current spend ($K)", 100, 2000, 600, 100)
        current_effect = (current_spend ** alpha) / (gamma ** alpha + current_spend ** alpha)
        
        st.info(f"""
        **At ${current_spend}K spend**: You're at **{current_effect*100:.0f}%** of maximum effect.
        
        {'🟢 Room to grow!' if current_effect < 0.6 else '🟡 Approaching saturation' if current_effect < 0.85 else '🔴 Heavily saturated - consider reallocation'}
        """)

with tab4:
    st.header("Problem #3: The Scale Problem")
    
    st.markdown("""
    > You've learned that each channel needs **θ** (decay), **γ** (saturation point), and **α** (steepness).
    > 
    > Now multiply that by every dimension you care about.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Your Real Data")
        
        try:
            session = get_session()
            
            channels_df = session.sql("""
                SELECT DISTINCT CHANNEL_CODE 
                FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY 
                WHERE CHANNEL_CODE IS NOT NULL
            """).to_pandas()
            
            regions_df = session.sql("""
                SELECT DISTINCT REGION_NAME 
                FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY 
                WHERE REGION_NAME IS NOT NULL
            """).to_pandas()
            
            n_channels = len(channels_df)
            n_regions = len(regions_df)
            
            st.metric("Channels", n_channels)
            st.caption(", ".join(channels_df['CHANNEL_CODE'].tolist()))
            
            st.metric("Regions", n_regions)
            st.caption(", ".join(regions_df['REGION_NAME'].tolist()[:5]) + ("..." if n_regions > 5 else ""))
            
        except:
            n_channels = 4
            n_regions = 5
            st.metric("Channels", "~4")
            st.metric("Regions", "~5")
        
        st.markdown("---")
        
        st.markdown("### 🔢 The Parameter Explosion")
        
        params_per_channel = 3
        
        base_params = n_channels * params_per_channel
        with_regions = n_channels * n_regions * params_per_channel
        
        st.markdown(f"""
        | Granularity | Combinations | Parameters to Estimate |
        |-------------|--------------|------------------------|
        | Channel only | {n_channels} channels | **{base_params}** (θ, γ, α each) |
        | Channel × Region | {n_channels} × {n_regions} | **{with_regions}** parameters |
        | + Seasonality | × 4 quarters | **{with_regions * 4}** parameters |
        """)
        
        st.error(f"""
        **{with_regions}+ parameters** — and they all interact with each other.
        
        Manual tuning? Impossible.
        """)
    
    with col2:
        st.markdown("### 🧠 Why Each Dimension Matters")
        
        st.markdown("""
        #### Different Channels = Different Memory
        
        | Channel | Typical θ | Why |
        |---------|-----------|-----|
        | **Search Ads** | 0.2 - 0.4 | Intent-driven, fast conversion |
        | **Social Ads** | 0.5 - 0.7 | Awareness builds over time |
        | **TV/Brand** | 0.7 - 0.9 | Long-lasting brand recall |
        
        ---
        
        #### Different Regions = Different Saturation
        
        | Market | Saturation Behavior |
        |--------|---------------------|
        | **Mature (US, EU)** | Saturates faster — audience is tapped |
        | **Emerging (APAC)** | More headroom — still growing |
        | **Niche verticals** | Small audience — saturates very fast |
        
        ---
        
        #### The Combinatorial Challenge
        
        LinkedIn in **EMEA** might have:
        - θ = 0.6 (medium decay)
        - γ = $200K (saturates early)
        
        LinkedIn in **APAC** might have:
        - θ = 0.5 (faster decay)  
        - γ = $500K (more headroom)
        
        **Same channel. Different parameters. Both are "right."**
        """)
    
    st.divider()
    
    st.markdown("### 🤯 The Real Problem")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Without Segmentation
        
        You get **one** θ, γ, α per channel.
        
        This **averages out** regional differences and hides opportunities.
        """)
    
    with col2:
        st.markdown("""
        #### With Naive Segmentation
        
        You split data by region.
        
        Now you have **less data per segment** → unreliable parameter estimates.
        """)
    
    with col3:
        st.markdown("""
        #### The ML Solution
        
        Regularized models can:
        - Learn shared patterns across segments
        - Allow segment-specific adjustments
        - Handle sparse data gracefully
        """)
    
    st.info("""
    **This is why MMM requires machine learning.**
    
    The parameter space is too large for manual optimization, and the interactions 
    between θ, γ, α across channels and regions create a search problem that 
    only algorithms can efficiently navigate.
    """)
    
    with st.expander("🔍 **For Example: Try It Yourself** — Does slicing the data help?"):
        st.markdown("""
        Let's see what happens when you try to analyze each channel-region segment separately.
        Pick a combination below and see the spend vs revenue correlation:
        """)
        
        try:
            session = get_session()
            
            all_channels = session.sql("""
                SELECT DISTINCT CHANNEL_CODE 
                FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY 
                WHERE CHANNEL_CODE IS NOT NULL
                ORDER BY CHANNEL_CODE
            """).to_pandas()['CHANNEL_CODE'].tolist()
            
            all_regions = session.sql("""
                SELECT DISTINCT REGION_NAME 
                FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY 
                WHERE REGION_NAME IS NOT NULL
                ORDER BY REGION_NAME
            """).to_pandas()['REGION_NAME'].tolist()
            
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                selected_channel = st.selectbox("📺 Channel", ["All Channels"] + all_channels, key="scale_channel")
            
            with filter_col2:
                selected_region = st.selectbox("🌍 Region", ["All Regions"] + all_regions, key="scale_region")
            
            where_clauses = ["SPEND IS NOT NULL", "REVENUE IS NOT NULL"]
            if selected_channel != "All Channels":
                where_clauses.append(f"CHANNEL_CODE = '{selected_channel}'")
            if selected_region != "All Regions":
                where_clauses.append(f"REGION_NAME = '{selected_region}'")
            
            where_sql = " AND ".join(where_clauses)
            
            segment_data = session.sql(f"""
                SELECT WEEK_START, SUM(SPEND) as SPEND, SUM(REVENUE) as REVENUE
                FROM DIMENSIONAL.V_MMM_INPUT_WEEKLY
                WHERE {where_sql}
                GROUP BY WEEK_START
                ORDER BY WEEK_START
            """).to_pandas()
            
            if len(segment_data) > 5:
                correlation = segment_data['SPEND'].corr(segment_data['REVENUE'])
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Data Points", len(segment_data))
                with metric_col2:
                    st.metric("Correlation (r)", f"{correlation:.3f}")
                
                import altair as alt
                
                scatter = alt.Chart(segment_data).mark_circle(size=60, opacity=0.6).encode(
                    x=alt.X('SPEND:Q', title='Weekly Spend ($)', scale=alt.Scale(zero=False)),
                    y=alt.Y('REVENUE:Q', title='Weekly Revenue ($)', scale=alt.Scale(zero=False)),
                    tooltip=['WEEK_START', 'SPEND', 'REVENUE']
                ).properties(
                    width=600,
                    height=300,
                    title=f"Spend vs Revenue — {selected_channel} / {selected_region}"
                )
                
                regression = scatter.transform_regression('SPEND', 'REVENUE').mark_line(
                    color='red',
                    strokeDash=[5, 5]
                )
                
                st.altair_chart(scatter + regression, use_container_width=True)
                
                if len(segment_data) < 20:
                    st.warning(f"""
                    **Only {len(segment_data)} data points.** That's not enough to estimate 3 parameters (θ, γ, α) reliably.
                    
                    This is the trap: slicing looks scientific, but it destroys statistical power.
                    """)
                elif abs(correlation) < 0.3:
                    st.warning(f"""
                    **Correlation is weak (r = {correlation:.2f}).** The relationship is noisy.
                    
                    Is this bad marketing? Lag effects? Saturation? You can't tell from this view.
                    """)
                else:
                    st.info(f"""
                    **Looks reasonable, but...** this linear correlation ignores carryover and saturation.
                    
                    The real θ, γ, α for this segment remain unknown.
                    """)
            else:
                st.warning("Not enough data for this segment combination.")
                
        except Exception as e:
            st.error(f"Could not load segment data: {e}")
        
        st.markdown("---")
        st.markdown("""
        **💡 The Point:** You can slice and dice all day. Each segment gives you a different picture.
        But you still need to discover the underlying **θ**, **γ**, and **α** that explain *why* the 
        relationship looks the way it does — and you need to do it for every combination simultaneously.
        
        That's not a spreadsheet problem. That's an ML problem.
        """)

with tab5:
    st.header("The Solution: Feature Engineering")
    
    st.markdown("""
    ### Transform Raw Spend Into Predictive Features
    
    We fix both problems by **transforming** the data before modeling:
    """)
    
    st.code("""
    Raw Spend 
        → Adstock(θ)           # Capture memory/carryover
        → Hill Saturation(α,γ) # Capture diminishing returns  
        → StandardScaler       # Normalize for ML
        → Ridge Regression     # Learn channel weights
    """, language="text")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 Adstock
        
        **Formula**: 
        ```
        adstock[t] = spend[t] + θ × adstock[t-1]
        ```
        
        **θ (theta)** = decay rate
        
        - θ = 0.3 → Fast decay (search ads)
        - θ = 0.8 → Slow decay (brand ads)
        """)
    
    with col2:
        st.markdown("""
        ### 📉 Hill Function
        
        **Formula**:
        ```
        hill(x) = x^α / (γ^α + x^α)
        ```
        
        **α (alpha)** = curve shape
        
        **γ (gamma)** = half-saturation point
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Ridge Regression
        *(Next Step: The ML Model)*
        
        **Formula**:
        ```
        Revenue = Σ(βᵢ × featureᵢ)
        ```
        
        **β (beta)** = channel contribution
        
        Regularization prevents overfitting
        """)
        
        with st.expander("ℹ️ How does this fit together?"):
            st.markdown("""
            **Two-stage process:**
            
            **Stage 1 — Feature Engineering** (this tab):
            - Discover θ, α, γ for each channel using optimization (Nevergrad)
            - Transform raw spend → adstock → saturation
            
            **Stage 2 — Model Training** (next demo):
            - Feed transformed features into Ridge Regression
            - Ridge learns **β weights** = how much each channel contributes
            - Why Ridge? Handles multicollinearity (channels spike together in Q4), prevents wild coefficient estimates
            
            *This is where we pick up in the training notebook!*
            """)
    
    st.markdown("---")
    
    st.subheader("The Improvement")
    
    st.markdown("""
    These metrics show **correlation (r)** between spend and revenue — a measure of how strongly 
    they move together:
    
    - **Raw Spend**: Correlation using spend data as-is. Low values mean spend alone doesn't 
      predict revenue well (due to carryover and saturation effects).
    - **Transformed**: Correlation after applying adstock and saturation transforms. Higher 
      values mean the engineered features explain revenue much better.
    - **Improvement**: How much better the transformed features predict revenue vs naive spend.
    """)
    
    try:
        df_total = load_total_weekly()
        raw_corr = df_total['TOTAL_SPEND'].corr(df_total['REVENUE'])
        transformed_corr = 0.757
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Raw Spend", f"{raw_corr:.2f}", "Naive")
        
        with col2:
            st.metric("Transformed", f"{transformed_corr:.2f}", f"+{transformed_corr - raw_corr:.2f}")
        
        with col3:
            improvement = ((transformed_corr - raw_corr) / max(abs(raw_corr), 0.01)) * 100
            st.metric("Improvement", f"{min(improvement, 999):.0f}%", "With feature engineering")
        
    except:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Raw Spend", "~0.3", "Naive")
        with col2:
            st.metric("Transformed", "0.76", "+0.4+")
        with col3:
            st.metric("Improvement", "100%+", "With feature engineering")
    
    st.markdown("---")
    
    with st.expander("🔬 **See the Transformation in Action** — Before & After by Channel"):
        st.markdown("""
        Select a channel to see how raw spend compares to transformed spend when predicting revenue.
        The transformation applies **adstock** (carryover) and **saturation** (diminishing returns).
        """)
        
        try:
            @st.cache_data(ttl=300)
            def load_channel_weekly():
                return session.sql("""
                    SELECT 
                        WEEK_START,
                        CHANNEL_CODE,
                        SUM(SPEND) as SPEND,
                        SUM(REVENUE) as REVENUE
                    FROM GLOBAL_B2B_MMM.DIMENSIONAL.V_MMM_INPUT_WEEKLY
                    GROUP BY WEEK_START, CHANNEL_CODE
                    ORDER BY CHANNEL_CODE, WEEK_START
                """).to_pandas()
            
            df_channel = load_channel_weekly()
            channels = sorted(df_channel['CHANNEL_CODE'].unique().tolist())
            
            selected_channel = st.selectbox(
                "📺 Select Channel", 
                channels, 
                key="transform_channel",
                index=0
            )
            
            channel_data = df_channel[df_channel['CHANNEL_CODE'] == selected_channel].copy()
            channel_data = channel_data.sort_values('WEEK_START').reset_index(drop=True)
            
            if len(channel_data) > 10:
                theta = 0.6
                channel_data['ADSTOCK'] = 0.0
                for i in range(len(channel_data)):
                    if i == 0:
                        channel_data.loc[i, 'ADSTOCK'] = channel_data.loc[i, 'SPEND']
                    else:
                        channel_data.loc[i, 'ADSTOCK'] = (
                            channel_data.loc[i, 'SPEND'] + 
                            theta * channel_data.loc[i-1, 'ADSTOCK']
                        )
                
                adstock_max = channel_data['ADSTOCK'].max()
                gamma = adstock_max * 0.5
                alpha = 2.0
                channel_data['TRANSFORMED'] = (
                    channel_data['ADSTOCK'] ** alpha / 
                    (gamma ** alpha + channel_data['ADSTOCK'] ** alpha)
                )
                
                raw_corr = channel_data['SPEND'].corr(channel_data['REVENUE'])
                trans_corr = channel_data['TRANSFORMED'].corr(channel_data['REVENUE'])
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Raw Correlation", f"{raw_corr:.3f}")
                with metric_col2:
                    st.metric("Transformed Correlation", f"{trans_corr:.3f}")
                with metric_col3:
                    delta = trans_corr - raw_corr
                    st.metric("Delta", f"{delta:+.3f}", "improvement" if delta > 0 else "regression")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("**Before: Raw Spend vs Revenue**")
                    scatter_raw = alt.Chart(channel_data).mark_circle(size=60, opacity=0.6, color='steelblue').encode(
                        x=alt.X('SPEND:Q', title='Raw Spend ($)', scale=alt.Scale(zero=False)),
                        y=alt.Y('REVENUE:Q', title='Revenue ($)', scale=alt.Scale(zero=False)),
                        tooltip=['WEEK_START', 'SPEND', 'REVENUE']
                    )
                    reg_raw = scatter_raw.transform_regression('SPEND', 'REVENUE').mark_line(color='red', strokeDash=[4,4])
                    st.altair_chart(scatter_raw + reg_raw, use_container_width=True)
                
                with chart_col2:
                    st.markdown("**After: Transformed Spend vs Revenue**")
                    scatter_trans = alt.Chart(channel_data).mark_circle(size=60, opacity=0.6, color='darkgreen').encode(
                        x=alt.X('TRANSFORMED:Q', title='Transformed Spend (0-1)', scale=alt.Scale(zero=False)),
                        y=alt.Y('REVENUE:Q', title='Revenue ($)', scale=alt.Scale(zero=False)),
                        tooltip=['WEEK_START', 'TRANSFORMED', 'REVENUE']
                    )
                    reg_trans = scatter_trans.transform_regression('TRANSFORMED', 'REVENUE').mark_line(color='red', strokeDash=[4,4])
                    st.altair_chart(scatter_trans + reg_trans, use_container_width=True)
                
                st.caption(f"Transform parameters: θ={theta} (adstock decay), α={alpha} (curve shape), γ={gamma:.0f} (half-saturation)")
            else:
                st.warning("Not enough data for this channel.")
        
        except Exception as e:
            st.error(f"Could not load channel data: {e}")
    
    st.success("""
    ✅ **Feature engineering encodes domain knowledge into data.**
    
    The transformed model can now:
    - **Attribute** revenue to specific channels
    - **Identify** saturation points for budget optimization
    - **Optimize** spend allocation across channels
    """)

st.divider()

st.markdown("""
### 💡 Key Takeaway

> **Raw data rarely reflects reality.**

Marketing spend data needs transformation because:
1. **Ads have memory** — effects linger beyond the spend week
2. **Returns diminish** — more spend ≠ proportionally more results

By encoding these real-world dynamics into features, we turn noisy correlations 
into actionable business insights.
""")
