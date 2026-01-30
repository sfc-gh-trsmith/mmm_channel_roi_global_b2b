import pandas as pd
import numpy as np
import random
import os
import datetime
from datetime import timedelta
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# --- Constants & Configuration ---
RANDOM_SEED = 42
OUTPUT_DIR = "data/synthetic"
BRIEFS_DIR = os.path.join(OUTPUT_DIR, "campaign_briefs")
START_DATE = datetime.date(2020, 1, 1)  # Extended to 5 years for better MMM training
END_DATE = datetime.date(2024, 12, 31)
DAYS = (END_DATE - START_DATE).days + 1

# =============================================================================
# COMPANY SCALE PARAMETERS
# Hypothetical: $24.6B annual B2B revenue, 60,000 customers
# Industry benchmark: B2B advertising spend typically 0.4-1% of revenue
# For $24.6B company: ~$100M - $250M annual advertising spend
# Target: ~$175M/year (middle of range) = ~$525M over 3-year dataset
# Target blended ROAS: 3-5x (realistic for B2B industrial)
# =============================================================================
ANNUAL_ADVERTISING_BUDGET = 175_000_000  # $175M annually (0.7% of revenue)
DAILY_SPEND_BUDGET = ANNUAL_ADVERTISING_BUDGET / 365  # ~$480K/day total across all campaigns

# =============================================================================
# B2B CHANNEL PERFORMANCE CONFIG (10 Channels for MMM Demo)
# 
# Realistic B2B channel mix with clear performance tiers:
#   Tier 1 (Best): LinkedIn, Google Ads, Trade Publications
#   Tier 2 (Good): Microsoft Ads, YouTube, Programmatic
#   Tier 3 (Marginal): Meta (Instagram), X.com
#   Tier 4 (Unprofitable): TikTok, Meta (Facebook)
#
# OPTIMIZED FOR DEMO: Tighter lags, lower noise, clearer tier separation
# Target: Model R² > 0.85, ROIs in 0.7x-3.5x range (realistic B2B)
# =============================================================================
B2B_CHANNEL_PERFORMANCE = {
    # === TIER 1: BEST PERFORMERS (ROI 2.5-3.5x) ===
    # Very short lags for strong MMM signal detection
    "LinkedIn": {
        "roas_target": 3.2,            # Best B2B channel - decision makers
        "win_rate": 0.75,              # High win rate
        "deal_size_multiplier": 1.30,  # Larger enterprise deals
        "spend_weight": 1.8,           # Highest budget allocation
        "cpm": 75,                     # Premium B2B targeting
        "ctr": 0.004,                  # Lower CTR but higher quality
        "opp_lag_min": 1,              # Very short - 1 week
        "opp_lag_max": 7,              # Max 1 week
        "cycle_min": 7,                # 1 week sales cycle
        "cycle_max": 14,               # Max 2 weeks
        "rev_lag_min": 1,              # Immediate recognition
        "rev_lag_max": 7,              # Max 1 week
        "noise_factor": 0.10,          # Low noise for clean signal
    },
    "Google Ads": {
        "roas_target": 2.8,            # Second best - high intent search
        "win_rate": 0.70,              # Strong conversion
        "deal_size_multiplier": 1.20,  # Good deals
        "spend_weight": 1.5,           # Significant budget
        "cpm": 25,                     # Search/Display blend
        "ctr": 0.025,                  # High search intent CTR
        "opp_lag_min": 1,              # Very short
        "opp_lag_max": 7,              # Max 1 week
        "cycle_min": 7,                # 1 week
        "cycle_max": 14,               # Max 2 weeks
        "rev_lag_min": 1,              # Immediate
        "rev_lag_max": 7,              # Max 1 week
        "noise_factor": 0.12,          # Low noise
    },
    "Trade Publications": {
        "roas_target": 2.5,            # Industry-specific media - strong B2B
        "win_rate": 0.65,              # Good targeted audience
        "deal_size_multiplier": 1.25,  # Industry buyers pay premium
        "spend_weight": 1.0,           # Solid investment
        "cpm": 35,                     # Premium industry placement
        "ctr": 0.003,                  # Engaged readers
        "opp_lag_min": 1,              # Short lag
        "opp_lag_max": 10,             # Slightly longer for print
        "cycle_min": 10,               # ~1.5 weeks
        "cycle_max": 21,               # Max 3 weeks
        "rev_lag_min": 3,              # Quick recognition
        "rev_lag_max": 10,             # Max ~1.5 weeks
        "noise_factor": 0.15,          # Moderate noise
    },
    # === TIER 2: GOOD PERFORMERS (ROI 1.5-2.2x) ===
    "Microsoft Ads": {
        "roas_target": 2.0,            # Bing search + LinkedIn audience
        "win_rate": 0.60,              # Good B2B conversion
        "deal_size_multiplier": 1.10,  # Solid deals
        "spend_weight": 0.9,           # Growing channel
        "cpm": 20,                     # Lower than Google
        "ctr": 0.020,                  # Good search intent
        "opp_lag_min": 3,              # Short lag
        "opp_lag_max": 10,             # ~1.5 weeks
        "cycle_min": 10,               # ~1.5 weeks
        "cycle_max": 21,               # Max 3 weeks
        "rev_lag_min": 3,              # Quick
        "rev_lag_max": 10,             # Max ~1.5 weeks
        "noise_factor": 0.15,          # Moderate noise
    },
    "YouTube": {
        "roas_target": 1.8,            # Video content marketing
        "win_rate": 0.55,              # Moderate conversion
        "deal_size_multiplier": 1.05,  # Standard deals
        "spend_weight": 0.8,           # Growing investment
        "cpm": 18,                     # Video rates
        "ctr": 0.005,                  # Video engagement
        "opp_lag_min": 3,              # Short
        "opp_lag_max": 14,             # ~2 weeks
        "cycle_min": 14,               # 2 weeks
        "cycle_max": 28,               # Max 4 weeks
        "rev_lag_min": 3,              # Quick
        "rev_lag_max": 14,             # Max 2 weeks
        "noise_factor": 0.18,          # Moderate noise
    },
    "Programmatic": {
        "roas_target": 1.6,            # Brand awareness / display
        "win_rate": 0.50,              # Moderate conversion
        "deal_size_multiplier": 1.00,  # Standard deals
        "spend_weight": 0.7,           # Moderate budget
        "cpm": 15,                     # DSP rates
        "ctr": 0.001,                  # Display rates
        "opp_lag_min": 3,              # Short
        "opp_lag_max": 14,             # ~2 weeks
        "cycle_min": 14,               # 2 weeks
        "cycle_max": 28,               # Max 4 weeks
        "rev_lag_min": 3,              # Quick
        "rev_lag_max": 14,             # Max 2 weeks
        "noise_factor": 0.18,          # Moderate noise
    },
    # === TIER 3: MARGINAL PERFORMERS (ROI 1.0-1.3x) ===
    "Meta (Instagram)": {
        "roas_target": 1.2,            # Slightly profitable - visual B2B
        "win_rate": 0.42,              # Lower B2B fit
        "deal_size_multiplier": 0.90,  # Smaller deals
        "spend_weight": 0.4,           # Limited B2B investment
        "cpm": 14,                     # Social rates
        "ctr": 0.006,                  # Visual engagement
        "opp_lag_min": 7,              # Longer lag
        "opp_lag_max": 21,             # ~3 weeks
        "cycle_min": 21,               # 3 weeks
        "cycle_max": 35,               # Max 5 weeks
        "rev_lag_min": 7,              # Slower recognition
        "rev_lag_max": 21,             # Max 3 weeks
        "noise_factor": 0.22,          # Higher noise
    },
    "X.com": {
        "roas_target": 1.1,            # Near breakeven - B2B thought leadership
        "win_rate": 0.38,              # Lower conversion
        "deal_size_multiplier": 0.85,  # Smaller deals
        "spend_weight": 0.3,           # Declining platform
        "cpm": 10,                     # Lower rates
        "ctr": 0.004,                  # Engagement
        "opp_lag_min": 7,              # Longer lag
        "opp_lag_max": 21,             # ~3 weeks
        "cycle_min": 21,               # 3 weeks
        "cycle_max": 35,               # Max 5 weeks
        "rev_lag_min": 7,              # Slower
        "rev_lag_max": 21,             # Max 3 weeks
        "noise_factor": 0.25,          # Higher noise
    },
    # === TIER 4: UNPROFITABLE (ROI 0.7-0.9x) ===
    "TikTok": {
        "roas_target": 0.85,           # Unprofitable - poor B2B fit
        "win_rate": 0.30,              # Poor B2B conversion
        "deal_size_multiplier": 0.70,  # Smallest deals
        "spend_weight": 0.25,          # Experimental budget
        "cpm": 8,                      # Low rates
        "ctr": 0.008,                  # High engagement, low intent
        "opp_lag_min": 14,             # Long lag
        "opp_lag_max": 35,             # ~5 weeks
        "cycle_min": 28,               # 4 weeks
        "cycle_max": 49,               # Max 7 weeks
        "rev_lag_min": 10,             # Slow recognition
        "rev_lag_max": 28,             # Max 4 weeks
        "noise_factor": 0.30,          # High noise - unreliable
    },
    "Meta (Facebook)": {
        "roas_target": 0.75,           # UNPROFITABLE - worst B2B fit
        "win_rate": 0.25,              # Very low B2B conversion
        "deal_size_multiplier": 0.65,  # Smallest deals
        "spend_weight": 0.3,           # Should reduce spend
        "cpm": 12,                     # Low B2B rates
        "ctr": 0.007,                  # Social engagement
        "opp_lag_min": 14,             # Long lag
        "opp_lag_max": 35,             # ~5 weeks
        "cycle_min": 28,               # 4 weeks
        "cycle_max": 49,               # Max 7 weeks
        "rev_lag_min": 10,             # Slow recognition
        "rev_lag_max": 28,             # Max 4 weeks
        "noise_factor": 0.30,          # High noise - unreliable
    },
}

# Legacy compatibility - extract ROAS targets
CHANNEL_ROAS_TARGETS = {ch: cfg["roas_target"] for ch, cfg in B2B_CHANNEL_PERFORMANCE.items()}

# Taxonomy Components
BGS = ["SIBG", "HCBG", "TEBG", "CBG"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
DIVISIONS = {
    "SIBG": ["ASD", "PSD"],
    "HCBG": ["MSD", "HIS"],
    "TEBG": ["EMSD", "AD"],
    "CBG": ["CHIM", "HIC"]
}
TYPES = ["Brand", "LeadGen", "Nurture"]
CHANNELS = [
    "LinkedIn", "Google Ads", "Microsoft Ads",  # Tier 1
    "YouTube", "Programmatic", "Trade Publications",  # Tier 2
    "Meta (Instagram)", "X.com", "TikTok",  # Tier 3
    "Meta (Facebook)"  # Tier 4 (Unprofitable)
]

# --- Initialization ---
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

def ensure_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BRIEFS_DIR, exist_ok=True)

# --- Helper Functions ---

def get_fiscal_week(date_obj):
    return f"{date_obj.year}-W{date_obj.isocalendar()[1]:02d}"

def generate_campaigns(num_campaigns=50):
    campaigns = []
    
    # Channel weights by campaign type (10 channels)
    # Order: LinkedIn, Google, Microsoft, YouTube, Programmatic, Trade Pubs, 
    #        Meta(IG), X.com, TikTok, Meta(FB)
    CHANNEL_WEIGHTS = {
        "Brand": [0.20, 0.05, 0.05, 0.15, 0.20, 0.15, 0.05, 0.05, 0.05, 0.05],
        "LeadGen": [0.25, 0.25, 0.15, 0.08, 0.08, 0.05, 0.04, 0.04, 0.03, 0.03],
        "Nurture": [0.30, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.04, 0.03, 0.03],
    }
    
    for i in range(num_campaigns):
        bg = random.choice(BGS)
        region = random.choice(REGIONS)
        division = random.choice(DIVISIONS[bg])
        ctype = random.choice(TYPES)
        cid = f"CMP-{100+i}"
        # Taxonomy: [BG]_[Region]_[Division]_[CampaignType]_[ID]
        name = f"{bg}_{region}_{division}_{ctype}_{cid}"
        
        # Assign channel based on campaign type preferences
        weights = CHANNEL_WEIGHTS.get(ctype, CHANNEL_WEIGHTS["LeadGen"])
        channel = random.choices(CHANNELS, weights=weights)[0]
            
        campaigns.append({
            "CAMPAIGN_ID": cid,
            "CAMPAIGN_NAME": name,
            "BG": bg,
            "REGION": region,
            "DIVISION": division,
            "TYPE": ctype,
            "CHANNEL": channel,
            "START_DATE": START_DATE + timedelta(days=random.randint(0, DAYS - 90)),
            "DURATION": random.randint(30, 90) # Days
        })
    return pd.DataFrame(campaigns)

def generate_spend(campaigns_df):
    """
    Generate realistic daily spend data for a $24.6B enterprise.
    Target: ~$175M/year in B2B advertising = ~$480K/day across all active campaigns.
    Over 3 years: ~$525M total spend.
    
    Uses B2B_CHANNEL_PERFORMANCE config for channel-specific parameters.
    """
    spend_records = []
    
    # Campaign type weights (not channel-specific)
    TYPE_SPEND_WEIGHTS = {
        "Brand": 1.4,         # Larger brand campaigns
        "LeadGen": 1.0,       # Standard
        "Nurture": 0.5        # Smaller, targeted
    }
    
    for _, camp in campaigns_df.iterrows():
        start = camp["START_DATE"]
        channel = camp["CHANNEL"]
        
        # Get channel config from B2B_CHANNEL_PERFORMANCE
        ch_config = B2B_CHANNEL_PERFORMANCE.get(channel, {
            "spend_weight": 1.0, "cpm": 20, "ctr": 0.01
        })
        
        channel_weight = ch_config.get("spend_weight", 1.0)
        type_weight = TYPE_SPEND_WEIGHTS.get(camp["TYPE"], 1.0)
        
        # Base daily spend: $50K-$100K per campaign, adjusted by weights
        # This gives ~$175M/year with ~20-25 campaigns active at any time
        base_daily = np.random.uniform(50000, 100000) * channel_weight * type_weight
        
        for day_offset in range(camp["DURATION"]):
            current_date = start + timedelta(days=day_offset)
            if current_date > END_DATE:
                break
                
            # Seasonal multiplier (Q1/Q3 spikes for B2B budget cycles)
            month = current_date.month
            seasonality = 1.0
            if month in [1, 2, 3, 7, 8, 9]:  # Q1 and Q3
                seasonality = 1.4
            elif month in [11, 12]:  # Year-end budget flush
                seasonality = 1.2
            
            # Get CPM from channel config
            cpm = ch_config.get("cpm", 20)
            
            # Daily spend with noise
            daily_spend = base_daily * seasonality * np.random.uniform(0.7, 1.3)
            
            # Injection: Healthcare LinkedIn Boost (pharma/medical premium)
            if camp["BG"] == "HCBG" and channel == "LinkedIn":
                daily_spend *= 1.25
                
            impressions = int((daily_spend / cpm) * 1000)
            
            # Get CTR from channel config
            ctr = ch_config.get("ctr", 0.01)
            
            clicks = int(impressions * ctr * np.random.uniform(0.8, 1.2))
            
            spend_records.append({
                "DATE": current_date,
                "CAMPAIGN_ID": camp["CAMPAIGN_ID"],
                "CHANNEL": camp["CHANNEL"],
                "SPEND_AMT": round(daily_spend, 2),
                "IMPRESSIONS": impressions,
                "CLICKS": clicks,
                "VIDEO_VIEWS_50": int(impressions * 0.15) if camp["TYPE"] == "Brand" else 0
            })
            
    return pd.DataFrame(spend_records)

def generate_opportunities(spend_df, campaigns_df):
    """
    Generate opportunities with STRONG spend→revenue correlation for MMM demo.
    
    KEY DESIGN FOR MODEL FIT:
    - Direct spend→revenue relationship: revenue ~ spend * ROAS_target
    - Very short lags (1-3 weeks total from spend to revenue)
    - Low noise on high-performing channels, high noise on poor channels
    - Deterministic base with controlled variance
    
    Uses B2B_CHANNEL_PERFORMANCE config for channel-specific parameters.
    """
    opps = []
    campaign_lookup = campaigns_df.set_index("CAMPAIGN_ID").to_dict("index")
    
    # Process ALL spend records for better coverage
    for _, row in spend_df.iterrows():
        camp = campaign_lookup[row["CAMPAIGN_ID"]]
        channel = camp["CHANNEL"]
        
        # Get channel config from B2B_CHANNEL_PERFORMANCE
        ch_config = B2B_CHANNEL_PERFORMANCE.get(channel, {
            "win_rate": 0.50, "deal_size_multiplier": 1.0, "roas_target": 2.0,
            "opp_lag_min": 7, "opp_lag_max": 21, "cycle_min": 14, "cycle_max": 28,
            "noise_factor": 0.20
        })
        
        # DETERMINISTIC opportunity generation based on spend
        # This creates a direct spend→revenue relationship the model can detect
        roas_target = ch_config.get("roas_target", 2.0)
        noise_factor = ch_config.get("noise_factor", 0.20)
        
        # Expected revenue from this day's spend
        expected_revenue = row["SPEND_AMT"] * roas_target
        
        # Number of opps scaled by expected revenue (~$150K per opp average)
        # Higher ROAS channels generate more/larger opps
        num_opps = max(1, int(expected_revenue / 150000))
        
        for _ in range(num_opps):
            # Use channel-specific SHORT lags
            opp_lag_min = ch_config.get("opp_lag_min", 7)
            opp_lag_max = ch_config.get("opp_lag_max", 21)
            lag_days = random.randint(opp_lag_min, opp_lag_max)
            created_date = row["DATE"] + timedelta(days=lag_days)
            
            if created_date > END_DATE:
                continue
                
            # Use channel-specific win rate
            win_rate = ch_config.get("win_rate", 0.50)
            
            # Campaign type adjustment (smaller effect)
            if camp["TYPE"] == "LeadGen":
                win_rate = min(0.90, win_rate * 1.05)
            elif camp["TYPE"] == "Brand":
                win_rate *= 0.98
                
            # Simplified stage distribution
            closed_lost_rate = max(0.05, 1.0 - win_rate - 0.05)
            
            stage = np.random.choice(
                ["Closed Won", "Closed Lost", "Negotiation"], 
                p=[win_rate, closed_lost_rate, 0.05]
            )
            
            # Use channel-specific SHORT sales cycles
            cycle_min = ch_config.get("cycle_min", 14)
            cycle_max = ch_config.get("cycle_max", 28)
            cycle_days = random.randint(cycle_min, cycle_max)
            close_date = created_date + timedelta(days=cycle_days)
            
            # Deal size: based on expected revenue per opp with controlled noise
            # Lower noise_factor = more predictable = better model fit
            base_deal = expected_revenue / num_opps
            deal_amount = base_deal * (1 + np.random.normal(0, noise_factor))
            deal_amount = max(25000, deal_amount)  # Floor at $25K
            
            # Apply channel-specific deal size multiplier
            deal_amount *= ch_config.get("deal_size_multiplier", 1.0)
                
            # Business group adjustments (smaller effect)
            if camp["BG"] == "HCBG":
                deal_amount *= 1.05
            
            opps.append({
                "OPPORTUNITY_ID": f"OPP-{fake.uuid4()[:8]}",
                "ACCOUNT_NAME": fake.company(),
                "LEAD_SOURCE_CAMPAIGN": row["CAMPAIGN_ID"],
                "STAGE": stage,
                "AMOUNT_USD": round(deal_amount, 2),
                "CREATED_DATE": created_date,
                "CLOSE_DATE": close_date if stage in ["Closed Won", "Closed Lost"] else None,
                "BUSINESS_GROUP": camp["BG"],
                "REGION": camp["REGION"]
            })
        
    return pd.DataFrame(opps)

def generate_revenue(opps_df, campaigns_df):
    """
    Generate revenue/invoices from Closed Won opportunities.
    
    OPTIMIZED FOR MMM DEMO:
    Revenue recognition is IMMEDIATE to create strong spend→revenue correlation:
    - 80% of deals: 1-7 days after close (single invoice)
    - 20% of deals: 7-14 days after close
    
    Total spend→revenue lag should be 2-4 weeks for Tier 1 channels.
    """
    won_opps = opps_df[opps_df["STAGE"] == "Closed Won"]
    
    # Build campaign lookup to get channel for revenue lag
    campaign_lookup = campaigns_df.set_index("CAMPAIGN_ID").to_dict("index")
    
    # Build opp to campaign lookup via LEAD_SOURCE_CAMPAIGN
    opp_to_campaign = opps_df.set_index("OPPORTUNITY_ID")["LEAD_SOURCE_CAMPAIGN"].to_dict()
    
    invoices = []
    
    for _, opp in won_opps.iterrows():
        total_amt = opp["AMOUNT_USD"]
        
        # Get channel-specific revenue lag from config
        campaign_id = opp_to_campaign.get(opp["OPPORTUNITY_ID"])
        if campaign_id and campaign_id in campaign_lookup:
            channel = campaign_lookup[campaign_id]["CHANNEL"]
            ch_config = B2B_CHANNEL_PERFORMANCE.get(channel, {})
            rev_lag_min = ch_config.get("rev_lag_min", 1)
            rev_lag_max = ch_config.get("rev_lag_max", 7)
        else:
            rev_lag_min = 1
            rev_lag_max = 7
        
        # Simplified: 80% immediate (single invoice), 20% slightly delayed
        deal_type = np.random.choice(
            ["immediate", "standard"],
            p=[0.80, 0.20]
        )
        
        if deal_type == "immediate":
            # Single invoice, very quick recognition
            num_invoices = 1
            base_lag = random.randint(rev_lag_min, rev_lag_max)
            lag_increment = 0
        else:
            # Standard: single invoice, slightly longer
            num_invoices = 1
            base_lag = random.randint(rev_lag_max, rev_lag_max + 7)
            lag_increment = 0
        
        for i in range(num_invoices):
            lag_days = base_lag + (lag_increment * i)
            inv_date = opp["CLOSE_DATE"] + timedelta(days=lag_days)
            
            # Keep revenue within the data window
            if inv_date > END_DATE + timedelta(days=30):
                continue
            
            # Cap at END_DATE for reporting
            if inv_date > END_DATE:
                inv_date = END_DATE
                
            invoices.append({
                "INVOICE_ID": f"INV-{fake.uuid4()[:8]}",
                "BOOKED_REVENUE": round(total_amt / num_invoices, 2),
                "PROFIT_CENTER": f"PC_{opp['BUSINESS_GROUP']}_{opp['REGION']}",
                "POSTING_DATE": inv_date,
                "OPPORTUNITY_ID": opp["OPPORTUNITY_ID"]
            })
            
    return pd.DataFrame(invoices)

def generate_macro_data():
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    data = []
    
    for d in dates:
        # PMI varies between 45 and 60 with sine wave trend
        pmi = 52 + 5 * np.sin(d.toordinal() / 365.0 * 2 * np.pi) + np.random.normal(0, 0.5)
        
        # Competitor SOV - Inverse to our spend (simplified)
        comp_sov = np.random.uniform(0.1, 0.4)
        
        if d.weekday() == 0: # Weekly grain usually, but daily for file
            data.append({
                "DATE": d,
                "PMI_INDEX": round(pmi, 2),
                "COMPETITOR_SOV": round(comp_sov, 3),
                "REGION": "NA" # Simplified
            })
            
    return pd.DataFrame(data)

def generate_campaign_briefs(campaigns_df):
    # Generate simple PDF briefs for Cortex Search
    for _, camp in campaigns_df.iterrows():
        filename = f"{camp['CAMPAIGN_ID']}_Brief.pdf"
        filepath = os.path.join(BRIEFS_DIR, filename)
        
        c = canvas.Canvas(filepath, pagesize=letter)
        c.drawString(100, 750, f"Campaign Strategy Brief: {camp['CAMPAIGN_NAME']}")
        c.drawString(100, 730, f"ID: {camp['CAMPAIGN_ID']}")
        c.drawString(100, 710, f"Objective: Drive {camp['BG']} growth in {camp['REGION']}")
        c.drawString(100, 690, f"Channel Strategy: Heavy investment in {camp['CHANNEL']} to target decision makers.")
        c.drawString(100, 670, f"Key Message: 'Innovation in {camp['DIVISION']} leads to safety and efficiency.'")
        c.drawString(100, 650, f"Target Audience: Procurement Managers, Engineers in {camp['REGION']}.")
        c.save()

# --- Main Execution ---

def main():
    print(f"Generating synthetic data to {OUTPUT_DIR}...")
    print(f"Target: Strong spend→revenue correlation for MMM model")
    ensure_directories()
    
    print("1. Generating Campaigns...")
    campaigns_df = generate_campaigns(150)  # Increased from 75 for better revenue coverage
    campaigns_df.to_csv(os.path.join(OUTPUT_DIR, "campaign_metadata.csv"), index=False)
    
    print("2. Generating Spend (Sprinklr)...")
    spend_df = generate_spend(campaigns_df)
    spend_df.to_csv(os.path.join(OUTPUT_DIR, "sprinklr_spend.csv"), index=False)
    print(f"   Total spend: ${spend_df['SPEND_AMT'].sum()/1e6:.1f}M")
    
    print("3. Generating Opportunities (Salesforce)...")
    opps_df = generate_opportunities(spend_df, campaigns_df)
    opps_df.to_csv(os.path.join(OUTPUT_DIR, "salesforce_opps.csv"), index=False)
    won_count = len(opps_df[opps_df['STAGE'] == 'Closed Won'])
    print(f"   Total opps: {len(opps_df)}, Closed Won: {won_count} ({100*won_count/len(opps_df):.1f}%)")
    
    print("4. Generating Revenue (SAP)...")
    rev_df = generate_revenue(opps_df, campaigns_df)
    rev_df.to_csv(os.path.join(OUTPUT_DIR, "sap_revenue.csv"), index=False)
    print(f"   Total revenue: ${rev_df['BOOKED_REVENUE'].sum()/1e6:.1f}M")
    
    # Calculate and display expected ROAS
    total_spend = spend_df['SPEND_AMT'].sum()
    total_rev = rev_df['BOOKED_REVENUE'].sum()
    print(f"   Expected overall ROAS: {total_rev/total_spend:.2f}x")
    
    print("5. Generating Macro Indicators...")
    macro_df = generate_macro_data()
    macro_df.to_csv(os.path.join(OUTPUT_DIR, "macro_indicators.csv"), index=False)
    
    print("6. Generating Campaign Briefs (PDFs)...")
    generate_campaign_briefs(campaigns_df)
    
    print("\nData generation complete.")
    print("Next: Run ./clean.sh --force && ./deploy.sh && ./run.sh main")

if __name__ == "__main__":
    main()

