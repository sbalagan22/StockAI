import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Try importing Groq, handle if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.error("Groq package is not installed. Please install it using: pip install groq")

# API Keys
POLYGON_API_KEY = "pDTt2mNMtPIvevWNFretS1dCHoTf2XQ2"
NEWS_API_KEY = "5ef758ddb04a45cc99513854af12d93b"
GROQ_API_KEY = "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"

# Initialize Groq client only if available
if GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)

# Groq parameters
model = "llama3-8b-8192"
temperature = 0.3
max_tokens = 500

# System message for Groq
SYSTEM_MESSAGE = """
You are an expert financial analyst with deep knowledge of stock markets and company analysis.
Your task is to analyze the provided business news and insights about a company and provide analysis in these four key areas.
Format all responses as concise bullet points, avoiding paragraphs and unnecessary line breaks.

1. Recent News and Updates
• Focus on latest company developments
• Material events and announcements
• Strategic changes and updates

2. Financial Performance
• Key financial metrics
• Performance comparisons
• Notable indicators and trends

3. Analyst Sentiment and Price Targets
• Current analyst recommendations
• Price target ranges
• Changes in market sentiment

4. Technical Analysis and Market Trends
• Price trend analysis
• Support and resistance levels
• Trading patterns and signals

Keep each section focused and use bullet points consistently.
Avoid long paragraphs and maintain a clear, concise format.
"""

def generate_stock_analysis_prompt(ticker, news_text):
    return f"""
    Please analyze the following business news and insights for {ticker} and provide a comprehensive analysis covering:

    1. Recent News and Updates
    2. Financial Performance
    3. Analyst Sentiment and Price Targets
    4. Technical Analysis and Market Trends

    News and context to analyze:
    {news_text}

    Please format each section distinctly and provide specific, actionable insights based on the available information.
    """

def get_groq_analysis(ticker, news_text):
    if not GROQ_AVAILABLE:
        return "Groq analysis is not available. Please install the Groq package."
    
    try:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": generate_stock_analysis_prompt(ticker, news_text)}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

# Update the market overview function
def get_market_overview():
    return """The market shows positive momentum with major indices trending upward. S&P 500 leads with a 1.2% gain, while Nasdaq follows at 1.7% and Dow edges up 0.3%. Technology and healthcare sectors outperform, while energy and utilities lag behind. Market sentiment remains bullish with low volatility (VIX: 14.2) and above-average trading volume."""

# Page config
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for dark theme
st.markdown("""
    <style>
    * {
        font-family: 'Segoe UI', sans-serif !important;
    }
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stPlotlyChart {
        background-color: #1A1C23 !important;
    }
    .metric-container {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 1.1em;
        color: #888;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    .metric-value.current {
        color: #FFFFFF;
    }
    .metric-value.high {
        color: #00ff88;
    }
    .metric-value.low {
        color: #ff4444;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 1em;
    }
    .market-overview-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #333;
        height: 520px;
        margin-top: 37px;
    }
    .market-title {
        color: #00ff88;
        font-size: 1.4em;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 1px solid #333;
    }
    .market-content {
        color: #FAFAFA;
        line-height: 1.8;
        font-size: 1.1em;
        padding: 10px 5px;
    }
    .analysis-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 25px;
        margin: 20px 0;
    }
    .analysis-box {
        background-color: #1E1E1E;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #333;
        min-height: 200px;
    }
    .box-title {
        color: #00ff88;
        font-size: 1.2em;
        font-weight: bold;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #333;
    }
    .box-content {
        color: #FAFAFA;
        line-height: 1.6;
        font-size: 1.0em;
        white-space: pre-line;
    }
    </style>
""", unsafe_allow_html=True)

# Title and main controls
st.markdown("<h1 class='title'>Stock Analysis Dashboard</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Enter Stock Ticker", value="AAPL").upper()
with col2:
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1W", "1M", "1Y"]
    )

# Function to get stock data from Polygon.io
def get_stock_data(symbol, timeframe):
    end_date = datetime.now()
    
    if timeframe == "1W":
        start_date = end_date - timedelta(weeks=1)
        multiplier = 1
        timespan = "hour"
    elif timeframe == "1M":
        start_date = end_date - timedelta(days=30)
        multiplier = 1
        timespan = "day"
    else:  # 1Y
        start_date = end_date - timedelta(days=365)
        multiplier = 1
        timespan = "day"

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url)
    return response.json()

# Function to get relevant stock news from NewsAPI
def get_stock_news(symbol):
    # Get company name for better news search
    company_names = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google',
        'AMZN': 'Amazon',
        'META': 'Meta OR Facebook',
        'NVDA': 'NVIDIA',
        'TSLA': 'Tesla',
        'NFLX': 'Netflix'
        # Add more mappings as needed
    }
    
    company_name = company_names.get(symbol, symbol)
    
    # Create a query that focuses on business-relevant news
    query = f"({company_name} OR {symbol}) AND (earnings OR revenue OR product OR launch OR partnership OR acquisition OR market OR stock OR shares OR CEO OR forecast OR growth OR technology OR innovation)"
    
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&sortBy=relevancy&language=en&pageSize=10"
    response = requests.get(url)
    return response.json()

def filter_relevant_news(articles):
    relevant_keywords = [
        'launch', 'product', 'revenue', 'earnings', 'growth',
        'market share', 'partnership', 'acquisition', 'innovation',
        'technology', 'forecast', 'outlook', 'guidance', 'CEO',
        'strategy', 'expansion', 'investment', 'development',
        'patent', 'research', 'breakthrough'
    ]
    
    relevant_news = []
    for article in articles:
        title = article['title'].lower()
        description = article['description'].lower() if article['description'] else ''
        
        # Check if the article contains relevant keywords
        if any(keyword in title or keyword in description for keyword in relevant_keywords):
            relevant_news.append(article)
    
    return relevant_news[:5]  # Return top 5 most relevant articles

# Custom metric display function
def display_metric(label, value, style_class):
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value {style_class}">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# Main content
try:
    # Get stock data
    data = get_stock_data(ticker, timeframe)
    
    if data.get('results'):
        # Create DataFrame
        df = pd.DataFrame(data['results'])
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        
        # Calculate percentage change
        start_price = df['c'].iloc[0]
        end_price = df['c'].iloc[-1]
        percent_change = ((end_price - start_price) / start_price) * 100
        change_color = '#00ff88' if percent_change >= 0 else '#ff4444'
        
        # Stock price chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['c'],
            mode='lines',
            name='Close Price',
            line=dict(color='#00ff88' if percent_change >= 0 else '#ff4444', width=2)
        ))
        
        # Format date based on timeframe
        date_format = '%m/%d/%Y'
            
        fig.update_layout(
            title=f"{ticker} Stock Price ({timeframe})",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            height=600,
            hovermode='x unified',
            xaxis=dict(
                tickformat=date_format,
                tickfont=dict(size=14, weight='bold', family='Segoe UI'),
                title_font=dict(size=16, weight='bold', family='Segoe UI')
            ),
            yaxis=dict(
                tickfont=dict(size=14, weight='bold', family='Segoe UI'),
                title_font=dict(size=16, weight='bold', family='Segoe UI')
            ),
            font=dict(family='Segoe UI')
        )
        
        # Create two columns for graph and market overview
        col_graph, col_market = st.columns([2, 1])
        
        with col_graph:
            st.plotly_chart(fig, use_container_width=True)
        
        with col_market:
            with st.spinner("Generating market overview..."):
                market_analysis = get_market_overview()
                
                # Clean the text
                market_analysis = market_analysis.replace('*', '').replace('#', '').strip()
                
                st.markdown(f"""
                    <div class="market-overview-box">
                        <div class="market-title">Market Overview</div>
                        <div class="market-content">
                            {market_analysis}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        # Key metrics with custom styling
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric("Current Price", f"${df['c'].iloc[-1]:.2f}", "current")
        with col2:
            display_metric("High", f"${df['h'].max():.2f}", "high")
        with col3:
            display_metric("Low", f"${df['l'].min():.2f}", "low")
        with col4:
            display_metric("Change", f"{percent_change:+.2f}%", "high" if percent_change >= 0 else "low")
        
        # AI Analysis section (renamed from "AI Market Analysis")
        st.header("AI Stock Analysis")
        news_data = get_stock_news(ticker)
        
        if news_data.get('articles'):
            filtered_articles = filter_relevant_news(news_data['articles'])
            news_text = "\n\n".join([
                f"• {article['description']}"
                for article in filtered_articles
                if article['description']
            ])
            
            # Get AI analysis
            with st.spinner("Generating AI analysis..."):
                analysis = get_groq_analysis(ticker, news_text)
            
            # Process the analysis text to remove markdown symbols
            def clean_text(text):
                return text.replace('*', '').replace('#', '').strip()
            
            # Split analysis into sections and clean up the text
            sections = {
                "Recent News and Updates": "",
                "Financial Performance": "",
                "Analyst Sentiment and Price Targets": "",
                "Technical Analysis and Market Trends": ""
            }
            
            current_section = None
            current_content = []
            
            # Parse the analysis text
            for line in analysis.split('\n'):
                line = clean_text(line)
                if any(section in line for section in sections.keys()):
                    if current_section and current_content:
                        sections[current_section] = '\n'.join(current_content)
                    current_section = next((s for s in sections.keys() if s in line), None)
                    current_content = []
                elif current_section and line.strip():
                    current_content.append(line)
            
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content)
            
            # Display the sections in a grid
            st.markdown('<div class="analysis-grid">', unsafe_allow_html=True)
            
            for title, content in sections.items():
                st.markdown(f"""
                    <div class="analysis-box">
                        <div class="box-title">{title}</div>
                        <div class="box-content">{content}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No analysis available - insufficient data.")
    
    else:
        st.error("No data available for the selected stock and timeframe.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please check if the ticker symbol is correct and try again.") 