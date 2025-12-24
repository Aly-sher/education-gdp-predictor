# styles.py

def load_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
        color: #333;
    }
    
    /* Main Background */
    .stApp { 
        background-color: #f4f6f9; 
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a202c;
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
        color: #edf2f7 !important;
    }
    
    /* Executive Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border-left: 4px solid #3182ce;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-value { 
        font-size: 32px; 
        font-weight: 700; 
        color: #2d3748; 
    }
    .metric-label { 
        font-size: 14px; 
        font-weight: 600; 
        color: #718096; 
        text-transform: uppercase; 
        letter-spacing: 1px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3182ce;
        color: white;
        border-radius: 6px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background-color: #2b6cb0;
    }

    /* Footer */
    .footer-container {
        text-align: center;
        margin-top: 80px;
        padding: 20px;
        border-top: 1px solid #e2e8f0;
        color: #718096;
        font-size: 12px;
    }
    </style>
    """