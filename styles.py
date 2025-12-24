# styles.py

def load_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp { 
        background-color: #f8f9fa; 
    }
    
    /* Executive Metric Cards */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    .metric-value { 
        font-size: 28px; 
        font-weight: 700; 
        color: #2e7d32; 
    }
    .metric-label { 
        font-size: 14px; 
        font-weight: 600; 
        color: #6c757d; 
        text-transform: uppercase; 
        letter-spacing: 0.8px;
    }
    
    /* Recommendations */
    .rec-card {
        background: white;
        border-left: 6px solid #1e88e5;
        padding: 18px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .rec-title { 
        font-weight: 700; 
        color: #1565c0; 
        font-size: 16px; 
        margin-bottom: 6px; 
    }
    .rec-body { 
        color: #455a64; 
        font-size: 14px; 
        line-height: 1.6; 
    }

    /* Footer */
    .footer-container {
        text-align: center;
        margin-top: 60px;
        padding: 30px;
        border-top: 1px solid #eaeaea;
        color: #666;
    }
    .footer-author {
        font-weight: 700;
        letter-spacing: 1px;
        color: #333;
    }
    </style>
    """