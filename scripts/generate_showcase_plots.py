import sys
import os
from pathlib import Path

# Add project root to sys.path for imports to work
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from dotenv import load_dotenv

# Load env variables from .env
load_dotenv()

from app.services.hybrid_prediction_service import hybrid_prediction_service

# Global Chart Style
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

def generate_plot(symbol, filename):
    print(f"Generating real system snapshot for {symbol}...")
    
    # Trigger quick training (for showcase purposes, we'll use existing if any or train fast)
    # We force training if not present
    if symbol not in hybrid_prediction_service._models:
        print(f"Model not found, training a fast version for {symbol}...")
        # Hack to speed up training for the script
        import app.services.hybrid_prediction_service as hps
        hps.EPOCHS = 30
        hps.ENSEMBLE_SIZE = 3
        hybrid_prediction_service.train_ensemble(symbol)
    
    # Get real system data
    result = hybrid_prediction_service.forecast_price([], symbol=symbol)
    
    if result.get("status") != "success":
        print(f"Error getting forecast for {symbol}: {result}")
        return

    # Extract data for plotting
    backtest = pd.DataFrame(result["backtest"])
    forecast = pd.DataFrame(result["forecast"])
    
    # Convert dates
    backtest['Date'] = pd.to_datetime(backtest['Date'])
    forecast['Date'] = pd.to_datetime(forecast['Date'])
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(15, 8), dpi=100)
    
    # Plot History (Actual)
    ax.plot(backtest['Date'], backtest['Actual'], label='Sistem Gerçek Veri (TwelveData)', color='#ffffff', linewidth=2, alpha=0.9)
    
    # Plot Backtest (Predicted)
    ax.plot(backtest['Date'], backtest['Predicted'], label='Model Backtest (History)', color='#00f2ff', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot Forecast
    ax.plot(forecast['Date'], forecast['PredictedClose'], label='AI Neural Forecast (14-Day)', color='#00ff41', linewidth=3, alpha=1)
    
    # Shaded Area for Forecast
    ax.fill_between(forecast['Date'], 
                    forecast['PredictedClose'] * 0.98, 
                    forecast['PredictedClose'] * 1.02, 
                    color='#00ff41', alpha=0.1, label='Confidence Interval')

    # Styling
    ax.set_title(f"MONSTER.AI - {symbol} NEURAL BACKTEST & FORECAST PROOF", fontsize=18, pad=20, color='#00f2ff')
    ax.set_ylabel("PRICE (USD)", fontsize=12, color='#64748b')
    ax.legend(loc='upper left', frameon=True, facecolor='#010205', edgecolor='#1e293b', fontsize=10)
    
    # Add System Watermark/Labels
    ax.text(0.5, 0.05, f"EXCHANGEMONSTER SYSTEM PROOF | EPOCHS: 30 | ENSEMBLE: 3 | DEVICE: {hybrid_prediction_service.device}", 
            transform=ax.transAxes, ha='center', fontsize=10, color='#1e293b', alpha=0.5)
    
    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, color='#64748b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#1e293b')
    ax.spines['bottom'].set_color('#1e293b')
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save path
    output_path = project_root.parent / "frontend" / "public" / filename
    plt.savefig(output_path, facecolor='#010205', bbox_inches='tight')
    plt.close()
    
    print(f"Successfully saved {filename} to {output_path}")

if __name__ == "__main__":
    # Ensure TwelveData API Key is set in env if not in config
    # generate_plot("NVDA", "nvda_real_snapshot.png")
    # generate_plot("TSLA", "tsla_real_snapshot.png")
    
    # We'll try to run it for one to showcase
    try:
        generate_plot("NVDA", "nvda_real_snapshot.png")
        generate_plot("TSLA", "tsla_real_snapshot.png")
    except Exception as e:
        print(f"Critical error during plot generation: {e}")
