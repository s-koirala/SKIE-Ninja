# SKIE_Ninja NinjaTrader 8 Installation Guide

**Last Updated**: 2025-12-06
**Version**: 1.3 - Added AddDataSeries best practices and VIX data troubleshooting

---

## CRITICAL: Ensemble Strategy Requirements

The Python backtests use an **Ensemble Strategy** that requires BOTH:

1. **Technical Volatility Model** (vol_expansion_prob >= 0.40)
2. **VIX Sentiment Model** (sent_vol_prob >= 0.55)

### Why This Matters

| Implementation | Filters | 2024 Trades | 2024 P&L |
|----------------|---------|-------------|----------|
| Python Ensemble | Tech + Sentiment | 2,044 | +$88,164 |
| C# without VIX | Tech only | 5,956 | -$38,250 |

**Missing the sentiment filter results in 3x overtrading and losses.**

### Required Data

- ES E-mini futures (5-minute bars)
- **VIX Index data** (daily or intraday)

### VIX Data Sources in NinjaTrader

| Symbol | Provider | Notes |
|--------|----------|-------|
| $VIX | Most data feeds | Common default symbol |
| ^VIX | Kinetick (free) | Delayed 15 min |
| $VIX.X | CQG/Continuum | Real-time |
| VIX | Interactive Brokers | Real-time |

---

## NinjaTrader AddDataSeries Best Practices

When using secondary data series (like VIX) in NinjaTrader strategies, there are important limitations and best practices to follow.

### Key Rules for AddDataSeries()

1. **Call in State.Configure Only**
   ```csharp
   protected override void OnStateChange()
   {
       if (State == State.Configure)
       {
           AddDataSeries("$VIX", Data.BarsPeriodType.Day, 1);  // VIX daily
       }
   }
   ```

2. **Arguments Must Be Hardcoded**
   - Per NinjaTrader documentation: "Arguments supplied to AddDataSeries() should be hardcoded and NOT dependent on run-time variables."
   - Objects like `Instrument`, `BarsPeriod`, `TradingHours` are NOT available until `State.DataLoaded`
   - You CANNOT dynamically construct the instrument name from variables

3. **Understanding BarsInProgress**
   - Primary series = `BarsInProgress 0`
   - First AddDataSeries = `BarsInProgress 1`
   - OnBarUpdate fires for EVERY series update
   - Always filter: `if (BarsInProgress != 0) return;`

4. **Accessing Secondary Series Data**
   - Use plural forms: `Closes[1][0]` for secondary series close
   - `Close[0]` is shorthand for `Closes[BarsInProgress][0]`
   - Example: `double vixClose = Closes[1][0];`

5. **Data Must Exist for Backtesting**
   - **CRITICAL**: Historical data for the secondary instrument MUST exist in NinjaTrader for backtesting
   - If data doesn't exist, AddDataSeries silently fails
   - Strategy Analyzer has known limitations with secondary series

### VIX Data Availability Issue

**Problem**: In Strategy Analyzer backtests, VIX data may not be available even if AddDataSeries() is called correctly.

**Symptoms**:
- Sentiment filter disabled (warning in output tab)
- Trade count matches non-sentiment results (~4,357 vs expected ~2,044)
- `vixCloseHistory.Count` shows 0 or null

**Root Cause**: VIX historical data must be downloaded/available in NinjaTrader's database for the backtest period.

**Solutions**:

| Approach | Pros | Cons |
|----------|------|------|
| Download VIX history | Full feature parity | Requires data subscription |
| Embed VIX in model files | No external data needed | Larger model files, less flexible |
| CSV file loader | Works with any data | Additional complexity |
| Disable sentiment filter | Simple | Reduced accuracy, more trades |

### Downloading VIX Data

1. Open **Control Center** → **Tools** → **Historical Data**
2. Search for your VIX symbol (`$VIX`, `^VIX`, or `VIX`)
3. Select appropriate data series (Daily recommended)
4. Download historical data for your backtest period (2024+)
5. Verify data loaded: Open a chart with VIX to confirm

### RECOMMENDED: Add VIX to Chart as Secondary Data Series

**Important**: A strategy can only access data series that are explicitly added via `AddDataSeries()` in its code. However, when you add VIX as a secondary data series **to the same chart** via the UI, NinjaTrader loads that historical data and makes it available to the strategy.

**Steps to add VIX to your ES chart:**

1. Open your ES futures chart (5-minute bars)
2. Click the **Data Series** button on the Chart Toolbar (or right-click chart → "Data Series...")
3. In the Data Series window, click **Add** (lower left corner)
4. Configure the secondary series:
   - **Instrument**: `$VIX` (or `^VIX` for Kinetick)
   - **Type**: `Day`
   - **Value**: `1`
5. Click **OK** - VIX will appear as a secondary panel below ES
6. (Optional) Hide the VIX panel by dragging the panel divider, or minimize it
7. Run your strategy on this combined chart

**Why this works**: When VIX is added to the chart via UI, NinjaTrader loads its historical data. The strategy's `AddDataSeries("$VIX", BarsPeriodType.Day, 1)` call then accesses this pre-loaded data via `BarsArray[1]`, `Closes[1]`, etc.

**Note**: You cannot reference data from a **separate** chart window - the secondary instrument must be on the **same** chart as your strategy.

References:
- [NinjaTrader: Working with Multiple Data Series](https://ninjatrader.com/support/helpguides/nt8/working_with_multiple_data_series.htm)
- [NinjaTrader Forum: Set multiple instruments via chart UI](https://forum.ninjatrader.com/forum/ninjatrader-8/strategy-development/1070358-set-multiple-instruments-in-strategy-via-chart-ui)

### Alternative: Pre-computed VIX Features

If VIX data cannot be obtained, an alternative is to:
1. Pre-compute VIX sentiment features during Python model training
2. Export these features embedded in the model or as separate JSON files
3. Load from files instead of real-time calculation

This approach trades real-time flexibility for guaranteed data availability.

### References

- [NinjaTrader AddDataSeries() Documentation](https://ninjatrader.com/support/helpguides/nt8/adddataseries.htm)
- [Multi-Timeframe & Instruments Guide](https://ninjatrader.com/support/helpguides/nt8/multi-time_frame__instruments.htm)
- [NinjaTrader Forum: AddDataSeries Best Practices](https://forum.ninjatrader.com/forum/ninjatrader-8/platform-technical-support-aa/1305538-what-is-best-practice-for-using-adddataseries-with-a-strategy)

---

## CRITICAL: Model Retraining Requirements

**IMPORTANT**: ONNX models are static and decay rapidly. Per the project's validated methodology:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Train Window | 180 days | Grid search optimized |
| Test Window | 5 days | Bias-variance tradeoff |
| **Retraining Frequency** | **Weekly** | Walk-forward equivalent |

**Failure to retrain models weekly will result in significant performance degradation.**

The Python backtests showed **+$502,219 profit** using walk-forward (retraining every 5 days). A static ONNX model tested on the same period showed **-$194,512 loss**.

See [Model Updates](#model-updates-critical) section for retraining instructions.

---

## Prerequisites

- NinjaTrader 8.1.x installed
- Visual Studio 2019+ (Community edition is fine)
- Python 3.9+ with required packages

## Step 1: Install Python Dependencies

```bash
pip install onnxmltools onnx onnxruntime lightgbm scikit-learn pandas numpy
```

## Step 2: Export ONNX Models

Run the export script to train and export production models:

```bash
cd SKIE_Ninja
python src/python/export_onnx.py
```

This creates:
- `data/models/onnx/vol_expansion_model.onnx`
- `data/models/onnx/breakout_high_model.onnx`
- `data/models/onnx/breakout_low_model.onnx`
- `data/models/onnx/atr_forecast_model.onnx`
- `data/models/onnx/scaler_params.json`
- `data/models/onnx/strategy_config.json`

## Step 3: Build C# DLL

1. Open Visual Studio
2. Open solution: `src/csharp/SKIENinjaML/SKIENinjaML.csproj`
3. Build → Build Solution (Release mode)
4. Output: `bin/Release/net48/SKIENinjaML.dll`

**Required DLLs** (from NuGet packages):
- `SKIENinjaML.dll` (your built DLL)
- `Microsoft.ML.OnnxRuntime.dll`
- `onnxruntime.dll` (native)
- `Newtonsoft.Json.dll`

## Step 4: Copy Files to NinjaTrader

### DLL Files
Copy to: `Documents\NinjaTrader 8\bin\Custom\`

```
SKIENinjaML.dll
Microsoft.ML.OnnxRuntime.dll
Newtonsoft.Json.dll
```

**Native ONNX Runtime:**
Copy to: `Documents\NinjaTrader 8\bin\Custom\runtimes\win-x64\native\`

```
onnxruntime.dll
```

### Model Files
Create folder: `Documents\SKIE_Ninja\models\`

Copy all files from `data/models/onnx/`:
```
vol_expansion_model.onnx
breakout_high_model.onnx
breakout_low_model.onnx
atr_forecast_model.onnx
scaler_params.json
strategy_config.json
```

## Step 5: Import Strategy into NinjaTrader

1. Open NinjaTrader 8
2. Open **NinjaScript Editor**: Control Center → New → NinjaScript Editor
3. Right-click in editor → **References**
4. Add reference to `SKIENinjaML.dll`
5. Create new Strategy file and paste contents of `SKIENinjaStrategy.cs`
6. Compile (F5)

## Step 6: Configure Strategy

1. Open a chart (ES 5-minute recommended)
2. Right-click → Strategies
3. Add **SKIENinjaStrategy**
4. Configure parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Model Path | Documents\SKIE_Ninja\models | Path to ONNX files |
| Min Vol Expansion Prob | 0.40 | Entry threshold |
| Min Breakout Prob | 0.45 | Direction threshold |
| Max Contracts | 3 | Position size limit |
| Max Holding Bars | 20 | Time-based exit |
| Enable Logging | false | Debug output |

5. Enable strategy (paper trading first!)

## Step 7: Paper Trading Validation

Run in Sim101 account for 30-60 days before live trading.

**Expected Metrics:**
| Metric | Target Range |
|--------|-------------|
| Win Rate | 38-43% |
| Avg Trade | $20-30 |
| Daily Trades | 12-18 |
| Sharpe | 2.5-3.5 |

## Troubleshooting

### "Could not load file or assembly"
- Ensure all DLLs are in `bin\Custom`
- Check .NET Framework 4.8 compatibility
- Restart NinjaTrader after copying DLLs

### "Model files not found"
- Verify Model Path setting points to correct folder
- Ensure all 6 files (4 ONNX + 2 JSON) exist

### "Prediction error"
- Check scaler_params.json matches feature count
- Verify ONNX models are not corrupted
- Enable logging to see detailed errors

### Feature mismatch
- The strategy calculates ~47 features
- scaler_params.json must have matching count
- Re-run export_onnx.py if mismatch occurs

### VIX data NOT available (Sentiment filter disabled)
- **Symptom**: Output shows "WARNING: VIX data NOT available - sentiment filter DISABLED"
- **Result**: Trade count ~4,357 instead of expected ~2,044
- **Cause**: VIX historical data not loaded in NinjaTrader
- **Solution**: See [AddDataSeries Best Practices](#ninjatrader-adddataseries-best-practices) section
- **Steps**:
  1. Open Historical Data Manager (Tools → Historical Data)
  2. Search for VIX symbol (`$VIX`, `^VIX`, or `VIX`)
  3. Download historical data covering your backtest period
  4. Re-run backtest

## Model Updates (CRITICAL)

**Weekly retraining is REQUIRED to maintain performance.** See the warning at the top of this document.

### Automated Retraining Script

Use the dedicated retraining script:

```bash
cd SKIE_Ninja

# Retrain and automatically copy to NinjaTrader folder
python src/python/retrain_onnx_models.py --copy-to-ninjatrader

# Or retrain only (manual copy)
python src/python/retrain_onnx_models.py
```

The script will:
1. Backup existing models to `data/models/onnx/backups/`
2. Load latest market data
3. Train on most recent 180 days
4. Export fresh ONNX models
5. Optionally copy to NinjaTrader folder

### Manual Retraining Steps

If you prefer manual control:

1. Add new data to `data/raw/market/`
2. Run `python src/python/export_onnx.py`
3. Copy new ONNX files to NinjaTrader folder
4. Restart NinjaTrader

### Retraining Schedule

| Frequency | Expected Performance |
|-----------|---------------------|
| Weekly | Optimal (walk-forward equivalent) |
| Monthly | Degraded (model drift begins) |
| Quarterly | Significantly degraded |
| Never | Model failure (-$194K observed) |

**Set a calendar reminder for every Sunday** to run the retraining script before the Monday market open

## File Structure Summary

```
Documents\
├── NinjaTrader 8\
│   └── bin\
│       └── Custom\
│           ├── SKIENinjaML.dll
│           ├── Microsoft.ML.OnnxRuntime.dll
│           ├── Newtonsoft.Json.dll
│           └── runtimes\
│               └── win-x64\
│                   └── native\
│                       └── onnxruntime.dll
└── SKIE_Ninja\
    └── models\
        ├── vol_expansion_model.onnx
        ├── breakout_high_model.onnx
        ├── breakout_low_model.onnx
        ├── atr_forecast_model.onnx
        ├── scaler_params.json
        └── strategy_config.json
```

## Support

- Check `config/project_memory.md` for decision history
- Review `docs/BEST_PRACTICES.md` for lessons learned
- Strategy logic matches `src/python/strategy/volatility_breakout_strategy.py`
