# SKIE_Ninja NinjaTrader 8 Installation Guide

**Last Updated**: 2025-12-06
**Version**: 1.1 - Added critical retraining requirements

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
