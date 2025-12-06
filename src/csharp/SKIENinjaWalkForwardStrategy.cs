/*
 * SKIE_Ninja Walk-Forward Strategy
 *
 * NinjaTrader 8 strategy that automatically switches ONNX models based on the
 * current bar's date. This enables true walk-forward backtesting in a SINGLE
 * NT8 backtest run - no manual model swapping required.
 *
 * How it works:
 * 1. Pre-generate 70+ walk-forward model sets using export_walkforward_onnx.py
 * 2. Set WalkForwardPath to the walkforward_onnx folder
 * 3. Enable WalkForwardMode = true
 * 4. Run a single backtest covering 2024 (or your date range)
 * 5. The strategy automatically loads the correct model for each date
 *
 * This matches the Python walk-forward methodology that achieved +$502K profit.
 *
 * Author: SKIE_Ninja Development Team
 * Created: 2025-12-06
 */

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
using SKIENinjaML;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class SKIENinjaWalkForwardStrategy : Strategy
    {
        #region Variables
        // Walk-forward predictor (automatic model switching)
        private WalkForwardPredictor wfPredictor;

        // Static predictor (single model)
        private SKIENinjaPredictor staticPredictor;

        private double[] featureBuffer;
        private bool predictorReady = false;

        // Indicators for feature calculation
        private SMA sma5, sma10, sma20, sma50;
        private Bollinger bb20;

        // True Range history for manual ATR calculation (matching Python)
        private List<double> trueRangeHistory;

        // Feature storage (rolling windows)
        private List<double> closeHistory;
        private List<double> highHistory;
        private List<double> lowHistory;
        private List<double> volumeHistory;

        // Diagnostic counters
        private int barCount = 0;
        private int signalCount = 0;
        private int tradeCount = 0;

        // Position tracking for time-based exits
        private int entryBar = 0;

        // Model tracking
        private string currentModelInfo = "";
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Walk-Forward Mode", Description = "Enable automatic model switching based on bar date",
            Order = 0, GroupName = "1. Walk-Forward Settings")]
        public bool WalkForwardMode { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Walk-Forward Path", Description = "Path to walkforward_onnx folder with model_schedule.csv",
            Order = 1, GroupName = "1. Walk-Forward Settings")]
        public string WalkForwardPath { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Static Model Path", Description = "Path to static ONNX models (used when Walk-Forward Mode = false)",
            Order = 2, GroupName = "1. Walk-Forward Settings")]
        public string StaticModelPath { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 0.9)]
        [Display(Name = "Min Vol Expansion Prob", Description = "Minimum volatility expansion probability (0.40 = optimized)",
            Order = 3, GroupName = "2. Strategy Settings")]
        public double MinVolExpansionProb { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 0.9)]
        [Display(Name = "Min Breakout Prob", Description = "Minimum breakout probability (0.45 = optimized)",
            Order = 4, GroupName = "2. Strategy Settings")]
        public double MinBreakoutProb { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Max Contracts", Description = "Maximum position size",
            Order = 5, GroupName = "2. Strategy Settings")]
        public int MaxContracts { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Max Holding Bars", Description = "Maximum bars to hold position",
            Order = 6, GroupName = "2. Strategy Settings")]
        public int MaxHoldingBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Log signals to output window",
            Order = 7, GroupName = "3. Debug")]
        public bool EnableLogging { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Log Model Switches", Description = "Log each model switch event",
            Order = 8, GroupName = "3. Debug")]
        public bool LogModelSwitches { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "SKIE_Ninja Walk-Forward Strategy - Automatic model switching for true walk-forward backtesting";
                Name = "SKIENinjaWalkForwardStrategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 1;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 55;
                IsUnmanaged = false;

                // Default paths
                string docsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                WalkForwardPath = Path.Combine(docsPath, "SKIE_Ninja", "walkforward_models");
                StaticModelPath = Path.Combine(docsPath, "SKIE_Ninja", "models");

                // Default: Walk-Forward mode enabled
                WalkForwardMode = true;
                MinVolExpansionProb = 0.40;
                MinBreakoutProb = 0.45;
                MaxContracts = 3;
                MaxHoldingBars = 20;
                EnableLogging = true;
                LogModelSwitches = true;
            }
            else if (State == State.Configure)
            {
                // No additional data series needed
            }
            else if (State == State.DataLoaded)
            {
                Print("======================================================");
                Print("SKIE_Ninja Walk-Forward Strategy Initializing...");
                Print("======================================================");

                // Initialize indicators
                sma5 = SMA(5);
                sma10 = SMA(10);
                sma20 = SMA(20);
                sma50 = SMA(50);
                bb20 = Bollinger(2, 20);

                // Initialize history buffers
                closeHistory = new List<double>();
                highHistory = new List<double>();
                lowHistory = new List<double>();
                volumeHistory = new List<double>();
                trueRangeHistory = new List<double>();

                // Initialize appropriate predictor
                try
                {
                    if (WalkForwardMode)
                    {
                        InitializeWalkForwardMode();
                    }
                    else
                    {
                        InitializeStaticMode();
                    }
                }
                catch (Exception ex)
                {
                    Print("ERROR: Predictor initialization failed");
                    Print("Error: " + ex.Message);
                    if (ex.InnerException != null)
                        Print("Inner: " + ex.InnerException.Message);
                    Print("Stack: " + ex.StackTrace);
                }
            }
            else if (State == State.Terminated)
            {
                if (wfPredictor != null)
                {
                    Print("======================================================");
                    Print("Walk-Forward Summary:");
                    Print("  Model switches: " + wfPredictor.GetModelSwitchCount());
                    Print("======================================================");
                    wfPredictor.Dispose();
                }
                if (staticPredictor != null)
                {
                    staticPredictor.Dispose();
                }

                Print("======================================================");
                Print("SKIE_Ninja Strategy Terminated");
                Print("Bars processed: " + barCount);
                Print("Signals generated: " + signalCount);
                Print("Trades executed: " + tradeCount);
                Print("======================================================");
            }
        }

        private void InitializeWalkForwardMode()
        {
            Print("Mode: WALK-FORWARD (automatic model switching)");
            Print("Path: " + WalkForwardPath);

            if (!System.IO.Directory.Exists(WalkForwardPath))
            {
                Print("ERROR: Walk-forward path does not exist: " + WalkForwardPath);
                Print("Run export_walkforward_onnx.py first to generate models.");
                return;
            }

            wfPredictor = new WalkForwardPredictor();

            // Hook up logging
            wfPredictor.OnLog += (msg) => {
                if (EnableLogging)
                    Print("WF: " + msg);
            };

            wfPredictor.OnModelSwitch += (msg) => {
                if (LogModelSwitches)
                    Print(">>> MODEL SWITCH: " + msg);
            };

            wfPredictor.Initialize(WalkForwardPath);

            // Feature buffer will be sized on first model load
            featureBuffer = new double[42]; // Default feature count
            predictorReady = true;

            Print("Walk-Forward predictor ready with " + wfPredictor.GetSchedule().Count + " model periods");
        }

        private void InitializeStaticMode()
        {
            Print("Mode: STATIC (single model)");
            Print("Path: " + StaticModelPath);

            if (!System.IO.Directory.Exists(StaticModelPath))
            {
                Print("ERROR: Static model path does not exist: " + StaticModelPath);
                return;
            }

            staticPredictor = new SKIENinjaPredictor();
            staticPredictor.Initialize(StaticModelPath);

            int featureCount = staticPredictor.GetFeatureCount();
            featureBuffer = new double[featureCount];
            predictorReady = true;

            Print("Static predictor ready with " + featureCount + " features");
        }

        protected override void OnBarUpdate()
        {
            barCount++;

            if (!predictorReady)
            {
                if (barCount == 1)
                    Print("Predictor not ready - skipping bars");
                return;
            }

            // Update price history
            UpdateHistory();

            // Wait for enough history
            if (closeHistory.Count < 51)
            {
                if (EnableLogging && barCount % 10 == 0)
                    Print("Building history... " + closeHistory.Count + "/51 bars");
                return;
            }

            // In walk-forward mode, ensure correct model is loaded for current date
            if (WalkForwardMode && wfPredictor != null)
            {
                DateTime barDate = Time[0].Date;
                bool modelReady = wfPredictor.EnsureModelForDate(barDate);

                if (!modelReady)
                {
                    // Date is before first model - can't trade
                    if (barCount == 51) // First tradeable bar
                    {
                        Print("Date " + barDate.ToString("yyyy-MM-dd") + " is before first model - waiting...");
                    }
                    return;
                }
            }

            // Handle existing position - check for time-based exit
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                int barsHeld = CurrentBar - entryBar;
                if (barsHeld >= MaxHoldingBars)
                {
                    if (EnableLogging)
                    {
                        Print(String.Format(">>> TIME EXIT [{0}]: Held {1} bars",
                            Time[0], barsHeld));
                    }

                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("TimeExit", "SkieEntry");
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort("TimeExit", "SkieEntry");
                }
                return;
            }

            // Calculate features
            if (!CalculateFeatures())
            {
                if (EnableLogging)
                    Print("Feature calculation failed at bar " + CurrentBar);
                return;
            }

            // Get ML prediction
            TradingSignal signal;
            try
            {
                double currentATR = GetATR(14);

                if (WalkForwardMode && wfPredictor != null)
                {
                    signal = wfPredictor.GenerateSignal(featureBuffer, currentATR);
                }
                else if (staticPredictor != null)
                {
                    signal = staticPredictor.GenerateSignal(featureBuffer, currentATR);
                }
                else
                {
                    return;
                }
            }
            catch (Exception ex)
            {
                Print("Prediction error: " + ex.Message);
                return;
            }

            // Log diagnostics periodically
            if (EnableLogging && barCount % 100 == 0)
            {
                string modelInfo = WalkForwardMode && wfPredictor != null ?
                    wfPredictor.GetCurrentModelInfo() : "Static";

                Print(String.Format("[{0}] Bar {1} | Model: {2} | VolProb={3:F3}",
                    Time[0], barCount, modelInfo, signal.VolExpansionProb));
            }

            // Execute trade if signal passes thresholds
            if (signal.ShouldTrade && signal.Direction != 0)
            {
                signalCount++;

                if (EnableLogging)
                {
                    Print(String.Format(">>> SIGNAL [{0}]: Dir={1}, VolProb={2:F3}, BreakoutProb={3:F3}",
                        Time[0], signal.Direction, signal.VolExpansionProb, signal.BreakoutProb));
                }

                ExecuteTrade(signal);
            }
        }

        private void UpdateHistory()
        {
            closeHistory.Add(Close[0]);
            highHistory.Add(High[0]);
            lowHistory.Add(Low[0]);
            volumeHistory.Add(Volume[0]);

            // Calculate True Range (matching Python exactly)
            double tr;
            if (closeHistory.Count > 1)
            {
                double prevClose = closeHistory[closeHistory.Count - 2];
                double tr1 = High[0] - Low[0];
                double tr2 = Math.Abs(High[0] - prevClose);
                double tr3 = Math.Abs(Low[0] - prevClose);
                tr = Math.Max(tr1, Math.Max(tr2, tr3));
            }
            else
            {
                tr = High[0] - Low[0];
            }
            trueRangeHistory.Add(tr);

            // Keep last 51 bars
            while (closeHistory.Count > 51)
            {
                closeHistory.RemoveAt(0);
                highHistory.RemoveAt(0);
                lowHistory.RemoveAt(0);
                volumeHistory.RemoveAt(0);
                trueRangeHistory.RemoveAt(0);
            }
        }

        private bool CalculateFeatures()
        {
            if (closeHistory.Count < 51)
                return false;

            try
            {
                int idx = 0;

                // 1. Returns (lagged) - 6 features
                int[] lags = { 1, 2, 3, 5, 10, 20 };
                foreach (int lag in lags)
                {
                    if (CurrentBar >= lag && Close[lag] != 0)
                        featureBuffer[idx++] = (Close[0] - Close[lag]) / Close[lag];
                    else
                        featureBuffer[idx++] = 0;
                }

                // 2. Volatility features - 12 features
                int[] volPeriods = { 5, 10, 14, 20 };
                foreach (int period in volPeriods)
                {
                    double rv = CalculateRealizedVol(period);
                    featureBuffer[idx++] = rv;

                    double atr = GetATR(period);
                    featureBuffer[idx++] = atr;

                    featureBuffer[idx++] = Close[0] != 0 ? atr / Close[0] : 0;
                }

                // 3. Price position - 9 features
                int[] pricePeriods = { 10, 20, 50 };
                foreach (int period in pricePeriods)
                {
                    double highMax = MAX(High, period)[0];
                    double lowMin = MIN(Low, period)[0];

                    featureBuffer[idx++] = Close[0] != 0 ? (Close[0] - highMax) / Close[0] : 0;
                    featureBuffer[idx++] = Close[0] != 0 ? (Close[0] - lowMin) / Close[0] : 0;
                    featureBuffer[idx++] = Close[0] != 0 ? (highMax - lowMin) / Close[0] : 0;
                }

                // 4. Momentum - 6 features
                int[] momPeriods = { 5, 10, 20 };
                foreach (int period in momPeriods)
                {
                    if (CurrentBar >= period && Close[period] != 0)
                        featureBuffer[idx++] = (Close[0] - Close[period]) / Close[period];
                    else
                        featureBuffer[idx++] = 0;

                    double ma = GetSMA(period);
                    featureBuffer[idx++] = ma != 0 ? (Close[0] - ma) / ma : 0;
                }

                // 5. RSI - 2 features
                featureBuffer[idx++] = CalculateRSI(7);
                featureBuffer[idx++] = CalculateRSI(14);

                // 6. Bollinger Band position - 1 feature
                double bbUpper = bb20.Upper[0];
                double bbLower = bb20.Lower[0];
                double bbRange = bbUpper - bbLower;
                featureBuffer[idx++] = bbRange > 0.0001 ? (Close[0] - bbLower) / bbRange : 0.5;

                // 7. Volume features - 6 features
                int[] volumePeriods = { 5, 10, 20 };
                foreach (int period in volumePeriods)
                {
                    double volSma = CalculateVolumeSMA(period);
                    featureBuffer[idx++] = volSma;
                    featureBuffer[idx++] = Volume[0] / (volSma + 1);
                }

                return idx == 42;
            }
            catch (Exception ex)
            {
                Print("Feature error: " + ex.Message);
                return false;
            }
        }

        private double CalculateRealizedVol(int period)
        {
            if (closeHistory.Count < period + 1)
                return 0;

            List<double> returns = new List<double>();
            int startIdx = closeHistory.Count - period;

            for (int i = startIdx; i < closeHistory.Count; i++)
            {
                if (i > 0 && closeHistory[i - 1] != 0)
                {
                    double ret = (closeHistory[i] - closeHistory[i - 1]) / closeHistory[i - 1];
                    returns.Add(ret);
                }
            }

            if (returns.Count <= 1)
                return 0;

            double mean = returns.Average();
            double variance = returns.Sum(r => Math.Pow(r - mean, 2)) / (returns.Count - 1);
            return Math.Sqrt(variance);
        }

        private double CalculateRSI(int period)
        {
            if (closeHistory.Count < period + 1)
                return 50.0;

            double sumGain = 0;
            double sumLoss = 0;
            int startIdx = closeHistory.Count - period;

            for (int i = startIdx; i < closeHistory.Count; i++)
            {
                if (i > 0)
                {
                    double delta = closeHistory[i] - closeHistory[i - 1];
                    if (delta > 0)
                        sumGain += delta;
                    else
                        sumLoss += (-delta);
                }
            }

            double avgGain = sumGain / period;
            double avgLoss = sumLoss / period;

            if (avgLoss < 1e-10)
                return 100.0;

            double rs = avgGain / avgLoss;
            return 100.0 - (100.0 / (1.0 + rs));
        }

        private double GetATR(int period)
        {
            if (trueRangeHistory.Count < period)
                return 0;

            double sum = 0;
            int startIdx = trueRangeHistory.Count - period;
            for (int i = startIdx; i < trueRangeHistory.Count; i++)
            {
                sum += trueRangeHistory[i];
            }
            return sum / period;
        }

        private double GetSMA(int period)
        {
            switch (period)
            {
                case 5: return sma5[0];
                case 10: return sma10[0];
                case 20: return sma20[0];
                case 50: return sma50[0];
                default: return sma20[0];
            }
        }

        private double CalculateVolumeSMA(int period)
        {
            if (volumeHistory.Count < period)
                return Volume[0] > 0 ? Volume[0] : 1;

            double sum = 0;
            int startIdx = volumeHistory.Count - period;
            for (int i = startIdx; i < volumeHistory.Count; i++)
            {
                sum += volumeHistory[i];
            }
            return sum / period;
        }

        private void ExecuteTrade(TradingSignal signal)
        {
            int contracts = Math.Min(signal.Contracts, MaxContracts);
            contracts = Math.Max(1, contracts);

            double tp = Close[0] + signal.TakeProfitOffset;
            double sl = Close[0] + signal.StopLossOffset;

            // CRITICAL: Set methods MUST be called BEFORE entry methods
            if (signal.Direction == 1)
            {
                SetProfitTarget("SkieEntry", CalculationMode.Price, tp);
                SetStopLoss("SkieEntry", CalculationMode.Price, sl, false);
                EnterLong(contracts, "SkieEntry");
                entryBar = CurrentBar;
                tradeCount++;

                if (EnableLogging)
                {
                    Print(String.Format(">>> LONG {0} @ {1:F2}, TP={2:F2}, SL={3:F2}",
                        contracts, Close[0], tp, sl));
                }
            }
            else if (signal.Direction == -1)
            {
                SetProfitTarget("SkieEntry", CalculationMode.Price, tp);
                SetStopLoss("SkieEntry", CalculationMode.Price, sl, false);
                EnterShort(contracts, "SkieEntry");
                entryBar = CurrentBar;
                tradeCount++;

                if (EnableLogging)
                {
                    Print(String.Format(">>> SHORT {0} @ {1:F2}, TP={2:F2}, SL={3:F2}",
                        contracts, Close[0], tp, sl));
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution,
            string executionId, double price, int quantity,
            MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (EnableLogging)
            {
                Print(String.Format("[{0}] EXECUTED: {1} {2} @ {3:F2}",
                    time, execution.Order.OrderAction, quantity, price));
            }
        }
    }
}
