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
        private double[] sentimentFeatureBuffer;
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

        // VIX data for sentiment features
        private bool vixDataAvailable = false;
        private List<double> vixCloseHistory;
        private double lastVixClose = 0;
        private DateTime lastVixDate = DateTime.MinValue;

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

        [NinjaScriptProperty]
        [Display(Name = "VIX Symbol", Description = "VIX symbol for sentiment features (^VIX, $VIX.X, or VIX)",
            Order = 9, GroupName = "4. VIX Sentiment")]
        public string VixSymbol { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Sentiment Filter", Description = "Enable VIX-based sentiment filtering (requires VIX data)",
            Order = 10, GroupName = "4. VIX Sentiment")]
        public bool EnableSentimentFilter { get; set; }
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

                // Default paths - point to NT8 custom folder with deployed models
                string docsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                WalkForwardPath = Path.Combine(docsPath, "NinjaTrader 8", "bin", "Custom", "SKIE_Ninja_Models", "walkforward_onnx");
                StaticModelPath = Path.Combine(docsPath, "NinjaTrader 8", "bin", "Custom", "SKIE_Ninja_Models");

                // Default: Walk-Forward mode enabled
                WalkForwardMode = true;
                MinVolExpansionProb = 0.40;
                MinBreakoutProb = 0.45;
                MaxContracts = 3;
                MaxHoldingBars = 20;
                EnableLogging = true;
                LogModelSwitches = true;

                // VIX sentiment settings
                VixSymbol = "$VIX";  // Common symbol: $VIX (most data feeds), ^VIX (Kinetick), VIX (IB)
                EnableSentimentFilter = true;
            }
            else if (State == State.Configure)
            {
                // Add VIX data series for sentiment features
                if (EnableSentimentFilter && !string.IsNullOrEmpty(VixSymbol))
                {
                    try
                    {
                        // Add VIX as daily data series (BarsInProgress = 1)
                        AddDataSeries(VixSymbol, Data.BarsPeriodType.Day, 1);
                        Print("VIX data series configured: " + VixSymbol);
                    }
                    catch (Exception ex)
                    {
                        Print("WARNING: Could not add VIX data series: " + ex.Message);
                        Print("Sentiment filter will be disabled.");
                        EnableSentimentFilter = false;
                    }
                }
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

                // Initialize VIX history for sentiment features
                vixCloseHistory = new List<double>();
                sentimentFeatureBuffer = new double[28]; // 28 sentiment features

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
                Print("======================================================");
                Print("SKIE_Ninja Strategy FINAL SUMMARY");
                Print("======================================================");

                if (wfPredictor != null)
                {
                    Print("Mode: Walk-Forward");
                    Print("  Model switches: " + wfPredictor.GetModelSwitchCount());
                    wfPredictor.Dispose();
                }
                else
                {
                    Print("Mode: Static");
                }

                if (staticPredictor != null)
                {
                    staticPredictor.Dispose();
                }

                // Sentiment filter status
                Print("");
                Print("SENTIMENT FILTER STATUS:");
                if (EnableSentimentFilter)
                {
                    if (vixDataAvailable)
                    {
                        Print("  Status: ACTIVE");
                        Print("  VIX history: " + vixCloseHistory.Count + " days");
                        Print("  Last VIX: " + lastVixClose.ToString("F2") + " on " + lastVixDate.ToString("yyyy-MM-dd"));
                        Print("  NOTE: Both tech AND sentiment filters applied");
                    }
                    else
                    {
                        Print("  Status: DISABLED (VIX data not available)");
                        Print("  VIX history: " + (vixCloseHistory != null ? vixCloseHistory.Count.ToString() : "null") + " days (need 21)");
                        Print("  WARNING: Only tech filter applied - more trades than expected");
                    }
                }
                else
                {
                    Print("  Status: DISABLED (by user setting)");
                }

                Print("");
                Print("TRADING STATISTICS:");
                Print("  Bars processed: " + barCount);
                Print("  Signals generated: " + signalCount);
                Print("  Trades executed: " + tradeCount);

                // Expected trade counts for reference
                Print("");
                Print("EXPECTED TRADE COUNTS (2024 full year):");
                Print("  With sentiment filter: ~2,044 trades");
                Print("  Without sentiment filter: ~4,357 trades");
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
            // Handle VIX data updates (BarsInProgress == 1)
            if (BarsInProgress == 1 && EnableSentimentFilter)
            {
                UpdateVixHistory();
                return; // VIX updates don't generate signals
            }

            // Only process primary instrument (BarsInProgress == 0)
            if (BarsInProgress != 0) return;

            barCount++;

            // On primary series, also check if VIX data is available (for regular hours data)
            // VIX daily data may have arrived even if we didn't get an OnBarUpdate for it
            if (EnableSentimentFilter && !vixDataAvailable && BarsArray.Length > 1)
            {
                CheckVixDataAvailability();
            }

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

            // Calculate sentiment features if enabled and available
            double[] sentimentFeatures = null;
            if (EnableSentimentFilter && vixDataAvailable)
            {
                // Update VIX history from BarsArray if new day arrived
                UpdateVixFromBarsArray();

                if (CalculateSentimentFeatures())
                {
                    sentimentFeatures = sentimentFeatureBuffer;

                    // Log first successful sentiment calculation
                    if (signalCount == 0 && EnableLogging)
                    {
                        Print(String.Format("FIRST SIGNAL WITH SENTIMENT: Bar {0}, vixHistory={1}, VIX={2:F2}",
                            barCount, vixCloseHistory.Count, lastVixClose));
                    }
                }
                else if (signalCount == 0 && EnableLogging)
                {
                    Print(String.Format("DIAG: CalculateSentimentFeatures() FAILED at bar {0}. vixHistory={1}",
                        barCount, vixCloseHistory.Count));
                }
            }
            else if (EnableSentimentFilter && !vixDataAvailable)
            {
                // Log periodically while waiting for VIX data
                if (barCount == 52 || barCount == 100 || barCount == 500 || barCount == 1000)
                {
                    int vixBars = (CurrentBars != null && CurrentBars.Length > 1) ? CurrentBars[1] : -999;
                    Print(String.Format("DIAG Bar {0}: vixDataAvailable=FALSE, vixHistory={1}, CurrentBars[1]={2}",
                        barCount, vixCloseHistory.Count, vixBars));
                }

                if (barCount == 52)
                {
                    Print("======================================================");
                    Print("WARNING: VIX SENTIMENT FILTER DISABLED");
                    Print("  Reason: VIX historical data not available or insufficient");
                    Print("  VIX history count: " + (vixCloseHistory != null ? vixCloseHistory.Count.ToString() : "null") + " (need 21)");
                    Print("  BarsArray length: " + (BarsArray != null ? BarsArray.Length.ToString() : "null"));
                    if (BarsArray != null && BarsArray.Length > 1)
                    {
                        Print("  BarsArray[1] count: " + BarsArray[1].Count);
                        Print("  CurrentBars[1]: " + CurrentBars[1]);
                    }
                    Print("  Expected trade count: ~4,357 (vs ~2,044 with sentiment)");
                    Print("  ");
                    Print("  To fix: Add VIX as secondary data series to chart");
                    Print("  See docs/NINJATRADER_INSTALLATION.md for details");
                    Print("======================================================");
                }
            }

            // Get ML prediction
            TradingSignal signal;
            try
            {
                double currentATR = GetATR(14);

                if (WalkForwardMode && wfPredictor != null)
                {
                    // Pass sentiment features to walk-forward predictor
                    signal = wfPredictor.GenerateSignal(featureBuffer, sentimentFeatures, currentATR);
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

                string sentInfo = signal.SentimentVolProb >= 0 ?
                    String.Format(", SentProb={0:F3}", signal.SentimentVolProb) : "";

                Print(String.Format("[{0}] Bar {1} | Model: {2} | VolProb={3:F3}{4}",
                    Time[0], barCount, modelInfo, signal.VolExpansionProb, sentInfo));
            }

            // Execute trade if signal passes thresholds
            if (signal.ShouldTrade && signal.Direction != 0)
            {
                signalCount++;

                if (EnableLogging)
                {
                    string sentInfo = signal.SentimentVolProb >= 0 ?
                        String.Format(", SentProb={0:F3}", signal.SentimentVolProb) : "";

                    Print(String.Format(">>> SIGNAL [{0}]: Dir={1}, VolProb={2:F3}{3}, BreakoutProb={4:F3}",
                        Time[0], signal.Direction, signal.VolExpansionProb, sentInfo, signal.BreakoutProb));
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

        /// <summary>
        /// Update VIX history when new daily VIX data arrives.
        /// Called from OnBarUpdate when BarsInProgress == 1 (VIX data).
        /// </summary>
        private void UpdateVixHistory()
        {
            if (BarsInProgress != 1) return;

            // Safety check - ensure VIX series has data
            if (BarsArray[1].Count == 0)
            {
                return;
            }

            double vixClose = Closes[1][0];
            DateTime vixDate = Times[1][0].Date;

            // Only update if we have a new day's data
            if (vixDate > lastVixDate)
            {
                vixCloseHistory.Add(vixClose);
                lastVixClose = vixClose;
                lastVixDate = vixDate;

                // Keep last 25 days for MA calculations
                while (vixCloseHistory.Count > 25)
                {
                    vixCloseHistory.RemoveAt(0);
                }

                bool wasAvailable = vixDataAvailable;
                vixDataAvailable = vixCloseHistory.Count >= 21;

                // Log when we cross the 21-day threshold
                if (!wasAvailable && vixDataAvailable)
                {
                    Print("======================================================");
                    Print(String.Format("VIX DATA READY: {0} days of history, SENTIMENT FILTER NOW ACTIVE", vixCloseHistory.Count));
                    Print(String.Format("  Latest VIX: {0:F2} on {1}", vixClose, vixDate.ToString("yyyy-MM-dd")));
                    Print("======================================================");
                }

                // Log progress for first 5 days, day 10, day 15, day 20, and day 21
                if (EnableLogging && (vixCloseHistory.Count <= 5 || vixCloseHistory.Count == 10 ||
                    vixCloseHistory.Count == 15 || vixCloseHistory.Count == 20 || vixCloseHistory.Count == 21))
                {
                    Print(String.Format("VIX update: {0} = {1:F2} (history: {2} days, vixDataAvailable={3})",
                        vixDate.ToString("yyyy-MM-dd"), vixClose, vixCloseHistory.Count, vixDataAvailable));
                }
            }
        }

        /// <summary>
        /// Update VIX history from BarsArray when new daily bar arrives.
        /// Called each primary bar to check for new VIX data.
        /// </summary>
        private void UpdateVixFromBarsArray()
        {
            // Use direct BarsArray access to bypass CurrentBars[1] = -1 issue
            if (BarsArray == null || BarsArray.Length < 2)
                return;

            int vixTotalBars = BarsArray[1].Count;
            if (vixTotalBars < 2)
                return;

            try
            {
                // Get the most recent COMPLETED VIX bar using direct access
                // vixTotalBars-1 is the latest (possibly forming), so use vixTotalBars-2
                int mostRecentCompleted = vixTotalBars - 2;
                if (mostRecentCompleted < 0)
                    return;

                double vixClose = BarsArray[1].GetClose(mostRecentCompleted);
                DateTime vixDate = BarsArray[1].GetTime(mostRecentCompleted).Date;

                // Only update if we have a new day's data
                if (vixDate > lastVixDate)
                {
                    vixCloseHistory.Add(vixClose);
                    lastVixClose = vixClose;
                    lastVixDate = vixDate;

                    // Keep last 25 days for MA calculations
                    while (vixCloseHistory.Count > 25)
                    {
                        vixCloseHistory.RemoveAt(0);
                    }

                    // Update vixDataAvailable if we now have enough history
                    if (!vixDataAvailable && vixCloseHistory.Count >= 21)
                    {
                        vixDataAvailable = true;
                        Print("======================================================");
                        Print(String.Format("VIX DATA NOW AVAILABLE: {0} days of history", vixCloseHistory.Count));
                        Print("SENTIMENT FILTER ACTIVATED");
                        Print("======================================================");
                    }

                    if (EnableLogging && vixCloseHistory.Count <= 25)
                    {
                        Print(String.Format("VIX update: {0} = {1:F2} (history: {2} days, available: {3})",
                            vixDate.ToString("yyyy-MM-dd"), vixClose, vixCloseHistory.Count, vixDataAvailable));
                    }
                }
            }
            catch
            {
                // Silently ignore errors during VIX update
            }
        }

        /// <summary>
        /// Check if VIX data is available by directly checking the BarsArray.
        /// CRITICAL FIX: Use BarsArray[1].GetClose(index) instead of Closes[1][barsAgo]
        /// to bypass the CurrentBars[1] = -1 issue with daily bars that haven't fired OnBarUpdate.
        /// </summary>
        private void CheckVixDataAvailability()
        {
            try
            {
                // Check if VIX data series exists
                if (BarsArray == null || BarsArray.Length < 2)
                {
                    if (barCount == 52)
                    {
                        Print("WARNING: VIX data series not loaded. BarsArray length: " +
                            (BarsArray != null ? BarsArray.Length.ToString() : "null"));
                    }
                    return;
                }

                // Get the total number of VIX bars loaded (historical + current)
                // This is the ACTUAL count of bars, regardless of CurrentBars[1]
                int vixTotalBars = BarsArray[1].Count;

                // Log VIX data status on first few checks
                if (barCount <= 55 && barCount % 5 == 0)
                {
                    Print(String.Format("VIX data check (bar {0}): CurrentBars[1]={1}, BarsArray[1].Count={2}, vixHistory={3}",
                        barCount, CurrentBars[1], vixTotalBars, vixCloseHistory.Count));
                }

                // Need at least 22 bars for sentiment calculations (21 for history + 1 for current)
                if (vixTotalBars < 22)
                {
                    if (barCount == 52)
                    {
                        Print(String.Format("VIX data insufficient: only {0} bars in BarsArray[1] (need 22)", vixTotalBars));
                    }
                    return;
                }

                // If we have VIX bars but haven't populated enough history, bulk-load now
                // Use DIRECT BarsArray access to bypass CurrentBars[1] = -1 issue
                if (vixTotalBars >= 22 && vixCloseHistory.Count < 21)
                {
                    // Clear partial history and reload completely
                    vixCloseHistory.Clear();
                    lastVixDate = DateTime.MinValue;
                    lastVixClose = 0;

                    // Load the last 25 COMPLETED bars (excluding the current forming bar)
                    // Bar index is 0-based, so vixTotalBars-1 is the latest bar (possibly forming)
                    // We want vixTotalBars-2 as the most recent COMPLETED bar
                    int mostRecentCompletedBar = vixTotalBars - 2;
                    int barsToLoad = Math.Min(mostRecentCompletedBar + 1, 25);
                    int startIndex = mostRecentCompletedBar - barsToLoad + 1;

                    if (EnableLogging)
                    {
                        Print(String.Format("BULK LOADING VIX: {0} bars from index {1} to {2} (using BarsArray.GetClose)",
                            barsToLoad, startIndex, mostRecentCompletedBar));
                    }

                    // Load from oldest to newest using DIRECT BarsArray access
                    for (int barIndex = startIndex; barIndex <= mostRecentCompletedBar; barIndex++)
                    {
                        if (barIndex >= 0 && barIndex < vixTotalBars)
                        {
                            double vixClose = BarsArray[1].GetClose(barIndex);
                            DateTime vixDate = BarsArray[1].GetTime(barIndex).Date;

                            if (vixDate > lastVixDate)
                            {
                                vixCloseHistory.Add(vixClose);
                                lastVixClose = vixClose;
                                lastVixDate = vixDate;
                            }
                        }
                    }

                    vixDataAvailable = vixCloseHistory.Count >= 21;

                    if (vixDataAvailable)
                    {
                        Print("======================================================");
                        Print(String.Format("VIX DATA BULK LOADED: {0} days of history (using BarsArray.GetClose)", vixCloseHistory.Count));
                        Print("SENTIMENT FILTER NOW ACTIVE");
                        Print(String.Format("  Latest VIX: {0:F2} on {1}", lastVixClose, lastVixDate.ToString("yyyy-MM-dd")));
                        Print(String.Format("  VIX range loaded: index {0} to {1}", startIndex, mostRecentCompletedBar));
                        Print("  Expected trade reduction: ~4,357 → ~2,044");
                        Print("======================================================");
                    }
                    else
                    {
                        Print(String.Format("VIX data insufficient: only {0} unique days loaded (need 21), sentiment filter DISABLED",
                            vixCloseHistory.Count));
                    }
                }
            }
            catch (Exception ex)
            {
                Print("WARNING: Error checking VIX data: " + ex.Message);
                if (ex.StackTrace != null)
                {
                    Print("  Stack: " + ex.StackTrace.Substring(0, Math.Min(200, ex.StackTrace.Length)));
                }
            }
        }

        /// <summary>
        /// Calculate 28 sentiment features from VIX history.
        /// Features match Python historical_sentiment_loader.py exactly.
        /// Uses PREVIOUS day's VIX data to avoid look-ahead bias.
        /// </summary>
        private bool CalculateSentimentFeatures()
        {
            if (!vixDataAvailable || vixCloseHistory.Count < 21)
            {
                return false;
            }

            try
            {
                // Use previous day's VIX (look-ahead bias prevention)
                int n = vixCloseHistory.Count;
                double vixClose = vixCloseHistory[n - 2]; // Previous day

                // VIX moving averages (calculated from history ending at previous day)
                double vixMa5 = CalculateVixSMA(5);
                double vixMa10 = CalculateVixSMA(10);
                double vixMa20 = CalculateVixSMA(20);

                // VIX vs moving averages
                double vixVsMa10 = vixMa10 > 0 ? vixClose / vixMa10 : 1.0;
                double vixVsMa20 = vixMa20 > 0 ? vixClose / vixMa20 : 1.0;

                // VIX percentile (rolling 20-day)
                double vixPercentile20d = CalculateVixPercentile(20);

                // Regime indicators (thresholds from Python)
                double vixFearRegime = vixClose > 25.0 ? 1.0 : 0.0;
                double vixExtremeFear = vixClose > 30.0 ? 1.0 : 0.0;
                double vixComplacency = vixClose < 15.0 ? 1.0 : 0.0;

                // VIX spike detection (15% daily increase)
                double vixPctChange = 0;
                if (n >= 3)
                {
                    double prevVix = vixCloseHistory[n - 3];
                    vixPctChange = prevVix > 0 ? (vixClose - prevVix) / prevVix : 0;
                }
                double vixSpike = vixPctChange > 0.15 ? 1.0 : 0.0;

                // Normalized sentiment (-1 = extreme fear, +1 = extreme complacency)
                double vixSentiment = -Math.Max(-1, Math.Min(1, (vixClose - 20) / 15));

                // Contrarian signal (same as sentiment)
                double vixContrarianSignal = vixSentiment;

                // AAII proxy features (derived from VIX percentile, matching Python)
                double aaiiPct = vixPercentile20d;
                double aaiiBearish = 25 + (aaiiPct * 30); // Range: 25-55%
                double aaiiBullish = 50 - (aaiiPct * 30); // Range: 20-50%
                double aaiiSpread = aaiiBullish - aaiiBearish;
                double aaiiExtremeBullish = aaiiBullish > 50 ? 1.0 : 0.0;
                double aaiiExtremeBearish = aaiiBearish > 50 ? 1.0 : 0.0;
                double aaiiContrarianSignal = (aaiiBearish - aaiiBullish) / 100;

                // PCR proxy features (derived from VIX, matching Python)
                double vixNormalized = Math.Max(0, Math.Min(1, (vixClose - 12) / 25));
                double pcrTotal = 0.6 + (vixNormalized * 0.7); // Range: 0.6 - 1.3
                double pcrMa5 = pcrTotal; // Simplified - would need PCR history
                double pcrMa10 = pcrTotal;
                double pcrBullishExtreme = pcrTotal > 1.1 ? 1.0 : 0.0;
                double pcrBearishExtreme = pcrTotal < 0.8 ? 1.0 : 0.0;
                double pcrContrarianSignal = Math.Max(-1, Math.Min(1, (pcrTotal - 0.95) / 0.35));

                // Composite contrarian signal
                double compositeContrarian = (vixContrarianSignal + aaiiContrarianSignal + pcrContrarianSignal) / 3;

                // Fear and greed regimes
                double fearRegime = (vixFearRegime == 1 || aaiiExtremeBearish == 1 || pcrBullishExtreme == 1) ? 1.0 : 0.0;
                double greedRegime = (vixComplacency == 1 || aaiiExtremeBullish == 1 || pcrBearishExtreme == 1) ? 1.0 : 0.0;

                // Fill sentiment feature buffer (28 features in exact order from Python)
                int idx = 0;
                sentimentFeatureBuffer[idx++] = vixClose;           // sent_vix_close
                sentimentFeatureBuffer[idx++] = vixMa5;             // sent_vix_ma5
                sentimentFeatureBuffer[idx++] = vixMa10;            // sent_vix_ma10
                sentimentFeatureBuffer[idx++] = vixMa20;            // sent_vix_ma20
                sentimentFeatureBuffer[idx++] = vixVsMa10;          // sent_vix_vs_ma10
                sentimentFeatureBuffer[idx++] = vixVsMa20;          // sent_vix_vs_ma20
                sentimentFeatureBuffer[idx++] = vixPercentile20d;   // sent_vix_percentile_20d
                sentimentFeatureBuffer[idx++] = vixFearRegime;      // sent_vix_fear_regime
                sentimentFeatureBuffer[idx++] = vixExtremeFear;     // sent_vix_extreme_fear
                sentimentFeatureBuffer[idx++] = vixComplacency;     // sent_vix_complacency
                sentimentFeatureBuffer[idx++] = vixSpike;           // sent_vix_spike
                sentimentFeatureBuffer[idx++] = vixSentiment;       // sent_vix_sentiment
                sentimentFeatureBuffer[idx++] = vixContrarianSignal;// sent_vix_contrarian_signal
                sentimentFeatureBuffer[idx++] = aaiiBullish;        // sent_aaii_bullish
                sentimentFeatureBuffer[idx++] = aaiiBearish;        // sent_aaii_bearish
                sentimentFeatureBuffer[idx++] = aaiiSpread;         // sent_aaii_spread
                sentimentFeatureBuffer[idx++] = aaiiExtremeBullish; // sent_aaii_extreme_bullish
                sentimentFeatureBuffer[idx++] = aaiiExtremeBearish; // sent_aaii_extreme_bearish
                sentimentFeatureBuffer[idx++] = aaiiContrarianSignal;// sent_aaii_contrarian_signal
                sentimentFeatureBuffer[idx++] = pcrTotal;           // sent_pcr_total
                sentimentFeatureBuffer[idx++] = pcrMa5;             // sent_pcr_ma5
                sentimentFeatureBuffer[idx++] = pcrMa10;            // sent_pcr_ma10
                sentimentFeatureBuffer[idx++] = pcrBullishExtreme;  // sent_pcr_bullish_extreme
                sentimentFeatureBuffer[idx++] = pcrBearishExtreme;  // sent_pcr_bearish_extreme
                sentimentFeatureBuffer[idx++] = pcrContrarianSignal;// sent_pcr_contrarian_signal
                sentimentFeatureBuffer[idx++] = compositeContrarian;// sent_composite_contrarian
                sentimentFeatureBuffer[idx++] = fearRegime;         // sent_fear_regime
                sentimentFeatureBuffer[idx++] = greedRegime;        // sent_greed_regime

                return true;
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                {
                    Print("Sentiment feature error: " + ex.Message);
                }
                return false;
            }
        }

        /// <summary>
        /// Calculate VIX SMA over the specified period (using previous day's data).
        /// </summary>
        private double CalculateVixSMA(int period)
        {
            int n = vixCloseHistory.Count;
            if (n < period + 1) return lastVixClose;

            double sum = 0;
            // Start from n-1-period to n-2 (previous day, excluding current day)
            for (int i = n - 1 - period; i < n - 1; i++)
            {
                if (i >= 0) sum += vixCloseHistory[i];
            }
            return sum / period;
        }

        /// <summary>
        /// Calculate VIX percentile over the specified period.
        /// </summary>
        private double CalculateVixPercentile(int period)
        {
            int n = vixCloseHistory.Count;
            if (n < period + 1) return 0.5;

            double currentVix = vixCloseHistory[n - 2]; // Previous day
            int countBelow = 0;

            for (int i = n - 1 - period; i < n - 1; i++)
            {
                if (i >= 0 && vixCloseHistory[i] < currentVix)
                {
                    countBelow++;
                }
            }

            return (double)countBelow / period;
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
