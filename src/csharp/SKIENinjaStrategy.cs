/*
 * SKIE_Ninja Volatility Breakout Strategy
 *
 * NinjaTrader 8 strategy using ONNX ML models for:
 * - Volatility expansion prediction (entry filter)
 * - Breakout direction prediction (trade direction)
 * - ATR forecasting (dynamic exits)
 *
 * Author: SKIE_Ninja Development Team
 * Created: 2025-12-05
 * Updated: 2025-12-06 - Fixed settings, added diagnostics
 * Updated: 2025-12-06 - Fixed critical SetProfitTarget/SetStopLoss order bug,
 *                       added time-based exit logic, updated optimized thresholds
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
    public class SKIENinjaStrategy : Strategy
    {
        #region Variables
        private SKIENinjaPredictor predictor;
        private double[] featureBuffer;
        private bool predictorReady = false;

        // Indicators for feature calculation
        private SMA sma5, sma10, sma20, sma50;
        // NOTE: We calculate ATR manually using SMA (not Wilder smoothing)
        // to match Python training code
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
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Model Path", Description = "Path to ONNX model folder",
            Order = 1, GroupName = "1. ML Settings")]
        public string ModelPath { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 0.9)]
        [Display(Name = "Min Vol Expansion Prob", Description = "Minimum volatility expansion probability (0.40 = optimized)",
            Order = 2, GroupName = "2. Strategy Settings")]
        public double MinVolExpansionProb { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 0.9)]
        [Display(Name = "Min Breakout Prob", Description = "Minimum breakout probability (0.45 = optimized)",
            Order = 3, GroupName = "2. Strategy Settings")]
        public double MinBreakoutProb { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Max Contracts", Description = "Maximum position size",
            Order = 4, GroupName = "2. Strategy Settings")]
        public int MaxContracts { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Max Holding Bars", Description = "Maximum bars to hold position",
            Order = 5, GroupName = "2. Strategy Settings")]
        public int MaxHoldingBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Log all signals to output window",
            Order = 6, GroupName = "3. Debug")]
        public bool EnableLogging { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "SKIE_Ninja Volatility Breakout Strategy with ML";
                Name = "SKIENinjaStrategy";
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

                // === CORRECT DEFAULT SETTINGS (matching Python config) ===
                ModelPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
                    "SKIE_Ninja", "models");
                MinVolExpansionProb = 0.40;  // Optimized (Phase 12)
                MinBreakoutProb = 0.45;      // Optimized (Phase 12)
                MaxContracts = 3;
                MaxHoldingBars = 20;
                EnableLogging = true;        // Enable by default for debugging
            }
            else if (State == State.Configure)
            {
                // No additional data series needed - uses primary series
            }
            else if (State == State.DataLoaded)
            {
                Print("SKIE_Ninja: State.DataLoaded - Starting initialization...");

                // Initialize indicators
                sma5 = SMA(5);
                sma10 = SMA(10);
                sma20 = SMA(20);
                sma50 = SMA(50);

                // NOTE: We do NOT use NinjaTrader's ATR indicator because it uses
                // Wilder smoothing. Python training uses Simple Moving Average of
                // True Range. We calculate ATR manually to match.

                bb20 = Bollinger(2, 20);
                Print("SKIE_Ninja: Indicators initialized");

                // Initialize history buffers
                closeHistory = new List<double>();
                highHistory = new List<double>();
                lowHistory = new List<double>();
                volumeHistory = new List<double>();
                trueRangeHistory = new List<double>();
                Print("SKIE_Ninja: History buffers initialized");

                // Initialize predictor
                predictor = new SKIENinjaPredictor();
                predictorReady = false;
                Print("SKIE_Ninja: Predictor created, attempting to load models...");
                Print("SKIE_Ninja: Model Path = " + ModelPath);

                try
                {
                    // Check if path exists
                    if (!System.IO.Directory.Exists(ModelPath))
                    {
                        Print("SKIE_Ninja ERROR: Model path does not exist: " + ModelPath);
                        return;
                    }
                    Print("SKIE_Ninja: Model path exists, loading models...");

                    predictor.Initialize(ModelPath);
                    int featureCount = predictor.GetFeatureCount();
                    featureBuffer = new double[featureCount];
                    predictorReady = true;

                    Print("========================================");
                    Print("SKIE_Ninja Strategy Initialized");
                    Print("========================================");
                    Print("Model Path: " + ModelPath);
                    Print("Features: " + featureCount);
                    Print("Min Vol Prob: " + MinVolExpansionProb);
                    Print("Min Breakout Prob: " + MinBreakoutProb);
                    Print("Max Contracts: " + MaxContracts);
                    Print("Max Holding Bars: " + MaxHoldingBars);
                    Print("========================================");
                }
                catch (Exception ex)
                {
                    Print("SKIE_Ninja ERROR: Failed to initialize predictor");
                    Print("Error Type: " + ex.GetType().Name);
                    Print("Error: " + ex.Message);
                    if (ex.InnerException != null)
                    {
                        Print("Inner Error: " + ex.InnerException.Message);
                    }
                    Print("Stack: " + ex.StackTrace);
                    Print("Model Path: " + ModelPath);
                }
            }
            else if (State == State.Terminated)
            {
                if (predictor != null)
                {
                    predictor.Dispose();
                }

                Print("========================================");
                Print("SKIE_Ninja Strategy Terminated");
                Print("Bars processed: " + barCount);
                Print("Signals generated: " + signalCount);
                Print("Trades executed: " + tradeCount);
                Print("========================================");
            }
        }

        protected override void OnBarUpdate()
        {
            barCount++;

            // Check if predictor is ready
            if (!predictorReady)
            {
                if (barCount == 1)
                    Print("SKIE_Ninja: Predictor not ready - skipping");
                return;
            }

            // Update price history first
            UpdateHistory();

            // Wait for enough history
            if (closeHistory.Count < 51)
            {
                if (EnableLogging && barCount % 10 == 0)
                    Print("SKIE_Ninja: Building history... " + closeHistory.Count + "/51 bars");
                return;
            }

            // Handle existing position - check for time-based exit
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                // Time-based exit: close position if held too long
                int barsHeld = CurrentBar - entryBar;
                if (barsHeld >= MaxHoldingBars)
                {
                    if (EnableLogging)
                    {
                        Print(String.Format(">>> TIME EXIT [{0}]: Held {1} bars (max={2})",
                            Time[0], barsHeld, MaxHoldingBars));
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
                    Print("SKIE_Ninja: Feature calculation failed at bar " + CurrentBar);
                return;
            }

            // Get ML prediction
            TradingSignal signal;
            try
            {
                // Use our manually calculated ATR (SMA of TR) to match Python training
                double currentATR = GetATR(14);
                signal = predictor.GenerateSignal(featureBuffer, currentATR);
            }
            catch (Exception ex)
            {
                Print("SKIE_Ninja: Prediction error - " + ex.Message);
                return;
            }

            // Log every Nth bar for diagnostics
            if (EnableLogging && barCount % 50 == 0)
            {
                Print(String.Format("[{0}] Bar {1}: VolProb={2:F3}, HighProb={3:F3}, Dir={4}, Trade={5}",
                    Time[0], barCount, signal.VolExpansionProb, signal.BreakoutProb,
                    signal.Direction, signal.ShouldTrade));

                // Log key features for comparison with Python (every 500 bars)
                if (barCount % 500 == 0)
                {
                    Print(String.Format("  Features: rv5={0:F6}, atr14={1:F4}, rsi14={2:F2}, bb_pct={3:F4}",
                        featureBuffer[6], featureBuffer[13], featureBuffer[35], featureBuffer[37]));
                }
            }

            // Check if signal passes our thresholds
            // Note: The predictor uses its own config thresholds internally
            // We apply additional filtering here if needed
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
            // tr1 = high - low
            // tr2 = abs(high - prev_close)
            // tr3 = abs(low - prev_close)
            // tr = max(tr1, tr2, tr3)
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

            // Keep last 51 bars (need 50 for lookback + current)
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

                // 2. Volatility features - 12 features (4 periods x 3 metrics)
                int[] volPeriods = { 5, 10, 14, 20 };
                foreach (int period in volPeriods)
                {
                    double rv = CalculateRealizedVol(period);
                    featureBuffer[idx++] = rv;

                    double atr = GetATR(period);
                    featureBuffer[idx++] = atr;

                    featureBuffer[idx++] = Close[0] != 0 ? atr / Close[0] : 0;
                }

                // 3. Price position - 9 features (3 periods x 3 metrics)
                int[] pricePeriods = { 10, 20, 50 };
                foreach (int period in pricePeriods)
                {
                    double highMax = MAX(High, period)[0];
                    double lowMin = MIN(Low, period)[0];

                    featureBuffer[idx++] = Close[0] != 0 ? (Close[0] - highMax) / Close[0] : 0;
                    featureBuffer[idx++] = Close[0] != 0 ? (Close[0] - lowMin) / Close[0] : 0;
                    featureBuffer[idx++] = Close[0] != 0 ? (highMax - lowMin) / Close[0] : 0;
                }

                // 4. Momentum - 6 features (3 periods x 2 metrics)
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

                // 5. RSI - 2 features (manual calculation matching Python)
                featureBuffer[idx++] = CalculateRSI(7);
                featureBuffer[idx++] = CalculateRSI(14);

                // 6. Bollinger Band position - 1 feature
                double bbUpper = bb20.Upper[0];
                double bbLower = bb20.Lower[0];
                double bbRange = bbUpper - bbLower;
                featureBuffer[idx++] = bbRange > 0.0001 ? (Close[0] - bbLower) / bbRange : 0.5;

                // 7. Volume features - 6 features (3 periods x 2 metrics)
                int[] volumePeriods = { 5, 10, 20 };
                foreach (int period in volumePeriods)
                {
                    double volSma = CalculateVolumeSMA(period);
                    featureBuffer[idx++] = volSma;
                    featureBuffer[idx++] = Volume[0] / (volSma + 1);
                }

                // Verify we filled all 42 features
                if (idx != 42)
                {
                    Print("SKIE_Ninja ERROR: Expected 42 features, got " + idx);
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                Print("SKIE_Ninja: Feature error - " + ex.Message);
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
            // Use SAMPLE standard deviation (ddof=1) to match Python pandas.std()
            // Divide by (n-1), not n
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
            // Calculate ATR as Simple Moving Average of True Range
            // This matches Python: tr.rolling(period).mean()
            // NOT NinjaTrader's Wilder smoothing
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
            // per NinjaTrader documentation. Otherwise, stale values from
            // previous trades may be used instead of current signal values.
            // See: https://ninjatrader.com/support/helpguides/nt8/setprofittarget.htm

            if (signal.Direction == 1)
            {
                // Set TP/SL BEFORE entry (NT8 requirement)
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
                // Set TP/SL BEFORE entry (NT8 requirement)
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
