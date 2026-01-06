/*
SKIE_Ninja TCP Strategy
=======================

NinjaTrader 8 strategy that connects to the Python Signal Server via TCP.

This approach solves the PCR/AAII data availability issue:
- Python has access to all historical sentiment data used in training
- NT8 only sends 42 technical features
- Python calculates sentiment features and runs the full ensemble model
- Returns ShouldTrade, Direction, and probability signals

Usage:
1. Start Python signal server: python -m src.python.signal_server
2. Add this strategy to NinjaTrader chart
3. Strategy will connect to localhost:5555 and receive trading signals

Protocol (JSON over TCP, newline-delimited):
  Request:  {"timestamp": "...", "features": [42 floats], "atr": float}
  Response: {"should_trade": bool, "direction": int, "vol_prob": float, ...}

Author: SKIE_Ninja Development Team
Created: 2025-12-07
*/

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Net.Sockets;
using System.Text;
using System.Threading;
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
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class SKIENinjaTCPStrategy : Strategy
    {
        #region Variables
        // TCP Client
        private TcpClient tcpClient;
        private NetworkStream networkStream;
        private bool isConnected = false;
        private int connectionAttempts = 0;
        private const int MAX_CONNECTION_ATTEMPTS = 3;
        private DateTime lastConnectionAttempt = DateTime.MinValue;
        private const int CONNECTION_RETRY_SECONDS = 30;

        // Feature buffer (42 technical features)
        private double[] featureBuffer;

        // Indicators for feature calculation - MUST MATCH TRAINED MODEL
        private SMA sma5, sma10, sma20, sma50;
        private Bollinger bb20;
        private SMA volSma5, volSma10, volSma20;  // Volume SMAs

        // History buffers for feature calculation
        private List<double> trueRangeHistory;
        private List<double> logReturnHistory;

        // Diagnostic counters
        private int barCount = 0;
        private int signalCount = 0;
        private int tradeCount = 0;
        private int serverSignals = 0;
        private int serverTrades = 0;

        // Position tracking
        private int entryBar = 0;

        // Response parsing
        private StringBuilder receiveBuffer = new StringBuilder();
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Server Host", Description = "Python signal server host", Order = 1, GroupName = "TCP Connection")]
        public string ServerHost { get; set; }

        [NinjaScriptProperty]
        [Range(1, 65535)]
        [Display(Name = "Server Port", Description = "Python signal server port", Order = 2, GroupName = "TCP Connection")]
        public int ServerPort { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Auto Reconnect", Description = "Automatically reconnect if connection lost", Order = 3, GroupName = "TCP Connection")]
        public bool AutoReconnect { get; set; }

        [NinjaScriptProperty]
        [Range(1, 100)]
        [Display(Name = "Max Holding Bars", Description = "Maximum bars to hold position", Order = 1, GroupName = "Trade Management")]
        public int MaxHoldingBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable Logging", Description = "Enable detailed logging", Order = 1, GroupName = "Diagnostics")]
        public bool EnableLogging { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "SKIE_Ninja TCP Strategy - Connects to Python Signal Server";
                Name = "SKIENinjaTCPStrategy";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 2;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Day;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 50;

                // Default parameters
                ServerHost = "127.0.0.1";
                ServerPort = 5555;
                AutoReconnect = true;
                MaxHoldingBars = 20;
                EnableLogging = true;
            }
            else if (State == State.Configure)
            {
                // Add 5-minute secondary series if needed for feature calculation
            }
            else if (State == State.DataLoaded)
            {
                // Initialize indicators - MUST MATCH TRAINED MODEL FEATURES
                sma5 = SMA(5);
                sma10 = SMA(10);
                sma20 = SMA(20);
                sma50 = SMA(50);
                bb20 = Bollinger(2, 20);

                // Volume SMAs for volume_ratio features
                volSma5 = SMA(Volume, 5);
                volSma10 = SMA(Volume, 10);
                volSma20 = SMA(Volume, 20);

                // Initialize buffers
                featureBuffer = new double[42];
                trueRangeHistory = new List<double>();
                logReturnHistory = new List<double>();

                // Connect to server
                ConnectToServer();

                Print("======================================================");
                Print("SKIE_Ninja TCP Strategy Initialized");
                Print("======================================================");
                Print("Server: " + ServerHost + ":" + ServerPort);
                Print("Connection: " + (isConnected ? "CONNECTED" : "DISCONNECTED"));
                Print("======================================================");
            }
            else if (State == State.Terminated)
            {
                DisconnectFromServer();

                Print("======================================================");
                Print("SKIE_Ninja TCP Strategy FINAL SUMMARY");
                Print("======================================================");
                Print("  Bars processed: " + barCount);
                Print("  Signals from server: " + serverSignals);
                Print("  Trades from server: " + serverTrades);
                Print("  Trades executed: " + tradeCount);
                Print("======================================================");
            }
        }

        private void ConnectToServer()
        {
            if (isConnected)
                return;

            if ((DateTime.Now - lastConnectionAttempt).TotalSeconds < CONNECTION_RETRY_SECONDS)
                return;

            lastConnectionAttempt = DateTime.Now;
            connectionAttempts++;

            try
            {
                Print("Connecting to Python Signal Server at " + ServerHost + ":" + ServerPort + "...");

                tcpClient = new TcpClient();
                tcpClient.Connect(ServerHost, ServerPort);
                networkStream = tcpClient.GetStream();
                tcpClient.ReceiveTimeout = 5000;
                tcpClient.SendTimeout = 5000;

                isConnected = true;
                connectionAttempts = 0;

                Print("SUCCESS: Connected to Python Signal Server");
            }
            catch (Exception ex)
            {
                Print("ERROR: Failed to connect to server: " + ex.Message);
                isConnected = false;

                if (connectionAttempts >= MAX_CONNECTION_ATTEMPTS)
                {
                    Print("WARNING: Max connection attempts reached. Will retry in " + CONNECTION_RETRY_SECONDS + " seconds.");
                    connectionAttempts = 0;
                }
            }
        }

        private void DisconnectFromServer()
        {
            try
            {
                if (networkStream != null)
                {
                    networkStream.Close();
                    networkStream = null;
                }

                if (tcpClient != null)
                {
                    tcpClient.Close();
                    tcpClient = null;
                }

                isConnected = false;
                Print("Disconnected from Python Signal Server");
            }
            catch (Exception ex)
            {
                Print("Error disconnecting: " + ex.Message);
            }
        }

        protected override void OnBarUpdate()
        {
            barCount++;

            // Skip if not enough bars
            if (CurrentBar < BarsRequiredToTrade)
                return;

            // Try to reconnect if needed
            if (!isConnected && AutoReconnect)
            {
                ConnectToServer();
            }

            // Time-based exit
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                int barsInPosition = CurrentBar - entryBar;
                if (barsInPosition >= MaxHoldingBars)
                {
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("TimeExit", "LongEntry");
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort("TimeExit", "ShortEntry");

                    if (EnableLogging)
                        Print("[" + Time[0] + "] TIME EXIT after " + barsInPosition + " bars");
                    return;
                }
            }

            // Only enter new trades if flat
            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // Calculate features
            if (!CalculateFeatures())
            {
                if (EnableLogging && barCount % 100 == 0)
                    Print("Feature calculation failed at bar " + CurrentBar);
                return;
            }

            // Get signal from Python server
            var signal = GetSignalFromServer();

            if (signal == null)
            {
                if (EnableLogging && barCount % 100 == 0)
                    Print("No signal from server at bar " + CurrentBar);
                return;
            }

            serverSignals++;

            // Execute trade if signal passes
            bool shouldTrade = signal.ContainsKey("should_trade") && (bool)signal["should_trade"];
            int direction = signal.ContainsKey("direction") ? Convert.ToInt32(signal["direction"]) : 0;

            if (shouldTrade && direction != 0)
            {
                serverTrades++;

                double tpOffset = signal.ContainsKey("tp_offset") ? Convert.ToDouble(signal["tp_offset"]) : 0;
                double slOffset = signal.ContainsKey("sl_offset") ? Convert.ToDouble(signal["sl_offset"]) : 0;

                if (direction > 0)
                {
                    EnterLong(1, "LongEntry");
                    if (tpOffset > 0) SetProfitTarget("LongEntry", CalculationMode.Ticks, tpOffset / TickSize);
                    if (slOffset > 0) SetStopLoss("LongEntry", CalculationMode.Ticks, slOffset / TickSize, false);
                    entryBar = CurrentBar;
                    tradeCount++;

                    if (EnableLogging)
                    {
                        double volProb = signal.ContainsKey("vol_prob") ? Convert.ToDouble(signal["vol_prob"]) : 0;
                        double sentProb = signal.ContainsKey("sent_prob") ? Convert.ToDouble(signal["sent_prob"]) : 0;
                        Print(String.Format("[{0}] LONG SIGNAL | VolProb={1:F3}, SentProb={2:F3}",
                            Time[0], volProb, sentProb));
                    }
                }
                else if (direction < 0)
                {
                    EnterShort(1, "ShortEntry");
                    if (tpOffset > 0) SetProfitTarget("ShortEntry", CalculationMode.Ticks, tpOffset / TickSize);
                    if (slOffset > 0) SetStopLoss("ShortEntry", CalculationMode.Ticks, slOffset / TickSize, false);
                    entryBar = CurrentBar;
                    tradeCount++;

                    if (EnableLogging)
                    {
                        double volProb = signal.ContainsKey("vol_prob") ? Convert.ToDouble(signal["vol_prob"]) : 0;
                        double sentProb = signal.ContainsKey("sent_prob") ? Convert.ToDouble(signal["sent_prob"]) : 0;
                        Print(String.Format("[{0}] SHORT SIGNAL | VolProb={1:F3}, SentProb={2:F3}",
                            Time[0], volProb, sentProb));
                    }
                }
            }

            // Periodic status logging
            if (EnableLogging && barCount % 500 == 0)
            {
                Print(String.Format("[{0}] Status: Bars={1}, ServerSignals={2}, ServerTrades={3}, Executed={4}",
                    Time[0], barCount, serverSignals, serverTrades, tradeCount));
            }
        }

        private Dictionary<string, object> GetSignalFromServer()
        {
            if (!isConnected)
                return null;

            try
            {
                // Build request
                var request = new Dictionary<string, object>
                {
                    { "timestamp", Time[0].ToString("o") },
                    { "features", featureBuffer },
                    { "atr", GetATR(14) }
                };

                string requestJson = JsonConvert.SerializeObject(request) + "\n";
                byte[] requestBytes = Encoding.UTF8.GetBytes(requestJson);

                // Send request
                networkStream.Write(requestBytes, 0, requestBytes.Length);

                // Read response
                byte[] responseBuffer = new byte[4096];
                int bytesRead = networkStream.Read(responseBuffer, 0, responseBuffer.Length);

                if (bytesRead == 0)
                {
                    isConnected = false;
                    return null;
                }

                string responseJson = Encoding.UTF8.GetString(responseBuffer, 0, bytesRead).Trim();

                // Parse response
                var response = JsonConvert.DeserializeObject<Dictionary<string, object>>(responseJson);

                return response;
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print("Server communication error: " + ex.Message);

                isConnected = false;
                return null;
            }
        }

        private bool CalculateFeatures()
        {
            try
            {
                // Update history buffers
                double trueRange = Math.Max(High[0] - Low[0],
                    Math.Max(Math.Abs(High[0] - Close[1]), Math.Abs(Low[0] - Close[1])));
                trueRangeHistory.Add(trueRange);
                if (trueRangeHistory.Count > 55)
                    trueRangeHistory.RemoveAt(0);

                double logReturn = Close[1] > 0 ? Math.Log(Close[0] / Close[1]) : 0;
                logReturnHistory.Add(logReturn);
                if (logReturnHistory.Count > 55)
                    logReturnHistory.RemoveAt(0);

                // ================================================================
                // FEATURES MUST MATCH EXACT ORDER FROM scaler_params.json
                // Trained model expects 42 features in specific order
                // ================================================================
                int idx = 0;

                // 1. Return features (6 features): return_lag1, lag2, lag3, lag5, lag10, lag20
                featureBuffer[idx++] = Close[1] > 0 ? (Close[0] - Close[1]) / Close[1] : 0;  // return_lag1
                featureBuffer[idx++] = Close[2] > 0 ? (Close[0] - Close[2]) / Close[2] : 0;  // return_lag2
                featureBuffer[idx++] = Close[3] > 0 ? (Close[0] - Close[3]) / Close[3] : 0;  // return_lag3
                featureBuffer[idx++] = Close[5] > 0 ? (Close[0] - Close[5]) / Close[5] : 0;  // return_lag5
                featureBuffer[idx++] = Close[10] > 0 ? (Close[0] - Close[10]) / Close[10] : 0;  // return_lag10
                featureBuffer[idx++] = Close[20] > 0 ? (Close[0] - Close[20]) / Close[20] : 0;  // return_lag20

                // 2. Volatility features period 5 (3 features): rv_5, atr_5, atr_pct_5
                featureBuffer[idx++] = GetRealizedVolatility(5);  // rv_5
                double atr5 = GetATR(5);
                featureBuffer[idx++] = atr5;  // atr_5
                featureBuffer[idx++] = Close[0] > 0 ? atr5 / Close[0] : 0;  // atr_pct_5

                // 3. Volatility features period 10 (3 features): rv_10, atr_10, atr_pct_10
                featureBuffer[idx++] = GetRealizedVolatility(10);  // rv_10
                double atr10 = GetATR(10);
                featureBuffer[idx++] = atr10;  // atr_10
                featureBuffer[idx++] = Close[0] > 0 ? atr10 / Close[0] : 0;  // atr_pct_10

                // 4. Volatility features period 14 (3 features): rv_14, atr_14, atr_pct_14
                featureBuffer[idx++] = GetRealizedVolatility(14);  // rv_14
                double atr14 = GetATR(14);
                featureBuffer[idx++] = atr14;  // atr_14
                featureBuffer[idx++] = Close[0] > 0 ? atr14 / Close[0] : 0;  // atr_pct_14

                // 5. Volatility features period 20 (3 features): rv_20, atr_20, atr_pct_20
                featureBuffer[idx++] = GetRealizedVolatility(20);  // rv_20
                double atr20 = GetATR(20);
                featureBuffer[idx++] = atr20;  // atr_20
                featureBuffer[idx++] = Close[0] > 0 ? atr20 / Close[0] : 0;  // atr_pct_20

                // 6. Price position period 10 (3 features): close_vs_high_10, close_vs_low_10, range_pct_10
                double high10 = GetHighestHigh(10);
                double low10 = GetLowestLow(10);
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - high10) / Close[0] : 0;  // close_vs_high_10
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - low10) / Close[0] : 0;   // close_vs_low_10
                featureBuffer[idx++] = Close[0] > 0 ? (high10 - low10) / Close[0] : 0;     // range_pct_10

                // 7. Price position period 20 (3 features): close_vs_high_20, close_vs_low_20, range_pct_20
                double high20 = GetHighestHigh(20);
                double low20 = GetLowestLow(20);
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - high20) / Close[0] : 0;  // close_vs_high_20
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - low20) / Close[0] : 0;   // close_vs_low_20
                featureBuffer[idx++] = Close[0] > 0 ? (high20 - low20) / Close[0] : 0;     // range_pct_20

                // 8. Price position period 50 (3 features): close_vs_high_50, close_vs_low_50, range_pct_50
                double high50 = GetHighestHigh(50);
                double low50 = GetLowestLow(50);
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - high50) / Close[0] : 0;  // close_vs_high_50
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - low50) / Close[0] : 0;   // close_vs_low_50
                featureBuffer[idx++] = Close[0] > 0 ? (high50 - low50) / Close[0] : 0;     // range_pct_50

                // 9. Momentum period 5 (2 features): momentum_5, ma_dist_5
                featureBuffer[idx++] = Close[0] - Close[5];  // momentum_5
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - sma5[0]) / Close[0] : 0;  // ma_dist_5

                // 10. Momentum period 10 (2 features): momentum_10, ma_dist_10
                featureBuffer[idx++] = Close[0] - Close[10];  // momentum_10
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - sma10[0]) / Close[0] : 0;  // ma_dist_10

                // 11. Momentum period 20 (2 features): momentum_20, ma_dist_20
                featureBuffer[idx++] = Close[0] - Close[20];  // momentum_20
                featureBuffer[idx++] = Close[0] > 0 ? (Close[0] - sma20[0]) / Close[0] : 0;  // ma_dist_20

                // 12. RSI features (2 features): rsi_7, rsi_14
                featureBuffer[idx++] = GetRSI(7);   // rsi_7
                featureBuffer[idx++] = GetRSI(14);  // rsi_14

                // 13. Bollinger Band (1 feature): bb_pct_20
                double bbWidth = bb20.Upper[0] - bb20.Lower[0];
                featureBuffer[idx++] = bbWidth > 0 ? (Close[0] - bb20.Lower[0]) / bbWidth : 0.5;  // bb_pct_20

                // 14. Volume features period 5 (2 features): volume_sma_5, volume_ratio_5
                featureBuffer[idx++] = volSma5[0];  // volume_sma_5
                featureBuffer[idx++] = volSma5[0] > 0 ? Volume[0] / volSma5[0] : 1.0;  // volume_ratio_5

                // 15. Volume features period 10 (2 features): volume_sma_10, volume_ratio_10
                featureBuffer[idx++] = volSma10[0];  // volume_sma_10
                featureBuffer[idx++] = volSma10[0] > 0 ? Volume[0] / volSma10[0] : 1.0;  // volume_ratio_10

                // 16. Volume features period 20 (2 features): volume_sma_20, volume_ratio_20
                featureBuffer[idx++] = volSma20[0];  // volume_sma_20
                featureBuffer[idx++] = volSma20[0] > 0 ? Volume[0] / volSma20[0] : 1.0;  // volume_ratio_20

                // Verify feature count (should be 42)
                if (idx != 42)
                {
                    Print("ERROR: Feature count mismatch. Expected 42, got " + idx);
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                if (EnableLogging)
                    Print("Feature calculation error: " + ex.Message);
                return false;
            }
        }

        #region Helper Methods
        private double GetATR(int period)
        {
            if (trueRangeHistory.Count < period)
                return 0;

            double sum = 0;
            int start = trueRangeHistory.Count - period;
            for (int i = start; i < trueRangeHistory.Count; i++)
                sum += trueRangeHistory[i];

            return sum / period;
        }

        private double GetRealizedVolatility(int period)
        {
            // Realized volatility = std(log returns) * sqrt(252)
            // Matches Python: df['close'].pct_change().rolling(period).std() * np.sqrt(252)
            if (logReturnHistory.Count < period)
                return 0;

            int start = logReturnHistory.Count - period;
            double sum = 0;
            for (int i = start; i < logReturnHistory.Count; i++)
                sum += logReturnHistory[i];
            double mean = sum / period;

            double variance = 0;
            for (int i = start; i < logReturnHistory.Count; i++)
                variance += Math.Pow(logReturnHistory[i] - mean, 2);
            variance /= (period - 1);

            return Math.Sqrt(variance) * Math.Sqrt(252);
        }

        private double GetRSI(int period)
        {
            if (CurrentBar < period)
                return 50;

            double gainSum = 0, lossSum = 0;
            for (int i = 0; i < period; i++)
            {
                double change = Close[i] - Close[i + 1];
                if (change > 0)
                    gainSum += change;
                else
                    lossSum -= change;
            }

            double avgGain = gainSum / period;
            double avgLoss = lossSum / period;

            if (avgLoss == 0)
                return 100;

            double rs = avgGain / avgLoss;
            return 100 - (100 / (1 + rs));
        }

        private double GetHighestHigh(int period)
        {
            double highest = High[0];
            for (int i = 1; i < period && i <= CurrentBar; i++)
            {
                if (High[i] > highest)
                    highest = High[i];
            }
            return highest;
        }

        private double GetLowestLow(int period)
        {
            double lowest = Low[0];
            for (int i = 1; i < period && i <= CurrentBar; i++)
            {
                if (Low[i] < lowest)
                    lowest = Low[i];
            }
            return lowest;
        }
        #endregion
    }
}
