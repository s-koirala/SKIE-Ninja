/*
SKIE Ninja Strategy - NinjaTrader 8 Client
==========================================

NinjaScript strategy that communicates with Python signal server via TCP socket.
Receives trade signals and executes orders through NinjaTrader.

Installation:
1. Copy this file to: Documents\NinjaTrader 8\bin\Custom\Strategies\
2. In NinjaTrader: Right-click Strategies folder > Compile
3. Start Python server: python ninja_signal_server.py
4. Apply strategy to ES 5-minute chart

Author: SKIE_Ninja Development Team
Created: 2025-12-15
*/

#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Xml.Serialization;
using Newtonsoft.Json;
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
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class SKIENinjaStrategy : Strategy
    {
        #region Variables
        private TcpClient tcpClient;
        private NetworkStream stream;
        private bool isConnected = false;
        private DateTime lastHeartbeat;
        private int currentPosition = 0;

        // VIX data series
        private double lastVixClose = 0;

        // Kill switch
        private double dailyPnL = 0;
        private DateTime lastResetDate;

        // Trade tracking
        private int tradesExecutedToday = 0;
        private Order entryOrder = null;
        private Order stopOrder = null;
        private Order targetOrder = null;

        // Heartbeat timer
        private System.Timers.Timer heartbeatTimer;
        private const int HEARTBEAT_INTERVAL_MS = 30000; // 30 seconds
        #endregion

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Python Server Host", Order = 1, GroupName = "Connection")]
        public string ServerHost { get; set; }

        [NinjaScriptProperty]
        [Range(1, 65535)]
        [Display(Name = "Python Server Port", Order = 2, GroupName = "Connection")]
        public int ServerPort { get; set; }

        [NinjaScriptProperty]
        [Range(1, 10)]
        [Display(Name = "Contracts Per Trade", Order = 3, GroupName = "Risk Management")]
        public int ContractsPerTrade { get; set; }

        [NinjaScriptProperty]
        [Range(100, 50000)]
        [Display(Name = "Daily Loss Limit ($)", Order = 4, GroupName = "Risk Management")]
        public double DailyLossLimit { get; set; }

        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "Max Trades Per Day", Order = 5, GroupName = "Risk Management")]
        public int MaxTradesPerDay { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Enable VIX Data", Order = 6, GroupName = "Data")]
        public bool EnableVixData { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "VIX Symbol", Order = 7, GroupName = "Data")]
        public string VixSymbol { get; set; }
        #endregion

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "SKIE Ninja Ensemble Strategy - Python Socket Bridge";
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
                TimeInForce = TimeInForce.Day;
                TraceOrders = true;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 200;
                IsUnmanaged = false;

                // Default parameters
                ServerHost = "localhost";
                ServerPort = 5555;
                ContractsPerTrade = 1;
                DailyLossLimit = 5000;
                MaxTradesPerDay = 10;
                EnableVixData = true;
                VixSymbol = "^VIX";
            }
            else if (State == State.Configure)
            {
                // Add VIX data series if enabled
                if (EnableVixData && !string.IsNullOrEmpty(VixSymbol))
                {
                    AddDataSeries(VixSymbol, Data.BarsPeriodType.Day, 1);
                }
            }
            else if (State == State.DataLoaded)
            {
                lastResetDate = DateTime.MinValue;
            }
            else if (State == State.Realtime)
            {
                // Connect to Python server when going live
                ConnectToServer();

                // Start heartbeat timer
                StartHeartbeatTimer();
            }
            else if (State == State.Terminated)
            {
                StopHeartbeatTimer();
                DisconnectFromServer();
            }
        }

        protected override void OnBarUpdate()
        {
            // Only process primary series (ES)
            if (BarsInProgress != 0)
            {
                // Update VIX close from secondary series
                if (BarsInProgress == 1 && EnableVixData)
                {
                    lastVixClose = Close[0];
                }
                return;
            }

            // Reset daily stats
            if (Time[0].Date != lastResetDate)
            {
                dailyPnL = 0;
                tradesExecutedToday = 0;
                lastResetDate = Time[0].Date;
                Print($"[SKIE] Daily reset - Date: {Time[0].Date:yyyy-MM-dd}");
            }

            // Kill switch check
            if (dailyPnL <= -DailyLossLimit)
            {
                Print($"[SKIE] KILL SWITCH - Daily loss limit reached: ${dailyPnL:F2}");
                return;
            }

            // Max trades check
            if (tradesExecutedToday >= MaxTradesPerDay)
            {
                return;
            }

            // Only process during RTH
            if (!IsRthSession())
            {
                return;
            }

            // Need connection to Python server
            if (!isConnected)
            {
                if (State == State.Realtime)
                {
                    ConnectToServer();
                }
                return;
            }

            // Send bar data to Python and get signal
            var signal = GetSignalFromPython();

            if (signal == null)
            {
                return;
            }

            // Execute signal
            ExecuteSignal(signal);
        }

        private bool IsRthSession()
        {
            // RTH: 9:30 AM - 4:00 PM ET
            var time = Time[0].TimeOfDay;
            return time >= new TimeSpan(9, 30, 0) && time < new TimeSpan(16, 0, 0);
        }

        private void ConnectToServer()
        {
            try
            {
                if (isConnected) return;

                tcpClient = new TcpClient();
                tcpClient.Connect(ServerHost, ServerPort);
                stream = tcpClient.GetStream();
                stream.ReadTimeout = 5000;
                stream.WriteTimeout = 5000;
                isConnected = true;
                lastHeartbeat = DateTime.Now;

                Print($"[SKIE] Connected to Python server at {ServerHost}:{ServerPort}");

                // Send status request
                var statusRequest = JsonConvert.SerializeObject(new { type = "STATUS" });
                SendMessage(statusRequest);
                var statusResponse = ReceiveMessage();
                Print($"[SKIE] Server status: {statusResponse}");
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Connection failed: {ex.Message}");
                isConnected = false;
            }
        }

        private void DisconnectFromServer()
        {
            try
            {
                if (stream != null) stream.Close();
                if (tcpClient != null) tcpClient.Close();
                isConnected = false;
                Print("[SKIE] Disconnected from Python server");
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Disconnect error: {ex.Message}");
            }
        }

        private void StartHeartbeatTimer()
        {
            try
            {
                heartbeatTimer = new System.Timers.Timer(HEARTBEAT_INTERVAL_MS);
                heartbeatTimer.Elapsed += OnHeartbeatTimer;
                heartbeatTimer.AutoReset = true;
                heartbeatTimer.Enabled = true;
                Print($"[SKIE] Heartbeat timer started (interval: {HEARTBEAT_INTERVAL_MS / 1000}s)");
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Failed to start heartbeat timer: {ex.Message}");
            }
        }

        private void StopHeartbeatTimer()
        {
            try
            {
                if (heartbeatTimer != null)
                {
                    heartbeatTimer.Stop();
                    heartbeatTimer.Dispose();
                    heartbeatTimer = null;
                    Print("[SKIE] Heartbeat timer stopped");
                }
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Error stopping heartbeat timer: {ex.Message}");
            }
        }

        private void OnHeartbeatTimer(object sender, System.Timers.ElapsedEventArgs e)
        {
            if (!isConnected) return;

            try
            {
                var heartbeat = JsonConvert.SerializeObject(new { type = "HEARTBEAT" });
                SendMessage(heartbeat);
                var response = ReceiveMessage();

                if (!string.IsNullOrEmpty(response) && response.Contains("HEARTBEAT_ACK"))
                {
                    lastHeartbeat = DateTime.Now;
                }
                else
                {
                    Print("[SKIE] Heartbeat failed - no ACK received");
                    // Attempt reconnection
                    isConnected = false;
                    ConnectToServer();
                }
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Heartbeat error: {ex.Message}");
                isConnected = false;
            }
        }

        private void SendMessage(string message)
        {
            if (!isConnected || stream == null) return;

            byte[] data = Encoding.UTF8.GetBytes(message);
            stream.Write(data, 0, data.Length);
        }

        private string ReceiveMessage()
        {
            if (!isConnected || stream == null) return null;

            byte[] buffer = new byte[4096];
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            return Encoding.UTF8.GetString(buffer, 0, bytesRead);
        }

        private TradeSignal GetSignalFromPython()
        {
            try
            {
                // Track current position
                currentPosition = Position.MarketPosition == MarketPosition.Long ? 1 :
                                  Position.MarketPosition == MarketPosition.Short ? -1 : 0;

                // Build request
                var request = new
                {
                    type = "BAR_UPDATE",
                    timestamp = Time[0].ToString("yyyy-MM-ddTHH:mm:ss"),
                    symbol = Instrument.FullName,
                    timeframe = "5min",
                    open = Open[0],
                    high = High[0],
                    low = Low[0],
                    close = Close[0],
                    volume = (long)Volume[0],
                    vix_close = lastVixClose,
                    position = currentPosition
                };

                string requestJson = JsonConvert.SerializeObject(request);
                SendMessage(requestJson);

                string responseJson = ReceiveMessage();
                if (string.IsNullOrEmpty(responseJson))
                {
                    return null;
                }

                var signal = JsonConvert.DeserializeObject<TradeSignal>(responseJson);

                // Log signal
                if (signal.action != "FLAT")
                {
                    Print($"[SKIE] Signal: {signal.action} | Confidence: {signal.confidence:F2} | " +
                          $"VolProb: {signal.vol_expansion_prob:F2} | TP: {signal.tp_price:F2} | SL: {signal.sl_price:F2}");
                }

                return signal;
            }
            catch (Exception ex)
            {
                Print($"[SKIE] Signal error: {ex.Message}");
                isConnected = false;
                return null;
            }
        }

        private void ExecuteSignal(TradeSignal signal)
        {
            if (signal == null || signal.action == "FLAT")
            {
                return;
            }

            // Check if already in position
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                return;
            }

            int contracts = Math.Min(signal.contracts, ContractsPerTrade);

            if (signal.action == "LONG")
            {
                EnterLong(contracts, "SKIE_Long");
                SetStopLoss("SKIE_Long", CalculationMode.Price, signal.sl_price, false);
                SetProfitTarget("SKIE_Long", CalculationMode.Price, signal.tp_price);
                tradesExecutedToday++;
                Print($"[SKIE] LONG entered - Contracts: {contracts} | TP: {signal.tp_price:F2} | SL: {signal.sl_price:F2}");
            }
            else if (signal.action == "SHORT")
            {
                EnterShort(contracts, "SKIE_Short");
                SetStopLoss("SKIE_Short", CalculationMode.Price, signal.sl_price, false);
                SetProfitTarget("SKIE_Short", CalculationMode.Price, signal.tp_price);
                tradesExecutedToday++;
                Print($"[SKIE] SHORT entered - Contracts: {contracts} | TP: {signal.tp_price:F2} | SL: {signal.sl_price:F2}");
            }
        }

        protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
        {
            if (position.MarketPosition == MarketPosition.Flat)
            {
                // Position closed - update daily P&L
                // FIXED: Use GetProfitLoss() for realized P&L, not GetUnrealizedProfitLoss()
                dailyPnL += position.GetProfitLoss(PerformanceUnit.Currency);
                Print($"[SKIE] Position closed - Daily P&L: ${dailyPnL:F2}");
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            Print($"[SKIE] Execution: {execution.Order.Name} | Price: {price:F2} | Qty: {quantity} | {marketPosition}");
        }
    }

    // Signal class for JSON deserialization
    public class TradeSignal
    {
        public string type { get; set; }
        public string action { get; set; }
        public double confidence { get; set; }
        public double vol_expansion_prob { get; set; }
        public double breakout_prob { get; set; }
        public double tp_price { get; set; }
        public double sl_price { get; set; }
        public int contracts { get; set; }
        public string reason { get; set; }
        public string timestamp { get; set; }
    }
}
