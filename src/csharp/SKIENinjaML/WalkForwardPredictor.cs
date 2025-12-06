/*
 * Walk-Forward ONNX Predictor
 *
 * Automatically switches between pre-trained ONNX models based on the current
 * bar's date. This enables true walk-forward simulation in a single NT8 backtest.
 *
 * The predictor loads a schedule of models, each valid for a specific date range.
 * When the bar date changes to a new period, the appropriate model is hot-loaded.
 *
 * Author: SKIE_Ninja Development Team
 * Created: 2025-12-06
 */

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;

namespace SKIENinjaML
{
    /// <summary>
    /// Model schedule entry - defines which model to use for each date range.
    /// </summary>
    public class ModelScheduleEntry
    {
        public int Fold { get; set; }
        public string ModelId { get; set; }
        public string FolderPath { get; set; }
        public DateTime ValidFrom { get; set; }
        public DateTime ValidUntil { get; set; }
        public int TrainSamples { get; set; }
    }

    /// <summary>
    /// Walk-Forward Predictor that automatically switches models based on date.
    /// Enables true walk-forward simulation in a single NinjaTrader backtest.
    /// </summary>
    public class WalkForwardPredictor : IDisposable
    {
        private List<ModelScheduleEntry> _schedule;
        private ModelScheduleEntry _currentEntry;
        private string _basePath;

        // Current loaded models
        private InferenceSession _volModel;
        private InferenceSession _highModel;
        private InferenceSession _lowModel;
        private InferenceSession _atrModel;
        private InferenceSession _sentimentVolModel;
        private ScalerParams _scalerParams;
        private ScalerParams _sentimentScalerParams;
        private StrategyConfig _config;
        private bool _hasSentimentModel = false;

        private SessionOptions _sessionOptions;
        private bool _isInitialized = false;
        private int _modelSwitchCount = 0;

        // Event for logging model switches
        public event Action<string> OnModelSwitch;
        public event Action<string> OnLog;

        /// <summary>
        /// Initialize the walk-forward predictor with a schedule file.
        /// </summary>
        /// <param name="walkForwardPath">Path to the walkforward_onnx folder containing model_schedule.csv</param>
        public void Initialize(string walkForwardPath)
        {
            _basePath = walkForwardPath;

            // Load model schedule
            string schedulePath = Path.Combine(walkForwardPath, "model_schedule.csv");
            if (!File.Exists(schedulePath))
            {
                throw new FileNotFoundException(
                    String.Format("Model schedule not found: {0}. Run export_walkforward_onnx.py first.", schedulePath));
            }

            _schedule = LoadSchedule(schedulePath);

            if (_schedule.Count == 0)
            {
                throw new InvalidOperationException("Model schedule is empty");
            }

            // Set up session options for model loading
            _sessionOptions = new SessionOptions();
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            Log(String.Format("Walk-Forward Predictor initialized with {0} model periods", _schedule.Count));
            Log(String.Format("Date range: {0:yyyy-MM-dd} to {1:yyyy-MM-dd}",
                _schedule.First().ValidFrom, _schedule.Last().ValidUntil));

            _isInitialized = true;
        }

        /// <summary>
        /// Load the model schedule from CSV file.
        /// </summary>
        private List<ModelScheduleEntry> LoadSchedule(string csvPath)
        {
            List<ModelScheduleEntry> entries = new List<ModelScheduleEntry>();
            string[] lines = File.ReadAllLines(csvPath);

            // Skip header
            for (int i = 1; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                string[] parts = line.Split(',');
                if (parts.Length < 6) continue;

                ModelScheduleEntry entry = new ModelScheduleEntry();
                entry.Fold = int.Parse(parts[0]);
                entry.ModelId = parts[1];
                entry.FolderPath = parts[2];
                entry.ValidFrom = DateTime.Parse(parts[3]);
                entry.ValidUntil = DateTime.Parse(parts[4]);
                entry.TrainSamples = int.Parse(parts[5]);

                entries.Add(entry);
            }

            // Sort by ValidFrom date
            entries = entries.OrderBy(e => e.ValidFrom).ToList();

            return entries;
        }

        /// <summary>
        /// Ensure the correct model is loaded for the given date.
        /// Returns true if models are ready, false if date is outside schedule.
        /// </summary>
        public bool EnsureModelForDate(DateTime barDate)
        {
            if (!_isInitialized)
            {
                throw new InvalidOperationException("Predictor not initialized. Call Initialize() first.");
            }

            // Check if current model is still valid
            if (_currentEntry != null &&
                barDate >= _currentEntry.ValidFrom &&
                barDate <= _currentEntry.ValidUntil)
            {
                return true; // Current model is still valid
            }

            // Find the appropriate model for this date
            ModelScheduleEntry newEntry = FindModelForDate(barDate);

            if (newEntry == null)
            {
                // Date is outside our schedule - use closest model
                if (barDate < _schedule.First().ValidFrom)
                {
                    // Before first model - can't trade (no trained model)
                    return false;
                }
                else if (barDate > _schedule.Last().ValidUntil)
                {
                    // After last model - use last model (stale but better than nothing)
                    newEntry = _schedule.Last();
                    Log(String.Format("WARNING: Date {0:yyyy-MM-dd} is after last model. Using stale model from {1}",
                        barDate, newEntry.ModelId));
                }
            }

            if (newEntry != null && newEntry != _currentEntry)
            {
                // Need to switch models
                LoadModels(newEntry);
                _currentEntry = newEntry;
                _modelSwitchCount++;

                string message = String.Format(
                    "Model Switch #{0}: Loaded fold_{1} (valid {2:yyyy-MM-dd} to {3:yyyy-MM-dd})",
                    _modelSwitchCount, newEntry.ModelId, newEntry.ValidFrom, newEntry.ValidUntil);

                Log(message);

                if (OnModelSwitch != null)
                {
                    OnModelSwitch(message);
                }
            }

            return _currentEntry != null;
        }

        /// <summary>
        /// Find the model entry that covers the given date.
        /// </summary>
        private ModelScheduleEntry FindModelForDate(DateTime date)
        {
            foreach (ModelScheduleEntry entry in _schedule)
            {
                if (date >= entry.ValidFrom && date <= entry.ValidUntil)
                {
                    return entry;
                }
            }
            return null;
        }

        /// <summary>
        /// Load models from the specified schedule entry.
        /// </summary>
        private void LoadModels(ModelScheduleEntry entry)
        {
            // Dispose previous models if any
            DisposeCurrentModels();

            string modelPath = entry.FolderPath;

            // Verify folder exists
            if (!Directory.Exists(modelPath))
            {
                throw new DirectoryNotFoundException(
                    String.Format("Model folder not found: {0}", modelPath));
            }

            // Load scaler params
            string scalerPath = Path.Combine(modelPath, "scaler_params.json");
            if (File.Exists(scalerPath))
            {
                string json = File.ReadAllText(scalerPath);
                _scalerParams = JsonConvert.DeserializeObject<ScalerParams>(json);
            }
            else
            {
                throw new FileNotFoundException(String.Format("Scaler params not found: {0}", scalerPath));
            }

            // Load sentiment scaler params if available
            string sentimentScalerPath = Path.Combine(modelPath, "sentiment_scaler_params.json");
            if (File.Exists(sentimentScalerPath))
            {
                string json = File.ReadAllText(sentimentScalerPath);
                _sentimentScalerParams = JsonConvert.DeserializeObject<ScalerParams>(json);
            }
            else
            {
                _sentimentScalerParams = null;
            }

            // Load strategy config
            string configPath = Path.Combine(modelPath, "strategy_config.json");
            if (File.Exists(configPath))
            {
                string json = File.ReadAllText(configPath);
                _config = JsonConvert.DeserializeObject<StrategyConfig>(json);
            }
            else
            {
                _config = new StrategyConfig();
            }

            // Load ONNX models
            _volModel = new InferenceSession(
                Path.Combine(modelPath, "vol_expansion_model.onnx"), _sessionOptions);
            _highModel = new InferenceSession(
                Path.Combine(modelPath, "breakout_high_model.onnx"), _sessionOptions);
            _lowModel = new InferenceSession(
                Path.Combine(modelPath, "breakout_low_model.onnx"), _sessionOptions);
            _atrModel = new InferenceSession(
                Path.Combine(modelPath, "atr_forecast_model.onnx"), _sessionOptions);

            // Load sentiment volatility model if available
            string sentimentModelPath = Path.Combine(modelPath, "sentiment_vol_model.onnx");
            if (File.Exists(sentimentModelPath) && _sentimentScalerParams != null)
            {
                _sentimentVolModel = new InferenceSession(sentimentModelPath, _sessionOptions);
                _hasSentimentModel = true;
                Log("  Loaded sentiment volatility model (28 sentiment features)");
            }
            else
            {
                _sentimentVolModel = null;
                _hasSentimentModel = false;
                Log("  WARNING: Sentiment model not found - using technical filter only");
            }
        }

        /// <summary>
        /// Dispose current models to free memory.
        /// </summary>
        private void DisposeCurrentModels()
        {
            if (_volModel != null) { _volModel.Dispose(); _volModel = null; }
            if (_highModel != null) { _highModel.Dispose(); _highModel = null; }
            if (_lowModel != null) { _lowModel.Dispose(); _lowModel = null; }
            if (_atrModel != null) { _atrModel.Dispose(); _atrModel = null; }
            if (_sentimentVolModel != null) { _sentimentVolModel.Dispose(); _sentimentVolModel = null; }
            _hasSentimentModel = false;
        }

        /// <summary>
        /// Scale features using stored scaler parameters.
        /// </summary>
        private float[] ScaleFeatures(double[] features)
        {
            // Use GetMeans()/GetScales() to handle both naming conventions
            List<double> means = _scalerParams.GetMeans();
            List<double> scales = _scalerParams.GetScales();

            float[] scaled = new float[features.Length];
            for (int i = 0; i < features.Length; i++)
            {
                scaled[i] = (float)((features[i] - means[i]) / scales[i]);
            }
            return scaled;
        }

        /// <summary>
        /// Scale combined features (technical + sentiment) for sentiment model.
        /// The sentiment model expects: [42 scaled tech features] + [28 scaled sentiment features]
        /// </summary>
        private float[] ScaleCombinedFeatures(double[] techFeatures, double[] sentimentFeatures)
        {
            if (_sentimentScalerParams == null)
            {
                throw new InvalidOperationException("Sentiment scaler not loaded");
            }

            // Scale technical features first
            float[] scaledTech = ScaleFeatures(techFeatures);

            // Scale sentiment features
            List<double> sentMeans = _sentimentScalerParams.GetMeans();
            List<double> sentScales = _sentimentScalerParams.GetScales();

            float[] scaledSent = new float[sentimentFeatures.Length];
            for (int i = 0; i < sentimentFeatures.Length; i++)
            {
                scaledSent[i] = (float)((sentimentFeatures[i] - sentMeans[i]) / sentScales[i]);
            }

            // Combine into single array
            float[] combined = new float[scaledTech.Length + scaledSent.Length];
            Array.Copy(scaledTech, 0, combined, 0, scaledTech.Length);
            Array.Copy(scaledSent, 0, combined, scaledTech.Length, scaledSent.Length);

            return combined;
        }

        /// <summary>
        /// Run classifier model and get probability of positive class.
        /// </summary>
        private double RunClassifier(InferenceSession session, float[] features)
        {
            DenseTensor<float> tensor = new DenseTensor<float>(features, new int[] { 1, features.Length });
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>();
            inputs.Add(NamedOnnxValue.CreateFromTensor("features", tensor));

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
            {
                // LightGBM classifier outputs: label (index 0), probabilities (index 1)
                DisposableNamedOnnxValue probOutput = results.ElementAt(1);
                object value = probOutput.Value;

                if (value == null)
                {
                    return 0.5;
                }

                // Handle LightGBM's nested output format
                if (value is System.Collections.IEnumerable enumerable)
                {
                    foreach (object item in enumerable)
                    {
                        if (item is DisposableNamedOnnxValue namedValue)
                        {
                            object innerValue = namedValue.Value;
                            if (innerValue is IDictionary<long, float> dict)
                            {
                                if (dict.ContainsKey(1))
                                {
                                    return dict[1];
                                }
                            }
                        }
                        else if (item is IDictionary<long, float> directDict)
                        {
                            if (directDict.ContainsKey(1))
                            {
                                return directDict[1];
                            }
                        }
                    }
                }

                return 0.5;
            }
        }

        /// <summary>
        /// Run regressor model and get prediction.
        /// </summary>
        private double RunRegressor(InferenceSession session, float[] features)
        {
            DenseTensor<float> tensor = new DenseTensor<float>(features, new int[] { 1, features.Length });
            List<NamedOnnxValue> inputs = new List<NamedOnnxValue>();
            inputs.Add(NamedOnnxValue.CreateFromTensor("features", tensor));

            using (IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs))
            {
                IEnumerable<float> output = results.First().AsEnumerable<float>();
                return output.First();
            }
        }

        /// <summary>
        /// Generate trading signal from features (backward compatible - no sentiment).
        /// Must call EnsureModelForDate() first to load appropriate model.
        /// </summary>
        public TradingSignal GenerateSignal(double[] features, double currentATR)
        {
            // Call full method with null sentiment features
            return GenerateSignal(features, null, currentATR);
        }

        /// <summary>
        /// Generate trading signal from features with optional sentiment data.
        /// Must call EnsureModelForDate() first to load appropriate model.
        /// </summary>
        /// <param name="techFeatures">42 technical features</param>
        /// <param name="sentimentFeatures">28 sentiment features (can be null if VIX not available)</param>
        /// <param name="currentATR">Current ATR for position sizing</param>
        public TradingSignal GenerateSignal(double[] techFeatures, double[] sentimentFeatures, double currentATR)
        {
            if (_currentEntry == null || _volModel == null)
            {
                throw new InvalidOperationException(
                    "No model loaded. Call EnsureModelForDate() first.");
            }

            TradingSignal signal = new TradingSignal();

            // Scale technical features
            float[] scaledTechFeatures = ScaleFeatures(techFeatures);

            // Get technical predictions
            double volProb = RunClassifier(_volModel, scaledTechFeatures);
            double highProb = RunClassifier(_highModel, scaledTechFeatures);
            double lowProb = RunClassifier(_lowModel, scaledTechFeatures);
            double predictedATR = RunRegressor(_atrModel, scaledTechFeatures);

            signal.VolExpansionProb = volProb;
            signal.PredictedATR = predictedATR;

            // 1. Technical volatility filter
            bool techVolPass = volProb >= _config.min_vol_expansion_prob;

            // 2. Sentiment volatility filter (if available)
            double sentVolProb = 0.0;
            bool sentVolPass = true; // Default to pass if no sentiment model

            if (_hasSentimentModel && sentimentFeatures != null && sentimentFeatures.Length == 28)
            {
                // Scale combined features and run sentiment model
                float[] scaledCombined = ScaleCombinedFeatures(techFeatures, sentimentFeatures);
                sentVolProb = RunClassifier(_sentimentVolModel, scaledCombined);
                signal.SentimentVolProb = sentVolProb;

                // Apply sentiment threshold
                sentVolPass = sentVolProb >= _config.min_sentiment_vol_prob;
            }
            else
            {
                // No sentiment model - skip sentiment filter
                signal.SentimentVolProb = -1.0; // Indicates not available
            }

            // 3. Apply ensemble 'agreement' mode: BOTH filters must pass
            bool passedVolFilters = false;
            if (_config.ensemble_mode == "agreement")
            {
                // Both must pass
                passedVolFilters = techVolPass && sentVolPass;
            }
            else
            {
                // Default: only technical filter
                passedVolFilters = techVolPass;
            }

            if (!passedVolFilters)
            {
                signal.ShouldTrade = false;
                signal.Direction = 0;
                return signal;
            }

            // 4. Direction from breakout probabilities
            bool highSignal = highProb >= _config.min_breakout_prob;
            bool lowSignal = lowProb >= _config.min_breakout_prob;

            if (highSignal && !lowSignal)
            {
                signal.Direction = 1;
                signal.BreakoutProb = highProb;
            }
            else if (lowSignal && !highSignal)
            {
                signal.Direction = -1;
                signal.BreakoutProb = lowProb;
            }
            else if (highSignal && lowSignal)
            {
                if (highProb > lowProb)
                {
                    signal.Direction = 1;
                    signal.BreakoutProb = highProb;
                }
                else
                {
                    signal.Direction = -1;
                    signal.BreakoutProb = lowProb;
                }
            }
            else
            {
                signal.ShouldTrade = false;
                signal.Direction = 0;
                return signal;
            }

            signal.ShouldTrade = true;

            // 5. Position sizing (inverse volatility)
            double volFactor = currentATR / (predictedATR + 1e-10);
            volFactor = Math.Max(0.5, Math.Min(2.0, volFactor));

            signal.Contracts = (int)(_config.base_contracts * volFactor * _config.vol_sizing_factor);
            signal.Contracts = Math.Max(1, Math.Min(signal.Contracts, _config.max_contracts));

            // 6. Dynamic exits based on predicted ATR
            double tpMult = _config.tp_atr_mult_base *
                (1 + _config.tp_adjustment_factor * (signal.BreakoutProb - 0.5));
            double slMult = _config.sl_atr_mult_base;

            signal.TakeProfitOffset = tpMult * predictedATR * signal.Direction;
            signal.StopLossOffset = -slMult * predictedATR * signal.Direction;

            return signal;
        }

        /// <summary>
        /// Get the number of technical features expected by the current model.
        /// </summary>
        public int GetFeatureCount()
        {
            if (_scalerParams != null)
            {
                return _scalerParams.n_features;
            }
            return 42; // Default expected feature count
        }

        /// <summary>
        /// Get the number of sentiment features expected (always 28 if sentiment model loaded).
        /// </summary>
        public int GetSentimentFeatureCount()
        {
            if (_hasSentimentModel && _sentimentScalerParams != null)
            {
                return _sentimentScalerParams.n_features;
            }
            return 0;
        }

        /// <summary>
        /// Check if sentiment model is loaded and available.
        /// </summary>
        public bool HasSentimentModel()
        {
            return _hasSentimentModel;
        }

        /// <summary>
        /// Get the current model's valid date range.
        /// </summary>
        public string GetCurrentModelInfo()
        {
            if (_currentEntry == null)
            {
                return "No model loaded";
            }
            return String.Format("Fold {0}: {1:yyyy-MM-dd} to {2:yyyy-MM-dd}",
                _currentEntry.ModelId, _currentEntry.ValidFrom, _currentEntry.ValidUntil);
        }

        /// <summary>
        /// Get the total number of model switches during the session.
        /// </summary>
        public int GetModelSwitchCount()
        {
            return _modelSwitchCount;
        }

        /// <summary>
        /// Get the model schedule for reference.
        /// </summary>
        public List<ModelScheduleEntry> GetSchedule()
        {
            return _schedule;
        }

        /// <summary>
        /// Get the current strategy config.
        /// </summary>
        public StrategyConfig GetConfig()
        {
            return _config;
        }

        private void Log(string message)
        {
            if (OnLog != null)
            {
                OnLog(message);
            }
        }

        /// <summary>
        /// Dispose resources.
        /// </summary>
        public void Dispose()
        {
            DisposeCurrentModels();
            if (_sessionOptions != null) _sessionOptions.Dispose();
        }
    }
}
