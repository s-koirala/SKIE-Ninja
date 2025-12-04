"""
Feature Selection Module
=========================
Implements various feature selection techniques to identify
the most predictive features for ML models.

Techniques:
1. Correlation Analysis - Remove highly correlated features
2. Variance Threshold - Remove near-constant features
3. Target Correlation - Rank by correlation with target
4. Feature Importance - Random Forest / XGBoost importance
5. Recursive Feature Elimination (RFE)
6. Statistical Tests - F-test, mutual information
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
import logging
import warnings
from scipy import stats
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Comprehensive feature selection pipeline.

    Combines multiple selection techniques to identify
    the most predictive features.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        target_col: str = 'target_direction_1',
        task: str = 'classification'
    ):
        """
        Initialize the feature selector.

        Args:
            features: DataFrame with all features and targets
            target_col: Name of target column
            task: 'classification' or 'regression'
        """
        self.features = features.copy()
        self.target_col = target_col
        self.task = task

        # Separate features and target
        target_cols = [c for c in features.columns if c.startswith('target_')]
        self.target = features[target_col].copy()
        self.X = features.drop(columns=target_cols, errors='ignore')

        # Track selection results
        self.selection_results = {}
        self.selected_features = None

        logger.info(f"FeatureSelector initialized with {len(self.X.columns)} features")
        logger.info(f"Target: {target_col}, Task: {task}")

    def remove_constant_features(self, threshold: float = 0.01) -> List[str]:
        """
        Remove features with near-zero variance.

        Args:
            threshold: Minimum variance threshold

        Returns:
            List of features to keep
        """
        logger.info(f"Removing constant features (variance < {threshold})...")

        # Handle NaN by filling with median
        X_filled = self.X.fillna(self.X.median())

        # Standardize first to make threshold meaningful
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_filled),
            columns=self.X.columns,
            index=self.X.index
        )

        # Calculate variance
        variances = X_scaled.var()

        # Keep features above threshold
        keep_features = variances[variances >= threshold].index.tolist()
        removed = len(self.X.columns) - len(keep_features)

        logger.info(f"Removed {removed} constant features, {len(keep_features)} remain")

        self.selection_results['constant'] = {
            'removed': removed,
            'kept': len(keep_features),
            'features': keep_features
        }

        return keep_features

    def remove_correlated_features(
        self,
        threshold: float = 0.95,
        method: str = 'pearson'
    ) -> List[str]:
        """
        Remove highly correlated features.

        Keeps the feature with higher target correlation when
        two features are correlated above threshold.

        Args:
            threshold: Correlation threshold (0-1)
            method: Correlation method ('pearson', 'spearman')

        Returns:
            List of features to keep
        """
        logger.info(f"Removing correlated features (>{threshold})...")

        # Calculate correlation matrix
        X_filled = self.X.fillna(self.X.median())
        corr_matrix = X_filled.corr(method=method).abs()

        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Calculate target correlations for tie-breaking
        target_corr = X_filled.corrwith(self.target.fillna(0)).abs()

        # Find pairs above threshold
        to_drop = set()
        for col in upper.columns:
            correlated = upper[col][upper[col] > threshold].index.tolist()
            for corr_col in correlated:
                # Keep feature with higher target correlation
                if target_corr.get(col, 0) >= target_corr.get(corr_col, 0):
                    to_drop.add(corr_col)
                else:
                    to_drop.add(col)

        keep_features = [c for c in self.X.columns if c not in to_drop]

        logger.info(f"Removed {len(to_drop)} correlated features, {len(keep_features)} remain")

        self.selection_results['correlated'] = {
            'removed': len(to_drop),
            'kept': len(keep_features),
            'features': keep_features,
            'dropped': list(to_drop)
        }

        return keep_features

    def rank_by_target_correlation(
        self,
        method: str = 'spearman',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank features by correlation with target.

        Args:
            method: Correlation method
            top_n: Return only top N features

        Returns:
            DataFrame with feature rankings
        """
        logger.info("Ranking features by target correlation...")

        X_filled = self.X.fillna(self.X.median())
        target_filled = self.target.fillna(self.target.median())

        correlations = X_filled.corrwith(target_filled, method=method)

        ranking = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'abs_correlation': correlations.abs().values
        }).sort_values('abs_correlation', ascending=False)

        if top_n:
            ranking = ranking.head(top_n)

        self.selection_results['target_correlation'] = ranking

        logger.info(f"Top 5 features by target correlation:")
        for _, row in ranking.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['correlation']:.4f}")

        return ranking

    def rank_by_mutual_information(
        self,
        top_n: Optional[int] = None,
        n_neighbors: int = 3
    ) -> pd.DataFrame:
        """
        Rank features by mutual information with target.

        Args:
            top_n: Return only top N features
            n_neighbors: Number of neighbors for MI estimation

        Returns:
            DataFrame with feature rankings
        """
        logger.info("Ranking features by mutual information...")

        # Prepare data
        X_filled = self.X.fillna(self.X.median())
        target_filled = self.target.fillna(self.target.median())

        # Remove any remaining NaN/inf
        valid_mask = ~(X_filled.isna().any(axis=1) | target_filled.isna())
        X_clean = X_filled[valid_mask]
        y_clean = target_filled[valid_mask]

        if len(X_clean) < 100:
            logger.warning("Not enough clean samples for MI calculation")
            return pd.DataFrame()

        # Calculate mutual information
        if self.task == 'classification':
            mi_scores = mutual_info_classif(
                X_clean, y_clean.astype(int),
                n_neighbors=n_neighbors,
                random_state=42
            )
        else:
            mi_scores = mutual_info_regression(
                X_clean, y_clean,
                n_neighbors=n_neighbors,
                random_state=42
            )

        ranking = pd.DataFrame({
            'feature': self.X.columns,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)

        if top_n:
            ranking = ranking.head(top_n)

        self.selection_results['mutual_info'] = ranking

        logger.info(f"Top 5 features by mutual information:")
        for _, row in ranking.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['mutual_info']:.4f}")

        return ranking

    def rank_by_statistical_test(
        self,
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank features by F-test statistics.

        Args:
            top_n: Return only top N features

        Returns:
            DataFrame with feature rankings
        """
        logger.info("Ranking features by F-test...")

        # Prepare data
        X_filled = self.X.fillna(self.X.median())
        target_filled = self.target.fillna(self.target.median())

        valid_mask = ~(X_filled.isna().any(axis=1) | target_filled.isna())
        X_clean = X_filled[valid_mask]
        y_clean = target_filled[valid_mask]

        if len(X_clean) < 100:
            logger.warning("Not enough clean samples for F-test")
            return pd.DataFrame()

        # Calculate F-scores
        if self.task == 'classification':
            f_scores, p_values = f_classif(X_clean, y_clean.astype(int))
        else:
            f_scores, p_values = f_regression(X_clean, y_clean)

        ranking = pd.DataFrame({
            'feature': self.X.columns,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)

        # Replace inf with max finite value
        max_finite = ranking['f_score'][np.isfinite(ranking['f_score'])].max()
        ranking['f_score'] = ranking['f_score'].replace([np.inf, -np.inf], max_finite)

        if top_n:
            ranking = ranking.head(top_n)

        self.selection_results['f_test'] = ranking

        return ranking

    def rank_by_random_forest(
        self,
        top_n: Optional[int] = None,
        n_estimators: int = 100,
        max_depth: int = 10
    ) -> pd.DataFrame:
        """
        Rank features by Random Forest importance.

        Args:
            top_n: Return only top N features
            n_estimators: Number of trees
            max_depth: Maximum tree depth

        Returns:
            DataFrame with feature rankings
        """
        logger.info("Ranking features by Random Forest importance...")

        # Prepare data
        X_filled = self.X.fillna(self.X.median())
        target_filled = self.target.fillna(self.target.median())

        valid_mask = ~(X_filled.isna().any(axis=1) | target_filled.isna())
        X_clean = X_filled[valid_mask]
        y_clean = target_filled[valid_mask]

        # Use subset for speed
        sample_size = min(50000, len(X_clean))
        if len(X_clean) > sample_size:
            indices = np.random.choice(len(X_clean), sample_size, replace=False)
            X_sample = X_clean.iloc[indices]
            y_sample = y_clean.iloc[indices]
        else:
            X_sample = X_clean
            y_sample = y_clean

        # Train Random Forest
        if self.task == 'classification':
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_sample, y_sample.astype(int))
        else:
            rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_sample, y_sample)

        # Get importance scores
        importance = rf.feature_importances_

        ranking = pd.DataFrame({
            'feature': self.X.columns,
            'rf_importance': importance
        }).sort_values('rf_importance', ascending=False)

        if top_n:
            ranking = ranking.head(top_n)

        self.selection_results['random_forest'] = ranking

        logger.info(f"Top 5 features by RF importance:")
        for _, row in ranking.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['rf_importance']:.4f}")

        return ranking

    def aggregate_rankings(
        self,
        methods: List[str] = None,
        top_n: int = 100
    ) -> pd.DataFrame:
        """
        Aggregate rankings from multiple methods.

        Uses average rank across methods to produce final ranking.

        Args:
            methods: List of methods to aggregate
            top_n: Number of top features to return

        Returns:
            DataFrame with aggregated rankings
        """
        if methods is None:
            methods = ['target_correlation', 'mutual_info', 'f_test', 'random_forest']

        logger.info(f"Aggregating rankings from {len(methods)} methods...")

        # Collect rankings
        rankings = {}
        for method in methods:
            if method not in self.selection_results:
                logger.warning(f"Method {method} not found, skipping...")
                continue

            result = self.selection_results[method]
            if isinstance(result, pd.DataFrame):
                # Create rank column
                result = result.reset_index(drop=True)
                result['rank'] = range(1, len(result) + 1)
                rankings[method] = result.set_index('feature')['rank']

        if not rankings:
            logger.warning("No rankings available")
            return pd.DataFrame()

        # Combine into single DataFrame
        rank_df = pd.DataFrame(rankings)

        # Fill missing ranks with max rank + 1
        max_rank = rank_df.max().max()
        rank_df = rank_df.fillna(max_rank + 1)

        # Calculate average rank
        rank_df['avg_rank'] = rank_df.mean(axis=1)
        rank_df = rank_df.sort_values('avg_rank')

        # Add aggregated ranking
        rank_df['final_rank'] = range(1, len(rank_df) + 1)

        if top_n:
            rank_df = rank_df.head(top_n)

        self.selection_results['aggregated'] = rank_df

        logger.info(f"Top 10 features by aggregated ranking:")
        for feature in rank_df.head(10).index:
            avg = rank_df.loc[feature, 'avg_rank']
            logger.info(f"  {feature}: avg_rank={avg:.1f}")

        return rank_df

    def select_features(
        self,
        n_features: int = 100,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        run_all_methods: bool = True
    ) -> List[str]:
        """
        Run full feature selection pipeline.

        Args:
            n_features: Number of features to select
            variance_threshold: Threshold for constant feature removal
            correlation_threshold: Threshold for correlated feature removal
            run_all_methods: Whether to run all ranking methods

        Returns:
            List of selected feature names
        """
        logger.info("=" * 60)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("=" * 60)

        # Step 1: Remove constant features
        keep_features = self.remove_constant_features(variance_threshold)

        # Subset to kept features
        original_X = self.X
        self.X = self.X[keep_features]

        # Step 2: Remove correlated features
        keep_features = self.remove_correlated_features(correlation_threshold)
        self.X = self.X[keep_features]

        # Step 3: Run ranking methods
        if run_all_methods:
            self.rank_by_target_correlation()
            self.rank_by_mutual_information()
            self.rank_by_statistical_test()
            self.rank_by_random_forest()

        # Step 4: Aggregate rankings
        aggregated = self.aggregate_rankings(top_n=n_features)

        # Get final selected features
        self.selected_features = aggregated.index.tolist()[:n_features]

        # Restore original X
        self.X = original_X

        logger.info("=" * 60)
        logger.info(f"SELECTED {len(self.selected_features)} FEATURES")
        logger.info("=" * 60)

        return self.selected_features

    def get_selected_features_df(self) -> pd.DataFrame:
        """Get DataFrame with only selected features."""
        if self.selected_features is None:
            raise ValueError("Run select_features() first")

        return self.X[self.selected_features]

    def save_results(self, output_dir: Union[str, Path]):
        """Save selection results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save selected features list
        if self.selected_features:
            with open(output_dir / 'selected_features.txt', 'w') as f:
                f.write(f"Selected {len(self.selected_features)} features\n")
                f.write("=" * 50 + "\n\n")
                for i, feat in enumerate(self.selected_features, 1):
                    f.write(f"{i:3d}. {feat}\n")

        # Save aggregated rankings
        if 'aggregated' in self.selection_results:
            self.selection_results['aggregated'].to_csv(
                output_dir / 'feature_rankings.csv'
            )

        # Save detailed results
        for method, result in self.selection_results.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(output_dir / f'ranking_{method}.csv', index=True)

        logger.info(f"Results saved to {output_dir}")


def select_best_features(
    features: pd.DataFrame,
    target_col: str = 'target_direction_1',
    n_features: int = 100,
    task: str = 'classification',
    output_dir: Optional[str] = None
) -> Tuple[List[str], FeatureSelector]:
    """
    Convenience function to run feature selection.

    Args:
        features: DataFrame with all features and targets
        target_col: Target column name
        n_features: Number of features to select
        task: 'classification' or 'regression'
        output_dir: Directory to save results

    Returns:
        Tuple of (selected feature names, FeatureSelector object)
    """
    selector = FeatureSelector(features, target_col, task)
    selected = selector.select_features(n_features=n_features)

    if output_dir:
        selector.save_results(output_dir)

    return selected, selector


if __name__ == "__main__":
    # Test feature selection
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("FEATURE SELECTION TEST")
    print("=" * 70)

    # Load sample data
    try:
        from data_collection.ninjatrader_loader import load_databento_data
        from feature_engineering.feature_pipeline import build_feature_matrix

        # Load ES data
        print("\nLoading ES data...")
        es_data = load_databento_data('ES')

        # Use subset for testing
        es_data = es_data.tail(50000)
        print(f"Using {len(es_data)} bars for testing")

        # Build features
        print("\nBuilding features...")
        features = build_feature_matrix(
            es_data,
            symbol='ES',
            include_lagged=True,
            include_interactions=True,
            include_targets=True,
            include_macro=False,  # Skip for speed
            include_sentiment=False,
            include_intermarket=False,
            include_alternative=False,
            dropna=True
        )

        print(f"Feature matrix shape: {features.shape}")

        # Run feature selection
        print("\nRunning feature selection...")
        selected, selector = select_best_features(
            features,
            target_col='target_direction_1',
            n_features=50,
            task='classification',
            output_dir=Path(__file__).parent.parent.parent.parent / 'data' / 'processed'
        )

        print(f"\nSelected {len(selected)} features:")
        for i, feat in enumerate(selected[:20], 1):
            print(f"  {i:2d}. {feat}")

        if len(selected) > 20:
            print(f"  ... and {len(selected) - 20} more")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 70)
