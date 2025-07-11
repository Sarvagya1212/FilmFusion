from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Univariate Analysis
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str, save=False, filename=None):
        pass


# Concrete Strategy for Numerical Features
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, save=False, filename=None, ax=None):
        print(f"\nDescriptive Statistics for Numerical Feature: {feature}")
        print(df[feature].describe())

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            created_fig = True

        sns.histplot(df[feature].dropna(), kde=True, bins=50, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

        if save and filename and created_fig:
            plt.savefig(filename, bbox_inches="tight")
        if created_fig:
            plt.tight_layout()
            plt.show()  # only if ax is not passed



# Concrete Strategy for Categorical Features
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str, hue=None, save=False, filename=None, ax=None):
        sns.set_style("whitegrid")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
            created_fig = True

        order = df[feature].value_counts().index  # ✅ FIX: define order

        if hue:
            sns.countplot(
                x=feature,
                data=df,
                hue=hue,
                palette="pastel",
                order=order,
                ax=ax
            )
        else:
            # Workaround for future Seaborn versions to allow `palette` without hue
            sns.countplot(
                x=feature,
                data=df,
                hue=feature,
                palette="pastel",
                order=order,
                legend=False,
                ax=ax
            )

        ax.set_title(f"Frequency of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=45)

        if save and filename and created_fig:
            plt.savefig(filename, bbox_inches="tight")
        if created_fig:
            plt.tight_layout()
            plt.show()



# Context Class (Manual strategy selection)
class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature: str, save=False, filename=None):
        self._strategy.analyze(df, feature, save=save, filename=filename)


# Auto Type Detection Analyzer (Recommended)
class AutoUnivariateAnalyzer:
    def __init__(self):
        self.num_strategy = NumericalUnivariateAnalysis()
        self.cat_strategy = CategoricalUnivariateAnalysis()

    def analyze(self, df: pd.DataFrame, feature: str, **kwargs):
        if pd.api.types.is_numeric_dtype(df[feature]):
            self.num_strategy.analyze(df, feature, **kwargs)
        else:
            self.cat_strategy.analyze(df, feature, **kwargs)


# Example usage
if __name__ == "__main__":
    # Example (uncomment and adjust for actual usage)
    
    # df = pd.read_csv("your_data.csv")
    # analyzer = AutoUnivariateAnalyzer()

    # Numerical feature
    # analyzer.analyze(df, "SalePrice", save=True, filename="sale_price_dist.png")

    # Categorical feature
    # analyzer.analyze(df, "Neighborhood", save=True, filename="neighborhood_dist.png")
    
    pass
