from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base Class for Bivariate Analysis Strategy

class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, tilte: str, ax=None, **kwargs):
        pass


# ----------------------------------------------------
# Numerical vs Numerical Strategy (Scatter Plot)
# ----------------------------------------------------
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, title: str, ax=None, **kwargs):
        sns.set_style("whitegrid")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True

        sns.scatterplot(x=feature1, y=feature2, data=df, ax=ax, **kwargs)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)

        if created_fig:
            plt.tight_layout()
            plt.show()


# ----------------------------------------------------
# Categorical vs Numerical Strategy (Box & Line Plot)
# ----------------------------------------------------
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str, title: str, ax=None, **kwargs):
        sns.set_style("whitegrid")
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            created_fig = True

        # Extract plot type (default = 'box')
        plot_type = kwargs.pop("plot_type", "box")
        order = df[feature1].value_counts().index
        hue = kwargs.pop("hue", None)

        if plot_type == "box":
            sns.boxplot(x=feature1, y=feature2, hue=hue, data=df, ax=ax, order=order, **kwargs)
        elif plot_type == "line":
            sns.lineplot(x=feature1, y=feature2, hue=hue, data=df, ax=ax, **kwargs)
        elif plot_type == "bar":
            sns.barplot(x=feature1, y=feature2, hue=hue, data=df, ax=ax, order=order, **kwargs)
        elif plot_type == "lineplot":
            sns.lineplot(x=feature1, y=feature2, data=df, ax=ax, hue=hue, **kwargs)
        else:
            raise ValueError("Invalid plot_type. Use 'box' or 'line'.")

        ax.set_title(title , fontsize=18)
        ax.set_xlabel(feature1, fontsize=12)
        ax.set_ylabel(feature2, fontsize=12)
        ax.tick_params(axis='x', rotation=45)

        if created_fig:
            plt.tight_layout()
            plt.show()

# ----------------------------------------------------
# Context Class for Strategy Management
# ----------------------------------------------------
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str, ax=None, **kwargs):
        self._strategy.analyze(df, feature1, feature2, ax=ax, **kwargs)


# ----------------------------------------------------
# Example Usage (Uncomment for testing)
# ----------------------------------------------------
if __name__ == "__main__":
    # df = pd.read_csv("your_data.csv")
    # analyzer = BivariateAnalyzer(NumericalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'engine_capacity', 'priceUSD')

    # analyzer.set_strategy(CategoricalVsNumericalAnalysis())
    # analyzer.execute_analysis(df, 'make', 'priceUSD', plot_type='line', hue='condition')
    pass
