import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Optional import for advanced missing data plots
try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False
    logging.warning("Missingno is not installed. Install it with `pip install missingno` for advanced visualizations.")

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


# Abstract Base Class
class MissingAnalyzer(ABC):
    def run(self, df: pd.DataFrame, kind="heatmap", save=False, filename="missing_plot.png"):
        if df.isnull().sum().sum() == 0:
            logging.info("✅ No missing values found in the dataset.")
            return

        self.show_missing(df)
        self.plot_missing(df, kind=kind, save=save, filename=filename)

    @abstractmethod
    def show_missing(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def plot_missing(self, df: pd.DataFrame, kind="heatmap", save=False, filename="missing_plot.png"):
        pass


# Concrete Class
class BasicAnalyzer(MissingAnalyzer):
    def show_missing(self, df: pd.DataFrame):
        missing = df.isnull().sum()
        percent = (missing / len(df)) * 100
        summary = pd.concat([missing, percent], axis=1, keys=["Missing Count", "Missing %"])
        logging.info("\nMissing Values Summary:")
        logging.info(summary[summary["Missing Count"] > 0].sort_values(by="Missing %", ascending=False))

    def plot_missing(self, df: pd.DataFrame, kind="heatmap", save=False, filename="missing_plot.png"):
        logging.info(f"\nGenerating {kind} visualization for missing values...")

        if kind == "bar":
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            missing.plot(kind="bar", figsize=(12, 6), color="coral")
            plt.title("Missing Values Count by Column")
            plt.ylabel("Missing Count")
            plt.xlabel("Columns")
            plt.xticks(rotation=45)
            plt.tight_layout()

        elif kind == "heatmap":
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Values Heatmap")

        elif kind == "msno_matrix" and MISSINGNO_AVAILABLE:
            msno.matrix(df)
            plt.title("Missing Data Matrix")

        elif kind == "msno_heatmap" and MISSINGNO_AVAILABLE:
            msno.heatmap(df)
            plt.title("Missing Values Correlation Heatmap")

        elif kind == "msno_dendrogram" and MISSINGNO_AVAILABLE:
            msno.dendrogram(df)
            plt.title("Missing Value Dendrogram")

        else:
            logging.warning(f"⚠️ Unknown kind '{kind}' or missingno not available. Falling back to seaborn heatmap.")
            plt.figure(figsize=(12, 8))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
            plt.title("Missing Values Heatmap")

        if save:
            plt.savefig(filename, bbox_inches="tight")
            logging.info(f"✅ Plot saved as '{filename}'")

        plt.show()


'''import pandas as pd
from missing_analyzer import BasicAnalyzer  # your custom module

# Load your data
df = pd.read_csv("your_data.csv")

# Create analyzer object
analyzer = BasicAnalyzer()

# Run a missingno matrix plot
analyzer.run(df, kind="msno_matrix")

# Run a missingno heatmap plot
analyzer.run(df, kind="msno_heatmap")

# Run a simple bar plot
analyzer.run(df, kind="bar", save=True, filename="missing_bar_plot.png")
'''