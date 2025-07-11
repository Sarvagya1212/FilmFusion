from abc import ABC, abstractmethod
import pandas as pd


class InspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df):
        pass

class ViewData(InspectionStrategy):
    def inspect(self, df):  
        print(f"\n- Shape : {df.shape}")
        print(f"Sample of :\n", df.head())


class TypeInspector(InspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nColumn Types and Non-null Counts:")
        df.info()


class StatsInspector(InspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        print("\nNumerical Stats:")
        print(df.describe())
        print("\nCategorical Stats:")
        print(df.describe(include=["O"]))

class Inspector:
    def __init__(self, method: InspectionStrategy):
        self._method = method

    def use(self, method: InspectionStrategy):
        self._method = method

    def inspect(self, df):
        self._method.inspect(df)


'''if __name__ == "__main__":
    
    df1 = pd.DataFrame({
        "A": [1, 2, 3],
        "B": ["x", "y", "z"]
    })
    df2 = pd.DataFrame({
        "C": [4.5, 6.7, 8.9],
        "D": ["foo", "bar", "baz"]
    })
    data = {"df1": df1, "df2": df2}

    # Use ViewData for multiple DataFrames
    inspector = Inspector(ViewData())
    inspector.inspect(data)

    # Now switch to single DataFrame inspection
    inspector.use(TypeInspector())
    inspector.inspect(df1)

    inspector.use(StatsInspector())
    inspector.inspect(df1)
'''