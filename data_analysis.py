import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

def load_data(file_path):
    """Load CSV file into DataFrame."""
    try:
        df = pd.read_csv(file_path)
        print(f"[INFO] Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def clean_data(df):
    """Fill missing values and remove duplicates."""
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    print("[INFO] Data cleaned")
    return df

def summary_statistics(df):
    """Return descriptive statistics."""
    return df.describe()

def correlation_heatmap(df):
    """Plot a correlation heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def auto_profile(df, output_file="report.html"):
    """Generate an HTML profiling report."""
    profile = ProfileReport(df, title="Autonomous Data Analyst Report")
    profile.to_file(output_file)
    print(f"[INFO] Profiling report saved to {output_file}")
