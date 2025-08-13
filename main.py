from data_analysis import load_data, clean_data, summary_statistics, correlation_heatmap, auto_profile

# Step 1: Load Data
file_path = "sample.csv"  # Replace with your CSV file name
df = load_data(file_path)

if df is not None:
    # Step 2: Clean Data
    df = clean_data(df)

    # Step 3: Summary Stats
    print("\n=== Summary Statistics ===")
    print(summary_statistics(df))

    # Step 4: Correlation Heatmap
    correlation_heatmap(df)

    # Step 5: Auto Profile Report
    auto_profile(df)
