import pandas as pd

# Read the CSV with error handling
try:
    df = pd.read_csv('yearwise_data.csv', on_bad_lines='skip', engine='python', sep=',')
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Clean up column names
    df.columns = df.columns.str.strip()
    
    # Remove extra whitespace from string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # Remove rows with all NaN values
    df = df.dropna(how='all')
    
    # Save cleaned version
    df.to_csv('yearwise_data_cleaned.csv', index=False)
    print(f"\n✅ Cleaned CSV saved to yearwise_data_cleaned.csv ({len(df)} rows)")
    print("\nSample data:")
    print(df[['player', 'year', 'runs']].head(10))
    
    # Now replace original
    import shutil
    shutil.copy('yearwise_data_cleaned.csv', 'yearwise_data.csv')
    print("\n✅ Replaced original yearwise_data.csv")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
