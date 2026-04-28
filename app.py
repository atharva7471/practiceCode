import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_folders = ['csvfiles']

output_folder = 'results'
processed_folder = 'processed'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(processed_folder, exist_ok=True)

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        # Forward fill
        df = df.ffill()

        if df.empty:
            print(f"Skipping: {file_path}")
            return

        # Select numeric
        df_numeric = df.select_dtypes(include='number')

        if df_numeric.empty:
            print(f"No numeric data: {file_path}")
            return

        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        # ----------------------------
        # 2. PAIRPLOT (limit columns!)
        # ----------------------------
        selected_cols = df_numeric.columns[:4]   # take first 4 columns only

        pairplot = sns.pairplot(df_numeric[selected_cols])
        pairplot.fig.suptitle(f"Pairplot - {file_name}", y=1.02)

        pairplot_path = os.path.join(output_folder, base_name + "_pairplot.png")
        pairplot.savefig(pairplot_path)
        plt.close()

        # ----------------------------
        # 3. COMBINED IMAGE (subplot)
        # ----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(14,6))

        # Heatmap
        sns.heatmap(df_numeric, ax=axes[0], cmap='viridis')
        axes[0].set_title("Heatmap")

        # Correlation Heatmap (better than squeezing pairplot here)
        corr = df_numeric.corr()
        sns.heatmap(corr, annot=True, ax=axes[1], cmap='coolwarm')
        axes[1].set_title("Correlation")

        combined_path = os.path.join(output_folder, base_name + "_combined.png")
        plt.tight_layout()
        plt.savefig(combined_path)
        plt.close()

        # ----------------------------
        # 4. SAVE PROCESSED CSV
        # ----------------------------
        processed_path = os.path.join(processed_folder, file_name)
        df.to_csv(processed_path, index=False)

        print(f"Saved all plots for: {file_name}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Run loop
for folder in input_folders:
    for file in os.listdir(folder):
        if file.endswith('.csv'):
            file_path = os.path.join(folder, file)
            process_csv(file_path)