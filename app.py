# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from radar import calculate_derived_metrics, RADAR_PRESETS

st.set_page_config(page_title="Similar Players Search", layout="wide")

st.title("🔍 Find Similar Players (Multiple Files)")

# Upload multiple datasets
files = st.file_uploader(
    "Upload up to 20 Excel files with player data",
    type=["xls", "xlsx"],
    accept_multiple_files=True
)

if files:
    if len(files) > 20:
        st.warning("You uploaded more than 20 files. Only the first 20 will be used.")
        files = files[:20]

    all_dfs = []
    for f in files:
        df = pd.read_excel(f)
        df = calculate_derived_metrics(df)
        df["Source"] = f.name
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Choose radar preset
    preset = st.selectbox("Select radar preset", list(RADAR_PRESETS.keys()))
    params = [p for p in RADAR_PRESETS[preset] if p in combined_df.columns]

    if params:
        # Optional filters
        if 'Age' in combined_df.columns:
            max_age = st.slider("Maximum age", 15, 45, 30)
            combined_df = combined_df[combined_df['Age'] <= max_age]
        if 'Minutes played' in combined_df.columns:
            min_minutes = st.slider("Minimum minutes played", 0, 3000, 500)
            combined_df = combined_df[combined_df['Minutes played'] >= min_minutes]

        # Reference player
        player_name = st.selectbox("Select the reference player", combined_df['Player'].unique())
        ref_data = combined_df[combined_df['Player'] == player_name].iloc[0]

        # Calculate reference player percentiles (within their own dataset source)
        ref_df = combined_df[combined_df["Source"] == ref_data["Source"]]
        ref_percentiles = {}
        for param in params:
            series = ref_df[param].dropna()
            val = ref_data[param]
            ref_percentiles[param] = percentileofscore(series, val, kind='rank') if len(series) > 0 else 0

        # Calculate distances to all other players
        distances = []
        for _, row in combined_df.iterrows():
            if row['Player'] == player_name and row["Source"] == ref_data["Source"]:
                continue
            dist = 0
            count = 0
            for param in params:
                if pd.notna(row[param]):
                    # Compare using percentiles within the player's own dataset
                    source_df = combined_df[combined_df["Source"] == row["Source"]]
                    perc = percentileofscore(source_df[param].dropna(), row[param], kind='rank')
                    dist += (perc - ref_percentiles[param]) ** 2
                    count += 1
            if count > 0:
                distances.append((
                    row['Player'],
                    row['Team'],
                    row['Position'],
                    row["Source"],
                    np.sqrt(dist / count)
                ))

        # Display results
        results_df = pd.DataFrame(distances, columns=["Player", "Team", "Position", "Source", "Distance"])
        results_df = results_df.sort_values("Distance")
        st.subheader("Most similar players")
        st.dataframe(results_df)

        # Radar comparison: reference vs most similar
        if not results_df.empty:
            top_player = results_df.iloc[0]['Player']
            top_source = results_df.iloc[0]['Source']
            sim_data = combined_df[(combined_df['Player'] == top_player) & (combined_df['Source'] == top_source)].iloc[0]

            ref_vals = [ref_percentiles[p] for p in params]
            sim_source_df = combined_df[combined_df["Source"] == top_source]
            sim_vals = [
                percentileofscore(sim_source_df[p].dropna(), sim_data[p], kind='rank') for p in params
            ]

            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
            angles += angles[:1]
            ref_vals += ref_vals[:1]
            sim_vals += sim_vals[:1]

            ax.plot(angles, ref_vals, label=player_name, color='blue')
            ax.fill(angles, ref_vals, alpha=0.25, color='blue')
            ax.plot(angles, sim_vals, label=top_player, color='red')
            ax.fill(angles, sim_vals, alpha=0.25, color='red')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(params, fontsize=8)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_title(f"{player_name} vs {top_player}")
            ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

            st.pyplot(fig)

    else:
        st.warning("The selected preset has no available metrics in these datasets.")
else:
    st.info("Please upload up to 20 Excel files to start.")
