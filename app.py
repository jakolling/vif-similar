# app.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from radar import calculate_derived_metrics, RADAR_PRESETS

# Configure page
st.set_page_config(page_title="Similar Players Search", layout="wide")
st.title("🔍 Find Similar Players (Multiple Files)")

# File uploader
files = st.file_uploader(
    "Upload up to 20 Excel files with player data",
    type=["xls", "xlsx"],
    accept_multiple_files=True
)

if files:
    if len(files) > 20:
        st.warning("You uploaded more than 20 files. Only the first 20 will be used.")
        files = files[:20]

    # Process files
    all_dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, engine='openpyxl')
            df = calculate_derived_metrics(df)
            df["Source"] = f.name
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Error reading {f.name}: {str(e)}")
            continue

    if not all_dfs:
        st.error("No valid data files could be processed.")
        st.stop()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Metric selection system
    st.sidebar.header("Metric Selection Options")
    use_presets = st.sidebar.checkbox("Use Preset Metrics", value=True)
    
    if use_presets:
        selected_presets = st.sidebar.multiselect(
            "Select up to 3 radar presets",
            list(RADAR_PRESETS.keys()),
            default=list(RADAR_PRESETS.keys())[0],
            max_selections=3
        )
        params = []
        for preset in selected_presets:
            params.extend([p for p in RADAR_PRESETS[preset] if p in combined_df.columns])
        params = list(dict.fromkeys(params))
    else:
        available_metrics = [col for col in combined_df.columns if col not in ['Player', 'Team', 'Position', 'Source', 'Age', 'Minutes played']]
        params = st.sidebar.multiselect(
            "Select metrics manually",
            available_metrics,
            default=available_metrics[:5] if len(available_metrics) >= 5 else available_metrics
        )

    if not params:
        st.warning("Please select at least one metric to continue.")
        st.stop()

    # Filters
    if 'Age' in combined_df.columns:
        max_age = st.slider("Maximum age", 15, 45, 30)
        combined_df = combined_df[combined_df['Age'] <= max_age]
    if 'Minutes played' in combined_df.columns:
        min_minutes = st.slider("Minimum minutes played", 0, 3000, 500)
        combined_df = combined_df[combined_df['Minutes played'] >= min_minutes]

    # Reference player
    player_name = st.selectbox("Select the reference player", combined_df['Player'].unique())
    ref_data = combined_df[combined_df['Player'] == player_name].iloc[0]

    # Calculate percentiles
    ref_df = combined_df[combined_df["Source"] == ref_data["Source"]]
    ref_percentiles = {}
    for param in params:
        series = ref_df[param].dropna()
        val = ref_data[param]
        ref_percentiles[param] = percentileofscore(series, val, kind='rank') if len(series) > 0 else 0

    # Calculate distances
    distances = []
    for _, row in combined_df.iterrows():
        if row['Player'] == player_name and row["Source"] == ref_data["Source"]:
            continue
        dist = 0
        count = 0
        for param in params:
            if pd.notna(row[param]):
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

    # Display results with checkboxes
    results_df = pd.DataFrame(distances, columns=["Player", "Team", "Position", "Source", "Distance"])
    results_df = results_df.sort_values("Distance")
    results_df['Compare'] = False  # Default unchecked
    
    st.subheader("Most similar players")
    edited_df = st.data_editor(
        results_df,
        column_config={
            "Compare": st.column_config.CheckboxColumn(
                "Compare on Radar",
                help="Select players to include in radar comparison",
                default=False,
            )
        },
        hide_index=True,
        use_container_width=True
    )

    # Radar comparison
    selected_players = edited_df[edited_df['Compare']]
    
    if not selected_players.empty:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        angles = np.linspace(0, 2*np.pi, len(params), endpoint=False).tolist()
        angles += angles[:1]
        
        # Reference player
        ref_vals = [ref_percentiles[p] for p in params]
        ref_vals_plot = ref_vals + ref_vals[:1]
        ax.plot(angles, ref_vals_plot, label=player_name, color='blue')
        ax.fill(angles, ref_vals_plot, alpha=0.25, color='blue')
        
        # Selected players
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        for idx, (_, row) in enumerate(selected_players.iterrows()):
            player_data = combined_df[(combined_df['Player'] == row['Player']) & 
                                    (combined_df['Source'] == row['Source'])].iloc[0]
            source_df = combined_df[combined_df["Source"] == row["Source"]]
            player_vals = [
                percentileofscore(source_df[p].dropna(), player_data[p], kind='rank') 
                for p in params
            ]
            player_vals_plot = player_vals + player_vals[:1]
            ax.plot(angles, player_vals_plot, label=row['Player'], color=colors[idx % len(colors)])
            ax.fill(angles, player_vals_plot, alpha=0.25, color=colors[idx % len(colors)])
        
        # Format radar
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(params, fontsize=8)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_title(f"{player_name} vs Selected Players")
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        st.pyplot(fig)
    else:
        st.info("Select players from the table above to compare on radar")

else:
    st.info("Please upload Excel files to start")
