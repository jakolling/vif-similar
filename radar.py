import os
import numpy as np
import pandas as pd
from scipy.stats import zscore, percentileofscore
from mplsoccer import Radar, FontManager, grid
import matplotlib.pyplot as plt


# === Métricas específicas de goleiros ===
GK_METRICS = [
    "Save rate, %",
    "Prevented goals per 90",
    "Conceded goals per 90",
    "Shots against per 90",
    "Clean sheets",
    "Back passes received as GK per 90",
    "xG against",
    "xG against per 90",
    "Prevented goals",
    "Shots against",
    "Conceded goals",
    "Exits per 90"
]

# === CONFIGURAÇÃO DE FONTES ===
URL = ('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
       'RobotoSlab%5Bwght%5D.ttf')
roboto_bold = FontManager(URL)

def calculate_derived_metrics(df):
    def calculate_zscore_composite(components):
        z_scores = {}
        for col, weight in components.items():
            if col in df.columns and df[col].notna().any():
                if '%' in col or 'percentage' in col.lower():
                    series = df[col].fillna(0)
                    if series.max() > 1:
                        series = series / 100
                    z_scores[col] = zscore(series) * weight
                else:
                    z_scores[col] = zscore(df[col].fillna(0)) * weight
        return z_scores

    derived_metrics = [
        "npxG", "npxG per 90", "npxG per Shot", "Box Threat",
        "xG Buildup", "Creativity", "Progression", "Defence",
        "Involvement", "Discipline", "G-xG", "Poaching",
        "Finishing", "Aerial Threat", "Passing Quality", "Aerial Defence"
    ]
    for metric in derived_metrics:
        if metric not in df.columns:
            df[metric] = np.nan

    if all(col in df.columns for col in ["xG", "Penalties taken"]):
        df["npxG"] = zscore(df["xG"] - (df["Penalties taken"] * 0.81))
    
    if all(col in df.columns for col in ["Goals", "xG"]):
        df["G-xG"] = zscore(df["Goals"] - df["xG"])
    
    if "npxG" in df.columns and "Minutes played" in df.columns:
        df["npxG per 90"] = zscore(df["npxG"] / (df["Minutes played"] / 90))
    
    if "npxG" in df.columns and "Shots" in df.columns:
        df["npxG per Shot"] = zscore(df["npxG"] / df["Shots"])

    if all(col in df.columns for col in ["npxG per 90", "Touches in box per 90"]):
        with np.errstate(divide='ignore', invalid='ignore'):
            box_threat = np.where(
                df["Touches in box per 90"].fillna(0) == 0,
                np.nan,
                df["npxG per 90"] / np.log(df["Touches in box per 90"] + 1)
            )
            df["Box Threat"] = zscore(box_threat)

    xgb_weights = {
        "xA per 90": 3.0,
        "Shot assists per 90": 3.0,
        "npxG per 90": 2.5,
        "Key passes per 90": 2.5,
        "Deep completions per 90": 2.5,
        "Deep completed crosses per 90": 2.0,
        "Second assists per 90": 1.5,
        "Accurate passes %": 1.0
    }
    
    xgb_z_scores = calculate_zscore_composite(xgb_weights)
    if xgb_z_scores:
        total_weight = sum(w for k, w in xgb_weights.items() if k in xgb_z_scores)
        df["xG Buildup"] = sum(xgb_z_scores.values()) / total_weight

    creativity_metrics = {
        "volume": {
            "Smart passes per 90": 1.0,
            "Through passes per 90": 0.8,
            "Passes to penalty area per 90": 0.6,
        },
        "accuracy": {
            "Accurate smart passes, %": 1.0,
            "Accurate through passes, %": 0.9,
            "Accurate passes to penalty area, %": 0.8,
            "Accurate passes %": 0.6 if "Accurate passes %" in df.columns else 0,
        }
    }
    
    if all(col in df.columns for col in list(creativity_metrics["volume"].keys()) + 
                               list(k for k in creativity_metrics["accuracy"].keys() 
                                   if creativity_metrics["accuracy"][k] > 0)):
        
        volume_scores = []
        for metric, weight in creativity_metrics["volume"].items():
            if df[metric].notna().any():
                z = zscore(df[metric].fillna(0)) * weight
                volume_scores.append(z)
        
        accuracy_scores = []
        for metric, weight in creativity_metrics["accuracy"].items():
            if weight > 0 and metric in df.columns and df[metric].notna().any():
                scaled = (df[metric].fillna(0) / 100 * weight)
                accuracy_scores.append(scaled)
        
        if volume_scores and accuracy_scores:
            df["Creativity"] = (0.6 * np.mean(volume_scores, axis=0) + 
                              0.4 * np.mean(accuracy_scores, axis=0))
            min_val = df["Creativity"].min()
            max_val = df["Creativity"].max()
            if max_val != min_val:
                df["Creativity"] = ((df["Creativity"] - min_val) / 
                                  (max_val - min_val)) * 100
            else:
                df["Creativity"] = 50

    progression_volume = {
        "Progressive passes per 90": 1.0,
        "Progressive runs per 90": 0.9,
        "Dribbles per 90": 0.7,
        "Accelerations per 90": 0.6
    }

    progression_accuracy = {
        "Accurate progressive passes, %": 1.0,
        "Successful dribbles, %": 0.8,
        "Accurate passes %": 0.3
    }

    vol_scores = []
    for col, weight in progression_volume.items():
        if col in df.columns and df[col].notna().any():
            vol_scores.append(zscore(df[col].fillna(0)) * weight)

    acc_scores = []
    for col, weight in progression_accuracy.items():
        if col in df.columns and df[col].notna().any():
            series = df[col].fillna(0)
            if series.max() > 1:
                series = series / 100
            acc_scores.append(zscore(series) * weight)

    if vol_scores and acc_scores:
        df["Progression"] = 0.6 * np.mean(vol_scores, axis=0) + 0.4 * np.mean(acc_scores, axis=0)

    defence_weights = {
        "Successful defensive actions per 90": 1.5,
        "PAdj Interceptions": 1.5,
        "Shots blocked per 90": 1.0,
        "PAdj Sliding tackles": 1.0,
        "Defensive duels per 90": 0.5,
        "Defensive duels won, %": 3.0,
        "Aerial duels won, %": 1.5
    }

    defence_z_scores = calculate_zscore_composite(defence_weights)
    if defence_z_scores:
        total_weight = sum(w for k, w in defence_weights.items() if k in defence_z_scores)
        df["Defence"] = sum(defence_z_scores.values()) / total_weight

    involvement_components = {
        'Passes per 90': 0.20,
        'Received passes per 90': 0.15,
        'Touches per 90': 0.05,
        'Defensive duels per 90': 0.10,
        'PAdj Interceptions': 0.10,
        'Successful defensive actions per 90': 0.05,
        'Touches in box per 90': 0.10,
        'Offensive duels per 90': 0.10,
        'Progressive runs per 90': 0.05,
        'Aerial duels per 90': 0.10
    }

    involvement_z_scores = calculate_zscore_composite(involvement_components)
    if involvement_z_scores:
        df["Involvement"] = sum(involvement_z_scores.values())

    disc_cols = ["Fouls per 90", "Yellow cards per 90", "Red cards per 90"]
    if all(col in df.columns for col in disc_cols):
        penalty = (
            df["Fouls per 90"] * 1.0 + 
            df["Yellow cards per 90"] * 2.0 + 
            df["Red cards per 90"] * 4.0
        )
        df["Discipline"] = -zscore(penalty)

    if all(col in df.columns for col in ["Goals", "Penalties taken", "Minutes played"]):
        df["Non-penalty goals per 90"] = (df["Goals"] - df["Penalties taken"]) / (df["Minutes played"] / 90)

    poaching_components = {
        "npxG per Shot": 0.35,
        "Goal conversion, %": 0.30,
        "Touches in box per 90": -0.25,
        "Received passes per 90": -0.20,
        "Non-penalty goals per 90": 0.40
    }

    poaching_z_scores = calculate_zscore_composite(poaching_components)
    if poaching_z_scores:
        df["Poaching"] = sum(poaching_z_scores.values())

    finishing_components = {
        "Goal conversion, %": 0.35,
        "Non-penalty goals per 90": 0.3,
        "Shots on target, %": 0.25,
        "G-xG": 0.15,
        "npxG per Shot": 0.10
    }
    
    finishing_z_scores = calculate_zscore_composite(finishing_components)
    if finishing_z_scores:
        df["Finishing"] = sum(finishing_z_scores.values())

    aerial_components = {
        "Head goals per 90": 0.35,
        "Aerial duels per 90": 0.20,
        "Aerial duels won, %": 0.20,
    }
    
    aerial_z_scores = calculate_zscore_composite(aerial_components)
    if aerial_z_scores:
        df["Aerial Threat"] = sum(aerial_z_scores.values())

    passing_components = {
        "Passes to final third per 90": ("Accurate passes to final third, %", 0.35),
        "Forward passes per 90": ("Accurate forward passes, %", 0.30),
        "Long passes per 90": ("Accurate long passes, %", 0.15),
        "Lateral passes per 90": ("Accurate lateral passes, %", 0.10),
        "Back passes per 90": ("Accurate back passes, %", 0.05),
        "Passes per 90": ("Accurate passes, %", 0.05)
    }

    passing_scores = []
    for vol_col, (acc_col, weight) in passing_components.items():
        if vol_col in df.columns and acc_col in df.columns:
            vol_z = zscore(df[vol_col].fillna(0))
            acc_series = df[acc_col].fillna(0)
            if acc_series.max() > 1:
                acc_series = acc_series / 100
            acc_z = zscore(acc_series)
            combined = (0.3 * vol_z + 0.7 * acc_z) * weight
            passing_scores.append(combined)

    if passing_scores:
        total_weight = sum(weight for vol_col, (acc_col, weight) in passing_components.items() if acc_col in df.columns)
        df["Passing Quality"] = sum(passing_scores) / total_weight

    aerial_defence_components = {
        "Aerial duels per 90": 0.35,
        "Aerial duels won, %": 0.40,
        "PAdj Interceptions": 0.10,
        "Shots blocked per 90": 0.10
    }

    aerial_defence_z_scores = calculate_zscore_composite(aerial_defence_components)
    if aerial_defence_z_scores:
        df["Aerial Defence"] = sum(aerial_defence_z_scores.values())

    for metric in derived_metrics:
        if metric in df.columns:
            if df[metric].nunique() > 1:
                min_val = df[metric].min()
                max_val = df[metric].max()
                df[metric] = ((df[metric] - min_val) / (max_val - min_val)) * 100
            else:
                df[metric] = 50

    negative_metrics = [
        "Conceded goals per 90",
        "xG against per 90",
        "Fouls per 90",
        "Yellow cards per 90",
        "Red cards per 90"
    ]

    for metric in negative_metrics:
        if metric in df.columns:
            if df[metric].nunique() > 1:
                max_val = df[metric].max()
                min_val = df[metric].min()
                df[metric] = ((max_val - df[metric]) / (max_val - min_val)) * 100
            else:
                df[metric] = 50

    return df

RADAR_PRESETS = {
    "goalkeeper": [
        "Save rate, %",
        "Prevented goals per 90",
        "Conceded goals per 90",
        "Shots against per 90",
        "Clean sheets",
        "Back passes received as GK per 90",
        "Passes per 90",
        "Accurate passes, %",
        "Long passes per 90",
        "Accurate long passes, %",
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Exits per 90"
    ],
    "general_summary": [
        "Involvement",
        "Box Threat",
        "Creativity",
        "xG Buildup",
        "Progression",
        "Defence",
        "Discipline",
        "Passing Quality"
    ],
    "center_back": [
        "Defence",
        "Aerial Defence",
        "Aerial Threat",
        "Passing Quality",
        "Progression",
        "Discipline",
        "Involvement"
    ],
    "full_back": [
        "Progression",
        "Creativity",
        "Passing Quality",
        "Defence",
        "Aerial Defence",
        "Involvement",
        "Discipline",
        "Deep completed crosses per 90"
    ],
    "defensive_midfielder": [
        "Defence",
        "Progression",
        "Passing Quality",
        "Discipline",
        "Involvement",
        "xG Buildup",
        "Creativity",
        "Successful defensive actions per 90",
        "pAdj interceptions"        
    ],
    "central_midfielder": [
        "Progression",
        "Passing Quality",
        "Creativity",
        "xG Buildup",
        "Involvement",
        "Defence",
        "Discipline"
    ],
    "attacking_midfielder": [
        "Creativity",
        "xG Buildup",
        "Progression",
        "Passing Quality",
        "Box Threat",
        "Involvement",
        "Finishing",
        "xA per 90",
        "Deep completions per 90"
    ],
    "winger": [
        "Progression",
        "Creativity",
        "Box Threat",
        "Passing Quality",
        "Crossing",
        "Involvement",
        "Finishing",
        "Successful attacking actions per 90",
        "Deep completions per 90",
        "Deep completed crosses per 90"
    ],
    "striker": [
        "npxG per 90",
        "npxG per Shot",
        "Finishing",
        "Poaching",
        "Aerial Threat",
        "Box Threat",
        "Involvement",
        "Touches in box per 90"
    ],
    "playmaking": [
        "Creativity",
        "Passing Quality",
        "Progression",
        "xG Buildup",
        "Successful attacking actions per 90",
        "Deep completions per 90",
        "Involvement",
        "Key passes per 90"
    ],
    "defensive_actions": [
        "Defence",
        "Aerial Defence",
        "PAdj Interceptions",
        "Successful defensive actions per 90",
        "Defensive duels won, %",
        "Shots blocked per 90",
        "Discipline"
    ],
    "crossing": [
        "Crossing",
        "Accurate crosses, %",
        "Deep completed crosses per 90",
        "xA per 90",
        "Shot assists per 90",
        "Creativity",
        "Passing Quality"
    ],
    "aerial_duels": [
        "Aerial Threat",
        "Aerial Defence",
        "Aerial duels per 90",
        "Aerial duels won, %",
        "Head goals per 90",
        "Involvement",
        "Defence"
    ],
    "shooting": [
        "npxG per 90",
        "npxG per Shot",
        "Finishing",
        "Goal conversion, %",
        "Shots on target, %",
        "G-xG",
        "Box Threat"
    ],
    "counter_attack": [
        "Progression",
        "Accelerations per 90",
        "Dribbles per 90",
        "Successful dribbles, %",
        "npxG per 90",
        "Finishing",
        "Box Threat"
    ],
    "build_up": [
        "Passing Quality",
        "Progression",
        "xG Buildup",
        "Deep completions per 90",
        "Involvement",
        "Creativity",
        "Discipline"
    ]
}

def ask_include_sc_data(country_path):
    include_sc = input("\nDeseja incluir dados do Skill Corner na análise? (s/n): ").lower()
    if include_sc != 's':
        return None, None
    
    sc_files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' in f.lower()]
    if not sc_files:
        print("Nenhum arquivo do Skill Corner encontrado na pasta do país.")
        return None, None
    
    print("\nArquivos do Skill Corner disponíveis:")
    for i, file in enumerate(sc_files, 1):
        print(f"{i}. {file}")
    
    try:
        file_index = int(input("\nSelecione o arquivo do Skill Corner correspondente (número): ")) - 1
        selected_sc_file = sc_files[file_index]
        sc_file_path = os.path.join(country_path, selected_sc_file)
        sc_df = pd.read_excel(sc_file_path)
        print(f"\nArquivo do Skill Corner carregado: {selected_sc_file}")
        
        sc_metrics = [col for col in sc_df.columns if col not in ['Short Name', 'Team', 'Position', 'Minutes played']]
        print("\nMétricas disponíveis no Skill Corner:")
        for i, metric in enumerate(sc_metrics, 1):
            print(f"{i}. {metric}")
        
        selected_indices = input("\nSelecione as métricas para incluir (números separados por vírgula, deixe em branco para todas): ").strip()
        if selected_indices:
            selected_indices = [int(i.strip())-1 for i in selected_indices.split(",") if i.strip().isdigit()]
            selected_metrics = [sc_metrics[i] for i in selected_indices if 0 <= i < len(sc_metrics)]
        else:
            selected_metrics = sc_metrics
        
        return sc_df, selected_metrics
    except Exception as e:
        print(f"Erro ao carregar dados do Skill Corner: {str(e)}")
        return None, None

def merge_sc_data(player_data, sc_df, selected_sc_metrics, player_name):
    if sc_df is None or selected_sc_metrics is None:
        return player_data
    
    try:
        sc_player_data = sc_df[sc_df['Short Name'].str.contains(player_name, case=False, na=False)]
        if not sc_player_data.empty:
            sc_player_data = sc_player_data.iloc[0]
            
            for metric in selected_sc_metrics:
                if metric in sc_player_data.index and pd.notna(sc_player_data[metric]):
                    player_data[metric] = sc_player_data[metric]
            
            print(f"\nDados do Skill Corner incluídos para {player_name}")
        else:
            print(f"\nJogador {player_name} não encontrado nos dados do Skill Corner")
    except Exception as e:
        print(f"\nErro ao mesclar dados do Skill Corner: {str(e)}")
    
    return player_data

def create_single_radar():
    base_path = r'C:\Users\jakol\OneDrive\Data'

    countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    country_index = int(input("\nSelecione um país (número): ")) - 1
    selected_country = countries[country_index]
    country_path = os.path.join(base_path, selected_country)

    files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
    print(f"\nArquivos Wyscout em {selected_country}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    file_index = int(input("\nSelecione um arquivo (número): ")) - 1
    selected_file = files[file_index]
    file_path = os.path.join(country_path, selected_file)
    df = pd.read_excel(file_path)
    print(f"\nArquivo carregado: {selected_file}")

    sc_df, selected_sc_metrics = ask_include_sc_data(country_path)

    df = calculate_derived_metrics(df)

    print("\nJogadores disponíveis:")
    print(df['Player'].tolist())
    
    player_name = input("\nDigite o nome do jogador: ")
    player_data = df[df['Player'] == player_name].iloc[0]

    player_data = merge_sc_data(player_data, sc_df, selected_sc_metrics, player_name)

    print("\nCores disponíveis para o gráfico:")
    color_choices = {
        "Verde petróleo": "#2A9D8F",
        "Laranja queimado": "#E76F51",
        "Azul escuro": "#264653",
        "Amarelo mostarda": "#E9C46A",
        "Roxo escuro": "#6A0572",
        "Azul claro": "#2196F3",
        "Vermelho": "#D72638",
        "Cinza escuro": "#4B4E6D",
        "Verde limão": "#A8DADC",
        "Rosa": "#F28482",
        "Marrom": "#8D6E63",
        "Preto": "#000000",
    }

    for i, (name, code) in enumerate(color_choices.items(), 1):
        print(f"{i}. {name} ({code})")

    selected_color_index = input("\nEscolha uma cor para o jogador 1 (número): ").strip()
    try:
        selected_color_index = int(selected_color_index)
        if 1 <= selected_color_index <= len(color_choices):
            selected_color_name = list(color_choices.keys())[selected_color_index - 1]
            primary_color = color_choices[selected_color_name]
            edge_color = "#000000" if primary_color != "#000000" else "#ffffff"
        else:
            raise ValueError
    except:
        print("Cor inválida, usando verde petróleo como padrão.")
        primary_color = "#2A9D8F"
        edge_color = "#264653"

    selected_color_index_2 = input("Escolha uma cor para o jogador 2 (número): ").strip()
    try:
        selected_color_index_2 = int(selected_color_index_2)
        if 1 <= selected_color_index_2 <= len(color_choices):
            selected_color_name_2 = list(color_choices.keys())[selected_color_index_2 - 1]
            primary_color_2 = color_choices[selected_color_name_2]
            edge_color_2 = "#000000" if primary_color_2 != "#000000" else "#ffffff"
        else:
            raise ValueError
    except:
        print("Cor inválida, usando laranja queimado como padrão.")
        primary_color_2 = "#E76F51"
        edge_color_2 = "#cc5500"

    print("\nPresets de radar (selecione até 3):")
    preset_keys = list(RADAR_PRESETS.keys())
    for i, preset in enumerate(preset_keys, 1):
        print(f"{i}. {preset}")
    
    selected_presets = []
    while len(selected_presets) < 3:
        preset_choice = input(f"\nEscolha o preset {len(selected_presets)+1} (número, ou deixe em branco para parar): ").strip()
        if not preset_choice:
            if len(selected_presets) == 0:
                print("Você deve selecionar pelo menos um preset.")
                continue
            break
            
        try:
            preset_index = int(preset_choice) - 1
            if 0 <= preset_index < len(preset_keys):
                selected_preset = preset_keys[preset_index]
                if selected_preset not in selected_presets:
                    selected_presets.append(selected_preset)
                    print(f"Preset selecionado: {selected_preset}")
                else:
                    print("Este preset já foi selecionado.")
            else:
                print("Número inválido. Por favor, escolha um número da lista.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

    all_params = []
    for preset in selected_presets:
        all_params.extend(RADAR_PRESETS[preset])
    all_params = list(dict.fromkeys(all_params))
    
    if selected_sc_metrics:
        all_params.extend([m for m in selected_sc_metrics if m not in all_params])
    
    params = [p for p in all_params if p in player_data.index and pd.notna(player_data[p])]
    
    if not params:
        print("Erro: Nenhum dos parâmetros selecionados existe no dataframe.")
        print("Colunas disponíveis:", player_data.index.tolist())
        return

    print(f"\nMétricas sendo usadas no radar ({len(params)}):")
    print(", ".join(params))

    values = []
    for param in params:
        if param in GK_METRICS:
            series = df[df["Position"].str.contains("GK", na=False)][param].dropna()
        else:
            series = df[param].dropna()
        
        player_value = player_data[param] if pd.notna(player_data[param]) else 0
        perc = percentileofscore(series, player_value, kind='rank') if len(series) > 0 else 0
        values.append(perc)

    minutes_played = int(player_data["Minutes played"]) if pd.notna(player_data["Minutes played"]) else 0

    fig, axs = grid(figheight=6, grid_height=0.80, title_height=0.15, endnote_height=0.04,
                    title_space=0, endnote_space=0, grid_key='radar', axis=False)
    fig.set_size_inches(10, 6)

    radar = Radar(params, [0]*len(params), [100]*len(params),
                  round_int=[False]*len(params),
                  num_rings=4, ring_width=1, center_circle_radius=1)

    radar.setup_axis(ax=axs['radar'])
    radar.draw_circles(ax=axs['radar'], facecolor='#f0f0f0', edgecolor='#cccccc')
    radar.spoke(ax=axs['radar'], color='#cccccc', linestyle='--', zorder=2)

    radar_poly, rings_outer, vertices = radar.draw_radar(
        values, ax=axs['radar'],
        kwargs_radar={'facecolor': primary_color, 'alpha': 0.7, 'lw': 3, 'edgecolor': edge_color},
        kwargs_rings={'facecolor': '#ffffff'}
    )

    axs['radar'].scatter(vertices[:, 0], vertices[:, 1],
                         c='#264653', edgecolors='black', s=40, zorder=3)

    radar.draw_range_labels(ax=axs['radar'], fontsize=7, fontproperties=roboto_bold.prop)
    radar.draw_param_labels(ax=axs['radar'], fontsize=7, fontproperties=roboto_bold.prop)

    team = player_data['Team']
    position = player_data['Position']
    league = input("Digite o nome da liga: ")
    season = input("Digite a temporada (ex: 2023/24): ")

    age = int(player_data["Age"]) if "Age" in player_data and pd.notna(player_data["Age"]) else "?"
    title_text = f"{player_name} | {team} | {age}y | {minutes_played} min"
    axs['title'].text(0.5, 0.75, title_text, fontsize=22, fontproperties=roboto_bold.prop,
                      ha='center', va='center', color='#000000')

    subtitle_text = f"{league} ({season}) | {position}"
    axs['title'].text(0.5, 0.45, subtitle_text, fontsize=16, fontproperties=roboto_bold.prop,
                      ha='center', va='center', color='#2A9D8F')

    axs['endnote'].text(0.99, 0.5, "made by Joao Alberto Kolling – jakolling@gmail.com",
                        fontsize=10, ha='right', va='center', color="#999999",
                        fontproperties=roboto_bold.prop)

    plt.show()

def compare_players():
    base_path = r'C:\Users\jakol\OneDrive\Data'

    print("\n=== PRIMEIRO JOGADOR ===")
    countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    country_index = int(input("\nSelecione país para o primeiro jogador (número): ")) - 1
    selected_country = countries[country_index]
    country_path = os.path.join(base_path, selected_country)

    files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
    print(f"\nArquivos em {selected_country}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    file_index = int(input("\nSelecione arquivo para o primeiro jogador (número): ")) - 1
    selected_file = files[file_index]
    file_path = os.path.join(country_path, selected_file)
    df1 = pd.read_excel(file_path)
    
    sc_df1, selected_sc_metrics1 = ask_include_sc_data(country_path)
    
    df1 = calculate_derived_metrics(df1)
    print(f"\nArquivo carregado: {selected_file}")

    print("\nJogadores disponíveis:")
    print(df1['Player'].tolist())
    
    player_name1 = input("\nDigite o nome do primeiro jogador: ")
    player_data1 = df1[df1['Player'] == player_name1].iloc[0]
    
    player_data1 = merge_sc_data(player_data1, sc_df1, selected_sc_metrics1, player_name1)

    print("\n=== SEGUNDO JOGADOR ===")
    countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    country_index = int(input("\nSelecione país para o segundo jogador (número): ")) - 1
    selected_country = countries[country_index]
    country_path = os.path.join(base_path, selected_country)

    files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
    print(f"\nArquivos em {selected_country}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    file_index = int(input("\nSelecione arquivo para o segundo jogador (número): ")) - 1
    selected_file = files[file_index]
    file_path = os.path.join(country_path, selected_file)
    df2 = pd.read_excel(file_path)
    
    sc_df2, selected_sc_metrics2 = ask_include_sc_data(country_path)
    
    df2 = calculate_derived_metrics(df2)
    print(f"\nArquivo carregado: {selected_file}")

    print("\nJogadores disponíveis:")
    print(df2['Player'].tolist())
    
    player_name2 = input("\nDigite o nome do segundo jogador: ")
    player_data2 = df2[df2['Player'] == player_name2].iloc[0]
    
    player_data2 = merge_sc_data(player_data2, sc_df2, selected_sc_metrics2, player_name2)

    print("\nCores disponíveis para o gráfico:")
    color_choices = {
        "Verde petróleo": "#2A9D8F",
        "Laranja queimado": "#E76F51",
        "Azul escuro": "#264653",
        "Amarelo mostarda": "#E9C46A",
        "Roxo escuro": "#6A0572",
        "Azul claro": "#2196F3",
        "Vermelho": "#D72638",
        "Cinza escuro": "#4B4E6D",
        "Verde limão": "#A8DADC",
        "Rosa": "#F28482",
        "Marrom": "#8D6E63",
        "Preto": "#000000",
    }

    for i, (name, code) in enumerate(color_choices.items(), 1):
        print(f"{i}. {name} ({code})")

    selected_color_index = input("\nEscolha uma cor para o jogador 1 (número): ").strip()
    try:
        selected_color_index = int(selected_color_index)
        if 1 <= selected_color_index <= len(color_choices):
            selected_color_name = list(color_choices.keys())[selected_color_index - 1]
            color1 = color_choices[selected_color_name]
        else:
            raise ValueError
    except:
        print("Cor inválida, usando verde petróleo como padrão.")
        color1 = "#2A9D8F"

    selected_color_index_2 = input("Escolha uma cor para o jogador 2 (número): ").strip()
    try:
        selected_color_index_2 = int(selected_color_index_2)
        if 1 <= selected_color_index_2 <= len(color_choices):
            selected_color_name_2 = list(color_choices.keys())[selected_color_index_2 - 1]
            color2 = color_choices[selected_color_name_2]
        else:
            raise ValueError
    except:
        print("Cor inválida, usando laranja queimado como padrão.")
        color2 = "#E76F51"

    print("\nPresets de radar (selecione até 3):")
    preset_keys = list(RADAR_PRESETS.keys())
    for i, preset in enumerate(preset_keys, 1):
        print(f"{i}. {preset}")
    
    selected_presets = []
    while len(selected_presets) < 3:
        preset_choice = input(f"\nEscolha o preset {len(selected_presets)+1} (número, ou deixe em branco para parar): ").strip()
        if not preset_choice:
            if len(selected_presets) == 0:
                print("Você deve selecionar pelo menos um preset.")
                continue
            break
            
        try:
            preset_index = int(preset_choice) - 1
            if 0 <= preset_index < len(preset_keys):
                selected_preset = preset_keys[preset_index]
                if selected_preset not in selected_presets:
                    selected_presets.append(selected_preset)
                    print(f"Preset selecionado: {selected_preset}")
                else:
                    print("Este preset já foi selecionado.")
            else:
                print("Número inválido. Por favor, escolha um número da lista.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

    all_params = []
    for preset in selected_presets:
        all_params.extend(RADAR_PRESETS[preset])
    all_params = list(dict.fromkeys(all_params))
    
    if selected_sc_metrics1:
        all_params.extend([m for m in selected_sc_metrics1 if m not in all_params])
    if selected_sc_metrics2:
        all_params.extend([m for m in selected_sc_metrics2 if m not in all_params])
    
    common_params = []
    for p in all_params:
        if p in df1.columns and p in df2.columns:
            common_params.append(p)
    
    if not common_params:
        print("Erro: Nenhum parâmetro em comum entre os dois datasets.")
        print("Colunas no primeiro dataset:", df1.columns.tolist())
        print("Colunas no segundo dataset:", df2.columns.tolist())
        return
    
    params = common_params

    values1 = []
    values2 = []
    for param in params:
        series1 = df1[param].dropna()
        player_value1 = player_data1[param] if pd.notna(player_data1[param]) else 0
        perc1 = percentileofscore(series1, player_value1, kind='rank') if len(series1) > 0 else 0
        values1.append(perc1)
        
        series2 = df2[param].dropna()
        player_value2 = player_data2[param] if pd.notna(player_data2[param]) else 0
        perc2 = percentileofscore(series2, player_value2, kind='rank') if len(series2) > 0 else 0
        values2.append(perc2)

    minutes_played1 = int(player_data1["Minutes played"]) if pd.notna(player_data1["Minutes played"]) else 0
    minutes_played2 = int(player_data2["Minutes played"]) if pd.notna(player_data2["Minutes played"]) else 0

    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=params,
        fill='toself',
        name=player_name1,
        line=dict(color=color1),
        fillcolor=f"rgba({int(color1[1:3], 16)}, {int(color1[3:5], 16)}, {int(color1[5:7], 16)}, 0.5)"
    ))

    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=params,
        fill='toself',
        name=player_name2,
        line=dict(color=color2),
        fillcolor=f"rgba({int(color2[1:3], 16)}, {int(color2[3:5], 16)}, {int(color2[5:7], 16)}, 0.5)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=['25%', '50%', '75%', '100%'],
                tickfont=dict(size=10)
            ),
            bgcolor='#f0f0f0',
            sector=[0, 360],
        ),
        showlegend=True,
        title={
            'text': f"<b>{player_name1} vs {player_name2}</b><br>"
                    f"<span style='font-size:12px; color:{color1}'>{player_name1}: {player_data1['Team']} | {player_data1['Position']} | {minutes_played1} min</span><br>"
                    f"<span style='font-size:12px; color:{color2}'>{player_name2}: {player_data2['Team']} | {player_data2['Position']} | {minutes_played2} min</span>",
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        font=dict(
            family="Roboto Slab, sans-serif",
            size=12,
            color="#000000"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        margin=dict(
            l=150,
            r=150,
            t=150,
            b=150
        ),
        autosize=True,
        height=700,
        width=700
    )

    league1 = input("\nDigite o nome da liga do primeiro jogador: ")
    season1 = input("Digite a temporada do primeiro jogador (ex: 2023/24): ")
    league2 = input("Digite o nome da liga do segundo jogador: ")
    season2 = input("Digite a temporada do segundo jogador (ex: 2023/24): ")

    fig.update_layout(
        annotations=[
            dict(
                text=f"<b>Presets:</b> {', '.join(selected_presets)}",
                x=0.5,
                y=-0.15,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="#666666"),
                align="center"
            ),
            dict(
                text=f"{league1} ({season1}) vs {league2} ({season2})",
                x=0.5,
                y=-0.20,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="#666666"),
                align="center"
            ),
            dict(
                text="made by Joao Alberto Kolling – jakolling@gmail.com",
                x=0.5,
                y=-0.25,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="#999999"),
                align="center"
            )
        ]
    )

    fig.show()

def search_players():
    base_path = r'C:\Users\jakol\OneDrive\Data'

    countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    country_index = int(input("\nSelecione um país (número): ")) - 1
    selected_country = countries[country_index]
    country_path = os.path.join(base_path, selected_country)

    files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
    print(f"\nArquivos em {selected_country}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    file_index = int(input("\nSelecione um arquivo (número): ")) - 1
    selected_file = files[file_index]
    file_path = os.path.join(country_path, selected_file)
    df = pd.read_excel(file_path)
    
    sc_df, selected_sc_metrics = ask_include_sc_data(country_path)
    
    print(f"\nArquivo carregado: {selected_file}")

    df = calculate_derived_metrics(df)

    has_age = 'Age' in df.columns
    has_minutes = 'Minutes played' in df.columns

    max_age = None
    if has_age:
        try:
            age_input = input("\nDigite a idade máxima desejada (deixe em branco para ignorar): ").strip()
            if age_input:
                max_age = int(age_input)
                if max_age < 15 or max_age > 50:
                    print("Idade inválida. O filtro de idade será ignorado.")
                    max_age = None
        except ValueError:
            print("Valor inválido. O filtro de idade será ignorado.")

    min_minutes = None
    if has_minutes:
        try:
            minutes_input = input("\nDigite os minutos mínimos jogados (deixe em branco para ignorar): ").strip()
            if minutes_input:
                min_minutes = int(minutes_input)
                if min_minutes < 0:
                    print("Valor inválido. O filtro de minutos será ignorado.")
                    min_minutes = None
        except ValueError:
            print("Valor inválido. O filtro de minutos será ignorado.")

    composite_metrics = [
        "npxG", "npxG per 90", "npxG per Shot", "Box Threat",
        "xG Buildup", "Creativity", "Progression", "Defence",
        "Involvement", "Discipline", "G-xG", "Poaching",
        "Finishing", "Aerial Threat", "Passing Quality", "Aerial Defence"
    ]
    
    available_metrics = [m for m in composite_metrics if m in df.columns and df[m].notna().any()]
    
    if selected_sc_metrics:
        available_metrics.extend([m for m in selected_sc_metrics if m not in available_metrics])
    
    if not available_metrics:
        print("\nNenhuma métrica composta disponível para busca.")
        return
    
    print("\nMétricas compostas disponíveis:")
    for i, metric in enumerate(available_metrics, 1):
        print(f"{i}. {metric}")

    try:
        selected_indices = input("\nDigite os números das métricas para filtrar (separados por vírgula): ")
        selected_indices = [int(i.strip()) - 1 for i in selected_indices.split(",") if i.strip().isdigit()]
        selected_metrics = [available_metrics[i] for i in selected_indices if 0 <= i < len(available_metrics)]
        
        if not selected_metrics:
            print("Nenhuma métrica válida selecionada.")
            return
    except:
        print("Entrada inválida. Por favor, tente novamente.")
        return

    min_percentiles = {}
    for metric in selected_metrics:
        while True:
            try:
                min_perc = float(input(f"Digite o percentil mínimo para {metric} (0-100): "))
                if 0 <= min_perc <= 100:
                    min_percentiles[metric] = min_perc
                    break
                else:
                    print("Por favor, digite um valor entre 0 e 100.")
            except ValueError:
                print("Entrada inválida. Digite um número.")

    filtered_df = df.copy()
    
    if max_age is not None and has_age:
        filtered_df = filtered_df[filtered_df['Age'] <= max_age]
    
    if min_minutes is not None and has_minutes:
        filtered_df = filtered_df[filtered_df['Minutes played'] >= min_minutes]
    
    for metric, min_perc in min_percentiles.items():
        valid_series = filtered_df[metric].dropna()
        if len(valid_series) == 0:
            print(f"\nAviso: A métrica {metric} não possui valores válidos para cálculo de percentis.")
            continue
            
        filtered_df[f'{metric}_percentile'] = filtered_df[metric].rank(pct=True, na_option='keep') * 100
        
        filtered_df = filtered_df[
            (filtered_df[f'{metric}_percentile'] >= min_perc) & 
            (filtered_df[metric].notna())
        ].copy()
    
    if filtered_df.empty:
        print("\nNenhum jogador encontrado com os critérios especificados.")
        return
    
    print(f"\n{len(filtered_df)} jogador(es) encontrado(s):")
    
    percentile_cols = [f'{m}_percentile' for m in selected_metrics if f'{m}_percentile' in filtered_df.columns]
    if percentile_cols:
        filtered_df['avg_percentile'] = filtered_df[percentile_cols].mean(axis=1)
        filtered_df = filtered_df.sort_values('avg_percentile', ascending=False)
    
    display_cols = ['Player', 'Team', 'Position']
    if has_age:
        display_cols.append('Age')
    if has_minutes:
        display_cols.append('Minutes played')
    display_cols += selected_metrics
    if percentile_cols:
        display_cols += percentile_cols
    
    print(filtered_df[display_cols].to_string(index=False))
    
    export = input("\nDeseja exportar os resultados para CSV? (s/n): ").lower()
    if export == 's':
        output_file = input("Digite o nome do arquivo de saída (sem extensão): ").strip()
        if not output_file:
            output_file = "jogadores_filtrados"
        output_file += ".csv"
        
        filtered_df.to_csv(output_file, index=False)
        print(f"Resultados exportados para {output_file}")

def find_similar_players():
    base_path = r'C:\Users\jakol\OneDrive\Data'
    
    print("\n=== JOGADOR DE REFERÊNCIA ===")
    countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis:")
    for i, country in enumerate(countries, 1):
        print(f"{i}. {country}")

    country_index = int(input("\nSelecione país para o jogador de referência (número): ")) - 1
    selected_country = countries[country_index]
    country_path = os.path.join(base_path, selected_country)

    files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
    print(f"\nArquivos em {selected_country}:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    file_index = int(input("\nSelecione arquivo para o jogador de referência (número): ")) - 1
    selected_file = files[file_index]
    file_path = os.path.join(country_path, selected_file)
    ref_df = pd.read_excel(file_path)
    
    sc_ref_df, selected_sc_metrics = ask_include_sc_data(country_path)
    
    ref_df = calculate_derived_metrics(ref_df)
    print(f"\nArquivo carregado: {selected_file}")

    print("\nJogadores disponíveis:")
    print(ref_df['Player'].tolist())
    
    ref_player_name = input("\nDigite o nome do jogador de referência: ")
    ref_player_data = ref_df[ref_df['Player'] == ref_player_name].iloc[0]
    
    ref_player_data = merge_sc_data(ref_player_data, sc_ref_df, selected_sc_metrics, ref_player_name)

    print("\n=== DATASETS PARA BUSCA ===")
    all_countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    print("\nPaíses disponíveis (máximo 20):")
    for i, country in enumerate(all_countries, 1):
        print(f"{i}. {country}")

    selected_countries_indices = input("\nSelecione países para busca (números separados por vírgula, máximo 20): ")
    try:
        selected_countries_indices = [int(i.strip())-1 for i in selected_countries_indices.split(",") if i.strip().isdigit()]
        selected_countries_indices = selected_countries_indices[:20]
        search_countries = [all_countries[i] for i in selected_countries_indices if 0 <= i < len(all_countries)]
        
        if not search_countries:
            print("Nenhum país válido selecionado. Usando apenas o país do jogador de referência.")
            search_countries = [selected_country]
    except:
        print("Entrada inválida. Usando apenas o país do jogador de referência.")
        search_countries = [selected_country]

    all_dfs = []
    for country in search_countries:
        country_path = os.path.join(base_path, country)
        files = [f for f in os.listdir(country_path) if f.endswith('.xlsx') and 'skillcorner' not in f.lower()]
        
        print(f"\nArquivos em {country}:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        file_indices = input(f"Selecione arquivos em {country} (números separados por vírgula, deixe em branco para todos): ")
        try:
            if file_indices.strip():
                selected_indices = [int(i.strip())-1 for i in file_indices.split(",") if i.strip().isdigit()]
                selected_files = [files[i] for i in selected_indices if 0 <= i < len(files)]
            else:
                selected_files = files
        except:
            print(f"Entrada inválida. Carregando todos os arquivos de {country}.")
            selected_files = files
        
        for file in selected_files:
            file_path = os.path.join(country_path, file)
            df = pd.read_excel(file_path)
            
            sc_df, _ = ask_include_sc_data(country_path)
            if sc_df is not None:
                for _, row in df.iterrows():
                    player_name = row['Player']
                    sc_player_data = sc_df[sc_df['Short Name'].str.contains(player_name, case=False, na=False)]
                    if not sc_player_data.empty:
                        sc_player_data = sc_player_data.iloc[0]
                        for col in sc_player_data.index:
                            if col not in row.index:
                                df.loc[df['Player'] == player_name, col] = sc_player_data[col]
            
            df = calculate_derived_metrics(df)
            df['Source'] = f"{country}/{file}"
            all_dfs.append(df)

    if not all_dfs:
        print("Nenhum dataset válido selecionado para busca.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    print("\nPresets de radar para similaridade (selecione até 3):")
    preset_keys = list(RADAR_PRESETS.keys())
    for i, preset in enumerate(preset_keys, 1):
        print(f"{i}. {preset}")
    
    selected_presets = []
    while len(selected_presets) < 3:
        preset_choice = input(f"\nEscolha o preset {len(selected_presets)+1} (número, ou deixe em branco para parar): ").strip()
        if not preset_choice:
            if len(selected_presets) == 0:
                print("Você deve selecionar pelo menos um preset.")
                continue
            break
            
        try:
            preset_index = int(preset_choice) - 1
            if 0 <= preset_index < len(preset_keys):
                selected_preset = preset_keys[preset_index]
                if selected_preset not in selected_presets:
                    selected_presets.append(selected_preset)
                    print(f"Preset selecionado: {selected_preset}")
                else:
                    print("Este preset já foi selecionado.")
            else:
                print("Número inválido. Por favor, escolha um número da lista.")
        except ValueError:
            print("Entrada inválida. Por favor, digite um número.")

    all_params = []
    for preset in selected_presets:
        all_params.extend(RADAR_PRESETS[preset])
    all_params = list(dict.fromkeys(all_params))
    
    if selected_sc_metrics:
        all_params.extend([m for m in selected_sc_metrics if m not in all_params])
    
    params = [p for p in all_params if p in ref_df.columns and p in combined_df.columns]
    
    if not params:
        print("Erro: Nenhum parâmetro em comum entre os datasets.")
        return

    print(f"\nMétricas usadas para similaridade ({len(params)}):")
    print(", ".join(params))

    if 'Age' in combined_df.columns:
        try:
            max_age = input("\nFiltrar por idade MÁXIMA (digite o número ou deixe em branco para ignorar): ").strip()
            if max_age:
                max_age = int(max_age)
                if not (16 <= max_age <= 45):
                    print("⚠️ Valor inválido. Use entre 16-45 anos. Filtro ignorado.")
                else:
                    combined_df = combined_df[combined_df['Age'] <= max_age].copy()
                    print(f"✅ Filtro ativo: Mostrando jogadores com até {max_age} anos")
        except ValueError:
            print("⚠️ Erro: Digite um número válido ou deixe em branco. Filtro ignorado.")

    if 'Minutes played' in combined_df.columns:
        try:
            min_minutes = input("\nMinutos mínimos jogados (deixe em branco para ignorar): ").strip()
            if min_minutes:
                min_minutes = int(min_minutes)
                combined_df = combined_df[combined_df['Minutes played'] >= min_minutes]
        except ValueError:
            print("Valor inválido. Ignorando filtro de minutos.")

    if 'Position' in combined_df.columns and 'Position' in ref_player_data:
        ref_position = ref_player_data['Position']
        print(f"\nPosição do jogador de referência: {ref_position}")
        use_position_filter = input("Deseja filtrar por posição similar? (s/n): ").lower() == 's'
        if use_position_filter and ref_position:
            position_groups = {
                'GK': ['GK'],
                'DF': ['CB', 'LB', 'RB', 'WB', 'FB', 'DF'],
                'MF': ['CM', 'DM', 'AM', 'LM', 'RM', 'MF'],
                'FW': ['CF', 'ST', 'LW', 'RW', 'WF', 'FW']
            }
            
            ref_group = None
            for group, positions in position_groups.items():
                if any(pos in ref_position for pos in positions):
                    ref_group = group
                    break
            
            if ref_group:
                combined_df = combined_df[
                    combined_df['Position'].apply(
                        lambda x: any(pos in str(x) for pos in position_groups[ref_group])
                    )
                ]

    ref_percentiles = {}
    for param in params:
        series = ref_df[param].dropna()
        ref_value = ref_player_data[param] if pd.notna(ref_player_data[param]) else 0
        perc = percentileofscore(series, ref_value, kind='rank') if len(series) > 0 else 0
        ref_percentiles[param] = perc

    for param in params:
        valid_players = combined_df[param].notna()
        combined_df.loc[valid_players, f'{param}_percentile'] = combined_df[valid_players].apply(
            lambda row: percentileofscore(combined_df[valid_players][param], row[param], kind='rank'), 
            axis=1
        )

    distances = []
    for _, row in combined_df.iterrows():
        if row['Player'] == ref_player_name:
            continue
            
        distance = 0
        valid_params = 0
        for param in params:
            if f'{param}_percentile' in row and pd.notna(row[f'{param}_percentile']):
                distance += (row[f'{param}_percentile'] - ref_percentiles[param])**2
                valid_params += 1
        
        if valid_params > 0:
            normalized_distance = (distance / valid_params)**0.5
            distances.append((row['Player'], row['Team'], row['Position'], 
                            row['Source'], normalized_distance))

    if not distances:
        print("Nenhum jogador encontrado com métricas comparáveis.")
        return

    results_df = pd.DataFrame(distances, 
                            columns=['Player', 'Team', 'Position', 'Source', 'Distance'])
    
    results_df = results_df.sort_values('Distance').head(20)

    print("\nJogadores mais similares:")
    print(results_df.to_string(index=False))
    
    if len(results_df) > 0:
        view_comparison = input("\nDeseja visualizar comparação com o jogador mais similar? (s/n): ").lower()
        if view_comparison == 's':
            most_similar = results_df.iloc[0]
            similar_player_name = most_similar['Player']
            similar_source = most_similar['Source']
            
            similar_df = None
            for df in all_dfs:
                if df['Player'].str.contains(similar_player_name).any() and df['Source'].iloc[0] == similar_source:
                    similar_df = df
                    break
            
            if similar_df is not None:
                similar_player_data = similar_df[similar_df['Player'] == similar_player_name].iloc[0]
                
                fig, axs = grid(figheight=6, grid_height=0.80, title_height=0.15, endnote_height=0.04,
                              title_space=0, endnote_space=0, grid_key='radar', axis=False)
                fig.set_size_inches(10, 6)

                radar = Radar(params, [0]*len(params), [100]*len(params),
                            round_int=[False]*len(params),
                            num_rings=4, ring_width=1, center_circle_radius=1)

                radar.setup_axis(ax=axs['radar'])
                radar.draw_circles(ax=axs['radar'], facecolor='#f0f0f0', edgecolor='#cccccc')
                radar.spoke(ax=axs['radar'], color='#cccccc', linestyle='--', zorder=2)

                ref_values = [ref_percentiles[p] for p in params]
                similar_values = []
                for param in params:
                    series = similar_df[param].dropna()
                    player_value = similar_player_data[param] if pd.notna(similar_player_data[param]) else 0
                    perc = percentileofscore(series, player_value, kind='rank') if len(series) > 0 else 0
                    similar_values.append(perc)

                radar_poly_ref, _, vertices_ref = radar.draw_radar(
                    ref_values, ax=axs['radar'],
                    kwargs_radar={'facecolor': primary_color, 'alpha': 0.5, 'lw': 3, 'edgecolor': edge_color},
                    kwargs_rings={'facecolor': '#ffffff'}
                )
                
                radar_poly_sim, _, vertices_sim = radar.draw_radar(
                    similar_values, ax=axs['radar'],
                    kwargs_radar={'facecolor': primary_color_2, 'alpha': 0.5, 'lw': 3, 'edgecolor': edge_color_2},
                    kwargs_rings={'facecolor': '#ffffff'}
                )

                axs['radar'].scatter(vertices_ref[:, 0], vertices_ref[:, 1],
                                   c='#264653', edgecolors='black', s=40, zorder=3)
                axs['radar'].scatter(vertices_sim[:, 0], vertices_sim[:, 1],
                                   c='#cc5500', edgecolors='black', s=40, zorder=3)

                radar.draw_range_labels(ax=axs['radar'], fontsize=7, fontproperties=roboto_bold.prop)
                radar.draw_param_labels(ax=axs['radar'], fontsize=7, fontproperties=roboto_bold.prop)

                ref_team = ref_player_data['Team']
                ref_position = ref_player_data['Position']
                ref_minutes = int(ref_player_data["Minutes played"]) if pd.notna(ref_player_data["Minutes played"]) else 0
                
                sim_team = similar_player_data['Team']
                sim_position = similar_player_data['Position']
                sim_minutes = int(similar_player_data["Minutes played"]) if pd.notna(similar_player_data["Minutes played"]) else 0

                title_text = f"{ref_player_name} vs {similar_player_name}"
                axs['title'].text(0.5, 0.75, title_text, fontsize=22, fontproperties=roboto_bold.prop,
                                ha='center', va='center', color='#000000')

                subtitle_text1 = f"{ref_player_name}: {ref_team} | {ref_position} | {ref_minutes} min"
                subtitle_text2 = f"{similar_player_name}: {sim_team} | {sim_position} | {sim_minutes} min | Distância: {most_similar['Distance']:.1f}"
                
                axs['title'].text(0.5, 0.45, subtitle_text1, fontsize=12, fontproperties=roboto_bold.prop,
                                ha='center', va='center', color='#264653')
                axs['title'].text(0.5, 0.25, subtitle_text2, fontsize=12, fontproperties=roboto_bold.prop,
                                ha='center', va='center', color='#cc5500')

                axs['endnote'].text(0.99, 0.5, "made by Joao Alberto Kolling – jakolling@gmail.com",
                                  fontsize=10, ha='right', va='center', color="#999999",
                                  fontproperties=roboto_bold.prop)

                plt.show()
            else:
                print("Não foi possível encontrar os dados do jogador similar para visualização.")

    export = input("\nDeseja exportar os resultados para CSV? (s/n): ").lower()
    if export == 's':
        output_file = input("Digite o nome do arquivo de saída (sem extensão): ").strip()
        if not output_file:
            output_file = "jogadores_similares"
        output_file += ".csv"
        
        results_df.to_csv(output_file, index=False)
        print(f"Resultados exportados para {output_file}")

def main():
    print("Escolha uma opção:")
    print("1. Radar individual")
    print("2. Comparar dois jogadores")
    print("3. Buscar jogadores por métricas")
    print("4. Buscar jogadores similares")
    choice = int(input("Digite sua escolha (1, 2, 3 ou 4): "))
    
    if choice == 1:
        create_single_radar()
    elif choice == 2:
        compare_players()
    elif choice == 3:
        search_players()
    elif choice == 4:
        find_similar_players()
    else:
        print("Escolha inválida")

if __name__ == "__main__":
    main()