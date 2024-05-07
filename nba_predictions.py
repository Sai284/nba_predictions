import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

def fetch_player_stats(player_name, seasons):
    nba_players = players.get_players()
    player = next((player for player in nba_players if player['full_name'] == player_name), None)
    all_stats = pd.DataFrame()

    if player:
        for season in seasons:
            gamelog = playergamelog.PlayerGameLog(player_id=player['id'], season=season)
            season_stats = gamelog.get_data_frames()[0]
            if not season_stats.empty:
                season_stats['SEASON'] = season  # Add a column for the season
                all_stats = pd.concat([all_stats, season_stats], ignore_index=True)
        print("Total games fetched:", len(all_stats))  # Print total games fetched to debug
        return all_stats
    else:
        return None

def prepare_features(stats):
    stats['GAME_DATE'] = pd.to_datetime(stats['GAME_DATE'], format='%b %d, %Y')
    stats['AVG_LAST_5_GAMES_PTS'] = stats['PTS'].rolling(window=5).mean().shift(1)
    stats['INTERACTION_MIN_FG_PCT'] = stats['MIN'] * stats['FG_PCT']
    required_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'FG3A', 'FGA', 'FTA', 'TOV', 'AVG_LAST_5_GAMES_PTS', 'INTERACTION_MIN_FG_PCT']
    features = stats[required_columns].copy()
    features.fillna(method='ffill', inplace=True)  # Forward fill to propagate last valid observation forward
    features['PTS_NEXT_GAME'] = stats['PTS'].shift(-1)
    features.dropna(inplace=True)  # Final drop of any remaining NaN values
    print("Games included after feature preparation:", len(features))
    return features

def train_model(features):
    X = features.drop('PTS_NEXT_GAME', axis=1)
    y = features['PTS_NEXT_GAME']
    # Smaller test size if you wish to retain a split but visualize more comprehensively
    test_size = 0.1  # Smaller test portion
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    # Predict on both train and test data
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    combined_y = pd.concat([y_train, y_test])
    combined_predictions = np.concatenate([train_predictions, test_predictions])
    
    return combined_y, combined_predictions

def plot_actual_vs_predicted(y_test, predictions, player_name, seasons, error_threshold=5):
    plt.figure(figsize=(12, 8))
    max_points = max(max(y_test), max(predictions)) + 10
    errors = abs(y_test - predictions)
    colors = ['green' if error <= error_threshold else 'red' for error in errors]
    for i, error in enumerate(errors):
        plt.scatter(y_test.iloc[i], predictions[i], color=colors[i], s=100, edgecolors='black', alpha=0.75)

    plt.plot([0, max_points], [0, max_points], 'r--', label='Perfect Prediction Line')
    plt.fill_between([0, max_points], [0, max_points - error_threshold], [0, max_points + error_threshold], color='gray', alpha=0.2, label='Good Prediction Zone')

    plt.xlabel('Actual Points', fontsize=14)
    plt.ylabel('Predicted Points', fontsize=14)
    season_str = ', '.join(seasons)
    plt.title(f'Actual vs Predicted Points per Game for {player_name} ({season_str})', fontsize=16)

    plt.xlim(0, max_points)
    plt.ylim(0, max_points)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    handles = [
        mpatches.Patch(color='green', label='Prediction within 5 Points'),
        mpatches.Patch(color='red', label='Prediction outside 5 Points'),
        mpatches.Patch(color='gray', alpha=0.2, label='Good Prediction Zone'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Perfect Prediction Line')
    ]
    plt.legend(handles=handles, loc='upper left', fontsize=12)
    plt.show()

def main():
    player_name = input("Enter the player's full name (e.g., LeBron James): ")
    seasons = input("Enter comma-separated seasons (e.g., 2019,2020,2021): ").split(',')
    player_stats = fetch_player_stats(player_name, seasons)
    if player_stats is not None and not player_stats.empty:
        features = prepare_features(player_stats)
        y_test, predictions = train_model(features)
        plot_actual_vs_predicted(y_test, predictions, player_name, seasons)
    else:
        print("No data found for the player across specified seasons.")

if __name__ == "__main__":
    main()
