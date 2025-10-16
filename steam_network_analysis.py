# steam_game_analysis.py
import requests
import pandas as pd
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Configuration
STEAM_API_KEY = "7FD27F7F6FA7050EFC6344E0C607D777"
SEED_USERS = [
    '76561199219621394',
    '76561199233070339',
    '76561199204339365',
    '76561198247385388',
    '76561199471160990', 
    '76561198002458530',
    '76561198062983156',
    '76561197986155474',
    '76561198061533143',
    '76561197960786090',
    '76561198088439474',
    '76561198054716294',
    '76561198024130010',
    '76561197995935758',
    '76561198864937912',
    '76561198000423758',
    '76561197964242807',
    '76561198001483290',
    '76561198037353817',
    '76561198045626176',
    '76561198058404299',
    '76561198034900008',
    '76561197977289618',
    '76561198016996730',
    '76561198036684172',
    '76561198038984439',
    '76561198074193248',
    '76561198044630165',
    '76561198020043521',
    '76561198015741357'
]
MIN_PLAYTIME_MINUTES = 60

def get_owned_games_with_playtime(steam_id):
    """Get games owned by a Steam user with playtime data"""
    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"
    params = {
        'key': STEAM_API_KEY,
        'steamid': steam_id,
        'include_appinfo': 1,
        'include_played_free_games': 1,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'response' in data and 'games' in data['response']:
                return data['response']['games']
    except Exception as e:
        print(f"Error getting games for {steam_id}: {e}")
    return []

def collect_steam_data_with_playtime():
    """Collect game ownership data from Steam API, filtered by playtime"""
    user_games = {}
    game_names = {}
    playtime_data = {}
    
    print("Collecting Steam data with playtime filter...")
    for i, steam_id in enumerate(SEED_USERS):
        print(f"Processing user {i+1}/{len(SEED_USERS)}")
        
        games = get_owned_games_with_playtime(steam_id)
        if games:
            user_games[steam_id] = []
            playtime_data[steam_id] = {}
            
            for game in games:
                appid = str(game['appid'])
                playtime = game.get('playtime_forever', 0)
                
                if playtime >= MIN_PLAYTIME_MINUTES:
                    user_games[steam_id].append(appid)
                    playtime_data[steam_id][appid] = playtime
                    
                    if appid not in game_names:
                        game_names[appid] = game.get('name', f'Unknown_{appid}')
        
        import time
        time.sleep(0.5)
    
    print(f"Playtime filter: Only including games with ≥ {MIN_PLAYTIME_MINUTES} minutes played")
    return user_games, game_names, playtime_data

def build_game_network(user_games, game_names, playtime_data, top_n_games=75):
    """Build network focusing on top games by player count"""
    g = nx.Graph()
    
    # Calculate game popularity
    game_popularity = defaultdict(int)
    for user_id, games in user_games.items():
        for game_id in games:
            if playtime_data[user_id].get(game_id, 0) >= MIN_PLAYTIME_MINUTES:
                game_popularity[game_id] += 1
    
    # Get top N most popular games
    top_games = sorted(game_popularity.items(), key=lambda x: x[1], reverse=True)[:top_n_games]
    top_game_ids = [game_id for game_id, count in top_games]
    
    print(f"Analyzing top {len(top_game_ids)} games by player count")
    print(f"Player count range: {min([c for _, c in top_games])} - {max([c for _, c in top_games])}")
    
    # Show what games are included at the cutoff
    if len(top_games) == top_n_games:
        cutoff_players = top_games[-1][1]  # Player count of the least popular included game
        print(f"Cutoff: Games with at least {cutoff_players} players")
    
    # Add game nodes
    for game_id, player_count in top_games:
        name = game_names[game_id]
        g.add_node(game_id, 
                  name=name,
                  players=player_count)
    
    # Create edges based on shared players with playtime weighting
    game_players = defaultdict(set)
    for user_id, games in user_games.items():
        for game_id in games:
            if game_id in top_game_ids and playtime_data[user_id].get(game_id, 0) >= MIN_PLAYTIME_MINUTES:
                game_players[game_id].add(user_id)
    
    # Add edges between games
    for i, game1 in enumerate(top_game_ids):
        players1 = game_players[game1]
        
        for game2 in top_game_ids[i+1:]:
            players2 = game_players[game2]
            shared_players = players1.intersection(players2)
            
            if len(shared_players) > 0:
                total_playtime = 0
                for user_id in shared_players:
                    playtime1 = playtime_data[user_id].get(game1, 0)
                    playtime2 = playtime_data[user_id].get(game2, 0)
                    total_playtime += min(playtime1, playtime2)
                
                avg_playtime = total_playtime / len(shared_players) if shared_players else 0
                weight = len(shared_players) * (1 + avg_playtime / 500)
                
                g.add_edge(game1, game2, 
                          weight=weight,
                          shared_players=len(shared_players),
                          avg_playtime=avg_playtime)
    
    print(f"Game network built: {len(g.nodes())} games, {len(g.edges())} connections")
    
    # Calculate expected vs actual connections
    max_possible_edges = len(top_game_ids) * (len(top_game_ids) - 1) // 2
    actual_edges = len(g.edges())
    connection_density = actual_edges / max_possible_edges
    print(f"Network density: {connection_density:.3f} ({actual_edges}/{max_possible_edges} possible edges)")
    
    return g

def visualize_game_network(g, figsize=(16, 12)):
    """Visualize game network with player-based sizing"""
    plt.figure(figsize=figsize)
    
    # Use spring layout
    pos = nx.spring_layout(g, k=3, iterations=100, seed=42)
    
    # Node sizes based on player count
    player_counts = [g.nodes[game]['players'] for game in g.nodes()]
    min_size = 100
    max_size = 2000
    
    if max(player_counts) > min(player_counts):
        node_sizes = [
            min_size + (max_size - min_size) * 
            (g.nodes[game]['players'] - min(player_counts)) / 
            (max(player_counts) - min(player_counts))
            for game in g.nodes()
        ]
    else:
        node_sizes = [500] * len(g.nodes())
    
    # Use a single color for all nodes
    node_colors = 'lightblue'
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos,
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1)
    
    # Draw edges
    edge_weights = [g[u][v]['weight'] for u, v in g.edges()]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        
        if max_weight > min_weight:
            edge_widths = [0.5 + 3 * (weight - min_weight) / (max_weight - min_weight) 
                          for weight in edge_weights]
        else:
            edge_widths = [1.0] * len(edge_weights)
        
        for (u, v), width in zip(g.edges(), edge_widths):
            nx.draw_networkx_edges(g, pos, edgelist=[(u, v)],
                                  alpha=0.4,
                                  edge_color='gray',
                                  width=width)
    
    # Add labels for significant nodes (top 20 by player count)
    player_counts = [g.nodes[game]['players'] for game in g.nodes()]
    top_games = sorted([(game, g.nodes[game]['players']) for game in g.nodes()], 
                      key=lambda x: x[1], reverse=True)[:20]
    
    labels = {}
    for game, _ in top_games:
        name = g.nodes[game]['name']
        if len(name) > 20:
            labels[game] = f"{name[:20]}..."
        else:
            labels[game] = name
    
    nx.draw_networkx_labels(g, pos, labels, font_size=8, font_weight='bold')
    
    plt.title(f"Steam Game Network - {len(g.nodes())} Games\nNode size = Player count, Edge width = Connection strength", 
              fontsize=16, pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print top games by player count
    print(f"\nTop 10 games by player count:")
    for game, players in top_games[:10]:
        print(f"  {g.nodes[game]['name']}: {players} players")

def visualize_betweenness_centrality(g):
    """Create a single bar chart for Betweenness Centrality - the most important measure"""
    print("\n" + "="*70)
    print("BETWEENNESS CENTRALITY ANALYSIS")
    print("="*70)
    print("Betweenness identifies 'bridge' games that connect different communities")
    print("These are strategically important for marketing and recommendations\n")
    
    # Compute betweenness centrality
    print("Calculating betweenness centrality...")
    betweenness = nx.betweenness_centrality(g, weight='weight', normalized=True)
    
    # Get top 15 games by betweenness
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:15]
    
    # Prepare data for visualization
    games = [g.nodes[game_id]['name'] for game_id, _ in top_betweenness]
    scores = [score for _, score in top_betweenness]
    
    # Create the single bar chart
    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar chart
    bars = plt.barh(range(len(games)), scores, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Customize the chart
    plt.yticks(range(len(games)), games, fontsize=11)
    plt.xlabel('Betweenness Centrality Score', fontsize=12, fontweight='bold')
    plt.title('Top 15 Games by Betweenness Centrality\n"Bridge Games" That Connect Different Gaming Communities', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add explanatory text
    plt.figtext(0.02, 0.02, 
                "• High betweenness = Games that connect different player communities\n" +
                "• Strategic importance: Target these for cross-promotion and recommendations",
                fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nTOP BRIDGE GAMES ANALYSIS:")
    print("-" * 50)
    for i, (game_id, score) in enumerate(top_betweenness[:10], 1):
        game_name = g.nodes[game_id]['name']
        player_count = g.nodes[game_id]['players']
        print(f"{i:2d}. {game_name}")
        print(f"    Betweenness: {score:.4f} | Players: {player_count}")
        
        # Show what this game connects to
        neighbors = list(g.neighbors(game_id))
        if neighbors:
            top_neighbors = sorted([(n, g[game_id][n]['weight']) for n in neighbors], 
                                 key=lambda x: x[1], reverse=True)[:3]
            neighbor_names = [g.nodes[n]['name'] for n, _ in top_neighbors]
            print(f"    Connects to: {', '.join(neighbor_names)}")
        print()

def analyze_network_properties(g):
    """Analyze key network properties"""
    print(f"\n" + "="*60)
    print("NETWORK PROPERTIES ANALYSIS")
    print("="*60)
    
    # Basic network metrics
    print(f"\nNetwork Metrics:")
    print(f"  Number of games: {len(g.nodes())}")
    print(f"  Number of connections: {len(g.edges())}")
    print(f"  Network density: {nx.density(g):.4f}")
    print(f"  Average clustering: {nx.average_clustering(g, weight='weight'):.4f}")
    
    # Most connected games
    weighted_degrees = [(g.nodes[game]['name'], g.degree(game, weight='weight')) 
                       for game in g.nodes()]
    print(f"\nTop 5 most connected games:")
    for name, degree in sorted(weighted_degrees, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {name}: {degree:.1f}")
    
    # Strongest connections
    strong_connections = []
    for u, v, data in g.edges(data=True):
        game1 = g.nodes[u]['name']
        game2 = g.nodes[v]['name']
        strong_connections.append((game1, game2, data['weight']))
    
    strong_connections.sort(key=lambda x: x[2], reverse=True)
    print(f"\nTop 5 strongest game connections:")
    for game1, game2, weight in strong_connections[:5]:
        print(f"  {game1} ↔ {game2}: {weight:.1f}")

def main():
    print("=== Steam Game Network Analysis ===")
    print(f"Using minimum playtime filter: {MIN_PLAYTIME_MINUTES} minutes")
    
    # Collect data
    user_games, game_names, playtime_data = collect_steam_data_with_playtime()
    
    if not user_games:
        print("No data collected.")
        return
    
    print(f"Data collected: {len(user_games)} users, {len(game_names)} total games")
    
    # Build game network
    print("\nBuilding game network...")
    game_network = build_game_network(user_games, game_names, playtime_data)
    
    # Visualize main network
    print("\nCreating game network visualization...")
    visualize_game_network(game_network)
    
    # Analyze network properties
    analyze_network_properties(game_network)
    
    # Visualize single centrality measure - Betweenness
    visualize_betweenness_centrality(game_network)

if __name__ == "__main__":
    main()