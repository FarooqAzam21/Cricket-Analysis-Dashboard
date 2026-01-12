from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
from langchain_community.llms import Ollama

def find_similar_players(df, player_name, top_n=5):
    """Find players with similar statistical profiles and roles using KNN."""
    metrics = ['matches', 'Innings', 'runs', 'wickets', 'average', 'strike_rate', 'bowling_average', 'economy']
    
    # Filter only relevant metrics available in df
    available_metrics = [m for m in metrics if m in df.columns]
    
    # Prepare data for KNN
    features = df[available_metrics].copy()
    for col in available_metrics:
        features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
    
    # Role-based similarity feature
    # We find the target player's role and create a binary feature for "matching role"
    try:
        target_role = df[df['player'] == player_name]['role'].iloc[0]
        features['same_role_bonus'] = (df['role'] == target_role).astype(int) * 2.0 # Weighting role significantly
    except:
        pass

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Train KNN
    knn = NearestNeighbors(n_neighbors=top_n+1, metric='cosine')
    knn.fit(scaled_features)
    
    # Find index of the player
    try:
        player_idx = df[df['player'] == player_name].index[0]
        distances, indices = knn.kneighbors(scaled_features[player_idx].reshape(1, -1))
        
        # Get results (excluding the player themselves)
        similar_indices = indices.flatten()[1:]
        return df.iloc[similar_indices], distances.flatten()[1:]
    except IndexError:
        return pd.DataFrame(), []

def get_ollama_response(prompt, context_data=""):
    """Get a response from Ollama with data context."""
    try:
        llm = Ollama(model="llama2") # Or "llama3" if available
        
        full_prompt = f"""
        You are a Cricket Expert Assistant. Use the following data context to answer the user's question.
        
        Data Context Summary:
        {context_data}
        
        User Question: {prompt}
        
        Assistant:"""
        
        return llm.invoke(full_prompt)
    except Exception as e:
        return f"Ollama Error: Make sure Ollama is running locally. Error: {e}"
