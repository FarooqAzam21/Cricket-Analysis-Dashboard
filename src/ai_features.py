import streamlit as st
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
    
    # Train KNN with parallel processing for faster computation
    knn = NearestNeighbors(n_neighbors=top_n+1, metric='cosine', n_jobs=-1)
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
    """Get a response from Ollama with data context. Supports multiple model fallbacks."""
    try:
        # Try to use llama2 first (most common), then fallback to other models
        model_options = ["llama2", "mistral", "neural-chat"]
        llm = None
        
        for model in model_options:
            try:
                llm = Ollama(model=model, base_url="http://localhost:11434")
                # Quick connectivity check
                response = llm.invoke("ping")
                if response:
                    break
            except:
                continue
        
        if llm is None:
            # If Ollama is not available, use a simple rule-based response
            return generate_cricket_response(prompt, context_data)
        
        full_prompt = f"""You are a Cricket Expert Assistant. Use the following data context to answer the user's question concisely (1-2 sentences).

Data Context Summary:
{context_data}

User Question: {prompt}

Answer:"""
        
        response = llm.invoke(full_prompt)
        return response if response else generate_cricket_response(prompt, context_data)
    except Exception as e:
        # Fallback to rule-based response if Ollama fails
        import traceback
        print(f"Ollama Error: {traceback.format_exc()}")
        return generate_cricket_response(prompt, context_data)


def generate_cricket_response(prompt, context_data=""):
    """Generate a response without LLM using rule-based logic."""
    prompt_lower = prompt.lower()
    
    # Simple rule-based responses
    if any(word in prompt_lower for word in ["best", "top", "highest", "most"]):
        if "batsman" in prompt_lower or "batter" in prompt_lower:
            return "Based on the data, the batsmen with the highest runs are shown in the context. These players demonstrate consistent performance across multiple matches."
        elif "bowler" in prompt_lower or "bowling" in prompt_lower:
            return "The top bowlers are those with significant wickets and good economy rates. They've proven their ability to take wickets consistently."
        else:
            return "The top performers in this dataset have demonstrated exceptional consistency and skill across multiple matches."
    
    elif "average" in prompt_lower or "performance" in prompt_lower:
        return "The average metric indicates consistent performance. Higher averages suggest better ability to score runs or maintain economy rates."
    
    elif "prediction" in prompt_lower or "next" in prompt_lower:
        return "Predictions depend on recent form, player consistency, and opponent strength. Our ML models analyze historical data to forecast next match performance."
    
    elif "compare" in prompt_lower or "difference" in prompt_lower:
        return "When comparing players, consider runs scored, average, strike rate, and consistency across different formats and opposition."
    
    else:
        return "I can help you analyze cricket player statistics. Try asking about top players, format comparisons, or performance predictions based on the data."        