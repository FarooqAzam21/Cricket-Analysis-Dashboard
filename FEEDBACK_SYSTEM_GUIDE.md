# Smart Scout Feedback System

## Overview
The Smart Scout feature now includes a machine learning feedback system that allows users to rate similarity results and improve recommendations over time.

## How It Works

### 1. User Feedback Collection
- After finding similar players, each result displays **👍 (Good)** and **👎 (Bad)** buttons
- Users can rate whether each similarity match is accurate
- Feedback is stored in the database with:
  - Username
  - Source player (who you searched for)
  - Similar player (who was recommended)
  - Format (T20, ODI, Test)
  - Rating (good/bad)
  - Timestamp

### 2. Database Storage
All feedback is saved in the `scout_feedback` table:
```sql
CREATE TABLE scout_feedback (
    id INTEGER PRIMARY KEY,
    username TEXT,
    source_player TEXT,
    similar_player TEXT,
    format TEXT,
    rating TEXT,
    timestamp DATETIME
)
```

### 3. Future Model Training
The collected feedback can be used to:
- **Adjust feature weights**: If users consistently rate certain player pairs as "good", the model can learn which statistical features are most important
- **Penalize bad matches**: Player pairs rated as "bad" can be given lower similarity scores
- **Personalized recommendations**: Different users may have different preferences for what makes players "similar"

## Usage
1. Go to **Smart Scout (AI)** in the sidebar
2. Select a player and format
3. Click **Find Similar Players**
4. Rate each result with 👍 or 👎
5. Your feedback is automatically saved!

## Viewing Feedback Stats
You can query the feedback database to see patterns:
```python
from src.database import get_feedback_stats
stats = get_feedback_stats()
print(stats)
```

This will show which player comparisons get the most positive/negative ratings.

## Next Steps for Model Improvement
To actually retrain the model based on feedback:
1. Collect sufficient feedback data (100+ ratings recommended)
2. Analyze which features correlate with "good" ratings
3. Adjust the KNN algorithm's feature weights
4. Implement a weighted similarity metric based on feedback patterns
