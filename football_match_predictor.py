import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import pickle
import io

# Function to load the trained model
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

github_url = "https://raw.githubusercontent.com/celinelooi/silver-pancake/main/trained_model2.pkl"

# Download and load the model file from GitHub
with urlopen(github_url) as response:
    model = pickle.load(response)

# Title and description
st.title('Football Match Predictor')
st.write('Enter the details of the match to predict the outcome.')

# Display a football image
st.image('football.jpg', caption='Football Match Predictor', use_column_width=True)

# Team names from the CSV
teams = ['Arsenal', 'Liverpool', 'Manchester City', 'Aston Villa', 'Tottenham Hotspur', 'Manchester United', 
         'Newcastle United', 'West Ham United', 'Chelsea', 'Bournemouth', 'Brighton and Hove Albion', 
         'Wolverhampton Wanderers', 'Fulham', 'Crystal Palace', 'Brentford', 'Everton', 'Nottingham Forest', 
         'Luton Town', 'Burnley', 'Sheffield United', 'Leicester City', 'Leeds United', 'Southampton']

# Input fields for match details
team = st.selectbox('Team', options=teams)
opponent = st.selectbox('Opponent', options=teams)
if team == opponent:
    st.error("Team and opponent cannot be the same. Please select different teams.")
    st.stop()

venue = st.selectbox('Venue', options=['Home', 'Away'])
# Set the minimum date to a year ago from today
min_date = datetime.now().date() - timedelta(days=365)
date = st.date_input('Date', min_value=min_date)
time = st.time_input('Kick-off Time')

# Convert inputs to model input format
team_code = teams.index(team)  # Encoding team based on their index
venue_code = 1 if venue == 'Home' else 0
opp_code = teams.index(opponent)  # Encoding opponents based on their index
hour = time.hour
day_code = date.weekday()  # Monday is 0 and Sunday is 6

# Randomized default values for hidden features within a realistic range
ema_goals_scored = np.random.uniform(1.0, 3.0)
days_since_last = np.random.randint(3, 14)
rank_difference = np.random.randint(-10, 10)
form_last_5 = np.random.uniform(0, 3)
interaction = np.random.uniform(0, 1)

# Prepare the features as expected by the model
features = pd.DataFrame({
    'team_code': [team_code],
    'venue_code': [venue_code],
    'opp_code': [opp_code],
    'hour': [hour],
    'day_code': [day_code],
    'ema_goals_scored': [ema_goals_scored],
    'days_since_last': [days_since_last],
    'rank_difference': [rank_difference],
    'form_last_5': [form_last_5],
    'interaction': [interaction]
})

# Prediction button
if st.button('Predict Result'):
    # Ensure the features match the expected input format
    expected_features = ['ema_goals_scored', 'days_since_last', 'rank_difference', 'form_last_5', 'interaction']
    if set(expected_features).issubset(features.columns):
        # Predict
        prediction = model.predict(features[expected_features])
        result = 'Win' if prediction[0] == 1 else 'Lose'
        
        # Display the prediction
        st.write(f'The prediction for {team} against {opponent} is: {result}')
    else:
        st.error(f"Feature mismatch: Model expects {expected_features} but received {list(features.columns)}")
