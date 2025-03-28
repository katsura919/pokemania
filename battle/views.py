import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# Load Pokémon dataset
pokemon_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "pokemon.csv"))
pokemon_dict = pokemon_df.set_index("#").to_dict(orient="index")

# Load trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pokemon_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

# Pokémon Type Effectiveness Matrix
type_chart = {
    "Normal": {"Rock": 0.5, "Ghost": 0, "Steel": 0.5},
    "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": 0.5, "Dragon": 0.5, "Steel": 2},
    "Water": {"Fire": 2, "Water": 0.5, "Grass": 0.5, "Ground": 2, "Rock": 2, "Dragon": 0.5},
    "Electric": {"Water": 2, "Electric": 0.5, "Grass": 0.5, "Ground": 0, "Flying": 2, "Dragon": 0.5},
    "Grass": {"Fire": 0.5, "Water": 2, "Grass": 0.5, "Poison": 0.5, "Ground": 2, "Flying": 0.5, "Bug": 0.5, "Rock": 2, "Dragon": 0.5, "Steel": 0.5},
    "Ice": {"Fire": 0.5, "Water": 0.5, "Ice": 0.5, "Ground": 2, "Flying": 2, "Dragon": 2, "Steel": 0.5},
    "Fighting": {"Normal": 2, "Ice": 2, "Rock": 2, "Dark": 2, "Steel": 2, "Poison": 0.5, "Flying": 0.5, "Psychic": 0.5, "Bug": 0.5, "Ghost": 0, "Fairy": 0.5},
}

def get_type_effectiveness(attacker_type1, attacker_type2, defender_type1, defender_type2):
    """Calculate type effectiveness based on Pokémon types"""
    effectiveness = 1.0

    for atk_type in [attacker_type1, attacker_type2]:
        if atk_type is None:
            continue
        for def_type in [defender_type1, defender_type2]:
            if def_type is None:
                continue
            effectiveness *= type_chart.get(atk_type, {}).get(def_type, 1.0)

    return effectiveness

# Function to extract features
def get_features(first_pokemon, second_pokemon):
    p1 = pokemon_dict.get(first_pokemon)
    p2 = pokemon_dict.get(second_pokemon)

    if not p1 or not p2:
        return None

    # Compute Type Effectiveness
    type_effectiveness = get_type_effectiveness(p1["Type 1"], p1.get("Type 2"), p2["Type 1"], p2.get("Type 2"))

    # Features: HP, Attack, Defense, Sp. Atk, Sp. Def, Speed, Legendary, Type Effectiveness
    features = [
        p1["HP"], p1["Attack"], p1["Defense"], p1["Sp. Atk"], p1["Sp. Def"], p1["Speed"], int(p1["Legendary"]),
        p2["HP"], p2["Attack"], p2["Defense"], p2["Sp. Atk"], p2["Sp. Def"], p2["Speed"], int(p2["Legendary"]),
        type_effectiveness  # This makes it 15 features
    ]

    return np.array(features).reshape(1, -1)

@csrf_exempt
def predict_winner(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            first_pokemon = int(data.get("first_pokemon"))
            second_pokemon = int(data.get("second_pokemon"))

            features = get_features(first_pokemon, second_pokemon)

            if features is None:
                return JsonResponse({"error": "Invalid Pokémon ID"}, status=400)

            # Predict winner
            prediction = model.predict(features)
            winner_id = first_pokemon if prediction[0][0] > 0.5 else second_pokemon

            # Get winner's name
            winner_name = pokemon_dict[winner_id]["Name"]

            return JsonResponse({"winner_id": winner_id, "winner_name": winner_name})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request"}, status=400)
