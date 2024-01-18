import json
import numpy as np
import tensorflow as tf
from flask import Flask, request
import pickle
from cb_recommendation import get_cb_recommendations
from collab_recommendation import get_collab_recommendations

# ================INIT FLASK========================
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
# ===================================================

# ================LOAD MODEL=========================
folder = './model/'

cb_df = pickle.load(open(folder + 'cb_df.pkl', 'rb'))
cosine_sim_df = pickle.load(open(folder + 'cosine_sim_df.pkl', 'rb'))

collab_df = pickle.load(open(folder + 'collab_df.pkl', 'rb'))
user_id_encoded = pickle.load(open(folder + 'user_id_encoded.pkl', 'rb'))
farmer_id_encoded = pickle.load(open(folder + 'farmer_id_encoded.pkl', 'rb'))
model = tf.keras.models.load_model(folder + '/model_collab')
print(model.summary())
# ===================================================

# ================ENDPOINT===========================
@app.route("/recommendation/cb", methods=["POST"])
def cb_recommendation():
    data = request.json
    farmer_id = data['farmer_id']
    k = data.get('k', None)  # Use get method to get the value of 'k', default to None if not present

    if farmer_id not in cosine_sim_df.index:
        return json.dumps({'success': False, 'message': 'Farmer not found!', 'data': []})

    if k is None or not (1 <= k <= len(cosine_sim_df.columns)):
        k = len(cosine_sim_df.columns)

    result = get_cb_recommendations(farmer_id, cosine_sim_df, cb_df, k)

    result_json = result.to_json(orient="records")

    return json.dumps({'success': True, 'message': 'Success retrieve content based filtering data!', 'data': json.loads(result_json)})

@app.route("/recommendation/collab", methods=["POST"])
def collab_recommendation():
    data = request.json
    user_id = data['user_id']
    k = data.get('k', None)  # Use get method to get the value of 'k', default to None if not present

    if user_id not in user_id_encoded.keys():
        return json.dumps({'success': False, 'message': 'User not found!', 'data': []})

    if k is None or not (1 <= k <= len(farmer_id_encoded)):
        k = len(farmer_id_encoded)

    result = get_collab_recommendations(user_id, cb_df, collab_df, model, user_id_encoded, farmer_id_encoded, k)

    result_json = result.to_json(orient="records")

    return json.dumps({'success': True, 'message': 'Success retrieve collaborative filtering data!', 'data': json.loads(result_json)})
# ===================================================

# ================MAIN===============================
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000)
# ===================================================
