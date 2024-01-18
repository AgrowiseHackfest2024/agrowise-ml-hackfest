

import numpy as np

def get_collab_recommendations(user_id, items, collab_df, model, user_id_encoded, farmer_id_encoded, k=5):
  # ambil data user yang sudah memberikan rating
  farmer_reviewed_by_user = collab_df[collab_df['user_id'] == user_id]

  farmer_not_reviewed = items[~items['farmer_id'].isin(  # ambil data user yang belum memberikan rating
    farmer_reviewed_by_user['farmer_id'])]

  farmer_not_reviewed = list(
    set(farmer_not_reviewed['farmer_id'].tolist()).intersection(set(farmer_id_encoded.keys())))  # ambil data user yang belum memberikan rating

  farmer_not_reviewed = [[user_id_encoded.get(user_id), farmer_id_encoded.get(
    farmer_id)] for farmer_id in farmer_not_reviewed]  # encode user dan farmer

  user_encoder = user_id_encoded.get(user_id)  # encode user

  user_farmer_array = np.hstack(
    ([[user_encoder]] * len(farmer_not_reviewed), farmer_not_reviewed))  # gabungkan user dan farmer

  ratings = model.predict(user_farmer_array).flatten()  # prediksi rating

  top_ratings_indices = ratings.argsort()[-k:][::-1]  # ambil 5 teratas

  farmer_encoded = {i: x for i, x in enumerate(farmer_id_encoded)}

  recommended_farmer_ids = [farmer_encoded.get(
      farmer_not_reviewed[x][0]) for x in top_ratings_indices]  # decode farmer
  
  recommended_farmer = items[items['farmer_id'].isin(recommended_farmer_ids)]

  return recommended_farmer


 