import numpy as np
import tensorflow as tf

def get_collab_recommendations(user_id, items, collab_df, model, user_id_encoded, farmer_id_encoded, k=5):
    farmer_reviewed_by_user = collab_df[collab_df['user_id'] == user_id]
    farmer_not_reviewed = items[~items['farmer_id'].isin(farmer_reviewed_by_user['farmer_id'])]

    farmer_not_reviewed = list(set(farmer_not_reviewed['farmer_id'].tolist()).intersection(set(farmer_id_encoded.keys())))

    farmer_not_reviewed = [[user_id_encoded.get(user_id), farmer_id_encoded.get(farmer_id)] for farmer_id in farmer_not_reviewed]

    user_farmer_array = np.array(farmer_not_reviewed)
    user_farmer_array = tf.convert_to_tensor(user_farmer_array, dtype=tf.int64)  # Convert to tf.int64

    ratings = model.predict(user_farmer_array).flatten()

    top_ratings_indices = ratings.argsort()[-k:][::-1]

    farmer_encoded = {i: x for i, x in enumerate(farmer_id_encoded)}
    recommended_farmer_ids = [farmer_encoded.get(farmer_not_reviewed[x][1]) for x in top_ratings_indices]

    recommended_farmer = items[items['farmer_id'].isin(recommended_farmer_ids)]

    return recommended_farmer
