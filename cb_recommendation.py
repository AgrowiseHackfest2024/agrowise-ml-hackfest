import pandas as pd

def get_cb_recommendations(farmer_id, similarity_data, items, k=5):
  index = similarity_data.loc[farmer_id].to_numpy().argpartition(
      range(-1, -k, -1))
  
  closest = similarity_data.columns[index[-1:-(k+2):-1]]

  closest = closest.drop(farmer_id, errors='ignore')

  return pd.DataFrame(closest).merge(items).head(k)
