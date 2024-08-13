import pandas as pd
import numpy as np

def read_orders_file(filename):
    df = pd.read_excel(filename, sheet_name='Sheet3')
    df["order_id"] = df["order_id"] - 1
    df["uuid"] = range(0, len(df))
    return df.to_numpy()

def read_coordinates_file(filename):

  # Excel dosyasını oku
  df = pd.read_excel(filename, sheet_name='Sheet1')

  # Gerekli kolonları çek
  item_ids = df['item_id'].values
  x_coords = df['x'].values
  y_coords = df['y'].values

  # Distance matrix oluşturmak için boş bir numpy array yarat
  num_items = len(item_ids)
  distance_matrix = np.zeros((num_items, num_items))

  # Distance matrix hesapla
  for i in range(num_items):
      for j in range(num_items):
          if i != j:
              distance_matrix[i, j] = min(x_coords[i] + x_coords[j], 400 - x_coords[i] - x_coords[j]) + (abs(y_coords[i] - y_coords[j])+1) * 5

  # # Distance matrix'i DataFrame olarak yarat
  # distance_df = pd.DataFrame(distance_matrix, index=item_ids, columns=item_ids)

  #Origin distance matrix:
  # origin distance matrix oluşturmak için boş bir numpy array yarat
  origin_distance_matrix = np.zeros(num_items)

  # Distance matrix hesapla
  for i in range(num_items):
      origin_distance_matrix[i] = (x_coords[i]+5) + (min(abs(5 - y_coords[i]), abs(4-y_coords[i]))*2-1)*5
  # # Distance matrix'i DataFrame olarak yarat
  # origin_distance_df = pd.DataFrame(origin_distance_matrix, index=item_ids, columns=item_ids)

  return distance_matrix, origin_distance_matrix
