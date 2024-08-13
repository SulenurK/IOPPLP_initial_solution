import os
from main import main

coordinates_file = os.path.join("input", "item_coordinates_4000.xlsx")
input_folder = "input"
input_pattern = "instance_"

input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.startswith(input_pattern) and f.endswith("xlsm")]

num_pickers = range(2,6)
num_run = range(1,2)

for inf in input_files:
  print()
  print(f"# ======= {inf} =======")
  for np in num_pickers:
    for nr in num_run:
      print(f"python main.py --coordinates-file {coordinates_file} --orders-file {inf} --num-pickers {np}")
