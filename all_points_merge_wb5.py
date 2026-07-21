import glob
import pickle

all_points = []
# Find all the partial pickle files
for file in glob.glob("wb5_feasible_points_part_*.pkl"):
    with open(file, "rb") as f:
        all_points.extend(pickle.load(f))

# Save the final combined dataset
with open("wb5_feasible_points_FINAL.pkl", "wb") as f:
    pickle.dump(all_points, f)

print(f"Merged {len(all_points)} total feasible points!")