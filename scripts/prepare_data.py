import os
import pandas as pd
from PIL import Image
import pandas as pd
from tqdm import tqdm

svs_dir = "svi_computed"
output_dir = "SVs_merged"
os.makedirs(output_dir, exist_ok=True)

df1 = pd.read_csv("one_building_detected.csv")
df2 = pd.read_csv("two_building_detected.csv")
df3 = pd.read_csv("three_building_detected.csv")
df  = pd.read_csv("building_detection_results.csv")
sub1 = df[df["parcel_id"].isin(df1["parcel_id"])]
sub2 = df[df["parcel_id"].isin(df2["parcel_id"])]
sub3 = df[df["parcel_id"].isin(df3["parcel_id"])]
df = pd.concat([sub1, sub2, sub3], axis=0)
all_ids = set(df['parcel_id'].to_list())

# prepare df for inference
n = 1
for each_df in [sub1, sub2, sub3]:
    parcels = list(set(each_df["parcel_id"].to_list()))
    parcel_svi = [f'{p_id}_merged.jpg' for p_id in parcels]
    dic = {'svi_merged': parcel_svi}
    pd.DataFrame(dic).to_csv(f"building_detected_{n}.csv")
    n += 1

# combine three perspectives into a single image
for pid in tqdm(all_ids, ):
    img_files = df.loc[df["parcel_id"] == pid, 'file'].tolist()

    if not all(os.path.exists(f) for f in img_files):
        print(f"missing: {pid}")
        continue

    if os.path.exists(os.path.join(output_dir, f"{pid}_merged.jpg")):
        continue

    imgs = [Image.open(f) for f in img_files]

    widths, heights = zip(*(img.size for img in imgs))
    max_width = max(widths)
    total_height = sum(heights) + 2 * 20

    new_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))

    y_offset = 0
    for i, img in enumerate(imgs):
        x_offset = (max_width - img.width) // 2
        new_img.paste(img, (x_offset, y_offset))
        y_offset += img.height
        if i < 2:
            y_offset += 20
        out_path = os.path.join(output_dir, f"{pid}_merged.jpg")
        new_img.save(out_path)
