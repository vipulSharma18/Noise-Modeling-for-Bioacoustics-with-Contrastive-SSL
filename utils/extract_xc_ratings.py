import pandas as pd
import json

# Read the input CSV file
input_file = 'configs/preprocessing_reproducibility_backup/cbc2020/xeno_canto_top100.csv'
output_file = 'configs/preprocessing_reproducibility_backup/cbc2020/xeno_canto_ratings.csv'

# Read the CSV file
df = pd.read_csv(input_file)

# Extract ratings from JSON response
ratings = []
for response in df['response']:
    try:
        json_data = json.loads(response)
        rating = json_data['recordings'][0]['q']
        ratings.append(rating)
    except (json.JSONDecodeError, KeyError, IndexError):
        ratings.append(None)

# Create new dataframe with extracted data
result_df = pd.DataFrame({
    'xc_id': df['ids'],
    'rating': ratings
})

# Save to new CSV file
result_df.to_csv(output_file, index=False)

print(f"Processed {len(df)} records")
print("Sample row:")
print(result_df.iloc[0].to_dict()) 