import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

#Load the dataset
df = pd.read_csv('../data/raw/cobs_dataset.csv')              # CSV

print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print(df.dtypes)
print(df.head(3).to_string())

#Type distribution
type_counts = df['type_code'].value_counts()
print('\nType distribution:')
print(type_counts)
print(f'Total docs: {len(df)}')

#Plot type distribution
plt.figure(figsize=(10, 6))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='viridis')
plt.title('Type Distribution')
plt.xlabel('Type Code')
plt.ylabel('Count')
plt.show()
