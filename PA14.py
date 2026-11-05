import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

pd.set_option("display.max_rows", 25)

homes = pd.read_csv("homes.csv")

priceFloor = homes[["Price", "Floor"]]
school = homes[["School"]]

# Define a standardization scaler to transform values
scaler = StandardScaler()

# Apply scaler to the priceFloor data
scaled = scaler.fit_transform(priceFloor)

homes_standardized = pd.DataFrame(scaled, columns=["Price", "Floor"])
print("Standardized data: \n", homes_standardized)

# Define a normalization scaler to transform values
scaler = MinMaxScaler()

# Apply scaler to the priceFloor data
normalized = scaler.fit_transform(priceFloor)

homes_normalized = pd.DataFrame(normalized, columns=["Price", "Floor"])
print("Normalized data: \n", homes_normalized)

# Define the OrdinalEncoder() function
ordinal_encoder = OrdinalEncoder()

# Create a dataframe of the ordinal encoder function fit to the school data,
# with the column labeled encoding
school_encoded_vals = pd.DataFrame(
    ordinal_encoder.fit_transform(school),
    columns=["encoding"]
)

# Join the new column to the school data
school_encoded = school.join(school_encoded_vals)

print("Encoded data: \n", school_encoded)

# Create a discretizer with equal widths and 3 bins
discretizer_eqwidth = KBinsDiscretizer(
    n_bins=3,
    encode="ordinal",
    strategy="uniform"
)

# Fit the discretizer to the Floor feature from the priceFloor data.
# Reshape the feature to an array with dimensions (50,1).
discretizer_eqwidth.fit(np.reshape(priceFloor["Floor"], (50, 1)))

print("Bin widths: \n", discretizer_eqwidth.bin_edges_[0])
