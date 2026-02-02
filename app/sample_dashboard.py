import streamlit as st
import pandas as pd

file1 = "19ZR8Usyitrp1_7lsLpHfxFMrXt2edIvZ"
url1 = f"https://drive.google.com/uc?id={file1}"
file2 = "1xzW4fH4tH9Dl2PDRp4lE9g7a2_1Bzy8L"
url2 = f"https://drive.google.com/uc?id={file2}"

df1 = pd.read_csv(url1)
df2 = pd.read_csv(url2)

df = df1.join(df2, how='inner')

from sklearn.model_selection import train_test_split

# Splits, ensuring 'target' distribution is the same in both
train, sample_df = train_test_split(
    df, test_size=0.1, stratify=df['isFraud'], random_state=42
)

st.title("My Data App")
st.dataframe(sample_df)