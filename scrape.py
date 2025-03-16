import pandas as pd

def getDetails():
    df=pd.read_csv("./assets/tshirts.csv")
    df=df.drop(["Position","Free Delivery"],axis=1)
    sample_records_full = df.sample(n=10).to_dict(orient='records')
    return sample_records_full