from dataclasses import dataclass
import pandas as pd


@dataclass
class ReadReadyModels():

    def read(self):
        try:
            data = pd.read_excel("src/ready_data/gender_avg.xlsx")
            return data
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return None
