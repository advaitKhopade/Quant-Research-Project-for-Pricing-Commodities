import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import csv

def load_data(csv_file):
    # Load the data from the CSV file
    data = pd.read_csv(csv_file)
    
    # Display the first few rows of the data
    print(data)
    
    
    csv_file = "Nat_Gas.csv"

    # Initialize an empty list to store the data
    data = []
    date=[]
    Price=[]
# Open the CSV file and read its contents
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
    
        # Iterate over each row in the CSV file and append it to the data list
        for row in csv_reader:
            data.append(row)

        for i in data:
            date.append(i[0])
            Price.append(i[1])
        # Print date and time
        for d, p in zip(date, Price):
            print("Date:", d, "Time:", p)
    

# Display the number of rows in the CSV file
    print("Number of rows:", len(data))

load_data("Nat_Gas.csv")
