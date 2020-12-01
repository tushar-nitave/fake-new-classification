import pandas as pd
import matplotlib.pyplot as plt
from preprocess import pre_process


class EDA:
    def __init__(self, data):
        self.data = data

    def basic_info(self):

        print("\nData Size: {}".format(self.data.shape))
        print("\nData Description \n\n {}".format(self.data.head()))
        print("\nData Types \n\n {}".format(self.data.dtypes))
        print("\nColumns {}".format(self.data.columns))

        # number of duplicate rows
        print("\nDuplicate Row: {}".format(self.data[self.data.duplicated()]))

        # number of missing values
        print("\nMissing Values: {}".format(self.data.isnull().sum()))

        # histogram plot
        self.data.label.value_counts().nlargest(40).plot(kind="bar", figsize=(10, 5))
        plt.title("Distribution of Labels")
        plt.ylabel("Frequency (Label Count)")
        plt.xlabel("Labels")
        plt.show()


if __name__ == "__main__":

    data = pd.read_csv("../train.csv", sep=",")
    obj = EDA(data)
    obj.basic_info()
    # data = pre_process(data)
    # data.to_csv("processed_train.csv", index=False)
