import pandas as pd
from src.utils.metrics import row_wise_f1_score
from src.model.knn import knn_model


def main():

    df = pd.DataFrame()
    knn_model()
    row_wise_f1_score()

    return df


if __name__ == '__main__':
    main()
    print('zy updated')
