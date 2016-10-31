# -*- coding: utf-8 -*-

"""Main entry point of our project"""

# Useful starting lines
from cross_validation import *
from proj1_helpers import *

if __name__ == '__main__':
    # Load the input data
    DATA_TRAIN_PATH = '../train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    # Format the input data
    mean_x = np.mean(tX, axis=0)
    std_x = np.std(tX, axis=0)
    tx_cleaned = remove_outliers(tX, mean_x, std_x)
    tx_scaled = rescale(tx_cleaned)
    y0 = format_y(y)

    # Compute the weights
    loss, weights = least_squares(y, tX)

    print(loss, weights)

    # Load the test data
    DATA_TEST_PATH = '../test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    # Compute and create the submission file
    OUTPUT_PATH = '../submission.csv'
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
