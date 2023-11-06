import numpy as np
import argparse
from scipy.linalg import pinv, svd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Homework 2",
                                     epilog="CSCI 4360/6360 Data Science II: Fall 2023",
                                     add_help="How to use",
                                     prog="python homework2.py <arguments>")
    parser.add_argument("-f", "--infile", required=True,
                        help="Dynamic texture file, a NumPy array.")
    parser.add_argument("-q", "--dimensions", required=True, type=int,
                        help="Number of state-space dimensions to use.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path where the 1-step prediction will be saved as a NumPy array.")

    args = vars(parser.parse_args())

    # args
    input_file = args['infile']
    q = args['dimensions']
    output_file = args['output']

    # np load texture data
    M = np.load(input_file)

    def shape(M, q, output):
        f = M.shape
        h = M.shape
        w = M.shape

        Y = M.reshape((f, h * w)).T
        U = svd(Y, full_matrices=False)
        S = svd(Y, full_matrices=False)
        Vt = svd(Y, full_matrices=False)
        C = U[:, :q]
        X = np.diag(S[:q]).dot(Vt[:q, :])

        # pseudo inverse
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        A = X2.dot(pinv(X1))

        # prediction
        x_next = A.dot(X[:, -1])
        y_next = C.dot(x_next)
        y_next_reshaped = y_next.reshape((h, w))

        # save
        np.save(output, y_next_reshaped)

    shape(M, q, output_file)
