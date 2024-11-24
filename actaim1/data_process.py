import open3d as o3d
import numpy as np
import pandas as pd

object_id = 100282
root = "./dataset"


def array_in_list(a, L):
    for i in range(len(L)):
        array_equals = a == L[0]
        array_equal = array_equals[0] and array_equals[1] and array_equals[2]

        if array_equal:
            return True
    return False


def main():
    # pd_file = '/' + str(object_id) + '_inter.csv'
    pd_file = "/" + "interact.csv"
    df = pd.read_csv(root + pd_file)
    df["qual"] = df["label"]

    success_points = []
    for i in range(len(df)):
        if df.loc[i, "label"].astype(np.long) > 0:
            point = df.loc[i, "x":"z"].to_numpy(np.single)
            success_points.append(point)
        else:
            point = df.loc[i, "x":"z"].to_numpy(np.single)
            if array_in_list(point, success_points):
                df.loc[i, "qual"] = 1

    new_save_file = root + "/" + str(object_id) + "_new_inter.csv"
    df.to_csv(new_save_file, index=False)


if __name__ == "__main__":
    main()
