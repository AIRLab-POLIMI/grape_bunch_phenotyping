"""
Add the target (grape bunch volume and weight) to the features.
The target is extracted from the lab analyses (excel file).
The target are:
- grapes_no: number of grapes in the grape bunch
- volume: #grapes * avgGrapeVolume before the harvest, real volume at the harvest
- weight: #grapes * avgGrapeWeight before the harvest, real weight at the harvest
"""

import pandas as pd
import csv
import argparse


def extract_avg_grape_volume_weight_frCurves(df_curves, bunch_id, plant_date):
    plant_no = bunch_id[3:len(bunch_id)]

    avg_grape_volume = df_curves[(df_curves['Pianta'] == int(plant_no)) & (df_curves['Data'] == plant_date)]['Vbac']
    avg_grape_weight = df_curves[(df_curves['Pianta'] == int(plant_no)) & (df_curves['Data'] == plant_date)]['Pbac']

    return extract_value_frSeries(avg_grape_volume), extract_value_frSeries(avg_grape_weight)


def extract_value_frSeries(series):

    if len(series.index) > 0:
        value = series.values[0]
    else:
        value = 0.0

    return value


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_annotations_path", help="path of the CSV file with the grape bunches features", default="/home/user/red_globe_2021_datasets/volume_regression_csv/redglobe_2021_features.csv")
    parser.add_argument("--csv_lab_analyses_path", help="path of the CSV file with the lab analyses (target)", default="/home/user/red_globe_2021_datasets/volume_regression_csv/redglobe_2021_lab_analyses.xlsx")
    parser.add_argument("--csv_complete_path", help="path of the CSV file with the complete dataset (features + target)", default="/home/user/red_globe_2021_datasets/volume_regression_csv/redglobe_2021_dataset.csv")
    args = vars(parser.parse_args())

    csv_annotations_path = args["csv_annotations_path"]
    csv_lab_analyses_path = args["csv_lab_analyses_path"]
    csv_complete_path = args["csv_complete_path"]

    df_annot = pd.read_csv(csv_annotations_path)
    df_harvest = pd.read_excel(csv_lab_analyses_path, sheet_name='Vendemmia')
    df_curves = pd.read_excel(csv_lab_analyses_path, sheet_name='Curve di maturazione')

    with open(csv_complete_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["bunch_id", "west", "cut", "date", "area", "avg_depth", "not_occluded", "grapes_no", "volume", "weight"])

    for index, row in df_annot.iterrows():
        bunch_id = row['bunch_id']

        grapes_no = df_harvest[df_harvest['Codice grappolo'] == bunch_id]['N. bacche'].values[0]

        if row['date'] != "2021-09-06":
            avg_grape_volume, avg_grape_weight = extract_avg_grape_volume_weight_frCurves(df_curves, bunch_id, row['date'])
            grape_volume = grapes_no*avg_grape_volume
            grape_weight = grapes_no*avg_grape_weight
        else:
            grape_volume = extract_value_frSeries(df_harvest[df_harvest['Codice grappolo'] == bunch_id]['Volume (ml)'])
            grape_weight = extract_value_frSeries(df_harvest[df_harvest['Codice grappolo'] == bunch_id]['Peso grappolo (g)'])

        with open(csv_complete_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([bunch_id, row['west'], row['cut'], row['date'], row['area'], row['avg_depth'], row['not_occluded'], grapes_no, grape_volume, grape_weight])
