"""
Used to read and write csv data, should be replaced by csv library!
All implementations assume that the first row contains column headers
"""
import csv


def write_as_csv(rows: list[dict], f_out: str):
    """writes rows to a .csv file"""
    with open(f_out, "w") as f:
        keys = [key for key in rows[0]]
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = []
            for key in keys:
                value = row[key]
                if type(value) == str:
                    values.append("\"" + value + "\"")
                elif value is None:
                    values.append("")
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def read_csv(f_in: str):
    """
    Reads a csv file
    """
    with open(f_in, "r") as f:
        lines = [line for line in csv.reader(f)]
    names = lines[0]
    result = []
    for line in lines[1:]:
        result.append({names[i]: line[i] for i in range(len(names))})
    return result


def merge_csv_files(files: list[str], f_out: str):
    """
    Merges multiple csv files
    """
    data = []
    for file_in in files:
        data.append(read_csv(file_in))

    empty_row = {}
    for file_data in data:
        empty_row |= {key: None for key in file_data[0]}

    all_rows = []
    for file_data in data:
        all_rows += [empty_row | row for row in file_data]

    write_as_csv(all_rows, f_out)
