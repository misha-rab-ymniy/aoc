import csv

def convert_txt_to_csv(txt_file, csv_file, delimiter):
    try:
        with open(txt_file, 'r') as file:
            lines = file.readlines()
            data = []
            approxim_value = 0.0
            for i in range(len(lines)):
                line_data = lines[i].strip().split(delimiter)
                approxim_value += float(line_data[1])

                if i % 5 == 4:
                    data.append([line_data[0], approxim_value / 5, line_data[2]])
                    approxim_value = 0.0
        
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iterations', 'time', 'pi'])
            writer.writerows(data)
        
        print("Данные сохранены в файл:", csv_file)
    except FileNotFoundError:
        print(f"Файл {txt_file} не существует.")

files = ["CPU/MultiNilakanthaMethod",
        "GPU/MultiNilakanthaMethod",
        "CPU/NilakanthaMethod",
        "GPU/NilakanthaMethod",
        "CPU/LeibnizMethod",
        "GPU/LeibnizMethod"]

csv_file = 'output.csv'
delimiter = ' '

for file in files:
    convert_txt_to_csv(f"{file}.txt", f"{file}.csv", delimiter)