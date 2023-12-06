import csv

PI = "0.314159265358979323846264338327"

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
                    accur = 0
                    for i in range(len(PI)):
                        if PI[i] == line_data[2][i]:
                            accur += 1
                        else:
                            break
                    data.append([line_data[0], approxim_value / 5, line_data[2], accur-3])
                    # print(type(line_data[2]))
                    approxim_value = 0.0
        
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['iterations', 'time', 'pi', 'accuracy'])
            writer.writerows(data)
        
        print("Данные сохранены в файл:", csv_file)
    except FileNotFoundError:
        print(f"Файл {txt_file} не существует.")

files = ["CPU/MultiNilakanthaMethod",
        "CPU_NO_HT/MultiNilakanthaMethod",
        "GPU/MultiNilakanthaMethod",
        "CPU/NilakanthaMethod",
        "CPU_NO_HT/NilakanthaMethod",
        "GPU/NilakanthaMethod",
        "CPU/LeibnizMethod",
        "CPU_NO_HT/LeibnizMethod",
        "GPU/LeibnizMethod"]

delimiter = ' '

for file in files:
    convert_txt_to_csv(f"{file}.txt", f"{file}.csv", delimiter)