compile_and_run_all:
	g++ -o clear clear.cpp
	nvcc -arch=sm_75 -o gpu_test gpu.cu -lgmp
	g++ -O3 -o cpu_test cpu.cpp -lgmp
	./clear
	./gpu_test
	./cpu_test

compile_all:
	g++ -o clear clear.cpp
	nvcc -arch=sm_75 -o gpu_test gpu.cu -lgmp
	g++ -O3 -o cpu_test cpu.cpp -lgmp

run_all:
	./clear
	./gpu_test
	./cpu_test

compile_and_run_cpu:
	g++ -O3 -o cpu_test cpu.cpp -lgmp
	./cpu_test

run_cpu:
	./cpu_test

compile_and_run_gpu:
	nvcc -arch=sm_75 -o gpu_test gpu.cu -lgmp
	./gpu_test

run_gpu:
	./gpu_test

clear_txt: 
	./clear

run_converter:
	python3 converter.py

run_table:
	python3 table.py