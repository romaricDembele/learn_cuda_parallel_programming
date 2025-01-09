PROGRAM_ENTRYPOINT = src/main.cu
EXECUTATBLE_FILE = ./main

start: build run

build:
	nvcc -o main ${PROGRAM_ENTRYPOINT}

run:
	${EXECUTATBLE_FILE}

clean:
	rm ${EXECUTATBLE_FILE}
