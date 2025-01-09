PROGRAM_ENTRYPOINT = src/main.cu
EXECUTATBLE_FILE = ./main

build:
	nvcc -o main ${PROGRAM_ENTRYPOINT}

run:
	${EXECUTATBLE_FILE}

clean:
	rm ${EXECUTATBLE_FILE}
