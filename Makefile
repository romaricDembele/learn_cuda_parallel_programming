PROGRAM_ENTRYPOINT = src/Main.cu
EXECUTATBLE_FILE = ./program

start: build run

build:
	nvcc ${PROGRAM_ENTRYPOINT} src/AddVectors.cu src/HelloWorld.cu src/DeviceInfo.cu -o program

run:
	${EXECUTATBLE_FILE}

clean:
	rm ${EXECUTATBLE_FILE}
