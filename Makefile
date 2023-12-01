NAME?=tts
GPUS?=0
SAVE_DIR?=/hdd/aidar/tts_save_dir
DATA_DIR?=/hdd/aidar/tts_data_dir
NOTEBOOK_DIR?=/hdd/aidar/notebooks/tts
USER?=$(shell whoami)
UID?=$(shell id -u)
GID?=$(shell id -g)

.PHONY: build run

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS) \
	--ipc=host \
	--net=host \
	-v $(PWD):/workspace \
	-v $(SAVE_DIR):/workspace/saved \
	-v $(DATA_DIR):/workspace/data \
	-v $(NOTEBOOK_DIR):/workspace/notebooks \
	--name=$(NAME) \
	$(NAME) \
	bash
