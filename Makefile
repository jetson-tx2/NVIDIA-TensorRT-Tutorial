.PHONY: build run start_jupyter

build:
	docker build -t optimization .

run: build
	docker run \
	  --runtime "nvidia" \
	  -it --rm \
	  -p 8888:8888 \
	  optimization

start_jupyter:
	jupyter lab \
	  --allow-root \
	  --ip=0.0.0.0 \
	  --NotebookApp.token=""
