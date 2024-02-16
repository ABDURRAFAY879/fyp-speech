# Build docker

```
docker build -t speech .
```

# Run the docker container
```
docker run -p 8000:8000 speech
docker run -v "$(pwd)":/app -p 8000:8000 speech

```


Build the Docker image:

```docker-compose build```
Run the Docker container:

```docker-compose up```


Bash

docker exec -it speech bash