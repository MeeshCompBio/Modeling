# Modeling App
A quick example on how to use pydantic and FastAPI to serve up an XGB model using the iris dataset

commands to get it running after you cd into the directory
```bash
docker build -t meeshcompbio/modelingapp .
docker run -d -p 8888:80 meeshcompbio/modelingapp
```

head to http://127.0.0.1:8888/docs to see the working api swagger