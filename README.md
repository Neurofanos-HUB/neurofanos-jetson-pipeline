# Neurofanos Jetson Pipeline

Source code for the CI/CD pipelines to run our AI models on NVIDIA Jetsons.

(NOTE: Currently a placeholder ONNX model is used, will connect with actual model in the future and inference using Jetson)

![diagram](/images/system_design.png)

## Getting started
* Run `make init`
* Edit the `.env`
* Run `make up`
* Open `127.0.0.1/health` in the browser. You can check the server status here.
* API documentation is automatically generated at `127.0.0.1/docs`

## The project structure
    |--docker
    |--src
        |--ml_pipelines
        |--models
        |--scripts
        |--tests
            |--unit_tests
            |--integration_tests
        |--utils
    |--server.py

* The `server.py` file is the entrypoint for our REST API service.
* Trained ML models in ONNX format should be in the `ml_pipelines` folder.
* The `models` folder stores Pydantic schemas. It defines the data schema in the ML model.

## How to use
* `make restart` will stop all the containers, remove them and start the application.
* `make bash` will create a new bash session in the container.