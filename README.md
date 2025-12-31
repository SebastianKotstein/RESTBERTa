# RESTBERTa
This repository is the official implementation of the journal article ["RESTBERTa: a Transformer-based question answering approach for semantic search in Web API documentation"](https://link.springer.com/article/10.1007/s10586-023-04237-x) (see citation). It contains notebooks for fine-tuning pre-trained BERT models to the Web API integration tasks of parameter matching and endpoint discovery, notebooks for data preparation and evaluation, and a Flask application to demonstrate the capabilities of RESTBERTa.
Additional materials (datasets and reports of executed notebooks) can be found on [Zenodo](https://zenodo.org/records/10118349). The fine-tuned models (best checkpoints) are available on [Hugging Face](https://huggingface.co/SebastianKotstein).

## Inference (Jupyter Notebook)
This repository includes a [Jupyter Notebook](https://github.com/SebastianKotstein/RESTBERTa/blob/master/Inference.ipynb) that demonstrates the application of a RESTBERTa model.
To execute this notebook, download the repository including the processing pipeline contained in [tools](https://github.com/SebastianKotstein/RESTBERTa/tree/master/tools), which is used by the notebook.

## Web API and UI for Inference
We created a Flask app that enables the application of a RESTBERTa model through a Web API and UI.
To use this application, download this repository, navigate to [tools](https://github.com/SebastianKotstein/RESTBERTa/tree/master/tools), and create a docker image with:
```
docker build -t restberta-core .
```
Use one of the following commands to start the application. Set the ```MODEL``` parameter to specify the RESTBERTa model that should be loaded.

### Parameter Matching
Run the following command to start a container with the RESTBERTa model that has been fine-tuned to parameter matching exclusively:
```
docker run -d -p 80:80 -e MODEL=SebastianKotstein/restberta-qa-parameter-matching --name pm-cpu restberta-core
```
### Endpoint Discovery
Run the following command to start a container with the RESTBERTa model that has been fine-tuned to endpoint discovery exclusively:
```
docker run -d -p 80:80 -e MODEL=SebastianKotstein/restberta-qa-endpoint-discovery --name ed-cpu restberta-core
```
### Parameter Matching and Endpoint Discovery
Run the following command to start a container with the RESTBERTa model that has been fine-tuned to both tasks:
```
docker run -d -p 80:80 -e MODEL=SebastianKotstein/restberta-qa-pm-ed --name pm-ed-cpu restberta-core
```
### Custom Models
To start a container with a custom model hosted on Hugging Face, you can specify any model repository using the ```MODEL``` parameter. Additionally, if the model repository is private, the ```TOKEN``` parameter allows you to set a Hugging Face access token, e.g.:
```
docker run -d -p 80:80 -e MODEL=my-user/my-qa-model -e TOKEN=hf_12345678 --name my-model-cpu restberta-core
```
### Web UI
To use the Web UI, open a browser and navigate to http://localhost:80.

### Web API
The Web API is exposed over the same endpoints as the Web UI. For proper routing of HTTP requests to the Web API, it is therefore essential to set the ```Accept``` and ```Content-Type``` headers and
specify either ```application/json``` or one of the application-specific MIME-types (see OpenAPI documentation) as request and response format.
Use the following cURL to make a prediction:
```
curl -L 'http://localhost:80/predict' \
-H 'Accept: application/vnd.skotstein.restberta-core.results.v1+json' \
-H 'Content-Type: application/vnd.skotstein.restberta-core.schemas.v1+json' \
-d ' {
 "schemas":[
    {
      "schemaId": "s00",
      "name": "My schema",
      "value": "state units auth.key location.city location.city_id location.country location.lat location.lon location.postal_code",
      "queries": [
        {
          "queryId": "q0",
          "name": "My query",
          "value": "The ZIP of the city",
          "verboseOutput": true
        }
      ]
    }
  ]
}'
```

## Citation
```bibtex
@ARTICLE{10.1007/s10586-023-04237-x0,
  author={Kotstein, Sebastian and Decker, Christian},
  journal={Cluster Computing}, 
  title={RESTBERTa: a Transformer-based question answering approach for semantic search in Web API documentation}, 
  year={2024},
  volume={},
  number={},
  pages={},
  publisher={Springer}
  doi={10.1007/s10586-023-04237-x}}
```
