# RESTBERTa
This repository is the official implementation of the journal article ["RESTBERTa: a Transformer-based question answering approach for semantic search in Web API documentation"](https://link.springer.com/article/10.1007/s10586-023-04237-x) (see citation). It contains notebooks for fine-tuning pre-trained BERT models to the Web API integration tasks of parameter matching and endpoint discovery, plus notebooks for data preparation and evaluation.
Additional materials (datasets and reports of executed notebooks) can be found on [Zenodo](https://zenodo.org/records/10118349). The fine-tuned models (best checkpoints) are available on [Hugging Face](https://huggingface.co/SebastianKotstein).

## Web API and UI for Inference
We implemented a Flask application that places our model behind a Web API and UI for inference.
To use this application, navigate to [tools](https://github.com/SebastianKotstein/RESTBERTa/tree/master/tools) and create a docker image with:
```
docker build -t restberta-core .
```
Start the docker container with:
```
docker run -d -p 80:80 --name pm-cpu restberta-core
```
If you want to start the application for another Web API integration task, i.e., with another RESTBERTa model (e.g., for endpoint discovery, see [RESTBERTa](https://github.com/SebastianKotstein/RESTBERTa)), specify the model as ENV parameter:
```
docker run -d -p 80:80 -e MODEL=SebastianKotstein/restberta-qa-endpoint-discovery --name ed-cpu restberta-core
```
### Web UI
To open the Web UI, use a browser and navigate to http://localhost:80.

### Web API
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
