'''
Copyright 2023 Sebastian Kotstein

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''


from flask import Flask, request, jsonify, render_template, Response, url_for, send_from_directory
from pipeline.pipeline import Pipeline, InvalidRequestException
import json
from datetime import datetime
from werkzeug.exceptions import HTTPException, BadRequest
from flask_swagger_ui import get_swaggerui_blueprint
from representations import *
from content_negotiation import *

import os

app = Flask(__name__,static_folder='static')

MODEL_PM = "SebastianKotstein/restberta-qa-parameter-matching"
MODEL_ED = "SebastianKotstein/restberta-qa-endpoint-discovery"
MODEL_ED_PM = "SebastianKotstein/restberta-qa-pm-ed"

#model = os.getenv("MODEL",default=MODEL_PM)
if "MODEL" in os.environ:
    model = os.environ["MODEL"]
else:
    model = MODEL_PM

#best_size = os.getenv("BEST_SIZE",default=20)
if "BEST_SIZE" in os.environ:
    best_size = int(os.environ["BEST_SIZE"])
else:
    best_size = 20

#cache_size = os.getenv("CACHE",default=100)   
if "CACHE" in os.environ:
    cache_size = int(os.environ["CACHE"])
else:
    cache_size = 100

print("Model: ",model)
pipeline = Pipeline(model,best_size,cache_size)

SWAGGER_URL = '/docs' 
OPEN_API_FILE = '/openapi.yml'  

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    OPEN_API_FILE,
    config={  # Swagger UI config overrides
        'app_name': "Test application"
    },
    # oauth_config={  # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    #    'clientId': "your-client-id",
    #    'clientSecret': "your-client-secret-if-required",
    #    'realm': "your-realms",
    #    'appName': "your-app-name",
    #    'scopeSeparator': " ",
    #    'additionalQueryStringParams': {'test': "hello"}
    # }
)

if model == MODEL_PM:
    header = "RESTBERTa Parameter Matching"
    paragraph = "Schema"
    answer = "Parameter"
    example = "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units"
elif model == MODEL_ED:
    header = "RESTBERTa Endpoint Discovery"
    paragraph = "List of Endpoints"
    answer = "Endpoint"
    example = "auth.post users.get users.post users.{userId}.address.get users.{userId}.address.put users.{userId}.delete users.{userId}.get users.{userId}.put"
else:
    header = "RESTBERTa"
    paragraph = "Schema"
    answer = "Answer"
    example = "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units"
    
app.register_blueprint(swaggerui_blueprint)

@app.route("/predict",methods=["POST"])
@produces(MIME_TYPE_APPLICATION_JSON,MIME_TYPE_RESULTS_V1_JSON, default_mime_type=MIME_TYPE_RESULTS_V1_JSON)
@consumes(MIME_TYPE_SCHEMAS_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def api():
    args = request.args
    top_answers_n = None

    suppress_duplicates = False
    if "duplicates" in args and args["duplicates"] == "suppress":
        suppress_duplicates = True

    if "top" in args and args["top"]:
        top_answers_n = int(args["top"])
    try:
        response_payload = pipeline.process(request.json,top_answers_n,suppress_duplicates)
        response_payload["_links"] = [
            {
                "rel":"prediction",
                "href": url_for("api")
            },
            {
                "rel":"base",
                "href": url_for("base")
            }
        ]

        response = jsonify(response_payload)
        #response.mimetype=MIME_TYPE_RESULTS_V1_JSON
        return response
    except InvalidRequestException as e:
        raise BadRequest(description = e.message)


@app.route("/",methods=["GET"])
@produces(MIME_TYPE_APPLICATION_XHTML_XML,MIME_TYPE_TEXT_HTML,MIME_TYPE_HYPERMEDIA_V1_JSON,MIME_TYPE_APPLICATION_JSON)
def base():
    if MIME_TYPE_APPLICATION_XHTML_XML in list(request.accept_mimetypes.values()) or MIME_TYPE_TEXT_HTML in list(request.accept_mimetypes.values()):
        return render_template("index.html", header=header, paragraph=paragraph, answer=answer, example=example)
    else:
        response = jsonify({
            "_links":[
                {
                    "rel":"prediction",
                    "href": url_for("api")
                },
                {
                    "rel":"self",
                    "href": url_for("base")
                }
            ]
        })
        response.mimetype = MIME_TYPE_HYPERMEDIA_V1_JSON
        return response

@app.route('/openapi.yml')
def send_docs():
    return send_from_directory(app.static_folder, 'OpenAPI.yml')

@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "status": e.code,
        "error": e.name,
        "message": e.description,
        "path": request.url,
        "_links":[
                {
                    "rel":"base",
                    "href": url_for("base")
                },
                {
                    "rel":"docs",
                    "href": SWAGGER_URL
                }
            ]
    })
    response.content_type = MIME_TYPE_ERROR_V1_JSON
    return response

'''
if __name__ == '__main__':
    pipeline = Pipeline("SebastianKotstein/restberta-qa-parameter-matching",20,2)

    input = {
        "schemas":[
            {
                "schemaId": "s1",
                "name": "testSchema",
                "value": "auth.key location.city location.city_id location.country location.lat location.lon location.postal_code state units", 
                "queries":[
                    {
                        "queryId": "q1",
                        "name": "first query",
                        "value": "The ZIP",
                        "verboseOutput":False
                    },
                    {
                        "queryId": "q2",
                        "name": "second query",
                        "value": "The auth token",
                        "verboseOutput":False
                    }
                    
                ]
            },
            {
                "schemaId": "s2",
                "name": "schemaWoQueries",
                "value": "none",
                "queries":[]
            }
        ]
    }

    results = pipeline.process(input)
    print(json.dumps(results, indent=2))
'''
