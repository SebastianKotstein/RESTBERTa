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

from flask import Flask, Response, url_for

app = Flask(__name__)

@app.route("/", method=["GET"])
def get_root():
    pass

@app.route("/jobs", method=["GET"])
def get_jobs():
    pass

@app.route("/jobs", method=["POST"])
def create_job(job):
    pass

@app.route("/jobs/<int:job_id>", method=["GET"])
def get_job(job_id: int):
    pass

@app.route("/jobs/<int:job_id>", method=["PUT"])
def update_job(job_id: int):
    pass

@app.route("/jobs/<int:job_id>", method=["DELETE"])
def delete_job(job_id: int):
    pass


@app.route("/jobs/<int:job_id>/schemas", method=["GET"])
def get_schemas(job_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas", method=["POST"])
def create_schema(job_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>", method=["GET"])
def get_schema(job_id: int, schema_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>", method=["PUT"])
def update_schema(job_id: int, schema_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>", method=["DELETE"])
def delete_schema(job_id: int, schema_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries", method=["GET"])
def get_queries(job_id: int, schema_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries", method=["POST"])
def create_query(job_id: int, schema_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries/<int:query_id>", method=["GET"])
def get_query(job_id: int, schema_id: int, query_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries/<int:query_id>", method=["PUT"])
def update_query(job_id: int, schema_id: int, query_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries/<int:query_id>", method=["DELETE"])
def delete_query(job_id: int, schema_id: int, query_id: int):
    pass

@app.route("/jobs/<int:job_id>/schemas/<int:schema_id>/queries/<int:query_id>/results", method=["GET"])
def get_results(job_id: int, schema_id: int, query_id: int):
    pass










