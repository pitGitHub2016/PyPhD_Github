#!flask/bin/python
import sys, pandas as pd, datetime, requests, sqlite3, time, os
import numpy
import numpy as np
#sys.path.insert(0, '/opt/torus-engine/torus/')
from flask import Flask, render_template, jsonify, json, Response, stream_with_context, request, flash
import matplotlib.pyplot as plt
import pickle
from time import gmtime, strftime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('VendorHomePage.html')

@app.route('/stream')
def stream_view():
    def stream_template(template_name, **context):
        app.update_template_context(context)
        t = app.jinja_env.get_template(template_name)
        # app.jinja_env.lstrip_blocks = True
        app.keep_trailing_newline = True
        # app.jinja_env.trim_blocks = True
        rv = t.stream(context)
        rv.disable_buffering()
        return rv

    def generate():
        while True:
            yield "{}\n".format(datetime.datetime.now().isoformat(" ", "seconds"))
            time.sleep(1)

    #rows = generate()
    #return Response(stream_template('template.html', rows=rows))
    return Response(generate())
    #return Response(stream_with_context(stream_template('template.html', rows=rows)))

@app.route('/betmechstipsGetVendorSubscriber/<wixUserId>', methods=['POST', 'GET'])
def getSpecificSubscriber(wixUserId):
    dataSubscriber = tr.getDBdf('sqlite', selDB, 'SELECT expires_in, vendorClientId, wixUserId FROM BetmechsVendorSubscribers WHERE wixUserId LIKE "' + str(wixUserId) + '"').reset_index(drop=True)

    jsonfiles = json.loads(dataSubscriber.to_json(orient='records'))
    return jsonify({'jsonfiles': jsonfiles})