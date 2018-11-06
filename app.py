# Copyright 2016-2018 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/kiosk-data-processing/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""REST API for data pre-processing and post-processing transformations"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import errno
import logging
from multiprocessing import Pool

from decouple import config
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np

import data_processing


app = Flask(__name__)

CORS(app)

PROCESSING_FUNCTIONS = {
    'preprocessing': {
        'noramlize': data_processing.preprocessing.noramlize,
    },
    'postprocessing': {
        'deepcell': data_processing.postprocessing.deepcell,
        'mibi': data_processing.postprocessing.mibi,
        'watershed': data_processing.postprocessing.watershed
    },
}


@app.before_first_request
def setup_logging():
    """Set up logging to send INFO to stderr"""
    if not app.debug:
        # In production mode, add log handler to sys.stderr.
        app.logger.setLevel(logging.INFO)


@app.route('/health', methods=['GET'])
def healthcheck():
    """Health check end-point"""
    return jsonify({'status': 'OK'}), 200


@app.route('/<process_type>/<func>', methods=['POST'])
def process(process_type, func):
    try:
        # first, verify the route parameters
        func = str(func).lower()
        process_type = str(process_type).lower()
        processing_function = PROCESSING_FUNCTIONS[process_type][func]
    except KeyError as err:
        return jsonify({'error': 'Not Found: {}'.format(err)}), 404

    try:
        # second, verify the post request data
        request_json = request.get_json(force=True)
        app.logger.info(json.dumps(request_json, indent=4))
    except Exception as err:
        errmsg = 'Invalid JSON: {}'.format(err)
        app.logger.error(errmsg)
        return jsonify({'error': errmsg}), 400

    try:
        images = [np.array(d['image']) for d in request_json['instances']]
    except Exception as err:
        errmsg = 'Could not process JSON data as images: {}'.format(err)
        app.logger.error(errmsg)
        return jsonify({'error': errmsg}), 400

    try:
        pool = Pool()
        processed = pool.map(processing_function, images)
        pool.close() 
        pool.join()
        return jsonify({'processed': processed}), 200
    except Exception as err:
        errmsg = 'Error applying mibi post-processing: {}'.format(err)
        app.logger.error(errmsg)
        return jsonify({'error': errmsg}), 500


if __name__ == '__main__':
    DEBUG = config('DEBUG', default=False, cast=bool)
    FLASK_PORT = config('PORT', default=8080, cast=int)

    app.run(debug=DEBUG, host='0.0.0.0', port=FLASK_PORT)
