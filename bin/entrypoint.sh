#!/bin/bash
gunicorn app:app -b 0.0.0.0:$LISTEN_PORT