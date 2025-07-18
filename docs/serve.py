#!/usr/bin/env python3
"""
Simple HTTP server for serving the demo webpage.
Run this script and open http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os

# Change to the docs directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8000

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    print("Press Ctrl+C to stop the server")
    httpd.serve_forever()