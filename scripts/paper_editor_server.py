#!/usr/bin/env python3
"""
Simple HTTP server for editing paper metadata
Provides API endpoints to load and save prototxt files
Runs on a separate port from MkDocs (default: 8001)
"""

import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import google.protobuf as pb
import google.protobuf.text_format

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proto import efficient_paper_pb2 as eppb


class PaperEditorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for paper editor API"""

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed_url = urlparse(self.path)

        if parsed_url.path == '/api/load-paper':
            self.handle_load_paper(parsed_url)
        elif parsed_url.path == '/api/get-baseline-methods':
            self.handle_get_baseline_methods()
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)

        if parsed_url.path == '/api/save-paper':
            self.handle_save_paper()
        elif parsed_url.path == '/api/upload-cover':
            self.handle_upload_cover()
        else:
            self.send_error(404, "Not Found")

    def handle_load_paper(self, parsed_url):
        """Load paper data from prototxt file"""
        try:
            # Parse query parameters
            query_params = parse_qs(parsed_url.query)
            path = query_params.get('path', [''])[0]

            if not path:
                self.send_json_response({'error': 'No path specified'}, 400)
                return

            # Construct full file path
            file_path = os.path.join(os.getcwd(), path)

            if not os.path.exists(file_path):
                self.send_json_response({'error': f'File not found: {path}'}, 404)
                return

            # Read and parse prototxt
            pinfo = eppb.PaperInfo()
            with open(file_path, 'r') as f:
                pb.text_format.Merge(f.read(), pinfo)

            # Convert to JSON-serializable dict
            data = {
                'paper': {
                    'title': pinfo.paper.title,
                    'abbr': pinfo.paper.abbr,
                    'url': pinfo.paper.url,
                    'authors': list(pinfo.paper.authors),
                    'institutions': list(pinfo.paper.institutions),
                },
                'pub': {
                    'where': pinfo.pub.where,
                    'year': pinfo.pub.year,
                },
                'code': {
                    'type': pinfo.code.type,
                    'url': pinfo.code.url,
                },
                'keyword': {
                    'words': [eppb.Keyword.Word.Name(w) for w in pinfo.keyword.words],
                },
                'cover': {
                    'url': pinfo.cover.url,
                },
                'baseline': {
                    'methods': list(pinfo.baseline.methods),
                },
            }

            self.send_json_response(data)

        except Exception as e:
            print(f"Error loading paper: {e}", file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def handle_get_baseline_methods(self):
        """Get all paper abbr in format 'year/abbr' from meta directory"""
        try:
            methods_set = set()
            meta_dir = "./meta"

            # Scan all prototxt files and collect year/abbr
            for year_dir in os.listdir(meta_dir):
                year_path = os.path.join(meta_dir, year_dir)
                if not os.path.isdir(year_path):
                    continue

                for filename in os.listdir(year_path):
                    if not filename.endswith(".prototxt"):
                        continue

                    filepath = os.path.join(year_path, filename)
                    try:
                        pinfo = eppb.PaperInfo()
                        with open(filepath, "r") as f:
                            pb.text_format.Merge(f.read(), pinfo)

                        # Get abbr from paper info
                        abbr = pinfo.paper.abbr
                        if abbr:
                            # Format as year/abbr
                            method_name = f"{year_dir}/{abbr}"
                            methods_set.add(method_name)
                    except Exception as e:
                        # Skip files that can't be parsed
                        continue

            # Convert to sorted list
            methods_list = sorted(list(methods_set))

            self.send_json_response({'methods': methods_list})

        except Exception as e:
            print(f"Error getting baseline methods: {e}", file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def handle_save_paper(self):
        """Save paper data to prototxt file"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request_data = json.loads(body)

            path = request_data.get('path')
            data = request_data.get('data')

            if not path or not data:
                self.send_json_response({'error': 'Missing path or data'}, 400)
                return

            # Construct full file path
            file_path = os.path.join(os.getcwd(), path)

            # Create PaperInfo protobuf
            pinfo = eppb.PaperInfo()

            # Paper section
            pinfo.paper.title = data['paper']['title']
            pinfo.paper.abbr = data['paper']['abbr']
            pinfo.paper.url = data['paper']['url']
            pinfo.paper.authors.extend(data['paper']['authors'])
            pinfo.paper.institutions.extend(data['paper']['institutions'])

            # Publication section
            pinfo.pub.where = data['pub']['where']
            pinfo.pub.year = data['pub']['year']

            # Code section
            pinfo.code.type = data['code']['type']
            pinfo.code.url = data['code']['url']

            # Keywords section
            for word_str in data['keyword']['words']:
                word_enum = eppb.Keyword.Word.Value(word_str)
                pinfo.keyword.words.append(word_enum)

            # Cover section
            pinfo.cover.url = data['cover']['url']

            # Baseline section
            pinfo.baseline.methods.extend(data['baseline']['methods'])

            # Write to file
            with open(file_path, 'w') as f:
                f.write(str(pinfo))

            self.send_json_response({'success': True, 'message': 'Paper saved successfully. Run ./start_editor.sh to see updates.'})

        except Exception as e:
            print(f"Error saving paper: {e}", file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def handle_upload_cover(self):
        """Handle cover image upload"""
        try:
            # Parse multipart form data
            content_type = self.headers['Content-Type']
            if not content_type or not content_type.startswith('multipart/form-data'):
                self.send_json_response({'error': 'Invalid content type'}, 400)
                return

            # Get content length
            content_length = int(self.headers['Content-Length'])

            # Read the body
            body = self.rfile.read(content_length)

            # Parse boundary
            boundary = content_type.split('boundary=')[1].encode()

            # Simple multipart parser
            parts = body.split(b'--' + boundary)

            file_data = None
            file_name = None
            prototxt_path = None

            for part in parts:
                if b'Content-Disposition' in part:
                    # Parse headers and content
                    header_end = part.find(b'\r\n\r\n')
                    if header_end == -1:
                        continue

                    headers = part[:header_end].decode('utf-8', errors='ignore')
                    content = part[header_end + 4:]

                    # Remove trailing boundary markers
                    if content.endswith(b'\r\n'):
                        content = content[:-2]

                    # Check if this is the file field
                    if 'name="file"' in headers:
                        # Extract filename
                        if 'filename="' in headers:
                            start = headers.index('filename="') + 10
                            end = headers.index('"', start)
                            file_name = headers[start:end]
                            file_data = content

                    # Check if this is the path field
                    elif 'name="path"' in headers:
                        prototxt_path = content.decode('utf-8', errors='ignore').strip()

            if not file_data or not file_name or not prototxt_path:
                self.send_json_response({'error': 'Missing file or path'}, 400)
                return

            # Extract paper info from path (e.g., meta/2025/P0JBYHCN.prototxt)
            path_parts = prototxt_path.split('/')
            if len(path_parts) < 3:
                self.send_json_response({'error': 'Invalid path format'}, 400)
                return

            year = path_parts[1]
            paper_id = path_parts[2].replace('.prototxt', '')

            # Create notes directory if it doesn't exist
            notes_dir = os.path.join(os.getcwd(), 'notes', year, paper_id)
            os.makedirs(notes_dir, exist_ok=True)

            # Get file extension
            file_ext = os.path.splitext(file_name)[1]
            if not file_ext:
                file_ext = '.png'

            # Save file with a standard name
            saved_filename = f'cover{file_ext}'
            file_path = os.path.join(notes_dir, saved_filename)

            with open(file_path, 'wb') as f:
                f.write(file_data)

            # Update note.md if it exists
            note_md_path = os.path.join(notes_dir, 'note.md')
            if os.path.exists(note_md_path):
                try:
                    import re
                    with open(note_md_path, 'r', encoding='utf-8') as f:
                        note_content = f.read()

                    # Use regex to replace any ![111](...) pattern with the new cover image
                    # This handles ../../blank.jpg, cover.png, or any other existing path
                    updated_content = re.sub(
                        r'!\[111\]\([^)]+\)',
                        f'![111]({saved_filename})',
                        note_content
                    )

                    with open(note_md_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)

                    print(f"Updated note.md with cover image reference: {saved_filename}")
                except Exception as e:
                    print(f"Warning: Could not update note.md: {e}", file=sys.stderr)

            # Return response with filename and preview URL
            preview_url = f'/notes/{year}/{paper_id}/{saved_filename}'

            self.send_json_response({
                'success': True,
                'filename': saved_filename,
                'url': preview_url,
                'message': 'Image uploaded successfully. Run ./start_editor.sh to see updates.'
            })

        except Exception as e:
            print(f"Error uploading cover: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': str(e)}, 500)

    def send_json_response(self, data, status=200):
        """Send JSON response"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))


def main():
    """Start the API server"""
    port = 8001  # Use different port from MkDocs (which uses 8000)
    server_address = ('', port)

    # Change to project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    httpd = HTTPServer(server_address, PaperEditorHandler)
    print(f"Starting Paper Editor API server on http://localhost:{port}")
    print(f"MkDocs should run on http://localhost:8000")
    print(f"API endpoints:")
    print(f"  - GET  http://localhost:{port}/api/load-paper?path=...")
    print(f"  - POST http://localhost:{port}/api/save-paper")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == '__main__':
    main()
