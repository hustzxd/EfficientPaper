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
        elif parsed_url.path == '/api/get-keywords':
            self.handle_get_keywords()
        elif parsed_url.path == '/api/get-institutions':
            self.handle_get_institutions()
        elif parsed_url.path == '/api/load-note':
            self.handle_load_note(parsed_url)
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        """Handle POST requests"""
        parsed_url = urlparse(self.path)

        if parsed_url.path == '/api/save-paper':
            self.handle_save_paper()
        elif parsed_url.path == '/api/upload-cover':
            self.handle_upload_cover()
        elif parsed_url.path == '/api/delete-paper':
            self.handle_delete_paper()
        elif parsed_url.path == '/save_note':
            self.handle_save_note()
        elif parsed_url.path == '/api/add-from-arxiv':
            self.handle_add_from_arxiv()
        elif parsed_url.path == '/api/upload-github':
            self.handle_upload_github()
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

    def handle_get_keywords(self):
        """Get all available keywords from proto definition"""
        try:
            # Get all keyword enum values from the protobuf
            keyword_enum = eppb.Keyword.Word

            # Convert enum names to readable labels
            def to_label(enum_name):
                """Convert enum_name like 'attention_sparsity' to 'Attention Sparsity'"""
                return ' '.join(word.capitalize() for word in enum_name.split('_'))

            # Get all values from the enum descriptor
            keywords = []
            for name, value in keyword_enum.items():
                keywords.append({
                    'value': name,
                    'label': to_label(name)
                })

            # Sort by value (enum name)
            keywords.sort(key=lambda x: x['value'])

            self.send_json_response({'keywords': keywords})

        except Exception as e:
            print(f"Error getting keywords: {e}", file=sys.stderr)
            self.send_json_response({'error': str(e)}, 500)

    def handle_get_institutions(self):
        """Get all unique institutions from existing papers"""
        try:
            institutions_set = set()
            meta_dir = "./meta"

            # Scan all prototxt files and collect institutions
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

                        # Add all institutions from this paper
                        for institution in pinfo.paper.institutions:
                            if institution:  # Skip empty strings
                                institutions_set.add(institution)
                    except Exception as e:
                        # Skip files that can't be parsed
                        continue

            # Convert to sorted list
            institutions_list = sorted(list(institutions_set))

            self.send_json_response({'institutions': institutions_list})

        except Exception as e:
            print(f"Error getting institutions: {e}", file=sys.stderr)
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

    def handle_delete_paper(self):
        """Delete paper prototxt file and corresponding notes folder"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request_data = json.loads(body)

            path = request_data.get('path')

            if not path:
                self.send_json_response({'error': 'Missing path'}, 400)
                return

            # Construct full file path for prototxt
            prototxt_path = os.path.join(os.getcwd(), path)

            if not os.path.exists(prototxt_path):
                self.send_json_response({'error': f'File not found: {path}'}, 404)
                return

            # Extract year and paper_id from path (e.g., meta/2025/P0JBYHCN.prototxt)
            path_parts = path.split('/')
            if len(path_parts) < 3:
                self.send_json_response({'error': 'Invalid path format'}, 400)
                return

            year = path_parts[1]
            paper_id = path_parts[2].replace('.prototxt', '')

            # Construct notes folder path
            notes_folder = os.path.join(os.getcwd(), 'notes', year, paper_id)

            deleted_items = []

            # Delete prototxt file
            try:
                os.remove(prototxt_path)
                deleted_items.append(f'Metadata file: {path}')
                print(f"Deleted prototxt file: {prototxt_path}")
            except Exception as e:
                print(f"Error deleting prototxt file: {e}", file=sys.stderr)
                self.send_json_response({'error': f'Failed to delete metadata file: {str(e)}'}, 500)
                return

            # Delete notes folder if it exists
            if os.path.exists(notes_folder):
                try:
                    import shutil
                    shutil.rmtree(notes_folder)
                    deleted_items.append(f'Notes folder: notes/{year}/{paper_id}/')
                    print(f"Deleted notes folder: {notes_folder}")
                except Exception as e:
                    print(f"Error deleting notes folder: {e}", file=sys.stderr)
                    # Don't fail if notes folder deletion fails, prototxt is already deleted
                    deleted_items.append(f'Notes folder (partial): {str(e)}')
            else:
                deleted_items.append('Notes folder: (not found, skipped)')

            self.send_json_response({
                'success': True,
                'message': 'Paper deleted successfully. Run ./start_editor.sh to see updates.',
                'deleted': deleted_items
            })

        except Exception as e:
            print(f"Error deleting paper: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': str(e)}, 500)

    def handle_load_note(self, parsed_url):
        """Load note.md file content"""
        try:
            # Parse query parameters
            query_params = parse_qs(parsed_url.query)
            path = query_params.get('path', [''])[0]

            if not path:
                self.send_json_response({'error': 'No path specified'}, 400)
                return

            # Construct full file path
            file_path = os.path.join(os.getcwd(), path)

            # Check if file exists
            if not os.path.exists(file_path):
                # Return empty template if file doesn't exist
                self.send_json_response({
                    'exists': False,
                    'content': ''
                })
                return

            # Read note content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"Loaded note: {file_path}")

            self.send_json_response({
                'exists': True,
                'content': content
            })

        except Exception as e:
            print(f"Error loading note: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': str(e)}, 500)

    def handle_save_note(self):
        """Save note.md file"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request_data = json.loads(body)

            path = request_data.get('path')
            content = request_data.get('content')

            if not path:
                self.send_json_response({'error': 'Missing path'}, 400)
                return

            if content is None:
                self.send_json_response({'error': 'Missing content'}, 400)
                return

            # Construct full file path
            file_path = os.path.join(os.getcwd(), path)

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write note content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"Saved note: {file_path}")

            self.send_json_response({
                'success': True,
                'message': 'Note saved successfully'
            })

        except Exception as e:
            print(f"Error saving note: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': str(e)}, 500)

    def handle_add_from_arxiv(self):
        """Add paper from arXiv ID"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request_data = json.loads(body)

            arxiv_id = request_data.get('arxiv_id', '').strip()
            abbr = request_data.get('abbr', '').strip()

            if not arxiv_id:
                self.send_json_response({'error': 'Missing arXiv ID'}, 400)
                return

            print(f"Adding paper from arXiv: {arxiv_id}")

            # Run add_paper.py script
            import subprocess
            import random
            import string

            # Generate random abbr if not provided
            if not abbr:
                abbr = "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

            # Create a temporary file path to pass as argument (script expects -f flag or stdin)
            # We'll use stdin to pass both arxiv_id and abbr
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(f"{arxiv_id}\n")  # arXiv ID
                f.write(f"{abbr}\n")       # Paper abbr name
                temp_input_file = f.name

            try:
                # Run the script without -f flag, so it reads from stdin
                with open(temp_input_file, 'r') as input_file:
                    result = subprocess.run(
                        ['python', 'scripts/add_paper.py'],
                        stdin=input_file,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )

                if result.returncode == 0:
                    # Extract the prototxt path from output
                    prototxt_path = None
                    for line in result.stdout.split('\n'):
                        if 'Writing paper information into' in line:
                            # Extract path from message like: Writing paper information into ./meta/2025/ABCD1234.prototxt
                            import re
                            match = re.search(r'into\s+\./meta/(\d+)/([^/]+\.prototxt)', line)
                            if match:
                                year = match.group(1)
                                filename = match.group(2)
                                prototxt_path = f"meta/{year}/{filename}"

                    self.send_json_response({
                        'success': True,
                        'message': 'Paper added successfully from arXiv',
                        'abbr': abbr,
                        'prototxt_path': prototxt_path,
                        'output': result.stdout
                    })
                else:
                    raise Exception(f"Script failed: {result.stderr}")

            finally:
                # Clean up temp file
                import os as os_module
                try:
                    os_module.unlink(temp_input_file)
                except:
                    pass

        except subprocess.TimeoutExpired:
            print(f"Timeout adding paper from arXiv", file=sys.stderr)
            self.send_json_response({'error': 'Request timeout - arXiv might be slow'}, 500)
        except Exception as e:
            print(f"Error adding paper from arXiv: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            self.send_json_response({'error': str(e)}, 500)

    def handle_upload_github(self):
        """Upload changes to GitHub and deploy to GitHub Pages"""
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            request_data = json.loads(body)

            commit_message = request_data.get('commit_message', '').strip()

            if not commit_message:
                self.send_json_response({'error': 'Missing commit message'}, 400)
                return

            print(f"Uploading to GitHub with message: {commit_message}")

            # Run git commands
            import subprocess

            try:
                # Run git add .
                result_add = subprocess.run(
                    ['git', 'add', '.'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result_add.returncode != 0:
                    raise Exception(f"git add failed: {result_add.stderr}")

                # Run git commit
                result_commit = subprocess.run(
                    ['git', 'commit', '-m', commit_message],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                # It's ok if commit fails because there are no changes
                if result_commit.returncode != 0 and 'nothing to commit' not in result_commit.stdout:
                    raise Exception(f"git commit failed: {result_commit.stderr}")

                # Run git push
                result_push = subprocess.run(
                    ['git', 'push'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result_push.returncode != 0:
                    raise Exception(f"git push failed: {result_push.stderr}")

                # Run mkdocs build first to ensure notes/ and meta/ are copied
                print("[GitHub Upload] Running mkdocs build...")
                result_build = subprocess.run(
                    ['mkdocs', 'build'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result_build.returncode != 0:
                    raise Exception(f"mkdocs build failed: {result_build.stderr}")

                # Run mkdocs gh-deploy
                result_deploy = subprocess.run(
                    ['mkdocs', 'gh-deploy', '--force'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result_deploy.returncode != 0:
                    raise Exception(f"mkdocs gh-deploy failed: {result_deploy.stderr}")

                self.send_json_response({
                    'success': True,
                    'message': 'Successfully uploaded to GitHub and deployed to GitHub Pages',
                    'output': {
                        'commit': result_commit.stdout,
                        'push': result_push.stdout,
                        'build': result_build.stdout,
                        'deploy': result_deploy.stdout
                    }
                })

            except subprocess.TimeoutExpired:
                raise Exception("Operation timed out")

        except Exception as e:
            print(f"Error uploading to GitHub: {e}", file=sys.stderr)
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
    print(f"  - GET  http://localhost:{port}/api/get-baseline-methods")
    print(f"  - GET  http://localhost:{port}/api/get-keywords")
    print(f"  - GET  http://localhost:{port}/api/get-institutions")
    print(f"  - GET  http://localhost:{port}/api/load-note?path=...")
    print(f"  - POST http://localhost:{port}/api/save-paper")
    print(f"  - POST http://localhost:{port}/api/upload-cover")
    print(f"  - POST http://localhost:{port}/api/delete-paper")
    print(f"  - POST http://localhost:{port}/api/add-from-arxiv")
    print(f"  - POST http://localhost:{port}/api/upload-github")
    print(f"  - POST http://localhost:{port}/save_note")
    print("Press Ctrl+C to stop")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == '__main__':
    main()
