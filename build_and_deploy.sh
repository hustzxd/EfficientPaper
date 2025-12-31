#!/bin/bash
# Deploy script that ensures notes/ and meta/ are copied to site/
# Note: MkDocs build should be run first

set -e

echo "Ensuring notes/ and meta/ directories are in site/..."
if [ -d "notes" ]; then
    rm -rf site/notes
    cp -r notes site/
    echo "✓ Copied notes/"
fi

if [ -d "meta" ]; then
    rm -rf site/meta
    cp -r meta site/
    echo "✓ Copied meta/"
fi

echo "Deploying to GitHub Pages..."
mkdocs gh-deploy --force

echo "✓ Deployment complete!"
