# source ./setup.sh
export CURRENT_DIR=`pwd`
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}

protoc -I=${CURRENT_DIR}/proto --python_out=${CURRENT_DIR}/proto ${CURRENT_DIR}/proto/efficient_paper.proto
python proto/gene_template.py

python scripts/ddl.py
python scripts/split_by_year.py
python scripts/generate_baseline_methods_graph.py
python scripts/generate_search_data.py
python scripts/generate_readme_pages.py

if [ -z "$1" ]; then
    echo "Refresh done"
else
    echo "Upload changes to github"
    git add .
    git commit -m $1
    git push
    # Build site with MkDocs
    mkdocs build
    # Use custom build script that copies notes/ and meta/ directories
    ./build_and_deploy.sh
fi

