# source ./setup.sh
export CURRENT_DIR=`pwd`
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}

protoc -I=${CURRENT_DIR}/proto --python_out=${CURRENT_DIR}/proto ${CURRENT_DIR}/proto/efficient_paper.proto
python proto/gene_template.py

python scripts/ddl.py
python scripts/split_by_year.py
python scripts/generate_reference.py
python scripts/generate_baseline_methods_graph.py
python scripts/generate_search_data.py

if [ -z "$1" ]; then
    python scripts/generate_paper_list.py
    echo "Refresh Readme Done"
else
    python scripts/generate_paper_list.py
    echo "Refresh Readme Done"
    echo "Upload changes to github"
    git add .
    git commit -m $1
    git push
    mkdocs gh-deploy --force
fi

