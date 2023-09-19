export CURRENT_DIR=`pwd`
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}

protoc -I=${CURRENT_DIR}/proto --python_out=${CURRENT_DIR}/proto ${CURRENT_DIR}/proto/efficient_paper.proto
python proto/gene_template.py