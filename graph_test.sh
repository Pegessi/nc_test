export SAVE_PREFIX='./figure/test/gpt350M/'
export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_ancestor_nodes.txt'
export OP_LOG_PATH='/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log' # resnet50_once.log gpt3_350M_forward_once.log llama2_7B_once.log
export SCALE_UP=1000


cd forkmerge
python test_ancestor.py
cd ..

python plot_main.py