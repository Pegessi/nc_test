# export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_ancestor_nodes.txt'
# export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_fm_nodes.txt'
# export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/main_path_nodes.txt'
# export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_bc_topk.txt'
export CUSTOM_NODES_PATH='/data/wangzehua/Megatron-LM/nc_test/forkmerge/test_sp_nodes.txt'
export SAVE_PREFIX='./figure/test/gpt350M/' # gpt350M resnet32 resnet50 llama2_layer llama2_7B sdv1.5
export OP_LOG_PATH='/data/wangzehua/Megatron-LM/nc_test/logs/gpt3_350M_forward_once.log' # resnet50_once.log gpt3_350M_forward_once.log llama2_7B.log llama2_7B_once.log sdv1.5_once.log llama2_7B_layers.log lstm_once.log
export CUSTOM_EXPORT_FILENAME='graph_sp_nodes_test.pdf'
export SCALE_UP=2000
export ONLY_CUSTOM=0
export ENABLE_CUSTOM=1
# export SPLIT_INDEX=120


cd forkmerge
# python test_ancestor.py
# python fork_merge.py
# python test_graph.py
# python bc_graph.py
python shortest_path.py
cd ..

python plot_main.py