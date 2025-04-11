export SAVE_PREFIX="/data/wangzehua/Megatron-LM/nc_test/figure/paper/"
export OURS="DT-Control"
export DATA_PATH="/data/wangzehua/Megatron-LM/nc_test/plot_old/logs/exp-data.xlsx"

# python3 plot_intro.py   # fragmentation & time/recurisive
# python3 plot_old/plot.py
python3 plot_single_compute_overhead.py
# python3 memlogs/plot_mem_effi.py
# python3 plot_training_loss.py 
