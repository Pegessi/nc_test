import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# 定义日志文件路径
log_file_path = '/data/wangzehua/Megatron-LM/plot/memlogs/GMLAKE-2025-2-7-11-49-32-1899247-default.log'

# 提取变量来指定字体粗细、字体大小和图像边框大小
FONT_WEIGHT = 'bold'
FONT_SIZE = 28
TICK_FONT_SIZE = 20
BORDER_WIDTH = 5

def get_mem_events(filename: str):
    with open(filename, 'r') as f:
        data = f.readlines()
        data = [eval(row.replace('\n', '')) for row in data]
    EVENTS_MAP = {
        0: "ALLOC",             # API made to the caching allocator for new memory
        1: "FREE_REQUESTED",    # API call made to the caching allocator to free memory
        2: "FREE_COMPLETED",    # The allocator might have to delay a free because
                                # it is still in use on another stream via record_stream
                                # This event is generated when a free actually completes.
        3: "SEGMENT_ALLOC",     # a call to cudaMalloc to get more memory from the OS
        4: "SEGMENT_FREE",      # a call to cudaFree to return memory to the OS (e.g. to
                                # defragment or empty_caches)
        5: "SEGMENT_MAP",       # a call to cuMemMap (used with expandable_segments)
        6: "SEGMENT_UNMAP",     # unmap part of a segment (used with expandable segments)
        7: "SNAPSHOT",          # a call to snapshot, used to correlate memory snapshots to trace events
        8: "OOM"                # the allocator threw an OutOfMemoryError (addr_ is the amount of free
                                # bytes reported by cuda)
    }
    # data = data[2612:2695] # 2468 9605 16625 int(len(data)/4) for forward one layer
    # data = data[:10000] # 2468 9605 16625 int(len(data)/4)

    bytes_to_mib = 1 / (1024 * 1024)
    def bytes_to_mib_round(x):
        return round(float(x) * bytes_to_mib, 2)
    
    total_allocated_mem = 0
    peak_cudaMalloc_mem = 0

    max_allocated_mem = 0
    max_reserved_mem = 0

    ax = []
    ay1, ay2 = [], []


    alloc_counts, alloc_size = 0, 0
    free_counts, free_size = 0, 0

    first_time = data[0]['TIME']
    for i in range(len(data)): # {"TYPE":"0", "SIZE":268435456, "ADDR":140421338497024}
        if len(data[i]['TYPE']) > 1:
            continue
        data[i]['TYPE'] = EVENTS_MAP[int(data[i]['TYPE'])]
        any_change = False
        if data[i]['TYPE'] == "ALLOC":
            total_allocated_mem += int(data[i]["SIZE"])
            max_allocated_mem = max(total_allocated_mem, max_allocated_mem)
            any_change = True
            alloc_counts += 1
            alloc_size += int(data[i]["SIZE"])
        elif data[i]['TYPE'] == "FREE_COMPLETED":
            total_allocated_mem -= int(data[i]["SIZE"])  
            any_change = True
            free_counts += 1
            free_size += int(data[i]["SIZE"])
        elif data[i]['TYPE'] == "SEGMENT_ALLOC":
            peak_cudaMalloc_mem += int(data[i]["SIZE"])
            max_reserved_mem = max(max_reserved_mem, peak_cudaMalloc_mem)
            any_change = True
        elif data[i]['TYPE'] == "SEGMENT_FREE":
            peak_cudaMalloc_mem -= int(data[i]["SIZE"])
            any_change = True

        if any_change:
            corrected_time = data[i]['TIME'] - first_time
            ax.append(corrected_time)
            ay1.append(bytes_to_mib_round(total_allocated_mem))
            ay2.append(bytes_to_mib_round(peak_cudaMalloc_mem))
    
    frag_str = str(round((1-float(max_allocated_mem)/float(max_reserved_mem))*100, 2)) + '%'
    print("Max allocated:", bytes_to_mib_round(max_allocated_mem), 'MiB, Max reserved:', bytes_to_mib_round(max_reserved_mem), 
        'MiB, peak fragmentation:', frag_str, 'records amount:', len(ax))

    # time, alloc, reserve
    return ax, ay1, ay2



log_files = [
    '/data/wangzehua/Megatron-LM/plot/memlogs/GMLAKE-B7.8-4step-2025-2-7-15-34-59-2480311-default.log',
    '/data/wangzehua/Megatron-LM/plot/memlogs/DTR-B6.5-4step-2025-2-7-15-22-32-2450911-default.log',
    '/data/wangzehua/Megatron-LM/plot/memlogs/TORCH-2025-2-1-11-31-35-3009862-default.log'
]

labels = [
    ['GMLAKE_alloc', 'GMLAKE_reserve'],
    ['DTR_alloc', 'DTR_reserve'],
    ['TORCH_alloc', 'TORCH_reserve']
]

# 设置图像画布大小为16:9
plt.figure(figsize=(16, 8))
# plt.rcParams.update({'font.weight': FONT_WEIGHT, 'font.size': FONT_SIZE})

datas = []
for idx, file_path in enumerate(log_files):
    ax, y1, y2 = get_mem_events(file_path)
    ax = [x / 1000 for x in ax]
    plt.plot(ax, y1, label=labels[idx][0])
    plt.plot(ax, y2, label=labels[idx][1])
    # datas.append(ax, y1, y2)


def k_formatter(x, pos):
    return '{:.0f}K'.format(x / 1000)
# 绘制折线图

# 设置x轴y轴刻度字体大小
plt.xticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
plt.yticks(fontsize=TICK_FONT_SIZE, fontweight=FONT_WEIGHT)
plt.xlabel('Time/s', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
plt.ylabel('Memory Usage/MB', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
plt.gca().yaxis.set_major_formatter(FuncFormatter(k_formatter))
# plt.title('Memory Timeline', fontsize=FONT_SIZE + 4, fontweight=FONT_WEIGHT)
# 设置图像边框大小
for spine in plt.gca().spines.values():
    spine.set_linewidth(BORDER_WIDTH)
plt.grid(True, linestyle='-', axis='y', which='major', color='gray', alpha=0.5, zorder=0)
# 设置图例位置在左上角
plt.legend(fontsize=TICK_FONT_SIZE)
plt.tight_layout()
plt.savefig('size_vs_corrected_time.pdf', dpi=400)