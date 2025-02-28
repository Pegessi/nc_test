#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#define ROUND_UP(size, granularity) (((size) + (granularity) - 1) / (granularity)) * (granularity)

CUmemGenericAllocationHandle allocatePhysicalMemory(int device, size_t size) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    
    size_t granularity = 2097152;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    // Ensure size matches granularity requirements for the allocation
    size_t padded_size = ROUND_UP(size, granularity);

    // Allocate physical memory
    CUmemGenericAllocationHandle allocHandle;
    cuMemCreate(&allocHandle, padded_size, &prop, 0);

    return allocHandle;
}

int main() {
    cudaError_t err;
    void* devPtr1;
    void* devPtr2;
    void* devPtrCombined;
    size_t size1 = 512 * 1024 * 1024;  // 512 MB
    size_t size2 = 512 * 1024 * 1024;  // 512 MB
    size_t sizeCombined = 1024 * 1024 * 1024;  // 1024 MB

    // err = cudaMalloc(&devPtr1, size1);
    // err = cudaMalloc(&devPtr2, size2);
    // cudaFree(devPtr1);
    // cudaFree(devPtr2);

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();


    int deviceSupportsVmm;
    int device = 0;
    // CUresult result = cuDeviceGetAttribute(&deviceSupportsVmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);
    // if (deviceSupportsVmm != 0) {
        // `device` supports Virtual Memory Management
        start = std::chrono::high_resolution_clock::now();

        auto m1 = allocatePhysicalMemory(device, size1);
        auto m2 = allocatePhysicalMemory(device, size2);

        end = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "Time for cuMemCreate(512 512): " << duration2 << " ns" << std::endl;

        // CUdeviceptr ptr;
        // // `ptr` holds the returned start of virtual address range reserved.
        // CUresult result = cuMemAddressReserve(&ptr, size, 0, 0, 0); // alignment = 0 for default alignment
        start = std::chrono::high_resolution_clock::now();

        CUdeviceptr ptr;
        // `ptr`: address in the address range previously reserved by cuMemAddressReserve.
        // `allocHandle`: CUmemGenericAllocationHandle obtained by a previous call to cuMemCreate. 
        CUresult result = cuMemMap(ptr, sizeCombined, 0, m1, 0);
        result = cuMemMap(ptr, sizeCombined, size1, m2, 0);

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = device;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cuMemSetAccess(ptr, sizeCombined, &accessDesc, 1); 
        // 计算并输出操作 1 的时间
        end = std::chrono::high_resolution_clock::now();
        duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "Time for operation 2 (map map setAccess): " << duration2 << " ns" << std::endl;
    // }else{
    //     std::cout << deviceSupportsVmm << "\n";
    // }




    // 分配两个 512 MB 的设备内存
    err = cudaMalloc(&devPtr1, size1);
    if (err!= cudaSuccess) {
        std::cerr << "cudaMalloc failed for devPtr1: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&devPtr2, size2);
    if (err!= cudaSuccess) {
        std::cerr << "cudaMalloc failed for devPtr2: " << cudaGetErrorString(err) << std::endl;
        cudaFree(devPtr1);
        return 1;
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Time for cudaMalloc(512 512): " << duration1 << " ns" << std::endl;




    // 开始时间测量
    start = std::chrono::high_resolution_clock::now();
    // 释放两块内存并分配 1024 MB 内存
    cudaFree(devPtr1);
    cudaFree(devPtr2);
    err = cudaMalloc(&devPtrCombined, sizeCombined);
    if (err!= cudaSuccess) {
        std::cerr << "cudaMalloc failed for devPtrCombined: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // 计算并输出操作 1 的时间
    end = std::chrono::high_resolution_clock::now();
    duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Time for operation (free free malloc): " << duration1 << " ns" << std::endl;
    cudaFree(devPtrCombined);


    // start = std::chrono::high_resolution_clock::now();
    // err = cudaMalloc(&devPtr1, size1);
    // if (err!= cudaSuccess) {
    //     std::cerr << "cudaMalloc failed for devPtr1: " << cudaGetErrorString(err) << std::endl;
    //     return 1;
    // }
    // err = cudaMalloc(&devPtr2, size2);
    // if (err!= cudaSuccess) {
    //     std::cerr << "cudaMalloc failed for devPtr2: " << cudaGetErrorString(err) << std::endl;
    //     cudaFree(devPtr1);
    //     return 1;
    // }
    // end = std::chrono::high_resolution_clock::now();
    // duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    // std::cout << "Time for cudaMalloc (512 512): " << duration1 << " ns" << std::endl;

    // 释放内存
    cudaFree(devPtr1);
    cudaFree(devPtr2);
    cudaFree(devPtrCombined);

    return 0;
}