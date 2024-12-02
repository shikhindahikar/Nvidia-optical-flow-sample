#include <dlfcn.h>
#include "flowvec.h"
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"


// Constructor for loading the library
API::API(CUcontext context, CUstream input, CUstream output ) : ctx(context), inputFrame(input), outputFrame(output) {
    // Load the library
    try
    {
        uint32_t version = 0;

        void* nvofhandle = dlopen("libnvidia-opticalflow.so", RTLD_LAZY);
        if (!nvofhandle) {
            NVOF_THROW_ERROR("API library file not found. Please ensure that the NVIDIA driver is installed", NV_OF_ERR_OF_NOT_AVAILABLE);
        }
        libHandle = nvofhandle;

        typedef NV_OF_STATUS(NVOFAPI *PFNNvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST* cudaOf);
        PFNNvOFAPICreateInstanceCuda NvOFAPICreateInstanceCuda = (PFNNvOFAPICreateInstanceCuda)dlsym(libHandle, "NvOFAPICreateInstanceCuda");
        typedef NV_OF_STATUS(NVOFAPI *PFNNvOFGetMaxSupportedApiVersion)(uint32_t* apiVer);
        PFNNvOFGetMaxSupportedApiVersion NvOFGetMaxSupportedApiVersion = (PFNNvOFGetMaxSupportedApiVersion)dlsym(libHandle, "NvOFGetMaxSupportedApiVersion");
        if (!NvOFAPICreateInstanceCuda) {
            NVOF_THROW_ERROR("Cannot find NvOFAPICreateInstanceCuda() entry in API library", NV_OF_ERR_OF_NOT_AVAILABLE);
        }
        nvofFuncList.reset(new NV_OF_CUDA_API_FUNCTION_LIST());
        NvOFGetMaxSupportedApiVersion(&version);
        // printf("API API version: %d\n", version);
        NVOF_API_CALL(NvOFAPICreateInstanceCuda(version, nvofFuncList.get()));
        NVOF_API_CALL(nvofFuncList->nvCreateOpticalFlowCuda(ctx, &handle));
        NVOF_API_CALL(nvofFuncList->nvOFSetIOCudaStreams(handle, inputFrame, outputFrame));
        // std::cout << "API library loaded successfully" << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
}

CUstream API::getCudaStream(NV_OF_BUFFER_USAGE use) {
    if (use == NV_OF_BUFFER_USAGE_INPUT)
        return inputFrame;
    else
        return outputFrame;
}

void NvOFCudaBuffer::UploadData(const void* data) {
    CUstream stream = apihandler->getCudaStream(getBufferUsage());
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth()* getElementSize();
    cuCopy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
    cuCopy2d.srcHost = data;
    cuCopy2d.srcPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cuCopy2d.dstDevice = this->getCudaDevicePtr();
    cuCopy2d.dstPitch = m_strideInfo.strideInfo[0].strideXInBytes;
    cuCopy2d.Height   = getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));

    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height   = (getHeight() + 1)/2;
        cuCopy2d.srcHost  = ((const uint8_t *)data + (cuCopy2d.srcPitch * cuCopy2d.Height));
        cuCopy2d.dstY     = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
}

void NvOFCudaBuffer::DownloadData(void* data) {
    CUstream stream = apihandler->getCudaStream(getBufferUsage());
    CUDA_MEMCPY2D cuCopy2d;
    memset(&cuCopy2d, 0, sizeof(cuCopy2d));
    cuCopy2d.WidthInBytes = getWidth() * getElementSize();
    cuCopy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
    cuCopy2d.dstHost = data;
    cuCopy2d.dstPitch = cuCopy2d.WidthInBytes;
    cuCopy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cuCopy2d.srcDevice = this->getCudaDevicePtr();
    cuCopy2d.srcPitch = m_strideInfo.strideInfo[0].strideXInBytes;
    cuCopy2d.Height = getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12 ? (getHeight() + getHeight() /2) : getHeight();
    CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    if (getBufferFormat() == NV_OF_BUFFER_FORMAT_NV12)
    {
        cuCopy2d.Height = (getHeight() + 1) / 2;
        cuCopy2d.dstHost = ((uint8_t *)data + (cuCopy2d.dstPitch * cuCopy2d.Height));
        cuCopy2d.srcY = m_strideInfo.strideInfo[0].strideYInBytes;
        CUDA_DRVAPI_CALL(cuMemcpy2DAsync(&cuCopy2d, stream));
    }
    CUDA_DRVAPI_CALL(cuStreamSynchronize(stream));
}

// Destructor for unloading the library
API::~API() {
    if (handle)
        nvofFuncList->nvOFDestroy(handle);
    std::cout << "API handle destroyed successfully" << std::endl;
    if (libHandle)
        dlclose(libHandle);
    std::cout << "API library unloaded successfully" << std::endl;
}