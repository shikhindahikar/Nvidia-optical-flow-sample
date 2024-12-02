#pragma once
#include "NvOFInterface/nvOpticalFlowCommon.h"
#include "NvOFInterface/nvOpticalFlowCuda.h"
#include <memory>
#include <sstream>
#include <string.h>

#define H_BUFF 1080
#define W_BUFF 1920

extern uint8_t gridsize;

class NvOFException : public std::exception
{
public:
    NvOFException(const std::string& errorStr, const NV_OF_STATUS errorCode)
        : m_errorString(errorStr), m_errorCode(errorCode) {}
    virtual ~NvOFException() {}
    virtual const char* what() const throw() { return m_errorString.c_str(); }
    NV_OF_STATUS getErrorCode() const { return m_errorCode; }
    const std::string& getErrorString() const { return m_errorString; }
    static NvOFException makeNvOFException(const std::string& errorStr, const NV_OF_STATUS errorCode,
        const std::string& functionName, const std::string& fileName, int lineNo);

private:
    std::string m_errorString;
    NV_OF_STATUS m_errorCode;
};

inline NvOFException NvOFException::makeNvOFException(const std::string& errorStr, const NV_OF_STATUS errorCode, const std::string& functionName,
                                        const std::string& fileName, int lineNo)
{
    std::ostringstream errorLog;
    errorLog << functionName << " : " << errorStr << " at " << fileName << ";" << lineNo << std::endl;
    NvOFException exception(errorLog.str(), errorCode);
    return exception;
}

#define NVOF_THROW_ERROR(errorStr, errorCode)                                                           \
    do                                                                                                  \
    {                                                                                                   \
        throw NvOFException::makeNvOFException(errorStr, errorCode, __FUNCTION__, __FILE__, __LINE__);  \
    } while (0)

#define NVOF_API_CALL(nvOFAPI)                                                                          \
    do                                                                                                  \
    {                                                                                                   \
        NV_OF_STATUS errorCode = nvOFAPI;                                                               \
        if (errorCode != NV_OF_SUCCESS)                                                                 \
        {                                                                                               \
            std::ostringstream errorLog;                                                                \
            errorLog << #nvOFAPI << "returned error " << NV_OF_STATUS(errorCode);                                    \
            throw NvOFException::makeNvOFException(errorLog.str(), errorCode, __FUNCTION__, __FILE__, __LINE__);    \
        }                                                                                               \
    } while (0)

#define CUDA_DRVAPI_CALL( call )                                                                                                 \
    do                                                                                                                           \
    {                                                                                                                            \
        CUresult err__ = call;                                                                                                   \
        if (err__ != CUDA_SUCCESS)                                                                                               \
        {                                                                                                                        \
            const char *szErrName = NULL;                                                                                        \
            cuGetErrorName(err__, &szErrName);                                                                                   \
            std::ostringstream errorLog;                                                                                         \
            errorLog << "CUDA driver API error " << szErrName ;                                                                  \
            throw NvOFException::makeNvOFException(errorLog.str(), NV_OF_ERR_GENERIC, __FUNCTION__, __FILE__, __LINE__);         \
        }                                                                                                                        \
    }                                                                                                                            \
    while (0)


class API {
    public:
        API(CUcontext context, CUstream input, CUstream output);
        ~API();
        NV_OF_CUDA_API_FUNCTION_LIST* getAPI() { return nvofFuncList.get(); }
        CUcontext getContext() { return ctx; }
        NvOFHandle getHandle() { return handle; }
        CUstream getCudaStream(NV_OF_BUFFER_USAGE use);
    protected:
        void* libHandle;
    private:
        std::unique_ptr<NV_OF_CUDA_API_FUNCTION_LIST> nvofFuncList;
        CUcontext ctx;
        CUstream inputFrame;
        CUstream outputFrame;
        NvOFHandle handle;
};

class NvOFCudaBuffer {
public:
    uint32_t getWidth() { return m_width; }
    uint32_t getHeight() { return m_height; }
    uint32_t getElementSize() { return m_elementSize; }
    NV_OF_BUFFER_FORMAT getBufferFormat() { return m_eBufFmt; }
    NV_OF_BUFFER_USAGE getBufferUsage() { return m_eBufUsage; }

    void UploadData(const void* pData);

    void DownloadData(void* pData);

    void* getAPIResourceHandle() { return m_hGPUBuffer; }
    NvOFGPUBufferHandle getOFBufferHandle() { return m_hGPUBuffer; }
    CUdeviceptr getCudaDevicePtr() { return m_devicePtr; }
    NV_OF_CUDA_BUFFER_STRIDE_INFO getStrideInfo() { return m_strideInfo; }

    NvOFCudaBuffer(API* api, const NV_OF_BUFFER_DESCRIPTOR& desc) :
        apihandler(api),
        m_width(desc.width),
        m_height(desc.height),
        m_eBufUsage(desc.bufferUsage),
        m_eBufFmt(desc.bufferFormat),
        m_hGPUBuffer(nullptr),
        m_devicePtr(0)
    {
        NVOF_API_CALL(apihandler->getAPI()->nvOFCreateGPUBufferCuda(apihandler->getHandle(),
            &desc,
            NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,
            &m_hGPUBuffer));
        m_devicePtr = apihandler->getAPI()->nvOFGPUBufferGetCUdeviceptr(m_hGPUBuffer);
        NVOF_API_CALL(apihandler->getAPI()->nvOFGPUBufferGetStrideInfo(m_hGPUBuffer, &m_strideInfo));

        // put the correct element size
        if (m_eBufFmt == NV_OF_BUFFER_FORMAT_ABGR8)
        {
            m_elementSize = 4;
        }
        else if (m_eBufFmt == NV_OF_BUFFER_FORMAT_SHORT2)
        {
            m_elementSize = 4;
        }
        else if (m_eBufFmt == NV_OF_BUFFER_FORMAT_NV12)
        {
            m_elementSize = 1;
        }
    }

    ~NvOFCudaBuffer() {
        if (m_hGPUBuffer)
        {
            NVOF_API_CALL(apihandler->getAPI()->nvOFDestroyGPUBufferCuda(m_hGPUBuffer));
            m_hGPUBuffer = nullptr;
        }
    }

private:
    API* apihandler;
    uint32_t m_width;
    uint32_t m_elementSize;
    uint32_t m_height;
    NV_OF_BUFFER_USAGE m_eBufUsage;
    NV_OF_BUFFER_FORMAT m_eBufFmt;
    NvOFGPUBufferHandle m_hGPUBuffer;
    CUdeviceptr m_devicePtr;
    NV_OF_CUDA_BUFFER_STRIDE_INFO m_strideInfo;
};