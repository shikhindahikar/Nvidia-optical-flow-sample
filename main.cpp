#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "flowvec.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "cuda.h"


#define UNKNOWN_FLOW_THRESH 1e9
int m_ncols = 0;
int m_colorwheel[60][3];
uint8_t gridsize = 0;

void writeFlowtoFile(float* flowvec, uint16_t width, uint16_t height) {
    std::ofstream flowfile;
    flowfile.open("flowvec.txt", std::ios::app);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            flowfile << flowvec[(y * 2 * width) + 2 * x] << " " << flowvec[(y * 2 * width) + 2 * x + 1] << " ";
        }
    }
    flowfile << std::endl;
    flowfile.close();
}

static inline bool unknown_flow(float u, float v)
{
    return (fabs(u) >  UNKNOWN_FLOW_THRESH)
        || (fabs(v) >  UNKNOWN_FLOW_THRESH)
        || std::isnan(u) || std::isnan(v);
}

void SetColors(int r, int g, int b, int k)
{
    m_colorwheel[k][0] = r;
    m_colorwheel[k][1] = g;
    m_colorwheel[k][2] = b;
}

void MakeColorWheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow 
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    m_ncols = RY + YG + GC + CB + BM + MR;

    int i;
    int k = 0;

    for (i = 0; i < RY; i++) SetColors(255, 255 * i / RY, 0, k++);
    for (i = 0; i < YG; i++) SetColors(255 - 255 * i / YG, 255, 0, k++);
    for (i = 0; i < GC; i++) SetColors(0, 255, 255 * i / GC, k++);
    for (i = 0; i < CB; i++) SetColors(0, 255 - 255 * i / CB, 255, k++);
    for (i = 0; i < BM; i++) SetColors(255 * i / BM, 0, 255, k++);
    for (i = 0; i < MR; i++) SetColors(255, 0, 255 - 255 * i / MR, k++);
}

void ComputeColor(float fx, float fy, uint8_t* pix)
{
    float rad = sqrtf(fx * fx + fy * fy);
    float a = atan2f(-fy, -fx) / M_PI;
    float fk = (a + 1.0f) / 2.0f * (m_ncols - 1);
    int k0 = (int)fk;
    int k1 = (k0 + 1) % m_ncols;
    float f = fk - k0;
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++)
    {
        float col0 = m_colorwheel[k0][b] / 255.0f;
        float col1 = m_colorwheel[k1][b] / 255.0f;
        float col = (1 - f) * col0 + f * col1;
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75f; // out of range
        pix[2 - b] = (int)(255.0f * col);
    }
}

// Post processing to get the flow vectors in RGB format for viewing
void postProcessVectors(const NV_OF_FLOW_VECTOR* _flowvectors, uint8_t* output, uint16_t outwidth, uint16_t outheight) {
    // converting them to normal float values first
    std::unique_ptr<float[]> flowvec;
    flowvec.reset(new float[outwidth * outheight * 2]);

    for (uint32_t y = 0; y < outheight; ++y)
    {
        for (uint32_t x = 0; x < outwidth; ++x)
        {
            flowvec[(y * 2 * outwidth) + 2 * x] = (float)(_flowvectors[y * outwidth + x].flowx / 32.0f);
            flowvec[(y * 2 * outwidth) + 2 * x + 1] = (float)(_flowvectors[y * outwidth + x].flowy / 32.0f);
        }
    }

    // writeFlowtoFile(flowvec.get(), outwidth, outheight);
    // exit(0);

    float maxx = -999.0f;
    float maxy = -999.0f;
    float minx = 999.0f, miny = 999.0f;
    float maxrad = -1.0f;
    for (uint32_t n = 0; n < (outheight * outwidth); ++n)
    {
        float fx = flowvec[2 * n];
        float fy = flowvec[(2 * n) + 1];

        if (unknown_flow(fx, fy))
            return;
        maxx = std::max(maxx, fx);
        maxy = std::max(maxy, fy);
        minx = std::min(minx, fx);
        miny = std::min(miny, fy);
        float rad = sqrt(fx * fx + fy * fy);
        maxrad = std::max(maxrad, rad);
    }
    maxrad = std::max(maxrad, 1.0f);

    // post processing to get the flow vectors in RGB format for viewing
    for (uint32_t y = 0; y < outheight; ++y)
    {
        for (uint32_t x = 0; x < outwidth; ++x)
        {
            float fx = flowvec[(y * outwidth * 2) + (2 * x)];
            float fy = flowvec[(y * outwidth * 2) + (2 * x) + 1];
            uint8_t pix[3];
            if (unknown_flow(fx, fy))
            {
                pix[0] = pix[1] = pix[2] = 0;
            }
            else
            {
                ComputeColor(fx / maxrad, fy / maxrad, pix);
            }

            output[(y * outwidth * 3) + (3 * x)] = pix[0];
            output[(y * outwidth * 3) + (3 * x) + 1] = pix[1];
            output[(y * outwidth * 3) + (3 * x) + 2] = pix[2];
        }
    }
}

// Function to initialize NVOF parameters
NV_OF_INIT_PARAMS initializeOFParameters() {
    NV_OF_INIT_PARAMS initparams = { 0 };
    initparams.width = W_BUFF;
    initparams.height = H_BUFF;
    initparams.inputBufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
    initparams.mode = NV_OF_MODE_OPTICALFLOW;
    initparams.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)gridsize;
    initparams.enableOutputCost = NV_OF_FALSE;
    initparams.predDirection = NV_OF_PRED_DIRECTION_FORWARD;
    initparams.perfLevel = NV_OF_PERF_LEVEL_SLOW;
    initparams.enableExternalHints = NV_OF_FALSE;
    initparams.enableRoi = NV_OF_FALSE;
    initparams.enableGlobalFlow = NV_OF_FALSE;
    initparams.hintGridSize = (NV_OF_HINT_VECTOR_GRID_SIZE)0;
    
    return initparams;
}

// Function to create and upload input buffer
NvOFCudaBuffer* createAndUploadInputBuffer(API* nvofobj, uint8_t* frameData) {
    NV_OF_BUFFER_DESCRIPTOR bufferDesc;
    bufferDesc.width = W_BUFF;
    bufferDesc.height = H_BUFF;
    bufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    bufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;
    
    NvOFCudaBuffer* buffer = new NvOFCudaBuffer(nvofobj, bufferDesc);
    buffer->UploadData(frameData);
    
    return buffer;
}

// Function to calculate output buffer dimensions
void calculateOutputDimensions(uint32_t& outwidth, uint32_t& outheight) {
    outheight = H_BUFF / gridsize;
    outwidth = W_BUFF / gridsize;
}

// Function to create output buffer
NvOFCudaBuffer* createOutputBuffer(API* nvofobj, uint32_t outwidth, uint32_t outheight) {
    NV_OF_BUFFER_DESCRIPTOR outbufferDesc;
    outbufferDesc.width = outwidth;
    outbufferDesc.height = outheight;
    outbufferDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    outbufferDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;
    
    return new NvOFCudaBuffer(nvofobj, outbufferDesc);
}

// Function to prepare execution input parameters
NV_OF_EXECUTE_INPUT_PARAMS prepareExecutionInputParams(NvOFCudaBuffer* inbuffer, NvOFCudaBuffer* refbuffer) {
    NV_OF_EXECUTE_INPUT_PARAMS inparams;
    memset(&inparams, 0, sizeof(NV_OF_EXECUTE_INPUT_PARAMS));
    
    inparams.inputFrame = inbuffer->getOFBufferHandle();
    inparams.referenceFrame = refbuffer->getOFBufferHandle();
    inparams.externalHints = (NvOFGPUBufferHandle)nullptr;
    inparams.disableTemporalHints = NV_OF_FALSE;
    inparams.hPrivData = (NvOFPrivDataHandle)nullptr;
    inparams.numRois = 0;
    inparams.padding = 0;
    inparams.padding2 = 0;
    
    return inparams;
}

// Function to prepare execution output parameters
NV_OF_EXECUTE_OUTPUT_PARAMS prepareExecutionOutputParams(NvOFCudaBuffer* outbuffer) {
    NV_OF_EXECUTE_OUTPUT_PARAMS outparams;
    memset(&outparams, 0, sizeof(NV_OF_EXECUTE_OUTPUT_PARAMS));
    
    outparams.bwdOutputBuffer = nullptr;
    outparams.bwdOutputCostBuffer = nullptr;
    outparams.globalFlowBuffer = nullptr;
    outparams.hPrivData = nullptr;
    outparams.outputBuffer = outbuffer->getOFBufferHandle();
    outparams.outputCostBuffer = nullptr;
    
    return outparams;
}

// Main function to calculate optical flow
void calculateFlow(uint8_t* frame1, uint8_t* frame2, uint8_t* vecframe, 
                   CUcontext cuContext, CUstream instream, CUstream outstream) {
    // Create an instance of the API
    API* nvofobj = new API(cuContext, instream, outstream);
    
    // Initialize the optical flow parameters
    NV_OF_INIT_PARAMS initparams = initializeOFParameters();
    NVOF_API_CALL(nvofobj->getAPI()->nvOFInit(nvofobj->getHandle(), &initparams));
    
    // Create and upload input buffers
    NvOFCudaBuffer* inbuffer = createAndUploadInputBuffer(nvofobj, frame1);
    NvOFCudaBuffer* refbuffer = createAndUploadInputBuffer(nvofobj, frame2);
    
    // Calculate output buffer dimensions
    uint32_t outwidth, outheight;
    calculateOutputDimensions(outwidth, outheight);
    
    // Pointer for storing the flow vectors
    std::unique_ptr<NV_OF_FLOW_VECTOR[]> flowdata;
    flowdata.reset(new NV_OF_FLOW_VECTOR[outwidth * outheight * 2]);
    
    // Create output buffer
    NvOFCudaBuffer* outbuffer = createOutputBuffer(nvofobj, outwidth, outheight);
    
    // Prepare execution parameters
    NV_OF_EXECUTE_INPUT_PARAMS inparams = prepareExecutionInputParams(inbuffer, refbuffer);
    NV_OF_EXECUTE_OUTPUT_PARAMS outparams = prepareExecutionOutputParams(outbuffer);
    
    // Run Optical Flow
    nvofobj->getAPI()->nvOFExecute(nvofobj->getHandle(), &inparams, &outparams);
    
    // Download flow vectors
    outbuffer->DownloadData(flowdata.get());
    
    // Post-process vectors
    postProcessVectors((const NV_OF_FLOW_VECTOR*)flowdata.get(), (uint8_t*)vecframe, outwidth, outheight);
    
    // Clean up
    flowdata.reset();
    delete inbuffer;
    delete refbuffer;
    delete outbuffer;
    
    // Destroy NVOF session
    nvofobj->getAPI()->nvOFDestroy(nvofobj->getHandle());
}

int main(int argc, char* argv[]) {
    // Initialize CUDA
    cuInit(0);
    MakeColorWheel();

    // Give the input video file path and GPU number
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input file path>" << " <GPU number>" << "<Grid Size>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string inputVideoFile(argv[1]);

    gridsize = atoi(argv[3]);

    printf("Input video file: %s\n", inputVideoFile.c_str());

    std::string ffmpeg_path = "ffmpeg";

    // command to run ffmpeg
	std::string command = ffmpeg_path + " -i " + "\"" + inputVideoFile + "\"" + " -f image2pipe -pix_fmt abgr -vcodec rawvideo -";
	std::FILE* pipe = popen(command.c_str(), "r");
	if (!pipe) {
		throw std::runtime_error("Failed to open pipe");
	}

	// Allocate memory for two frames
    uint8_t* frame1 = (uint8_t*)malloc(H_BUFF * W_BUFF * 4 * sizeof(uint8_t));
    uint8_t* frame2 = (uint8_t*)malloc(H_BUFF * W_BUFF * 4 * sizeof(uint8_t));
    uint8_t* vecframe = (uint8_t*)malloc(H_BUFF / gridsize * W_BUFF / gridsize * 3 * sizeof(uint8_t));

    if (!frame1 || !frame2) {
        std::cerr << "Failed to allocate memory." << std::endl;
        pclose(pipe);
        free(frame1);
        free(frame2);
        throw std::runtime_error("Failed to allocate memory");
    }

    // Read the first frame into frame1
    if (fread(frame1, H_BUFF * W_BUFF * 4, 1, pipe) != 1) {
        std::cerr << "Failed to read the first frame." << std::endl;
        free(frame1);
        free(frame2);
        pclose(pipe);
        throw std::runtime_error("Failed to read the first frame");
    }

    // Create CUDA context
    CUcontext cuContext = nullptr;
    CUdevice cuDevice = 0;
    int device = atoi(argv[2]);
    cuDeviceGet(&cuDevice, device);
    cuCtxCreate(&cuContext, 0, cuDevice);

    // Copy the frame data to the input and reference streams
    CUstream instream = nullptr, outstream = nullptr;

    cuStreamCreate(&instream, CU_STREAM_DEFAULT);
    cuStreamCreate(&outstream, CU_STREAM_DEFAULT);

    // Run inference on each frame till last frame
	while (fread(frame2, H_BUFF * W_BUFF * 4, 1, pipe) == 1) {
        
        // Calculate the flow vectors
        calculateFlow(frame1, frame2, vecframe, cuContext, instream, outstream);

        // Display
        cv::imshow("Vectors", cv::Mat(H_BUFF / gridsize, W_BUFF / gridsize, CV_8UC3, vecframe));
        // cv::imshow("Original2", out);

        if (cv::waitKey(1) == 27) break;

        // Swap the frames
        memcpy(frame1, frame2, H_BUFF * W_BUFF * 4);

    }

    // free memory
    free(frame1);
    free(frame2);
    free(vecframe);
    pclose(pipe);

    cuStreamDestroy(instream);
    cuStreamDestroy(outstream);
    cuCtxDestroy(cuContext);

    // Close all windows
    cv::destroyAllWindows();    
    return 0;
}