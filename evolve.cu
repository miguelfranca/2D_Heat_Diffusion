#include "evolve.h"
#include <stdio.h>
#include <algorithm>
#include <cmath>

#define BLOCK_SIZE_X 8
#define BLOCK_SIZE_Y 16
#define WORK_PER_THREAD 2
#define TAU .3f // damping constant

typedef struct CUDA_Config {
    CUDA_Config() {}

    float a; // Diffusion constant
    float dx; // Horizontal grid spacing
    float dy; // Vertical grid spacing
    float dx2;
    float dy2;
    float dt; // Largest stable time step

    int numElements;
    int nx, ny;

    cudaStream_t comp_stream;
    cudaStream_t read_stream;

    float* d_O1; // device output 1
    float* d_O2; // device output 2

    bool* barriers; // true for each cell with a barrier, false otherwise

    dim3 numBlocks;
    dim3 threadsPerBlock;

    bool copy;
} CUDA_Config;

CUDA_Config config;

/* Convert 2D index layout to unrolled 1D layout
 *
 * \param[in] i      Row index
 * \param[in] j      Column index
 * \param[in] width  The width of the area
 * \param[in] array_index  The index (0 or 1) for the array position
 *
 * \returns An index in the unrolled 1D array.
 */
int __host__ __device__ getIndex(const int y, const int x, const int width,
                                 const int height,
                                 const int array_index)
{
    return (y * width + x) + (array_index * height * width);
}

__device__ void computeCurrentCell(const int x, const int y, const float* in, float* out,
                                   const int nx, const int ny, const float dx2, const float dy2,
                                   const float aTimesDt, const bool* barriers)
{
    float current_cell_0 = in[getIndex(y, x, nx, ny, 0)]; // Phi
    float current_cell_1 = in[getIndex(y, x, nx, ny, 1)]; // PI

    // this cell is a barrier
    if(barriers[getIndex(y, x, nx, ny, 0)])
        return;

    float left = x == 0 || barriers[getIndex(y, x - 1, nx, ny, 0)] ?
                 in[getIndex(y, x + 1, nx, ny, 0)] : in[getIndex(y, x - 1, nx, ny, 0)];

    float right = x == nx - 1 || barriers[getIndex(y, x + 1, nx, ny, 0)] ?
                  in[getIndex(y, x - 1, nx, ny, 0)] : in[getIndex(y, x + 1, nx, nx, 0)];

    float up = y == 0 || barriers[getIndex(y - 1, x, nx, ny, 0)] ?
               in[getIndex(y + 1, x, nx, ny, 0)] : in[getIndex(y - 1, x, nx, ny, 0)];

    float down = y == ny - 1 || barriers[getIndex(y + 1, x, nx, ny, 0)] ?
                 in[getIndex(y - 1, x, nx, ny, 0)] : in[getIndex(y + 1, x, nx, nx, 0)];

    // Eq.1 diffusion
    out[getIndex(y, x, nx, ny, 0)] = current_cell_0 + aTimesDt *
                                     ((left - 2.0 * current_cell_0 + right) / dx2 +
                                      (up - 2.0 * current_cell_0 + down) / dy2);

    // Eq.2 dampened wave equation
    // out[getIndex(y, x, nx, ny, 0)] = current_cell_0 + aTimesDt * current_cell_1;
    // out[getIndex(y, x, nx, ny, 1)] = current_cell_1 + aTimesDt *
    //                                  ((left - 2.0 * current_cell_0 + right) / dx2 +
    //                                   (up - 2.0 * current_cell_0 + down) / dy2 -
    //                                   TAU * current_cell_1);

    // Eq.2 dampened wave equation + diffused
    // out[getIndex(y, x, nx, ny, 0)] = current_cell_0 + aTimesDt * current_cell_1 + .002 * aTimesDt *
    //                                  ((left - 2.0 * current_cell_0 + right) / dx2 +
    //                                   (up - 2.0 * current_cell_0 + down) / dy2);
    // out[getIndex(y, x, nx, ny, 1)] = current_cell_1 + aTimesDt *
    //                                  ((left - 2.0 * current_cell_0 + right) / dx2 +
    //                                   (up - 2.0 * current_cell_0 + down) / dy2 -
    //                                   TAU * current_cell_1);
}

__global__ void evolveKernel(const float* Un, float* Unp1, const int nx, const int ny,
                             const float dx2, const float dy2, const float aTimesDt, const bool* barriers)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (int w = i * WORK_PER_THREAD; w < (i + 1) * WORK_PER_THREAD ; ++w) {
        if (w < nx) { // need to check this because cuda blocks can have threads that go past the image boundaries
            int j = threadIdx.y + blockIdx.y * blockDim.y;

            if (j < ny) {
                // Explicit scheme
                computeCurrentCell(w, j, Un, Unp1, nx, ny, dx2, dy2, aTimesDt, barriers);
            }
        }
    }
}

__global__ void addHeatKernel(const int center_x, const int center_y, const float amount,
                              float* d_O1,
                              const int nx, const int ny, const float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny) {
        float dx = x - center_x;
        float dy = y - center_y;
        float distance = dx * dx + dy * dy;

        if (distance <= radius * radius)
            // gaussian distribution
            d_O1[getIndex(y, x, nx, ny, 0)] = amount * exp(-distance / (radius * radius));
    }
}

__global__ void addBarrierKernel(const int center_x, const int center_y, bool* barriers, float* d_O1,
                                 const int nx, const int ny, const float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < nx && y < ny) {
        float dx = x - center_x;
        float dy = y - center_y;
        float distance = dx * dx + dy * dy;

        if (distance <= radius * radius)
            barriers[getIndex(y, x, nx, ny, 0)] = true;
    }
}

void __host__ d_prepare(float* h_O, bool* h_barriers, const int nx, const int ny)
{
    // data dimensions
    config.nx = nx;
    config.ny = ny;
    config.numElements = nx * ny;

    // initialize starting state
    float radius2 = (nx / 6.0) * (nx / 6.0);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            int index = getIndex(i, j, ny, nx, 0);
            // Distance of point i, j from the origin
            float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2);

            // if (ds2 < radius2)
            // if (i == 0)
            // h_O[index] = 100.0 * exp(-(i - nx / 2) * (i - nx / 2) / (10.*10.));
            // else
            h_O[index] = 0.0;
            h_barriers[index] = false;
        }
    }

    // simulation settings
    config.a = 1.0;
    config.dx = 0.005;
    config.dy = 0.005;
    config.dx2 = config.dx * config.dx;
    config.dy2 = config.dy * config.dy;
    config.dt = config.dx2 * config.dy2 / (2.0 * config.a * (config.dx2 + config.dy2));
    // config.dt = config.dx * config.dy / (2.0 * config.a * (config.dx + config.dy));

    // streams for reading and computation
    cudaStreamCreate(&config.comp_stream);
    cudaStreamCreate(&config.read_stream);

    // initialize memory on GPU
    cudaMalloc((void**)&config.d_O1, config.numElements * 2 * sizeof(float));
    cudaMalloc((void**)&config.d_O2, config.numElements * 2 * sizeof(float));

    cudaMalloc((void**)&config.barriers, config.numElements * sizeof(bool));
    cudaMemset(config.barriers, 0, config.numElements * sizeof(bool));

    // set PI initial values
    cudaMemset(&config.d_O1[config.numElements], 0, config.numElements * sizeof(float));
    cudaMemset(&config.d_O2[config.numElements], 0, config.numElements * sizeof(float));

    // copy initial state to GPU, Phi initial values
    cudaMemcpy(config.d_O1, h_O, config.numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(config.d_O2, h_O, config.numElements * sizeof(float), cudaMemcpyHostToDevice);

    // number of blocks and threads per block
    config.numBlocks = dim3(nx / BLOCK_SIZE_X + 1, ny / BLOCK_SIZE_Y + 1);
    config.threadsPerBlock = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    config.copy = false;
}

void __host__ d_launchKernel(const int step, const int outputEvery, float* h_O, bool* h_barriers)
{
    evolveKernel <<< config.numBlocks, config.threadsPerBlock, 0, config.comp_stream>>>
    (config.d_O1, config.d_O2, config.nx, config.ny, config.dx2, config.dy2,
     config.a * config.dt, config.barriers);

    if (config.copy) {
        int index = 0;
        cudaMemcpyAsync(h_O, &config.d_O1[config.numElements * index],
                        config.numElements * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        config.read_stream);

        cudaMemcpyAsync(h_barriers, config.barriers,
                        config.numElements,
                        cudaMemcpyDeviceToHost,
                        config.read_stream);

        cudaStreamSynchronize(config.read_stream);
        cudaError_t errorCode = cudaGetLastError();

        if (errorCode != cudaSuccess) {
            printf("Cuda error %d: %s\n", errorCode, cudaGetErrorString(errorCode));
            exit(0);
        }

        config.copy = false;
    }

    if (step % outputEvery == 0) {
        config.copy = true;
        cudaStreamSynchronize(config.comp_stream);
    }

    std::swap(config.d_O1, config.d_O2);
}

void __host__ d_finalize()
{
    cudaStreamSynchronize(config.read_stream);
    cudaStreamSynchronize(config.comp_stream);

    cudaFree(config.d_O1);
    cudaFree(config.d_O2);
    cudaStreamDestroy(config.comp_stream);
    cudaStreamDestroy(config.read_stream);
}

std::array<int, 3> scalarToRGB(double value)
{
    // Calculate the RGB values based on the hue
    // int hue = static_cast<int>(value * 240);
    int r = static_cast<int>(value * 480);
    int g = static_cast<int>(std::max(0., 240 - value * 240));
    int b = static_cast<int>(240 * (1 - value));

    // Return the RGB color as an array
    return { r, g, b };
}

void addHeat(const int x, const int y, const float amount, const float radius)
{
    addHeatKernel <<< config.numBlocks, config.threadsPerBlock, 0, config.comp_stream>>>(x, y,
            amount, config.d_O1, config.nx, config.ny, radius);
}

void addBarrier(const int x, const int y, const float radius)
{
    addBarrierKernel <<< config.numBlocks, config.threadsPerBlock, 0, config.comp_stream>>>(x,
            y, config.barriers, config.d_O1, config.nx, config.ny, radius);
}

//################## Wrappers #################
void prepare(float* h_O, bool* h_barriers, const int nx, const int ny)
{
    d_prepare(h_O, h_barriers, nx, ny);
}

void launchKernel(const int step, const int outputEvery, float* h_O, bool * h_barriers)
{
    d_launchKernel(step, outputEvery, h_O, h_barriers);
}

void finalize()
{
    d_finalize();
}
//###############################################