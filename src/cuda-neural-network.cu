#include <stdio.h>
#include <math.h>
#include <cuda-neural-network.h>

#define BACKPROP

#ifdef BACKPROP
float *d_error;
float *d_temp_error;
__device__ float d_all_out[ALL_OUT_SIZE];
#endif

float *d_network, *d_training_dataset, *d_testing_dataset, *_h_temp, *_d_temp, *d_expected;
int *training_dataset_labels, *d_training_labels, training_dataset_size, *testing_dataset_labels, *d_testing_labels, testing_dataset_size, *d_batch;

__device__ const int NODE_COUNTS[LAYER_COUNT] = {784, 128, 64, 10};
__device__ const activation ACTIVATIONS[LAYER_COUNT] = {None, ReLu, Sigmoid, SoftMax};

__device__ float d_temp, d_temp_array_1[MAX_NODES], d_temp_array_2[MAX_NODES], d_output[MAX_NODES];

inline __device__ float deviceVectorDotProduct(int vector_size, float *a, float *b, int bInc){
    register float tmp = 0.0;
    register int iB = 0;
    
    // #pragma unroll
    for(register int i = 0; i < vector_size; i++){
        tmp += a[i] * b[iB];
        iB += bInc;
    }
    
    return tmp;
}

inline __device__ float deviceTransferDerivative(float input, int layer){
    activation perceptron = ACTIVATIONS[layer];
    if(perceptron == Sigmoid) return input * (1.0 - input);
    if(perceptron == ReLu) return input > 0.0;
    if(perceptron == SoftMax) return 2.0 * input * (1.0 - input) / (float)NODE_COUNTS[layer];
    return 1.0; //None
}

#ifdef BACKPROP
inline __device__ void deviceBackPropError(float *network, float *error, float *expected){
    int x = threadIdx.x;
    register int block_index = NETWORK_SIZE - OUTPUT_SIZE;
    register int index = block_index + x;
    register float tmp;
    register int i, j, next_block_index, last_node_count, node_count;
    register int outIndex = ALL_OUT_SIZE - OUTPUT_SIZE;
    last_node_count = OUTPUT_SIZE;
    
    if(x < OUTPUT_SIZE){
        error[index] = (d_output[x] - expected[x]) * deviceTransferDerivative(d_all_out[outIndex + x], LAYER_COUNT - 1);
    }

    __syncthreads();

    for(i = LAYER_COUNT - 1; i >= 1; i--){
        node_count = last_node_count;
        last_node_count = NODE_COUNTS[i - 1];
        next_block_index = block_index;
        block_index -= node_count * last_node_count;

        if(x < node_count){
            index = block_index + x;
            for(j = 0; j < last_node_count; j++){
                error[index] = error[next_block_index + x] * network[index] * deviceTransferDerivative(d_all_out[outIndex + x], i - 1);
                index += node_count;
            }
        }

        __syncthreads();

        if(x < last_node_count && i > 1){
            index = block_index + x;
            tmp = error[index];

            for(j = 1; j < node_count; j++){
                tmp += error[index + j];
            }

            index -= last_node_count;

            error[index] = tmp * deviceTransferDerivative(d_all_out[outIndex + x], i - 1);
        }

        block_index -= last_node_count;
        outIndex -= last_node_count;
    }
}

__global__ void cudaBackPropError(float *network, float *error, float *expected){
    deviceBackPropError(network, error, expected);
    __syncthreads();
}

inline void _backPropError(){
    cudaBackPropError<<<1, MAX_NODES>>>(d_network, d_error, d_expected);
}
#endif

inline __device__ void deviceApplyErrorToPoint(float *network_point, float *error_point, float scalar){
    (*network_point) -= (*error_point) * scalar;
}

__global__ void cudaApplyErrorToPoint(float *network_point, float *error_point, float scalar){
    deviceApplyErrorToPoint(network_point, error_point, scalar);
}

inline __device__ void deviceApplyError(float *network, float *error, float scalar){
    int x = threadIdx.x;
    register int index = x;
    register int i, j, last_base_index;
    register int base_index = NODE_COUNTS[0];
    for(i = 1; i < LAYER_COUNT; i++){
        last_base_index = base_index;
        base_index = NODE_COUNTS[i];
        for(j = 0; j < last_base_index + 1; j++){
            if(x < base_index){
                deviceApplyErrorToPoint(&network[index], &error[index], scalar);
            }
            index += base_index;
        }
    }
}

__global__ void cudaApplyError(float *network, float *error, float scalar){
    deviceApplyError(network, error, scalar);
    __syncthreads();
}

// __global__ void cudaZeroError(float *error){
//     int x = threadIdx.x;
//     register int index = x;
//     register int i, j, last_base_index;
//     register int base_index = NODE_COUNTS[0];
//     for(i = 1; i < LAYER_COUNT; i++){
//         last_base_index = base_index;
//         base_index = NODE_COUNTS[i];
//         for(j = 0; j < last_base_index + 1; j++){
//             if(x < base_index){
//                 error[index] = 0.0;
//             }
//             index += base_index;
//         }
//     }
// }

inline void _applyError(float *d_target, float *d_error, float scalar){
    cudaApplyError<<<1, MAX_NODES>>>(d_target, d_error, scalar);
}

inline __device__ void deviceApplyPerceptron(float *a, int index, int nodeCount, activation perceptron){
    register float tmp = a[index];
    
    if(perceptron == SoftMax){
        register float sum = 0.0;
        for(register int i = 0; i < nodeCount; i++){
            sum += a[i];
        }

        tmp /= sum;
    }else{
        //ReLu branchless
        tmp *= !((perceptron == ReLu) && (tmp < 0.0));
        //Sigmoid branchless
        tmp = (perceptron != Sigmoid) * tmp + (perceptron == Sigmoid) / (1 + pow(M_E, -tmp));
    }

    a[index] = tmp;
}

inline __device__ void deviceEvalNeuralNetwork(float *input, float *network){
    int x = threadIdx.x;
    register float tmp = input[x];
    d_temp_array_1[x] = tmp;
    register int index = x;
    register int lastNodeCount;
    register int nodeCount = NODE_COUNTS[0];
    register int inRange = x < nodeCount;
    activation perceptron = ACTIVATIONS[0];
    #ifdef BACKPROP
    register int outIndex = 0;
    #endif
    
    if(perceptron == SoftMax){
        if(inRange) tmp = pow(M_E, tmp);
        __syncthreads();
    }

    if(inRange){
        deviceApplyPerceptron(d_temp_array_1, x, nodeCount, perceptron);
        #ifdef BACKPROP
        d_all_out[outIndex + x] = d_temp_array_1[x];
        #endif
    }

    #pragma unroll
    for(register int i = 1; i < LAYER_COUNT; i++){
        lastNodeCount = nodeCount;
        nodeCount = NODE_COUNTS[i];
        inRange = x < nodeCount;
        perceptron = ACTIVATIONS[i];
        __syncthreads();

        if(inRange){
            tmp = deviceVectorDotProduct(lastNodeCount, d_temp_array_1, &network[index], nodeCount) + network[index += nodeCount * lastNodeCount];
        }else{
            index += nodeCount * lastNodeCount;
        }

        index += nodeCount;

        if(inRange && perceptron == SoftMax) tmp = pow(M_E, tmp);
        
        d_output[x] = tmp;

        if(perceptron == SoftMax) __syncthreads();
        
        #ifdef BACKPROP
        outIndex += lastNodeCount;
        #endif

        if(inRange){
            deviceApplyPerceptron(d_output, x, nodeCount, perceptron);

            d_temp_array_1[x] = d_output[x];

            #ifdef BACKPROP
            d_all_out[outIndex + x] = d_temp_array_1[x];
            #endif
        }
    }

    d_output[x] = d_temp_array_1[x];
    __syncthreads();
}

__global__ void cudaEvalNeuralNetwork(float *input, float *network){
    deviceEvalNeuralNetwork(input, network);
}

__global__ void cudaCpyOut(float *out){
    int x = threadIdx.x;
    out[x] = d_output[x];
}

__global__ void cudaInc(float *a, float b){
    a[threadIdx.x] += b;
}

inline __device__ void devicePick(float *expected, int pick){
    int x = threadIdx.x;
    if(x == pick) expected[x] = 1.0;
    else expected[x] = 0.0;
}

__global__ void cudaPick(float *expected, int pick){
    devicePick(expected, pick);
    __syncthreads();
}

void pick(int pick){
    cudaPick<<<1, OUTPUT_SIZE>>>(d_expected, pick);
}

float *mallocNetwork(int *nodeCounts){
    int i, j, k, weights;
    
    float *network = (float *)malloc(NETWORK_SIZE * sizeof(float));

    i = 0;
    #pragma unroll
    for(j = 1; j < LAYER_COUNT; j++){
        weights = nodeCounts[j - 1] * nodeCounts[j];
        
        for(k = 0; k < weights; k++){
            //1.0 / (float)nodeCounts[j - 1];
            //rand() / (float)RAND_MAX;
            network[i] = rand() / (float)RAND_MAX;
            i++;
        }

        for(k = 0; k < nodeCounts[j]; k++){
            //0.0;
            //rand() / (float)RAND_MAX;
            network[i] = rand() / (float)RAND_MAX;
            i++;
        }
    }

    return network;
}

void initializeNetwork(float *h_network){
    cudaMalloc(&d_network, NETWORK_SIZE * sizeof(float));
    cudaMemcpy(d_network, h_network, NETWORK_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_error, NETWORK_SIZE * sizeof(float));
    cudaMalloc(&d_temp_error, NETWORK_SIZE * sizeof(float));
    cudaMalloc(&d_expected, OUTPUT_SIZE * sizeof(float));

    _h_temp = (float*)malloc(MAX_NODES * sizeof(float));

    cudaMalloc(&_d_temp, MAX_NODES * sizeof(float));

    // cudaZeroError<<<1, MAX_NODES>>>(d_error);
}

void setTrainingData(float *h_training_dataset, int *_training_dataset_labels, int _training_dataset_size){
    training_dataset_size = _training_dataset_size;

    cudaMalloc(&d_training_dataset, INPUT_SIZE * training_dataset_size * sizeof(float));
    cudaMemcpy(d_training_dataset, h_training_dataset, INPUT_SIZE * training_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_training_labels, training_dataset_size * sizeof(int));
    cudaMemcpy(d_training_labels, _training_dataset_labels, training_dataset_size * sizeof(int), cudaMemcpyHostToDevice);

    training_dataset_labels = (int*)malloc(training_dataset_size * sizeof(int));
    for(int i = 0; i < training_dataset_size; i++) training_dataset_labels[i] = _training_dataset_labels[i];
}

void freeTrainingData(){
    cudaFree(d_training_dataset);
    cudaFree(d_training_labels);
    free(training_dataset_labels);
}

void setTestingData(float *h_testing_dataset, int *_testing_dataset_labels, int _testing_dataset_size){
    testing_dataset_size = _testing_dataset_size;

    cudaMalloc(&d_testing_dataset, INPUT_SIZE * testing_dataset_size * sizeof(float));
    cudaMemcpy(d_testing_dataset, h_testing_dataset, INPUT_SIZE * testing_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_testing_labels, testing_dataset_size * sizeof(int));
    cudaMemcpy(d_testing_labels, _testing_dataset_labels, testing_dataset_size * sizeof(int), cudaMemcpyHostToDevice);

    testing_dataset_labels = (int*)malloc(testing_dataset_size * sizeof(int));
    for(int i = 0; i < testing_dataset_size; i++) testing_dataset_labels[i] = _testing_dataset_labels[i];
}

void freeTestingData(){
    cudaFree(d_testing_dataset);
    cudaFree(d_testing_labels);
    free(testing_dataset_labels);
}

inline void _swapPointers(float **a, float **b){
    float *tmp = *a;
    *a = *b;
    *b = tmp;
}

inline void _evalCudaNeuralNetwork(float *d_input){
    cudaEvalNeuralNetwork<<<1, MAX_NODES>>>(d_input, d_network);
    cudaDeviceSynchronize();
}

void evalCudaNeuralNetwork(float *d_input){
    _evalCudaNeuralNetwork(d_input);
}

void getNetwork(float *h_network){
    cudaMemcpy(h_network, d_network, NETWORK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}

void printLastOut(){
    cudaCpyOut<<<1, OUTPUT_SIZE>>>(_d_temp);
    cudaDeviceSynchronize();
    cudaMemcpy(_h_temp, _d_temp, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < OUTPUT_SIZE; i++){
        printf("out[%d]: %f\n", i, _h_temp[i]);
    }
}

__device__ void deviceTrainOnPoint(float *network, float *error, float *expected, float *datum, int label){
    deviceEvalNeuralNetwork(datum, network);
    if(threadIdx.x < OUTPUT_SIZE) devicePick(expected, label);
    __syncthreads();
    deviceBackPropError(network, error, expected);
    __syncthreads();
}

__global__ void cudaTrainOnPoint(float *network, float *error, float *expected, float *datum, int label){
    deviceTrainOnPoint(network, error, expected, datum, label);
}

__global__ void cudaTrain(float *network, float *error, float *expected, float *dataset, int *dataset_lables, int dataset_size){
    for(register int i = 0; i < dataset_size; i++) deviceTrainOnPoint(network, error, expected, &dataset[INPUT_SIZE * i], dataset_lables[i]);
    deviceApplyError(network, error, LEARNING_RATE);
    __syncthreads();
}

void trainOnBatch(int *batch, int batch_size){
    register int index = batch[0];
    cudaTrainOnPoint<<<1, MAX_NODES>>>(d_network, d_error, d_expected, &d_training_dataset[INPUT_SIZE * index], training_dataset_labels[index]);
    for(register int i = 1; i < batch_size; i++){
        index = batch[i];
        cudaTrainOnPoint<<<1, MAX_NODES>>>(d_network, d_temp_error, d_expected, &d_training_dataset[INPUT_SIZE * index], training_dataset_labels[index]);
        _applyError(d_error, d_temp_error, -1.0);
    }
    _applyError(d_network, d_error, LEARNING_RATE);
}

void train(){
    cudaTrain<<<1, MAX_NODES>>>(d_network, d_error, d_expected, d_training_dataset, d_training_labels, training_dataset_size);
}

float accuracy(){
    int accuracy = 0;
    register int i, j, pick;

    for(i = 0; i < testing_dataset_size; i++){
        _evalCudaNeuralNetwork(&d_testing_dataset[i * INPUT_SIZE]);
        cudaCpyOut<<<1, OUTPUT_SIZE>>>(_d_temp);
        cudaDeviceSynchronize();
        cudaMemcpy(_h_temp, _d_temp, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        pick = 0;
        for(j = 1; j < OUTPUT_SIZE; j++){
            if(_h_temp[j] > _h_temp[pick]) pick = j;
        }
        accuracy += (pick == testing_dataset_labels[i]);
    }

    return accuracy / (float)testing_dataset_size;
}

float loss(){
    float tmp, loss = 0.0;
    register int i, j;

    for(i = 0; i < testing_dataset_size; i++){
        _evalCudaNeuralNetwork(&d_testing_dataset[i * INPUT_SIZE]);
        cudaCpyOut<<<1, OUTPUT_SIZE>>>(_d_temp);
        cudaDeviceSynchronize();
        cudaMemcpy(_h_temp, _d_temp, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        for(j = 0; j < OUTPUT_SIZE; j++){
            tmp = (float)(j == testing_dataset_labels[i]) - _h_temp[j];
            loss += tmp * tmp;
        }
    }

    return loss / (float)testing_dataset_size;
}