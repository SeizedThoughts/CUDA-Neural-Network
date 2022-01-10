#include <stdio.h>
#include <math.h>
#include <cuda-neural-network.h>

float *d_network, *d_training_dataset, *d_testing_dataset, *_h_temp, *_d_temp;
int *training_dataset_labels, training_dataset_size, *testing_dataset_labels, testing_dataset_size;

__device__ const int NODE_COUNTS[LAYER_COUNT] = {784, 128, 128, 10};
__device__ const activation ACTIVATIONS[LAYER_COUNT] = {None, ReLu, Sigmoid, SoftMax};

__device__ float d_temp, d_temp_array_1[MAX_NODES], d_temp_array_2[MAX_NODES], d_output[MAX_NODES];

inline __device__ float deviceVectorDotProduct(int vector_size, float *a, float *b, int bInc){
    register float tmp = 0.0;
    register int iB = 0;
    
    // #pragma unroll
    for(int i = 0; i < vector_size; i++){
        tmp += a[i] * b[iB];
        iB += bInc;
    }
    
    return tmp;
}

inline __device__ void deviceApplyPerceptron(float *a, int index, int nodeCount, activation perceptron){
    register float tmp = a[index];
    
    if(perceptron == SoftMax){
        float sum = 0.0;
        for(int i = 0; i < nodeCount; i++){
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
    int index = x;
    int lastNodeCount;
    int nodeCount = NODE_COUNTS[0];
    int inRange = x < nodeCount;
    activation perceptron = ACTIVATIONS[0];
    
    if(perceptron == SoftMax){
        if(inRange) tmp = pow(M_E, tmp);
        __syncthreads();
    }

    if(inRange) deviceApplyPerceptron(d_output, x, nodeCount, perceptron);

    #pragma unroll
    for(int i = 1; i < LAYER_COUNT; i++){
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

        if(perceptron == SoftMax){
            if(inRange) tmp = pow(M_E, tmp);
            __syncthreads();
        }

        if(inRange){
            d_output[x] = tmp;

            deviceApplyPerceptron(d_output, x, nodeCount, perceptron);

            d_temp_array_1[x] = d_output[x];
        }
    }

    d_output[x] = d_temp_array_1[x];
    __syncthreads();
}

__global__ void cudaEvalNeuralNetwork(float *input, float *network){
    for(int i = 0; i < 100000; i++){
        deviceEvalNeuralNetwork(input, network);
    }
}

__global__ void cudaCpyOut(float *out){
    int x = threadIdx.x;
    out[x] = d_output[x];
}

__global__ void cudaInc(float *a, float b){
    a[threadIdx.x] += b;
}

float *mallocNetwork(int *nodeCounts){
    int i, j, k, weights;
    
    float *network = (float *)malloc(NETWORK_SIZE * sizeof(float));

    i = 0;
    #pragma unroll
    for(j = 1; j < LAYER_COUNT; j++){
        weights = nodeCounts[j - 1] * nodeCounts[j];
        
        for(k = 0; k < weights; k++){
            network[i] = 1.0 / (float)nodeCounts[j - 1];
            i++;
        }

        for(k = 0; k < nodeCounts[j]; k++){
            network[i] = 0.0;
            i++;
        }
    }

    return network;
}

void initializeNetwork(float *h_network){
    cudaMalloc(&d_network, NETWORK_SIZE * sizeof(float));
    cudaMemcpy(d_network, h_network, NETWORK_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    _h_temp = (float*)malloc(MAX_NODES * sizeof(float));

    cudaMalloc(&_d_temp, MAX_NODES * sizeof(float));
}

void getNetwork(float *h_network){
    cudaMemcpy(h_network, d_network, NETWORK_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}

void setTrainingData(float *h_training_dataset, int *_training_dataset_labels, int _training_dataset_size){
    training_dataset_size = _training_dataset_size;

    cudaMalloc(&d_training_dataset, DATUM_SIZE * training_dataset_size * sizeof(float));
    cudaMemcpy(d_training_dataset, h_training_dataset, DATUM_SIZE * training_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
    training_dataset_labels = (int*)malloc(training_dataset_size * sizeof(int));
    for(int i = 0; i < training_dataset_size; i++) training_dataset_labels[i] = _training_dataset_labels[i];
}

void freeTrainingData(){
    cudaFree(d_training_dataset);
    free(training_dataset_labels);
}

void setTestingData(float *h_testing_dataset, int *_testing_dataset_labels, int _testing_dataset_size){
    testing_dataset_size = _testing_dataset_size;

    cudaMalloc(&d_testing_dataset, DATUM_SIZE * testing_dataset_size * sizeof(float));
    cudaMemcpy(d_testing_dataset, h_testing_dataset, DATUM_SIZE * testing_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
    testing_dataset_labels = (int*)malloc(testing_dataset_size * sizeof(int));
    for(int i = 0; i < testing_dataset_size; i++) testing_dataset_labels[i] = _testing_dataset_labels[i];
}

void freeTestingData(){
    cudaFree(d_testing_dataset);
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

void printLastOut(){
    cudaCpyOut<<<1, MAX_NODES>>>(_d_temp);
    cudaDeviceSynchronize();
    cudaMemcpy(_h_temp, _d_temp, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < OUTPUT_SIZE; i++){
        printf("out[%d]: %f\n", i, _h_temp[i]);
    }
}

void evalCudaNeuralNetwork(float *d_input){
    _evalCudaNeuralNetwork(d_input);
}

inline float _lossCuda(float *d_dataset, int *dataset_labels, int datum_index){
    _evalCudaNeuralNetwork(&d_dataset[DATUM_SIZE * datum_index]);
    cudaMemcpy(_h_temp, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return 1.0 - _h_temp[dataset_labels[datum_index]];
}

float lossCuda(float *d_dataset, int *dataset_labels, int datum_index){
    return _lossCuda(d_dataset, dataset_labels, datum_index);
}

float _batchLossCuda(float *d_dataset, int *dataset_labels, int *batch, int batchSize){
    float loss = 0.0;

    for(int i = 0; i < batchSize; i++){
        loss += _lossCuda(d_dataset, dataset_labels, batch[i]);
    }

    return loss / (float)batchSize;
}

void trainCuda(int *batch, int batchSize, float trainingSpeed){
    int point;
    float deltaLoss, gradient;
    float currentLoss = _batchLossCuda(d_training_dataset, training_dataset_labels, batch, batchSize);

    for(int i = 0; i < STOCHASTIC_POINTS; i++){
        point = rand() % NETWORK_SIZE;
        for(int j = 0; j < STOCHASTIC_STEPS; j++){
            cudaInc<<<1, 1>>>(&d_network[point], DELTA_WEIGHT);
            deltaLoss = _batchLossCuda(d_training_dataset, training_dataset_labels, batch, batchSize) - currentLoss;
            gradient = deltaLoss / DELTA_WEIGHT;
            if(fabs(gradient) < 0.001) break;
            cudaInc<<<1, 1>>>(&d_network[point], -(gradient * trainingSpeed + DELTA_WEIGHT));
        }
    }
}

// float accuracy(){
//     int correct = 0;
//     float v;
//     int j, prediction;
//     float *h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));

//     for(int i = 0; i < testing_dataset_size; i++){
//         _evalCudaNeuralNetwork(&d_testing_dataset[DATUM_SIZE * i]);
//         cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
//         v = h_output[0];
//         prediction = 0;

//         for(j = 1; j < node_counts[LAYER_COUNT - 1]; j++){
//             if(h_output[j] > v){
//                 v = h_output[j];
//                 prediction = j;
//             }
//         }

//         if(prediction == testing_dataset_labels[i]) correct++;
//     }

//     return (float)correct / (float)testing_dataset_size;
// }

float confidenceRating(float *output, int *prediction){
    (*prediction) = 0;
    float total = 0.0;

    for(int i = 0; i < OUTPUT_SIZE; i++){
        total += output[i];
        if(output[i] > output[*prediction]) (*prediction) = i;
    }

    float lowerBound = total / (float)OUTPUT_SIZE;

    return (output[*prediction] - lowerBound) / (total - lowerBound);
}