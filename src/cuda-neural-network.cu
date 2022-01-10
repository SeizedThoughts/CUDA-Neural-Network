#include <stdio.h>
#include <math.h>
#include <cuda-neural-network.h>

float *d_network, *d_training_dataset, *d_testing_dataset, *_h_temp, *_d_temp;
int *training_dataset_labels, training_dataset_size, *testing_dataset_labels, testing_dataset_size, _network_size, _datum_size, _output_node_count;

#define MAX_NODES 784
#define LAYER_COUNT 4
#define NETWORK_SIZE = 784 * 128 + 128 * 128 + 128 * 10
__device__ const int NODE_COUNTS[LAYER_COUNT] = {784, 128, 128, 10};
__device__ const activation ACTIVATIONS[LAYER_COUNT] = {None, ReLu, Sigmoid, SoftMax};

__device__ float d_temp, d_temp_array_1[MAX_NODES], d_temp_array_2[MAX_NODES], d_output[MAX_NODES];

inline __device__ void deviceVectorDotProduct(int vector_size, float *a, float *b, int bInc, float *c){
    register float tmp = 0.0;
    
    for(int i = 0; i < vector_size; i++){
        tmp += a[i] * b[i * bInc];
    }
    
    (*c) = tmp;
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
    d_temp_array_1[x] = input[x];
    __syncthreads();
    int index = x;
    int lastNodeCount;
    int nodeCount = NODE_COUNTS[0];
    int inRange = x < nodeCount;
    activation perceptron = ACTIVATIONS[0];

    if(inRange) deviceApplyPerceptron(d_output, x, nodeCount, perceptron);

    #pragma unroll
    for(int i = 1; i < LAYER_COUNT; i++){
        lastNodeCount = nodeCount;
        nodeCount = NODE_COUNTS[i];
        inRange = x < nodeCount;
        perceptron = ACTIVATIONS[i];
        __syncthreads();

        if(inRange) deviceVectorDotProduct(lastNodeCount, d_temp_array_1, &network[index], nodeCount, &d_output[x]);

        index += nodeCount * lastNodeCount;

        d_output[x] += inRange * network[index];

        index += nodeCount;

        if(perceptron == SoftMax){
            if(inRange) d_output[x] = pow(M_E, d_output[x]);
            __syncthreads();
        }

        if(inRange) deviceApplyPerceptron(d_output, x, nodeCount, perceptron);

        d_temp_array_1[x] = d_output[x];
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

int mallocNetwork(int *nodeCounts, float **network){
    int i, j, k, weights;
    int networkSize = 0;

    #pragma unroll
    for(i = 1; i < LAYER_COUNT; i++){
        networkSize += nodeCounts[i] * (nodeCounts[i - 1] + 1);
    }

    *network = (float *)malloc(networkSize * sizeof(float));

    i = 0;
    #pragma unroll
    for(j = 1; j < LAYER_COUNT; j++){
        weights = nodeCounts[j - 1] * nodeCounts[j];
        
        for(k = 0; k < weights; k++){
            (*network)[i] = 1.0 / (float)nodeCounts[j - 1];
            i++;
        }

        for(k = 0; k < nodeCounts[j]; k++){
            (*network)[i] = 0.0;
            i++;
        }
    }

    return networkSize;
}

void initializeNetwork(float *h_network, activation *activations, int *node_counts){
    int i;

    _network_size = 0;
    _datum_size = node_counts[0];
    _output_node_count = node_counts[LAYER_COUNT - 1];

    #pragma unroll
    for(i = 1; i < LAYER_COUNT; i++){
        _network_size += node_counts[i] * (node_counts[i - 1] + 1);
    }

    cudaMalloc(&d_network, _network_size * sizeof(float));
    cudaMemcpy(d_network, h_network, _network_size * sizeof(float), cudaMemcpyHostToDevice);

    _h_temp = (float*)malloc(MAX_NODES * sizeof(float));

    cudaMalloc(&_d_temp, MAX_NODES * sizeof(float));
}

void getNetwork(float *h_network){
    cudaMemcpy(h_network, d_network, _network_size * sizeof(float), cudaMemcpyDeviceToHost);
}

void setTrainingData(float *h_training_dataset, int *_training_dataset_labels, int _training_dataset_size){
    training_dataset_size = _training_dataset_size;

    cudaMalloc(&d_training_dataset, _datum_size * training_dataset_size * sizeof(float));
    cudaMemcpy(d_training_dataset, h_training_dataset, _datum_size * training_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
    training_dataset_labels = (int*)malloc(training_dataset_size * sizeof(int));
    for(int i = 0; i < training_dataset_size; i++) training_dataset_labels[i] = _training_dataset_labels[i];
}

void freeTrainingData(){
    cudaFree(d_training_dataset);
    free(training_dataset_labels);
}

void setTestingData(float *h_testing_dataset, int *_testing_dataset_labels, int _testing_dataset_size){
    testing_dataset_size = _testing_dataset_size;

    cudaMalloc(&d_testing_dataset, _datum_size * testing_dataset_size * sizeof(float));
    cudaMemcpy(d_testing_dataset, h_testing_dataset, _datum_size * testing_dataset_size * sizeof(float), cudaMemcpyHostToDevice);
    
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
    cudaMemcpy(_h_temp, _d_temp, _output_node_count * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i = 0; i < _output_node_count; i++){
        printf("out[%d]: %f\n", i, _h_temp[i]);
    }
}

void evalCudaNeuralNetwork(float *d_input){
    _evalCudaNeuralNetwork(d_input);
}

inline float _lossCuda(float *d_dataset, int *dataset_labels, int datum_index){
    _evalCudaNeuralNetwork(&d_dataset[_datum_size * datum_index]);
    cudaMemcpy(_h_temp, d_output, _output_node_count * sizeof(float), cudaMemcpyDeviceToHost);

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
        point = rand() % _network_size;
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
//     float *h_output = (float*)malloc(_output_node_count * sizeof(float));

//     for(int i = 0; i < testing_dataset_size; i++){
//         _evalCudaNeuralNetwork(&d_testing_dataset[_datum_size * i]);
//         cudaMemcpy(h_output, d_output, _output_node_count * sizeof(float), cudaMemcpyDeviceToHost);
        
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

    for(int i = 0; i < _output_node_count; i++){
        total += output[i];
        if(output[i] > output[*prediction]) (*prediction) = i;
    }

    float lowerBound = total / (float)_output_node_count;

    return (output[*prediction] - lowerBound) / (total - lowerBound);
}