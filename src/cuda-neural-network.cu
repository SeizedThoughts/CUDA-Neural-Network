#include <stdio.h>
#include <math.h>
#include <cuda-neural-network.h>

float *d_network, *d_training_dataset, *d_testing_dataset, *d_output, *_h_temp, *_d_temp;
int *_node_counts, *d_node_counts, *training_dataset_labels, training_dataset_size, *testing_dataset_labels, testing_dataset_size, _max_nodes, _network_size, _datum_size, _output_node_count;
activation *_activations, *d_activations;

#define BLOCK_SIZE 1024
#define LAYER_COUNT 4

__device__ float temp_array[BLOCK_SIZE];
__device__ float temp;

inline __device__ void deviceVectorDotProduct(int vector_size, float *a, float *b, int aInc, int bInc, float *c, int inRange){
    (*c) = 0.0;
    for(int i = 0; i < vector_size * inRange; i++){
        (*c) += a[aInc * i] * b[bInc * i];
    }
}

inline __device__ void deviceApplyPerceptron(float *a, int index, int nodeCount, activation perceptron){
    float *point = &a[index];
    if(perceptron == SoftMax){
        (*point) = pow(M_E, (*point));

        atomicAdd(&temp, *point);

        (*point) /= temp;
    }else{
        //ReLu branchless
        (*point) *= !((perceptron == ReLu) && ((*point) < 0.0));
        //Sigmoid branchless
        (*point) = (perceptron != Sigmoid) * (*point) + (perceptron == Sigmoid) * (*point);
    }
}

inline __device__ void deviceSwapPtrs(float **ptr1, float **ptr2){
    float *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

inline __device__ void deviceEvalNeuralNetwork(float *input, float *network, int *nodeCounts, activation *perceptrons, float *output, int maxNodes){
    int x = threadIdx.x;
    int index = x;
    int lastNodeCount;
    int nodeCount = nodeCounts[0];
    int inRange = x < nodeCount;
    activation perceptron = perceptrons[0];

    if(inRange) deviceApplyPerceptron(output, x, nodeCount, perceptron);

    #pragma unroll
    for(int i = 1; i < LAYER_COUNT; i++){
        temp = 0.0;
        lastNodeCount = nodeCount;
        nodeCount = nodeCounts[i];
        inRange = x < nodeCount;
        perceptron = perceptrons[i];

        deviceVectorDotProduct(lastNodeCount, input, &network[index], 1, nodeCount, &output[x], inRange);

        __syncthreads();

        index += nodeCount * lastNodeCount;

        output[x] += inRange * network[index];

        __syncthreads();

        index += nodeCount;

        if(inRange) deviceApplyPerceptron(output, x, nodeCount, perceptron);

        __syncthreads();

        deviceSwapPtrs(&input, &output);
    }

    if(x < nodeCounts[LAYER_COUNT - 1] && LAYER_COUNT % 2 == 1) output[x] = input[x];
}

__global__ void cudaEvalNeuralNetwork(float *input, float *network, int *nodeCounts, activation *perceptrons, float *output, int maxNodes){
    int x = threadIdx.x;
    for(int i = 0; i < 100000; i++){
        temp_array[x] = input[x];
        __syncthreads();
        deviceEvalNeuralNetwork(temp_array, network, nodeCounts, perceptrons, output, maxNodes);
        __syncthreads();
    }
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
    _node_counts = (int*)malloc(LAYER_COUNT * sizeof(int));
    _activations = (activation*)malloc(LAYER_COUNT * sizeof(activation));
    
    #pragma unroll
    for(i = 0; i < LAYER_COUNT; i++){
        _node_counts[i] = node_counts[i];
        _activations[i] = activations[i];
    }

    cudaMalloc(&d_node_counts, LAYER_COUNT * sizeof(int));
    cudaMemcpy(d_node_counts, _node_counts, LAYER_COUNT * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_activations, LAYER_COUNT * sizeof(activation));
    cudaMemcpy(d_activations, _activations, LAYER_COUNT * sizeof(activation), cudaMemcpyHostToDevice);
    
    _network_size = 0;
    _datum_size = _node_counts[0];
    _output_node_count = _node_counts[LAYER_COUNT - 1];
    _max_nodes = _node_counts[0];

    #pragma unroll
    for(i = 1; i < LAYER_COUNT; i++){
        _network_size += _node_counts[i] * (_node_counts[i - 1] + 1);
        if(_node_counts[i] > _max_nodes) _max_nodes = _node_counts[i];
    }

    cudaMalloc(&d_network, _network_size * sizeof(float));
    cudaMemcpy(d_network, h_network, _network_size * sizeof(float), cudaMemcpyHostToDevice);

    _h_temp = (float*)malloc(_max_nodes * sizeof(float));

    cudaMalloc(&_d_temp, _max_nodes * sizeof(float));
    cudaMalloc(&d_output, _max_nodes * sizeof(float));
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
    cudaEvalNeuralNetwork<<<1, BLOCK_SIZE>>>(d_input, d_network, d_node_counts, d_activations, d_output, _max_nodes);
    cudaDeviceSynchronize();
}

void printLastOut(){
    cudaMemcpy(_h_temp, d_output, _output_node_count * sizeof(float), cudaMemcpyDeviceToHost);
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

float accuracy(){
    int correct = 0;
    float v;
    int j, prediction;
    float *h_output = (float*)malloc(_output_node_count * sizeof(float));

    for(int i = 0; i < testing_dataset_size; i++){
        _evalCudaNeuralNetwork(&d_testing_dataset[_datum_size * i]);
        cudaMemcpy(h_output, d_output, _output_node_count * sizeof(float), cudaMemcpyDeviceToHost);
        
        v = h_output[0];
        prediction = 0;

        for(j = 1; j < _node_counts[LAYER_COUNT - 1]; j++){
            if(h_output[j] > v){
                v = h_output[j];
                prediction = j;
            }
        }

        if(prediction == testing_dataset_labels[i]) correct++;
    }

    return (float)correct / (float)testing_dataset_size;
}

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