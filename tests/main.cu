#include <stdio.h>
#include <stdlib.h>

#include <cuda-neural-network.h>

#define USE_MNIST_LOADER

#include "mnist.h"

void makeBatch(int dataSetSize, int *batch, int batchSize){
    for(int i = 0; i < batchSize; i++){
        batch[i] = rand() % dataSetSize;

        for(int j = 0; j < i; j++){
            if(batch[j] == batch[i]){
                i--;
                break;
            }
        }

        //todo: remove stuff at the top of the rand range
    }
}

void load(const char *imagePath, const char *labelPath, mnist_data **mnistData, float **data, int **labels, unsigned int *count){
    mnist_load(imagePath, labelPath, mnistData, count);

    *data = (float*)malloc(784 * (*count) * sizeof(float*));
    *labels = (int*)malloc((*count) * sizeof(int));

    for(int i = 0; i < *count; i++){
        for(int j = 0; j < 28; j++){
            for(int k = 0; k < 28; k++){
                (*data)[784 * i + 28 * j + k] = (*mnistData)[i].data[j][k];
            }
        }
        (*labels)[i] = (*mnistData)[i].label;
    }
}

int networkFromFile(const char *file, float **network){
    FILE *f = fopen(file, "r");

    int networkSize = 0;
    char c;

    while((c = getc(f)) != EOF){
        if(c == ' ') networkSize++;
    }

    rewind(f);

    *network = (float*)malloc(networkSize * sizeof(float));

    for(int i = 0; i < networkSize; i++){
        fscanf(f, "%f ", &(*network)[i]);
    }

    fclose(f);

    return networkSize;
}

void writeNetworkToFile(const char *file, float *network, int networkSize){
    FILE *f = fopen(file, "w");

    for(int i = 0; i < networkSize; i++){
        fprintf(f, "%f ", network[i]);
    }

    fclose(f);
}

int main(void){
    srand(0);
    const char *networkFile = "tests/data/network.txt";
    
    float *network;
    int layerCount = 4;
    enum activation activations[layerCount] = {None, SoftMax, SoftMax, SoftMax};
    int nodeCounts[layerCount] = {784, 128, 128, 10};
    // int networkSize, i, j;
    
    FILE *f;

    f = fopen(networkFile, "a+");

    if(getc(f) != EOF){
        // networkSize = 
        networkFromFile(networkFile, &network);
    }else{
        // networkSize = 
        mallocNetwork(nodeCounts, &network);
    }

    // printf("Network Size: %d\n", networkSize);

    fclose(f);
    
    initializeNetwork(network, activations, nodeCounts);

    mnist_data *training_mnist;
    unsigned int trainingCount;
    float *trainingImgs;
    int *trainingLabels;
    load("tests/data/train-images-idx3-ubyte", "tests/data/train-labels-idx1-ubyte", &training_mnist, &trainingImgs, &trainingLabels, &trainingCount);

    mnist_data *testing_mnist;
    unsigned int testingCount;
    float *testingImgs;
    int *testingLabels;
    load("tests/data/t10k-images-idx3-ubyte", "tests/data/t10k-labels-idx1-ubyte", &testing_mnist, &testingImgs, &testingLabels, &testingCount);

    setTrainingData(trainingImgs, trainingLabels, trainingCount);
    setTestingData(testingImgs, testingLabels, testingCount);

    int batchSize = 1;
    int *batch = (int*)malloc(batchSize * sizeof(int));
    // float trainingSpeed = 0.01;

    /*
        100,000 nn evals (10 trials)
        model:
        784, 128, 128, 10
        None, SoftMax, SoftMax, SoftMax
        loop on host time: 33.7
        loop on device time: 32.5
        branchless Sigmoid & branchless ReLu 32.4
        SoftMax with atomic add 31.2
    */
    long int start;
    for(int i = 0; i < 10; i++){
        printf("Test: %d\n", i + 1);
        start = time(NULL);
        evalCudaNeuralNetwork(d_training_dataset);
        printf("    Time: %ld\n", time(NULL) - start);
    }
    printLastOut();

    // for(int ep = 0; ep < 100; ep++){
    //     printf("Epoch: %d\n", ep + 1);
    //     printf("    Pre-Epoch Accuracy: %f%%\n", 100.0 * accuracy());
    //     for(i = 0; i < training_dataset_size / batchSize; i++){
    //         for(j = 0; j < batchSize; j++){
    //             batch[j] = batchSize * i + j;
    //         }

    //         trainCuda(batch, batchSize, trainingSpeed);
    //         //printf("        Round %d complete.\n", i + 1);
    //     }

    //     getNetwork(network);
    //     writeNetworkToFile(networkFile, network, networkSize);
    //     printf("    Post-Epoch Accuracy: %f%%\n\n", 100.0 * accuracy());
    // }

    free(network);

    return 0;
}