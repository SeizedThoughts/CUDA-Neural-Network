#include <stdio.h>
#include <stdlib.h>

#define BACKPROP

#include <cuda-neural-network.h>

#define USE_MNIST_LOADER

#include "mnist.h"

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

void printImg(float *img){
    for(int i = 0; i < 28; i++){
        for(int j = 0; j < 28; j++){
            if(img[i * 28 + j] > 0.5) printf("&");
            else printf(".");
        }
        printf("\n");
    }
}

void shuffle(int *array, int size){
    int tmp, j;
    for(int i = size - 1; i >= 1; i--){
        j = rand() % i;
        tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

int *makeBatches(int data_set_size){
    int *batches = (int*)malloc(data_set_size * sizeof(int));

    for(int i = 0; i < data_set_size; i++){
        batches[i] = i;
    }

    shuffle(batches, data_set_size);

    return batches;
}

int main(void){
    srand(0);
    const char *networkFile = "tests/data/network.txt";
    
    float *network;
    int layerCount = 4;
    int nodeCounts[layerCount] = {784, 128, 64, 10};
    // int i, j;
    
    FILE *f;

    f = fopen(networkFile, "a+");

    // if(getc(f) != EOF){
    //     networkFromFile(networkFile, &network);
    // }else{
        network = mallocNetwork(nodeCounts);
    // }

    fclose(f);
    
    initializeNetwork(network);

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

    // float trainingSpeed = 0.01;

    /*
        100,000 nn evals (10 trials)
        model:
        784, 128, 128, 10
        None, SoftMax, SoftMax, SoftMax
        loop on host time: 33.7
        loop on device time: 32.5
        branchless Sigmoid & branchless ReLu 32.4

        1,000,000 nn evals
        model:
        784, 128, 128, 10
        None, ReLu, Sigmoid, SoftMax
        applyPerceptrons branchful: 306
        applyPerceptrons branchless ReLu: 307
        applyPerceptrons branchless ReLu & branchless Sigmoid:

        100,000 nn evals
        model:
        784, 128, 128, 10
        None, ReLu, Sigmoid, SoftMax
        deviceVectorDotProduct without tmp var in register: ~30
        deviceVectorDotProduct with tmp var in register: ~15
    */

    long int start;
    int batch_size = 500;
    int *batches = makeBatches(trainingCount);
    for(int e = 0; e < 10000; e++){
        start = time(NULL);
        for(int b = 0; b < trainingCount / batch_size; b++){
            trainOnBatch(&batches[b * batch_size], batch_size);
        }
        printf("Time: %d\n", (int)(time(NULL) - start));

        // printf("Number done: %d\n", trainingCount);
        // printf("Img: %d\n", batches[trainingCount - 1]);
        // printf("Expected: %d\n", training_dataset_labels[batches[trainingCount - 1]]);
        // printLastOut();
        // printImg(&trainingImgs[batches[trainingCount - 1] * INPUT_SIZE]);

        printf("Epoch: %d\n", e);
        printf("Loss: %f\n", loss());
        printf("Accuracy: %f\n", accuracy());


        getNetwork(network);
        writeNetworkToFile(networkFile, network, NETWORK_SIZE);
        batches = makeBatches(trainingCount);
    }

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

        // getNetwork(network);
        // writeNetworkToFile(networkFile, network, NETWORK_SIZE);
    //     printf("    Post-Epoch Accuracy: %f%%\n\n", 100.0 * accuracy());
    // }

    getNetwork(network);
    writeNetworkToFile(networkFile, network, NETWORK_SIZE);

    free(network);

    return 0;
}