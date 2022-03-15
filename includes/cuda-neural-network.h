#define LEARNING_RATE 0.001

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10
#define MAX_NODES 784
#define LAYER_COUNT 4
#define NETWORK_SIZE ((784 + 1) * 128 + (128 + 1) * 64 + (64 + 1) * 10)
#define ALL_OUT_SIZE (784 + 128 + 64 + 10)

// #define INPUT_SIZE 2
// #define OUTPUT_SIZE 10
// #define MAX_NODES 10
// #define LAYER_COUNT 4
// #define NETWORK_SIZE ((2 + 1) * 4 + (4 + 1) * 6 + (6 + 1) * 10)
// #define ALL_OUT_SIZE (2 + 4 + 6 + 10)

enum activation {
    None,
    Sigmoid,
    SoftMax,
    ReLu
};

extern float *d_network, *d_training_dataset, *d_testing_dataset, *_h_temp, *_d_temp;
extern int *_node_counts, *training_dataset_labels, training_dataset_size, *testing_dataset_labels, testing_dataset_size, _max_nodes, _network_size, _output_node_count;

extern __global__ void cudaInc(float *a, float b);

extern float *mallocNetwork(int *nodeCounts);
extern void initializeNetwork(float *h_network);
extern void getNetwork(float *h_network);
extern void setTrainingData(float *h_training_dataset, int *_training_dataset_labels, int _training_dataset_size);
extern void freeTrainingData();
extern void setTestingData(float *h_testing_dataset, int *_testing_dataset_labels, int _testing_dataset_size);
extern void freeTestingData();
extern void evalCudaNeuralNetwork(float *d_input);
extern float lossCuda(float *d_dataset, int *dataset_labels, int datum_index);
extern float _batchLossCuda(float *d_dataset, int *dataset_labels, int *batch, int batchSize);
extern void trainCuda(int *batch, int batchSize, float trainingSpeed);
extern float accuracy();
extern float confidenceRating(float *output, int *prediction);
extern void printLastOut();
extern void trainOnBatch(int *batch, int batch_size);
extern void train();
extern float loss();