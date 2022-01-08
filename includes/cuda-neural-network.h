#define DELTA_WEIGHT 0.01

#define STOCHASTIC_POINTS 1000
#define STOCHASTIC_STEPS 10

enum activation {
    None,
    Sigmoid,
    SoftMax,
    ReLu
};

extern float *d_network, *d_training_dataset, *d_testing_dataset, *d_output, *_h_temp, *_d_temp;
extern int *_node_counts, _layer_count, *training_dataset_labels, training_dataset_size, *testing_dataset_labels, testing_dataset_size, _max_nodes, _network_size, _datum_size, _output_node_count;
extern activation *_activations;

extern __global__ void cudaInc(float *a, float b);

extern int mallocNetwork(int *nodeCounts, int layerCount, float **network);
extern void initializeNetwork(float *h_network, activation *activations, int *node_counts, int layer_count);
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