
typedef struct {
    float *images;          // n_images * 784, normalized [0,1]
    unsigned char *labels;  // n_images, integer labels 0-9 each ranging 0 - 255
    int n_images;
    int n_rows;
    int n_cols;
} MNISTData;

MNISTData load_mnist(const char *image_path, const char *label_path);
void free_mnist(MNISTData *data);