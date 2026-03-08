
typedef struct
{
    float *images;
    unsigned char *labels;
    int n_images;
    int n_rows;
    int n_cols;
} MNISTData;

MNISTData load_mnist(const char *image_path, const char *label_path);
void free_mnist(MNISTData *data);
