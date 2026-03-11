
typedef struct
{
    float *images;
    unsigned char *labels;
    int n_images;
    int n_rows;
    int n_cols;
} MNISTData;

#ifdef __cplusplus
extern "C" {
#endif

MNISTData load_mnist(const char *image_path, const char *label_path);
void free_mnist(MNISTData *data);

#ifdef __cplusplus
}
#endif
