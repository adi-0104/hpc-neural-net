#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h>
#include "mnist.h"
// parse the data from folder
MNISTData load_mnist(const char *img_path, const char *label_path)
{
    FILE *f_img = fopen(img_path, "rb");
    FILE *f_label = fopen(label_path, "rb");

    if (!f_img || !f_label)
        return (MNISTData){0};

    // MNISTData *data = malloc(sizeof(MNISTData));
    MNISTData data;
    uint32_t magic, n_imgs, rows, cols, n_labels;

    // process Image dataset
    fread(&magic, 4, 1, f_img);
    fread(&n_imgs, 4, 1, f_img);
    fread(&rows, 4, 1, f_img);
    fread(&cols, 4, 1, f_img);

    data.n_images = ntohl(n_imgs);
    data.n_rows = ntohl(rows);
    data.n_cols = ntohl(cols);

    int pixels_per_image = data.n_cols * data.n_rows;
    data.images = malloc(sizeof(float) * data.n_images * pixels_per_image);

    // read pixel data for each image
    // create buffer for image pixels
    unsigned char *pix_buff = malloc(pixels_per_image);
    for (int i = 0; i < data.n_images; i++)
    {
        fread(pix_buff, 1, pixels_per_image, f_img);
        for (int p = 0; p < pixels_per_image; p++)
        {
            data.images[i * pixels_per_image + p] = (float)pix_buff[p] / 255.0f;
        }
    }

    // process labels
    fread(&magic, 4, 1, f_label);
    fread(&n_labels, 4, 1, f_label);

    data.labels = malloc(sizeof(unsigned char) * data.n_images);
    fread(data.labels, 1, data.n_images, f_label);

    // free
    free(pix_buff);
    fclose(f_img);
    fclose(f_label);

    return data;
}

void free_mnist(MNISTData *data)
{
    if (data == NULL)
        return;

    if (data->images)
        free(data->images);
    if (data->labels)
        free(data->labels);
}
