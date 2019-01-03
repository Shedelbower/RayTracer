#ifndef IMAGE_H
#define IMAGE_H

#include <math.h>
#include "RGBColor.cuh"
#include "Vector3.cuh"
#include "BVHNode.cuh"
#include "Primitive.cuh"
#include "Raycast.cuh"

#define COLOR_CHANNELS 3
#define PI 3.14159265358979323846

class Image {

public:

    float widthf, heightf;

    __device__ int getWidth() {
        return (int) this->widthf;
    }

    __device__ int getHeight() {
        return (int) this->heightf;
    }

    __device__ int size() {
        return 2 + this->getWidth() * this->getHeight() * COLOR_CHANNELS;
    }

    __device__ Image * getNext() {
        float * ptr = (float*) this;
        ptr += this->size();
        return (Image*)ptr;
    }

    __device__ RGBColor * getFirstPixel() {
        float * ptr = (float*) this;
        ptr += 2;
        return (RGBColor*)ptr;
    }

    __device__ RGBColor getPixel(int x, int y) {
        int width = this->getWidth();
        int i = (y * width + x);
        RGBColor * start = this->getFirstPixel();
        return *(start + i);
    }

    __device__ RGBColor sample(float u, float v) {
        int x = (int) (u * this->widthf);
        int y = (int) (v * this->heightf);
        return getPixel(x,y);
    }

    __device__ RGBColor sample(Vector3 unitVector) {
        float u = 0.5 + atan2f(unitVector.z, unitVector.x) / (2*PI);
        float v = 0.5 - asinf(unitVector.y) / PI;
        //return RGBColor(u,v,0.0); //TODO: REMOVE
        return this->sample(u, v);
    }
};



#endif // IMAGE_H