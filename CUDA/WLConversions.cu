// File for converting between the WL representation of objects to the C++ classes.

#include "Vector3.cuh"
#include "Ray.cuh"
#include "RGBColor.cuh"

// Vector3

__global__ void toVector3(float * input, int outputCount, Vector3 * output) {
    int outputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (outputIndex < outputCount) {
        float * x = input + outputIndex*3;
        float * y = x + 1;
        float * z = y + 1;

        output[outputIndex] = Vector3(*x, *y, *z);
    }
}



__global__ void fromVector3(Vector3 * input, int inputCount, float * output) {
    int inputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (inputIndex < inputCount) {

        Vector3 * curr = input + inputIndex;

        output[inputIndex*3] = curr->x;
        output[inputIndex*3 + 1] = curr->y;
        output[inputIndex*3 + 2] = curr->z;
    }
}



// Ray
__global__ void toRay(float * input, int outputCount, Ray * output) {
    int outputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (outputIndex < outputCount) {
        float * origin = input + outputIndex*6;
        float * direction = origin + 3;

        Ray * ray = output + outputIndex;
        Vector3 vOrigin = Vector3(origin[0], origin[1], origin[2]);
        Vector3 vDirection = Vector3(direction[0], direction[1], direction[2]);

        *ray = Ray(vOrigin, vDirection);
    }
}

__global__ void fromRay(Ray * input, int inputCount, float * output) {
    int inputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (inputIndex < inputCount) {

        Ray * ray = input + inputIndex;
        float * origin = output + inputIndex*6;
        float * direction = origin + 3;

        origin[0] = ray->origin.x;
        origin[1] = ray->origin.y;
        origin[2] = ray->origin.z;

        direction[0] = ray->direction.x;
        direction[1] = ray->direction.y;
        direction[2] = ray->direction.z;
    }
}

// RGBColor

__global__ void toRGBColor(float * input, int outputCount, RGBColor * output) {
    int outputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (outputIndex < outputCount) {
        float * red = input + outputIndex*3;
        float * green = red + 1;
        float * blue = green + 1;

        output[outputIndex] = RGBColor(*red, *green, *blue);
    }
}

__global__ void fromRGBColor(RGBColor * input, int inputCount, float * output) {
    int inputIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (inputIndex < inputCount) {

        RGBColor * curr = input + inputIndex;

        output[inputIndex*3] = curr->red;
        output[inputIndex*3 + 1] = curr->green;
        output[inputIndex*3 + 2] = curr->blue;
    }
}


