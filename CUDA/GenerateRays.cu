#include <math.h>
#include "Vector3.cuh"
#include "Ray.cuh"


//TODO: Cleanup to use Vector3 objects for (u,v,w) cam direction, 
__global__ void generateRays(float * u, float * v, float * w, float * eyePoint, float * cameraDirection, float viewPlaneDistance, float horizontalRaySpacing, float verticalRaySpacing, mint hres, mint vres, mint pixelCount, mint orthographic, mint sampleCountRoot, float * randomValues, mint randomCount, Ray * rays) {
    
    int pixelIndex = threadIdx.x + blockIdx.x*blockDim.x;

	if (pixelIndex < pixelCount) {

        float * etp = (float*)malloc(3 * sizeof(float)); // Vector from the Eye To Pixel of this ray
        
        // Get the screen space coordinates (col,row)
        int col = pixelIndex % hres;
        int row = pixelIndex / vres; // Integer division
        
        int sampleCount = sampleCountRoot * sampleCountRoot;

        for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {

            // Get the view space coordinates for the pixel (xv, yv, zv)
            float xv = horizontalRaySpacing * (col - hres/2.0);
            float yv = verticalRaySpacing * (row - vres/2.0);
            float zv = -viewPlaneDistance; // Distance from the eye point to the view plane

            // Offset the view space coordinates by random amount in the section specified by the current sample
            int subCol = sampleIndex % sampleCountRoot;
            int subRow = sampleIndex / sampleCountRoot; // Integer Division
            float hSubSpacing = horizontalRaySpacing / sampleCountRoot;
            float vSubSpacing = verticalRaySpacing / sampleCountRoot;
            // float subSpacing = raySpacing / sampleCountRoot;

            // TODO: Have the initial xv and yv not have the +0.5 to put in center.  X DONE, keeping this in as reminder
            xv += subCol * hSubSpacing;
            yv += subRow * vSubSpacing;

            // randX and randY are pre-computed random values between 0-1.
            float randX = *(randomValues + ((pixelIndex + sampleIndex) % randomCount));
            float randY = *(randomValues + ((pixelIndex + sampleIndex+13) % randomCount));
            
            xv += hSubSpacing * randX;
            yv += vSubSpacing * randY;

            // Get the vectors from the eye to the pixel in world space
            // using the orthonormal basis (u,v,w) to convert the view space coordinates.
            // Vector3 eyeToPixel = Vector3(   xv*u[0] + yv*v[0] + zv*w[0],
            //                                 xv*u[1] + yv*v[1] + zv*w[1],
            //                                 xv*u[2] + yv*v[2] + zv*w[2]
            //                             );
            etp[0] = xv*u[0] + yv*v[0] + zv*w[0];
            etp[1] = xv*u[1] + yv*v[1] + zv*w[1];
            etp[2] = xv*u[2] + yv*v[2] + zv*w[2];

            // float * origin = rayOrigins + pixelIndex*3*sampleCount + sampleIndex*3;
            // float * direction = rayDirections + pixelIndex*3*sampleCount + sampleIndex*3;

            Ray * ray = rays + pixelIndex*sampleCount + sampleIndex;
            Vector3 origin;
            Vector3 direction;
            
            if (orthographic) {
                // Origins should be each individual pixel's position
                // Directions should all be the camera direction
                    
                // Pixel position = eye point + eye-to-pixel vector   
                // origin[0] = eyePoint[0] + etp[0];
                // origin[1] = eyePoint[1] + etp[1];
                // origin[2] = eyePoint[2] + etp[2];

                origin = Vector3();
                origin.x = eyePoint[0] + etp[0];
                origin.y = eyePoint[1] + etp[1];
                origin.z = eyePoint[2] + etp[2];
                    
                // Direction = Camera Direciton
                // direction[0] = cameraDirection[0];
                // direction[1] = cameraDirection[1];
                // direction[2] = cameraDirection[2];

                direction = Vector3();
                direction.x = cameraDirection[0];
                direction.y = cameraDirection[1];
                direction.z = cameraDirection[2];

            } else {
                // Origins are all the same (eye point)
                // Directions are from the eye point to the pixel (normalized)

                // Origin = Eye Point
                // origin[0] = eyePoint[0];
                // origin[1] = eyePoint[1];
                // origin[2] = eyePoint[2];

                origin = Vector3();
                origin.x = eyePoint[0];
                origin.y = eyePoint[1];
                origin.z = eyePoint[2];

                // Direction = Normalized Eye-To-Pixel Vector
                float magnitude = sqrtf(etp[0]*etp[0] + etp[1]*etp[1] + etp[2]*etp[2]);

                // direction[0] = etp[0] / magnitude;
                // direction[1] = etp[1] / magnitude;
                // direction[2] = etp[2] / magnitude;

                direction = Vector3();
                direction.x = etp[0] / magnitude;
                direction.y = etp[1] / magnitude;
                direction.z = etp[2] / magnitude;
                
            }
        
            *ray = Ray(origin, direction);
        
        }

        free(etp);
        
    }

}
