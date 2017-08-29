#include <stdio.h>

#include "cu_engine.h"


int main (void)
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);

  if (device_count == 0)
  {
    printf ("There are no available device(s) that support CUDA\n");
  }
  else
  {
    printf ("Detected %d CUDA Capable device(s)\n", device_count);
  }

  int dev = 0;
  int driver_version = 0; 
  int runtime_version = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp device_prop;
  CHECK(cudaGetDeviceProperties(&device_prop, dev));
  printf("Device %d: \"%s\"\n", dev, device_prop.name);

  cudaDriverGetVersion(&driver_version);
  cudaRuntimeGetVersion(&runtime_version);
  printf("  CUDA Driver Version / Runtime Version  %d.%d / %d.%d\n", driver_version/1000,
                                                                     (driver_version%100)/10,
                                                                     runtime_version/1000,
                                                                     (runtime_version%100)/10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n", device_prop.major, 
                                                                     device_prop.minor);

  return 0;
}
