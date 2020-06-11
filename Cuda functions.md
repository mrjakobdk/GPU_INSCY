# Cuda functions

```c++
//copy to host
cudaMemcpy(h_dst, d_src, sizeof(int) * size, cudaMemcpyDeviceToHost);
//copy to device
cudaMemcpy(d_dst, h_src, sizeof(int) * size, cudaMemcpyHostToDevice);
```

```c++
//allocate memory
cudaMalloc(&d_array, size * sizeof(int));
```

```c++
//set memory
cudaMemset(d_array, value, size * sizeof(int));
```

```c++
//shared memory variable size
extern __shared__ int s[];
```

```c++
//shared memory fixed size
__shared__ int s[64];
```