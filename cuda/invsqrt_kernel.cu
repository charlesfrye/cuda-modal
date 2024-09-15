__global__ void invsqrt_kernel(float *out, const float *X, long long size)
{
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        float x = X[i];
        float half_x = x * 0.5f;

        int ii = __float_as_int(x);  // evil floating point bit-level hacking
        ii = 0x5f3759df - (ii >> 1); // what the fuck?
        float guess = __int_as_float(ii);

        guess = guess * (1.5f - (half_x * guess * guess));

        out[i] = guess;
    }
}
