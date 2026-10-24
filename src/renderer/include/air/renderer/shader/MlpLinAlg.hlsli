// The multilayer perceptron of MlpShared.h on the GPU's matrix units: Shader Model 6.10 linear
// algebra at thread scope (dx/linalg.h; each thread multiplies its own vector, the hardware gathers
// a wave's). Weights and activations are float16, products accumulate in at least float32 and are
// rounded to float16; outer products accumulate into float32 gradients. GPU only, cs_6_10 / lib_6_10.
//
// The matrices are MlpBuffers::matrices in the MulOptimal layout; multiplying by a transposed matrix
// loads the same memory as MulOptimalTranspose. Weight gradients go to MlpBuffers::gradients in the
// OuterProductOptimal layout through atomic InterlockedAccumulate.
#ifndef AIR_RENDERER_MLP_LINALG_HLSLI
#define AIR_RENDERER_MLP_LINALG_HLSLI

#include <dx/linalg.h>

#include "air/renderer/shader/MlpShared.h"

typedef vector<half, 64> MlpHalf;       // kMlpWidth
typedef vector<half, 16> MlpHalfOutput; // kMlpOutputs

typedef dx::linalg::Matrix<dx::linalg::ComponentType::F16, 64, 64, dx::linalg::MatrixUse::A,
                           dx::linalg::MatrixScope::Thread>
    MlpHiddenMatrix;
typedef dx::linalg::Matrix<dx::linalg::ComponentType::F16, 16, 64, dx::linalg::MatrixUse::A,
                           dx::linalg::MatrixScope::Thread>
    MlpOutputMatrix;
typedef dx::linalg::Matrix<dx::linalg::ComponentType::F16, 64, 16, dx::linalg::MatrixUse::A,
                           dx::linalg::MatrixScope::Thread>
    MlpOutputMatrixTransposed;

#ifdef AIR_RENDERER_MLP_HLSLI
// Between Mlp.hlsli's float32 vectors (the encodings write them) and float16 ones.
MlpHalf MlpToHalf(MlpVector x)
{
    MlpHalf h;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        h[i * 4u] = (half)x.v[i].x;
        h[i * 4u + 1u] = (half)x.v[i].y;
        h[i * 4u + 2u] = (half)x.v[i].z;
        h[i * 4u + 3u] = (half)x.v[i].w;
    }
    return h;
}

MlpVector MlpFromHalf(MlpHalf h)
{
    MlpVector x;
    AIR_UNROLL
    for (uint i = 0u; i < kMlpVector4s; ++i)
    {
        x.v[i] = float4(h[i * 4u], h[i * 4u + 1u], h[i * 4u + 2u], h[i * 4u + 3u]);
    }
    return x;
}
#endif

MlpHalf MlpLinAlgRelu(MlpHalf z)
{
    return max(z, (MlpHalf)0.0h);
}

// g where `activation` (a ReLU's output) is positive, 0 elsewhere.
MlpHalf MlpLinAlgReluGradient(MlpHalf g, MlpHalf activation)
{
    return select(activation > (MlpHalf)0.0h, g, (MlpHalf)0.0h);
}

// A forward pass, keeping the input of every matrix (the encoded input, then the hidden layers'
// activations) and the output.
struct MlpLinAlgForward
{
    MlpHalf input[kMlpMatrices];
    MlpHalfOutput output;
};

MlpLinAlgForward MlpLinAlgEvaluate(MlpBuffers b, MlpHalf x)
{
    ByteAddressBuffer matrices = ResourceDescriptorHeap[b.matrices];
    MlpLinAlgForward f;
    f.input[0] = x;
    MlpHiddenMatrix w0 = MlpHiddenMatrix::Load<dx::linalg::MatrixLayout::MulOptimal>(matrices, b.matrixOffset0, 0);
    f.input[1] = MlpLinAlgRelu(dx::linalg::Multiply<half>(w0, f.input[0]));
    MlpHiddenMatrix w1 = MlpHiddenMatrix::Load<dx::linalg::MatrixLayout::MulOptimal>(matrices, b.matrixOffset1, 0);
    f.input[2] = MlpLinAlgRelu(dx::linalg::Multiply<half>(w1, f.input[1]));
    MlpOutputMatrix w2 = MlpOutputMatrix::Load<dx::linalg::MatrixLayout::MulOptimal>(matrices, b.matrixOffset2, 0);
    f.output = dx::linalg::Multiply<half>(w2, f.input[2]);
    return f;
}

// Backpropagates the loss gradient dy (with respect to the outputs): accumulates every weight
// gradient (output gradient (outer) input) into the gradients and returns the gradient with respect
// to the encoded input.
MlpHalf MlpLinAlgBackpropagate(MlpBuffers b, MlpLinAlgForward f, MlpHalfOutput dy)
{
    ByteAddressBuffer matrices = ResourceDescriptorHeap[b.matrices];
    RWByteAddressBuffer gradients = ResourceDescriptorHeap[b.gradients];
    dx::linalg::OuterProduct<dx::linalg::ComponentType::F32>(dy, f.input[2]).InterlockedAccumulate(gradients,
                                                                                                  b.gradientOffset2);
    MlpOutputMatrixTransposed w2 =
        MlpOutputMatrixTransposed::Load<dx::linalg::MatrixLayout::MulOptimalTranspose>(matrices, b.matrixOffset2, 0);
    MlpHalf d1 = MlpLinAlgReluGradient(dx::linalg::Multiply<half>(w2, dy), f.input[2]);
    dx::linalg::OuterProduct<dx::linalg::ComponentType::F32>(d1, f.input[1]).InterlockedAccumulate(gradients,
                                                                                                  b.gradientOffset1);
    MlpHiddenMatrix w1 =
        MlpHiddenMatrix::Load<dx::linalg::MatrixLayout::MulOptimalTranspose>(matrices, b.matrixOffset1, 0);
    MlpHalf d0 = MlpLinAlgReluGradient(dx::linalg::Multiply<half>(w1, d1), f.input[1]);
    dx::linalg::OuterProduct<dx::linalg::ComponentType::F32>(d0, f.input[0]).InterlockedAccumulate(gradients,
                                                                                                  b.gradientOffset0);
    MlpHiddenMatrix w0 =
        MlpHiddenMatrix::Load<dx::linalg::MatrixLayout::MulOptimalTranspose>(matrices, b.matrixOffset0, 0);
    return dx::linalg::Multiply<half>(w0, d0);
}

#endif
