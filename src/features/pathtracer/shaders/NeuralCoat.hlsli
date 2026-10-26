// The clear coat's neural inner part (kCoatNeural, docs/architecture.md §8.17): a small MLP that
// corrects the approximation (CoatInnerApproximate) towards what the random walk (CoatInnerWalk)
// estimates, trained on the CPU from walks (NeuralCoat.cpp) and evaluated here by the GPU's kernels
// and the CPU's. Included by Bsdf.hlsli with AIR_BSDF_COATS; shared by HLSL and C++.
//
// The including code defines, before including Bsdf.hlsli:
//   float4 NeuralCoatWeights4(uint index)   parameters index .. index + 3 (index a multiple of 4)
//
// Network: kNeuralCoatInputs inputs (the first held at 1), two hidden layers of kNeuralCoatHidden
// ReLUs (the second with biases), three linear outputs z; the inner part is the approximation times
// exp(z) per colour channel. Parameters (float32): W0 [hidden][inputs], W1 [hidden][hidden], b1,
// W2 [4][hidden] (rows 0-2 used), b2 [4].
#ifndef AIR_PT_NEURAL_COAT_HLSLI
#define AIR_PT_NEURAL_COAT_HLSLI

AIR_CONST uint kNeuralCoatInputs = 32u;
AIR_CONST uint kNeuralCoatHidden = 16u;
AIR_CONST uint kNeuralCoatW0 = 0u;
AIR_CONST uint kNeuralCoatW1 = kNeuralCoatHidden * kNeuralCoatInputs;
AIR_CONST uint kNeuralCoatB1 = kNeuralCoatW1 + kNeuralCoatHidden * kNeuralCoatHidden;
AIR_CONST uint kNeuralCoatW2 = kNeuralCoatB1 + kNeuralCoatHidden;
AIR_CONST uint kNeuralCoatB2 = kNeuralCoatW2 + 4u * kNeuralCoatHidden;
AIR_CONST uint kNeuralCoatParameters = kNeuralCoatB2 + 4u;
AIR_CONST float kNeuralCoatMaxCorrection = 6.0f; // |z|: the correction stays within e^-6 .. e^6

// The inputs, four per float4.
struct NeuralCoatInput
{
    float4 v[8];
};

// The network's view of a shading situation: the directions outside and refracted into the layer,
// their half vectors and azimuth, how far the base's lobe peaks there, and the material.
NeuralCoatInput NeuralCoatEncode(BsdfMaterial m, float3 wo, float3 wi)
{
    float3 wo2 = CoatRefracted(wo);
    float3 wi2 = CoatRefracted(wi);
    float3 h = normalize(wo + wi);
    float3 h2 = normalize(wo2 + wi2);
    float lo = sqrt(wo.x * wo.x + wo.y * wo.y);
    float li = sqrt(wi.x * wi.x + wi.y * wi.y);
    float cosPhi = lo > 1e-4f && li > 1e-4f ? (wo.x * wi.x + wo.y * wi.y) / (lo * li) : 1.0f;
    float alpha = RoughnessToAlpha(m.roughness);
    float peak = log(1.0f + GgxD(h2, alpha)) * 0.125f;
    float spread = min((1.0f - h2.z) / (alpha + 0.01f), 4.0f);
    NeuralCoatInput x;
    x.v[0] = float4(1.0f, wo.z, wi.z, h.z);
    x.v[1] = float4(dot(wo, h), wo2.z, wi2.z, h2.z);
    x.v[2] = float4(dot(wo2, h2), cosPhi, peak, spread);
    x.v[3] = float4(m.baseColor.x, m.baseColor.y, m.baseColor.z, m.metallic);
    x.v[4] = float4(m.coatColor.x, m.coatColor.y, m.coatColor.z, m.coatRoughness);
    x.v[5] = float4(m.roughness, wo.z * wi.z, sin(kPi * wo2.z), sin(kPi * wi2.z));
    x.v[6] = float4(cos(kPi * wo2.z), cos(kPi * wi2.z), sin(kPi * h2.z), cos(kPi * h2.z));
    x.v[7] = float4(sin(kTwoPi * h2.z), cos(kTwoPi * h2.z), sin(kPi * cosPhi), cos(kPi * cosPhi));
    return x;
}

// Written out without loops: DXC's debug information misplaces the variables of unrolled loops in
// inlined functions, which the driver's verifier then prints at every pipeline creation.

// The hidden layers' activations, four per float4.
struct NeuralCoatHiddenLayer
{
    float4 v[4];
};

// The row of 32 weights at `row` times the inputs.
float NeuralCoatInputRow(uint row, NeuralCoatInput x)
{
    return dot(NeuralCoatWeights4(row), x.v[0]) + dot(NeuralCoatWeights4(row + 4u), x.v[1]) +
           dot(NeuralCoatWeights4(row + 8u), x.v[2]) + dot(NeuralCoatWeights4(row + 12u), x.v[3]) +
           dot(NeuralCoatWeights4(row + 16u), x.v[4]) + dot(NeuralCoatWeights4(row + 20u), x.v[5]) +
           dot(NeuralCoatWeights4(row + 24u), x.v[6]) + dot(NeuralCoatWeights4(row + 28u), x.v[7]);
}

// The row of 16 weights at `row` times a hidden layer.
float NeuralCoatHiddenRow(uint row, NeuralCoatHiddenLayer h)
{
    return dot(NeuralCoatWeights4(row), h.v[0]) + dot(NeuralCoatWeights4(row + 4u), h.v[1]) +
           dot(NeuralCoatWeights4(row + 8u), h.v[2]) + dot(NeuralCoatWeights4(row + 12u), h.v[3]);
}

float4 NeuralCoatRelu(float4 s)
{
    return float4(max(s.x, 0.0f), max(s.y, 0.0f), max(s.z, 0.0f), max(s.w, 0.0f));
}

// Four neurons of the first hidden layer, from row r on.
float4 NeuralCoatFirstNeurons(uint r, NeuralCoatInput x)
{
    uint row = kNeuralCoatW0 + r * kNeuralCoatInputs;
    return NeuralCoatRelu(float4(NeuralCoatInputRow(row, x), NeuralCoatInputRow(row + 32u, x),
                                 NeuralCoatInputRow(row + 64u, x), NeuralCoatInputRow(row + 96u, x)));
}

// Four neurons of the second, from row r on.
float4 NeuralCoatSecondNeurons(uint r, NeuralCoatHiddenLayer h)
{
    uint row = kNeuralCoatW1 + r * kNeuralCoatHidden;
    return NeuralCoatRelu(NeuralCoatWeights4(kNeuralCoatB1 + r) +
                          float4(NeuralCoatHiddenRow(row, h), NeuralCoatHiddenRow(row + 16u, h),
                                 NeuralCoatHiddenRow(row + 32u, h), NeuralCoatHiddenRow(row + 48u, h)));
}

// The correction z for an input.
float3 NeuralCoatCorrection(NeuralCoatInput x)
{
    NeuralCoatHiddenLayer h0;
    h0.v[0] = NeuralCoatFirstNeurons(0u, x);
    h0.v[1] = NeuralCoatFirstNeurons(4u, x);
    h0.v[2] = NeuralCoatFirstNeurons(8u, x);
    h0.v[3] = NeuralCoatFirstNeurons(12u, x);
    NeuralCoatHiddenLayer h1;
    h1.v[0] = NeuralCoatSecondNeurons(0u, h0);
    h1.v[1] = NeuralCoatSecondNeurons(4u, h0);
    h1.v[2] = NeuralCoatSecondNeurons(8u, h0);
    h1.v[3] = NeuralCoatSecondNeurons(12u, h0);
    float4 b2 = NeuralCoatWeights4(kNeuralCoatB2);
    return float3(b2.x + NeuralCoatHiddenRow(kNeuralCoatW2, h1), b2.y + NeuralCoatHiddenRow(kNeuralCoatW2 + 16u, h1),
                  b2.z + NeuralCoatHiddenRow(kNeuralCoatW2 + 32u, h1));
}

float3 NeuralCoatInner(BsdfContext c, float3 wi)
{
    float3 z = NeuralCoatCorrection(NeuralCoatEncode(c.material, c.wo, wi));
    float k = kNeuralCoatMaxCorrection;
    z = float3(clamp(z.x, -k, k), clamp(z.y, -k, k), clamp(z.z, -k, k));
    return CoatInnerApproximate(c, wi) * exp(z);
}

#endif
