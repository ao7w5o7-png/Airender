// Cutouts (MaterialData::alphaCutoff, renderer::Material::alphaCutoff): where a material's colour
// texture has an alpha below its cutoff the surface is a hole, as for leaves, chains and decals. The
// instances of such materials are non-opaque in the acceleration structure (the others are forced
// opaque, so rays over them never stop for a candidate); every query that sees a non-opaque candidate
// asks AlphaCutsOut before committing it. Shared by the GPU's queries (SceneAccess.hlsli,
// PathTracerKernel.hlsli) and the CPU reference's BVH (ReferenceScene.h), through the hooks
// LoadInstance, LoadMaterial, LoadIndex, LoadVertex, LoadTextureSet and MaterialTextureSample.
#ifndef AIR_PT_CUTOUT_HLSLI
#define AIR_PT_CUTOUT_HLSLI

// Whether the ray passes through triangle `primitive` of `instance` at `barycentrics` (of vertices 1
// and 2); textureSets: IntegratorParams::textureSets.
bool AlphaCutsOut(uint textureSets, uint instance, uint primitive, float2 barycentrics)
{
    InstanceData data = LoadInstance(instance);
    MaterialData m = LoadMaterial(data.materialIndex);
    if (!(m.alphaCutoff > 0.0f) || m.textureSet == kInvalidIndex)
    {
        return false;
    }
    Vertex v0 = LoadVertex(data, LoadIndex(data, primitive * 3u));
    Vertex v1 = LoadVertex(data, LoadIndex(data, primitive * 3u + 1u));
    Vertex v2 = LoadVertex(data, LoadIndex(data, primitive * 3u + 2u));
    float b0 = 1.0f - barycentrics.x - barycentrics.y;
    float2 uv = (float2(v0.u, v0.v) * b0 + float2(v1.u, v1.v) * barycentrics.x + float2(v2.u, v2.v) * barycentrics.y) *
                m.uvScale;
    TextureSetData set = LoadTextureSet(textureSets, m.textureSet);
    return MaterialTextureSample(set.color, uv).w < m.alphaCutoff;
}

#endif
